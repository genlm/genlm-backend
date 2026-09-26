import os
import sys
import queue
import asyncio
import warnings
import weakref
import threading
import torch
import logging

from genlm.backend.llm.base import AsyncLM, UNKNOWN_ADAPTER


try:
    # In-process v1 engine with the V2 (MRv2) model runner. These env vars only
    # take effect if set before vllm is first imported in this process.
    if "vllm" in sys.modules:
        warnings.warn(
            "vllm was imported before genlm.backend.llm.vllm; engine-mode env "
            "vars may not take effect.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # flashinfer's sampler JIT needs CUDA_HOME at runtime.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # The capture sampler installs through MRv2's ModelState.custom_sampler hook,
    # so every architecture must run under MRv2.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    import vllm.v1.worker.gpu.model_runner as _gpu_model_runner

    HAS_VLLM = True
except ImportError:  # pragma: no cover
    HAS_VLLM = False  # pragma: no cover

if not HAS_VLLM:

    class AsyncVirtualLM:  # pragma: no cover
        """Placeholder class when vLLM is not installed."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm' "
                "to use the vLLM-based AsyncLM model."
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm' "
                "to use the vLLM-based AsyncLM model."
            )

else:
    logging.getLogger("vllm").setLevel(logging.WARNING)

    _REQ_PREFIX = "genlm-"
    _STOP = object()  # engine-thread shutdown sentinel

    def _drop_rows(rows):  # pragma: no cover
        """Drops rows captured after cleanup."""

    def _weak_resolve(inst):  # pragma: no cover
        """Route captured rows to ``inst`` without keeping it alive.

        A strong callback would close a cycle through the engine ``inst`` owns,
        so only ``gc`` could reclaim it.
        """
        ref = weakref.ref(inst)

        def resolve(rows):
            target = ref()
            if target is not None:
                target._resolve(rows)

        return resolve

    def _engine_loop(work, ref):  # pragma: no cover
        """Run the engine thread without owning its instance.

        Holding the instance across a wait would keep ``__del__``, which stops
        this thread, from ever running.
        """
        while True:
            item = work.get()
            inst = ref()
            if inst is None:
                return
            stop = inst._engine_step(item)
            del inst
            if stop:
                return

    class _CaptureSampler:  # pragma: no cover
        """Sampler installed in the model runner's sampler slot.

        One of this backend's requests gets its full-vocabulary logprob row
        captured and zero sampled tokens reported; every other request goes to
        the wrapped sampler.
        """

        def __init__(self, base, resolve):
            self._base = base
            self._resolve = resolve  # ({req_id: row}) -> None, one call per forward

        def __getattr__(self, name):
            return getattr(self._base, name)

        def __call__(self, logits, input_batch):
            req_ids = input_batch.req_ids
            n = input_batch.num_reqs
            cu = input_batch.cu_num_logits_np
            ours, rows_at = [], []
            for i in range(n):
                if req_ids[i].startswith(_REQ_PREFIX):
                    ours.append(i)
                    lo, hi = int(cu[i]), int(cu[i + 1])
                    if hi > lo:  # mid-prefill chunks produce no logits
                        rows_at.append((req_ids[i], hi - 1))
            if rows_at:
                # index_select copies out of vLLM's live logits buffer, which
                # the next forward overwrites.
                idx = torch.tensor([pos for _, pos in rows_at], device=logits.device)
                block = torch.log_softmax(logits.index_select(0, idx).float(), dim=-1)
                self._resolve({rid: block[j] for j, (rid, _) in enumerate(rows_at)})
            if len(ours) == n:
                return SamplerOutput(
                    sampled_token_ids=logits.new_zeros((n, 1), dtype=torch.int64),
                    logprobs_tensors=None,
                    num_nans=None,
                    num_sampled=logits.new_zeros(n, dtype=torch.int32),
                    num_rejected=logits.new_zeros(n, dtype=torch.int32),
                )
            out = self._base(logits, input_batch)
            if ours:
                idx = torch.tensor(ours, device=out.num_sampled.device)
                out.num_sampled = out.num_sampled.clone()
                out.num_sampled[idx] = 0
            return out

    class BackendScheduler(AsyncScheduler):  # pragma: no cover
        """Scheduler for requests whose tokens arrive from the caller, not the sampler.

        Fed tokens travel on the next ``SchedulerOutput`` to the runner wrap below.
        This backend's requests keep ``num_output_placeholders`` at zero, or async
        run-ahead accounting corrupts.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fed_tokens = {}  # appends awaiting the runner
            self._fed_seq = 0  # attaches stamped; the runner wrap acks each one
            self._fed_acked = 0

        def feed_token(self, request, token):
            request.append_output_token_ids(int(token))
            self._fed_tokens[request.request_id] = int(token)

        def schedule(self, *args, **kwargs):
            output = super().schedule(*args, **kwargs)
            if self._fed_tokens:
                # Async run-ahead leaves at most two attaches unacked; more
                # means the runner wrap is not running.
                self._fed_seq += 1
                if self._fed_acked < self._fed_seq - 2:
                    raise RuntimeError(
                        "the runner is not consuming fed tokens: the "
                        "update_requests wrap is not installed or not running "
                        "(decodes would silently forward token 0)"
                    )
                seq = self._fed_seq
                output.fed_tokens = self._fed_tokens
                output.fed_ack = lambda: self._fed_ack(seq)
                self._fed_tokens = {}
            return output

        def _fed_ack(self, seq):
            if seq > self._fed_acked:
                self._fed_acked = seq

        def _update_after_schedule(self, scheduler_output):
            super()._update_after_schedule(scheduler_output)
            for req_id in scheduler_output.num_scheduled_tokens:
                if req_id.startswith(_REQ_PREFIX):
                    request = self.requests.get(req_id)
                    if request is not None:
                        request.num_output_placeholders = 0

    # ``LLM(...)`` builds its ModelState through this function, so the sampler
    # must be installed here; ``from_name`` raises unless the capture comes back
    # ``armed``.
    _PENDING_CAPTURE = None
    _orig_init_model_state = _gpu_model_runner.init_model_state

    def _init_model_state_with_capture(vllm_config, model, encoder_cache, device):
        state = _orig_init_model_state(vllm_config, model, encoder_cache, device)
        capture = _PENDING_CAPTURE
        if capture is not None:
            orig_custom = state.custom_sampler

            def custom_sampler(sampler):
                capture["armed"] = True
                custom = orig_custom(sampler)
                base, rejection = custom if custom is not None else (sampler, None)
                resolve = lambda rows: capture["resolve"](rows)  # noqa: E731
                return _CaptureSampler(base, resolve), rejection

            state.custom_sampler = custom_sampler
        return state

    _gpu_model_runner.init_model_state = _init_model_state_with_capture

    # A decode reads its input token from the runner's last-sampled buffer, so
    # fed tokens must land there after the stock update and before input
    # preparation, or the forward consumes a stale zero.
    _orig_update_requests = _gpu_model_runner.GPUModelRunner.update_requests

    def _update_requests_with_appends(self, scheduler_output):
        _orig_update_requests(self, scheduler_output)
        appends = getattr(scheduler_output, "fed_tokens", None)
        if appends:
            states = self.req_states
            idxs, tokens = [], []
            for req_id, token in appends.items():
                idx = states.req_id_to_index.get(req_id)
                if idx is not None:
                    idxs.append(idx)
                    tokens.append(token)
            if idxs:
                device = states.last_sampled_tokens.device
                both = torch.tensor(idxs + tokens, dtype=torch.int64, device=device)
                k = len(idxs)
                states.last_sampled_tokens[both[:k]] = both[k:].unsqueeze(1)
            scheduler_output.fed_ack()

    _gpu_model_runner.GPUModelRunner.update_requests = _update_requests_with_appends

    class AsyncVirtualLM(AsyncLM):  # pragma: no cover
        """Async language model using vLLM v1 with live engine requests.

        Concurrent calls are auto-batched. A one-token extension of an idle
        context appends to that context's engine request, and identical
        contexts in one batch share one row.
        """

        def __init__(self, llm_engine):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                llm_engine (LLM): The vLLM engine instance.
            """
            self.llm_engine = llm_engine
            self.tokenizer = llm_engine.get_tokenizer()
            self._params = SamplingParams(
                n=1, max_tokens=1 << 20, ignore_eos=True, detokenize=False
            )

            # name -> LoRARequest. vLLM caches adapter weights by int id, so ids
            # are never reused. Loop-side only; the engine thread never reads this map.
            self._lora_requests = {}
            self._next_lora_id = 1

            # Only the engine thread reads or writes the state below.
            self._requests = {}  # rid -> (tuple(ids), lora_name), recency order
            self._by_content = {}  # (tuple(ids), lora_name) -> rid
            self._pending = {}  # rid -> [(future, loop)] awaiting a row or an exception
            self._served = 0  # rows resolved; the engine thread's progress signal
            self._next_rid = 0
            self._max_requests = 1 << 30  # tightened by from_name

            self._work = queue.SimpleQueue()
            self._engine_thread = None  # started by from_name once the engine is wired
            self._sched = None
            self._core = (
                None  # in-process engine-core client, stepped by the engine thread
            )
            self._block_hasher = None

            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_name, engine_opts=None, **kwargs):
            """Create a `AsyncVirtualLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load.
                engine_opts (dict): Additional options to pass to the `LLM` engine.
                **kwargs: Additional arguments passed to `AsyncVirtualLM` constructor.

            Returns:
                (AsyncVirtualLM): An `AsyncVirtualLM` instance.
            """
            if not HAS_VLLM:
                raise ImportError(  # pragma: no cover
                    "vLLM not available. Install vLLM or use AsyncTransformer instead."
                )

            engine_opts = {
                "enable_prefix_caching": True,
                "disable_log_stats": True,
                "gpu_memory_utilization": 0.9,
                "async_scheduling": True,
                "scheduler_cls": BackendScheduler,
                **(engine_opts or {}),
            }

            global _PENDING_CAPTURE
            capture = {"armed": False, "resolve": None}
            _PENDING_CAPTURE = capture
            try:
                llm = LLM(model=model_name, tokenizer=model_name, **engine_opts)
            finally:
                _PENDING_CAPTURE = None
            if not capture["armed"]:
                raise RuntimeError(
                    "capture sampler was not installed: vLLM's "
                    "ModelState.custom_sampler hook has moved"
                )

            inst = cls(llm, **kwargs)
            capture["resolve"] = _weak_resolve(inst)
            inst._capture = capture

            # The engine thread drives the engine core directly: LLMEngine's output
            # processor would reject requests it never registered.
            inst._core = llm.llm_engine.engine_core
            engine_core = inst._core.engine_core
            sched = engine_core.scheduler
            if not isinstance(sched, BackendScheduler):
                raise RuntimeError(
                    f"engine scheduler is {type(sched).__name__}, not BackendScheduler"
                )
            inst._sched = sched
            inst._block_hasher = engine_core.request_block_hasher
            inst._max_requests = max(
                8, llm.llm_engine.vllm_config.scheduler_config.max_num_seqs - 8
            )
            inst._engine_thread = threading.Thread(
                target=_engine_loop,
                args=(inst._work, weakref.ref(inst)),
                name="genlm-engine",
                daemon=True,
            )
            inst._engine_thread.start()
            return inst

        @property
        def underlying_model(self):
            """Access the underlying model for advanced use cases."""
            engine_core = self.llm_engine.llm_engine.engine_core.engine_core
            return engine_core.model_executor.driver_worker.worker.model_runner.model

        # -- LoRA -------------------------------------------------------------

        def add_new_lora(self, lora_path, lora_name="lora_1"):
            """Register a LoRA adapter under ``lora_name``.

            Re-registering an existing name rebinds it to the weights at
            ``lora_path``; forwards still pending under the old weights fail.
            Forwards select the adapter per call via ``lora_name=``.

            Args:
                lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
                lora_name (str): Name to assign to the loaded adapter.
            """
            if lora_name in self._lora_requests:
                del self._lora_requests[lora_name]
                self._work.put(("abort_adapter", lora_name, None, None))
            lid = self._next_lora_id
            self._next_lora_id += 1
            self._lora_requests[lora_name] = LoRARequest(lora_name, lid, lora_path)

        def remove_lora(self, lora_name):
            """Unregister ``lora_name`` and evict its weights. Forwards still pending
            under it fail.

            Args:
                lora_name (str): Name of the adapter to remove.
            """
            req = self._lora_requests.pop(lora_name)
            self._work.put(("remove_lora", (lora_name, req.lora_int_id), None, None))

        def _abort_adapter(self, lora_name):
            """(engine thread) Abort every request under ``lora_name``."""
            self._abort(
                [rid for rid, (_, name) in self._requests.items() if name == lora_name],
                RuntimeError(
                    f"adapter {lora_name!r} was rebound or removed mid-forward"
                ),
            )

        # -- the batch --------------------------------------------------------

        async def next_token_logprobs(self, token_ids, lora_name=None):
            """Request log probabilities of next token asynchronously with auto-batching.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """
            rows = await self.batch_next_token_logprobs(
                [token_ids], lora_name=lora_name
            )
            return rows[0]

        async def batch_next_token_logprobs(self, token_ids_list, lora_name=None):
            """Batch request log probabilities for multiple token sequences asynchronously.

            Concurrent callers, batched or single, are dispatched together as
            one batch.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                (torch.Tensor): A ``[N, vocab]`` tensor of normalized log probabilities.
            """
            self._check_alive()
            if lora_name is not None and lora_name not in self._lora_requests:
                raise ValueError(UNKNOWN_ADAPTER.format(lora_name))
            if any(not token_ids for token_ids in token_ids_list):
                raise ValueError("token_ids must not be empty")

            loop = asyncio.get_running_loop()
            futures = [loop.create_future() for _ in token_ids_list]
            batch = await self._join_batch(
                [
                    ((tuple(token_ids), lora_name), loop, future)
                    for token_ids, future in zip(token_ids_list, futures)
                ]
            )
            if batch is not None:
                queries = self._bind_adapters(batch)
                if queries:
                    self._work.put(("queries", queries, None, None))
            return torch.stack(await asyncio.gather(*futures))

        def _check_alive(self):
            if self._engine_thread is None:
                raise RuntimeError(
                    "engine thread not running: this model was cleaned up"
                    if getattr(self, "_engine_cleaned", False)
                    else "engine thread not running; construct via from_name()"
                )

        def _bind_adapters(self, batch):
            """Resolve each query's adapter as the batch is dispatched.

            Nothing may yield between this and the ``_work`` put, or a rebind
            could land between them.
            """
            bound = []
            for key, loop, future in batch:
                lora_name = key[1]
                if lora_name is None:
                    bound.append((key, None, loop, future))
                elif lora_name in self._lora_requests:
                    bound.append((key, self._lora_requests[lora_name], loop, future))
                else:  # removed while this query sat in the batch
                    self._resolve(
                        loop, future, exc=ValueError(UNKNOWN_ADAPTER.format(lora_name))
                    )
            return bound

        # -- the engine thread --------------------------------------------------

        def _engine_step(self, item):
            """(engine thread) Handle one work item, then step the engine until no
            row is pending. A failure goes to every pending future. Returns whether
            the thread should stop."""
            stopping = item is _STOP
            try:
                if not stopping:
                    self._handle(item)
                stalled = 0
                steps = 0
                while not stopping and self._pending:
                    served = self._served
                    self._core.get_output()
                    steps += 1
                    while True:
                        try:
                            nxt = self._work.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is _STOP:
                            stopping = True
                            break
                        self._handle(nxt)
                        stalled = 0
                    if stopping:
                        break
                    if self._served == served:
                        stalled += 1
                        if stalled > 4096:
                            raise RuntimeError(
                                "engine made no progress on pending rows: "
                                f"{list(self._pending)}"
                            )
                    else:
                        stalled = 0
            except BaseException as exc:
                self._fail_pending(exc)
            if stopping:
                self._fail_pending(RuntimeError("backend was shut down"))
            return stopping

        def _handle(self, item):
            """(engine thread) Execute one work item. A barrier item resolves its
            own future with the result or the failure; a query batch leaves its
            futures in ``_pending`` for ``_resolve`` or ``_fail_pending``."""
            kind, arg, future, loop = item
            if kind == "queries":
                self._reconcile(arg)
                return
            try:
                if kind == "abort_adapter":
                    self._abort_adapter(arg)
                elif kind == "release":
                    self._evict_idle(len(self._requests))
                elif kind == "remove_lora":
                    lora_name, lora_int_id = arg
                    self._abort_adapter(lora_name)
                    # One step flushes the finished ids through the runner
                    # before the weights they used disappear.
                    self._core.get_output()
                    self.llm_engine.llm_engine.remove_lora(lora_int_id)
            except BaseException as exc:
                if future is not None:
                    self._resolve(loop, future, exc=exc)
                raise
            if future is not None:
                self._resolve(loop, future, None)

        def _reconcile(self, batch):
            """(engine thread) Reconcile one batch against the request table, each
            context before its extensions. A failure fails every query in the batch."""
            try:
                self._reconcile_inner(batch)
            except BaseException as exc:
                for _, _, loop, future in batch:
                    self._resolve(loop, future, exc=exc)
                raise

        def _reconcile_inner(self, batch):
            self._evict_under_pressure()
            grouped = {}
            for key, lora_request, loop, future in batch:
                entry = grouped.setdefault(key, (lora_request, []))
                entry[1].append((future, loop))
            for key in sorted(grouped, key=lambda k: len(k[0])):
                lora_request, waiters = grouped[key]
                ids, lora_name = key
                rid = None
                # A request is extendable only while no row is pending on it: one
                # request cannot serve both a context and its extension.
                cand = self._by_content.get((ids[:-1], lora_name))
                if cand is not None and cand not in self._pending:
                    request = self._sched.requests.get(cand)
                    if request is None:
                        self._forget(cand)  # engine dropped it (e.g. preempt races)
                    else:
                        rid = cand
                        self._sched.feed_token(request, ids[-1])
                        self._rekey(rid, key)
                if rid is None:
                    rid = self._new_rid()
                    self._sched.add_request(
                        Request(
                            request_id=rid,
                            prompt_token_ids=list(ids),
                            sampling_params=self._params,
                            pooling_params=None,
                            lora_request=lora_request,
                            block_hasher=self._block_hasher,
                        )
                    )
                    self._admit(rid, key)
                self._pending[rid] = waiters

            self._evict_over_cap()

        def _new_rid(self):
            self._next_rid += 1
            return f"{_REQ_PREFIX}{self._next_rid}"

        # -- delivery -----------------------------------------------------------

        def _resolve(self, rows):
            """(engine thread) Capture callback: resolve the forward's pending
            futures, one callback per event loop."""
            by_loop = {}
            for req_id, row in rows.items():
                waiters = self._pending.pop(req_id, None)
                if not waiters:
                    continue
                self._served += 1
                for i, (future, loop) in enumerate(waiters):
                    by_loop.setdefault(loop, []).append(
                        (future, row if i == 0 else row.clone())
                    )
            for loop, items in by_loop.items():

                def _set(items=items):
                    for future, result in items:
                        if not future.done():
                            future.set_result(result)

                try:
                    loop.call_soon_threadsafe(_set)
                except RuntimeError:
                    pass  # the loop closed; the rows have no reader

        def _fail_pending(self, exc):
            """(engine thread) Abort every request with a pending row."""
            self._abort(list(self._pending), exc)

        def _fail_waiters(self, waiters, exc):
            for future, loop in waiters:
                self._resolve(loop, future, exc=exc)

        @staticmethod
        def _resolve(loop, future, result=None, exc=None):
            def _set():
                if not future.done():
                    if exc is not None:
                        future.set_exception(exc)
                    else:
                        future.set_result(result)

            try:
                loop.call_soon_threadsafe(_set)
            except RuntimeError:
                pass  # the loop closed; the result has no reader

        async def _submit(self, kind, arg=None):
            self._check_alive()
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._work.put((kind, arg, future, loop))
            return await future

        # -- the request table ------------------------------------------------

        def _abort(self, rids, exc):
            """(engine thread) Fail the futures pending on ``rids`` and drop the
            requests from the table and the engine.

            Never raises: it runs in the engine thread's failure handler, where an
            exception would kill the thread.
            """
            rids = list(rids)
            for rid in rids:
                waiters = self._pending.pop(rid, None)
                if waiters:
                    self._fail_waiters(waiters, exc)
                self._forget(rid)
            if rids:
                try:
                    self._sched.finish_requests(rids, RequestStatus.FINISHED_ABORTED)
                except BaseException:  # pragma: no cover
                    logging.getLogger(__name__).exception(
                        "could not finish aborted requests %s", rids
                    )

        def _admit(self, rid, key):
            """Register a live request at ``key``. A newcomer takes the content
            index; a displaced incumbent stays in the table as an idle row."""
            self._requests[rid] = key
            self._by_content[key] = rid

        def _rekey(self, rid, key):
            """Re-key an extended request (re-insertion keeps recency order)."""
            self._forget(rid)
            self._admit(rid, key)

        def _forget(self, rid):
            """Drop ``rid`` from the table and, if it still holds it, the index."""
            key = self._requests.pop(rid, None)
            if key is not None and self._by_content.get(key) == rid:
                del self._by_content[key]

        def _evict_idle(self, n):
            """(engine thread) Finish up to ``n`` oldest requests with no pending row."""
            victims = []
            for rid in self._requests:
                if len(victims) >= n:
                    break
                if rid not in self._pending:
                    victims.append(rid)
            for rid in victims:
                self._forget(rid)
            if victims:
                self._sched.finish_requests(victims, RequestStatus.FINISHED_ABORTED)

        def _evict_over_cap(self):
            over = len(self._requests) - self._max_requests
            if over > 0:
                self._evict_idle(over)

        def _evict_under_pressure(self):
            """(engine thread) Evict the oldest idle requests when KV-cache usage reaches 90%."""
            if self._sched.kv_cache_manager.usage >= 0.9:
                self._evict_idle(max(1, len(self._requests) // 8))

        async def release_all(self):
            """Evict every idle request (end of an inference run)."""
            await self._submit("release")

        # -- sync paths -----------------------------------------------------

        def next_token_logprobs_sync(self, token_ids, lora_name=None):
            """Request log probabilities of next token synchronously.

            Does not support auto-batching. For batched sync calls, use
            ``batch_next_token_logprobs_sync`` instead.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            return asyncio.run(self.next_token_logprobs(token_ids, lora_name=lora_name))

        def batch_next_token_logprobs_sync(self, token_ids_list, lora_name=None):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            return asyncio.run(
                self.batch_next_token_logprobs(token_ids_list, lora_name=lora_name)
            )

        def cleanup(self):
            """Explicitly clean up GPU resources. Call this when done with the model."""
            self._cleanup_engine()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()
            return False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()
            return False

        def __del__(self):
            """Clean up resources on deletion."""
            self._cleanup_engine()

        def _cleanup_engine(self):
            """Clean up the vLLM engine and associated resources.

            This is invoked from both :meth:`cleanup` (explicit, during normal
            program flow) and :meth:`__del__` (implicit, possibly at
            interpreter shutdown). The narrow exception classes below cover
            the races and idempotency issues we know about:

            * ``ImportError`` / ``AttributeError`` arise when ``__del__`` runs
              after ``sys.meta_path`` is already torn down during interpreter
              shutdown.
            * ``AssertionError`` is raised by vLLM's
              ``destroy_distributed_environment`` if it's called twice.
            * ``RuntimeError`` can surface from CUDA when the driver is
              already being torn down.

            Anything else is re-raised so real bugs are not swallowed.
            """
            if getattr(self, "_engine_cleaned", False):
                return
            self._engine_cleaned = True
            thread = getattr(self, "_engine_thread", None)
            if thread is not None and thread.is_alive():
                self._work.put(_STOP)
                thread.join(timeout=10)
            self._engine_thread = None
            # Only release what the engine thread can no longer reach: a thread
            # that outlived its join would fault on the handles this drops.
            released = thread is None or not thread.is_alive()
            try:
                import gc

                capture = getattr(self, "_capture", None)
                if capture is not None:
                    capture["resolve"] = _drop_rows
                if released:
                    self._sched = self._core = self._block_hasher = None
                    self.llm_engine = None
                # vLLM's internals are cyclic; only a collection frees them.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                destroy_model_parallel()
                destroy_distributed_environment()
            except (ImportError, AttributeError, AssertionError, RuntimeError):
                pass
