"""vLLM backend: a batched next-token-logprobs server over resident engine
requests.

The public surface is ``next_token_logprobs`` / ``batch_next_token_logprobs``;
concurrent calls meet in an autobatch window and execute as one batch. Behind
the window, each distinct growing context holds a resident engine request:
extending a context by one token appends that token to its request (a request
whose next token hasn't arrived is skipped by the scheduler — resident and
free), and the full-vocabulary row for every step leaves through a capture
shim in the model runner's sampler slot. All engine interaction — residency,
eviction, stepping — happens on one backend-owned crank thread; the in-process
engine runs ``schedule()`` on its caller's thread, so that state is
single-threaded by construction. Callers only ever await rows — or the
exception that took them.
"""

import os
import sys
import queue
import asyncio
import warnings
import threading
import torch
import logging
from collections import Counter

from genlm.backend.llm.base import AsyncLM


try:
    # In-process v1 engine with the V2 (MRv2) model runner. These env vars must
    # be set BEFORE vllm is imported for the first time in this process.
    if "vllm" in sys.modules:
        warnings.warn(
            "vllm was imported before genlm.backend.llm.vllm; engine-mode env "
            "vars may not take effect.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # flashinfer's JIT needs CUDA_HOME at runtime; compute nodes lack it.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # One runner, one seam: the capture shim installs via MRv2's
    # ModelState.custom_sampler hook, so force MRv2 for every architecture.
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

    _REQ_PREFIX = "resident-"
    _STOP = object()  # crank-shutdown sentinel

    class _CaptureSampler:  # pragma: no cover
        """Sits in the model runner's sampler slot. For resident requests it
        captures the full-vocabulary log-probability row (device-resident) and
        reports zero sampled tokens — the scheduler then appends nothing and
        the request idles until its next token arrives from the caller. All
        other requests defer verbatim to the wrapped stock sampler."""

        def __init__(self, base, deliver):
            self._base = base
            self._deliver = deliver  # ({req_id: row}) -> None, one call per frame

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
                    if hi > lo:
                        # The last position's row is the next-token distribution
                        # at the request's current context (mid prefill chunks
                        # produce no logits and are skipped).
                        rows_at.append((req_ids[i], hi - 1))
            if rows_at:
                # One gather + one normalize for the whole frame; the copy also
                # un-aliases vLLM's live logits buffer, which the next forward
                # overwrites. Delivered rows are views of this block.
                idx = torch.tensor([pos for _, pos in rows_at], device=logits.device)
                block = torch.log_softmax(logits.index_select(0, idx).float(), dim=-1)
                self._deliver({rid: block[j] for j, (rid, _) in enumerate(rows_at)})
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
        """Two duties for caller-fed requests, both consequences of their
        tokens arriving from the caller instead of the sampler.

        ``feed_token`` runs on the engine thread between steps; each
        appended token also rides the next ``SchedulerOutput`` for a
        worker-side wrap to write into the runner's last-sampled buffer —
        that buffer, not the request's token list, is where a decode step
        reads its input token. Miss it and every decode forwards token 0
        while all scheduler-side bookkeeping looks healthy. And
        ``num_output_placeholders`` is zeroed for resident requests every
        schedule, or async run-ahead accounting corrupts.
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
                # The wrap acks after writing the last-sampled buffer. Allow two
                # outstanding attaches (async run-ahead executes behind the
                # schedule); further lag means the wrap is not running and every
                # decode is forwarding token 0.
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

    # from_name needs the shim installed while ``LLM(...)`` constructs the
    # model runner; the runner reaches its ModelState through this module
    # function, wrapped once here. The box carries the deliver callback and
    # an "armed" flag from_name asserts on — the hook moving in a vLLM
    # upgrade must fail loudly, not serve sampled tokens silently.
    _PENDING_CAPTURE = None
    _orig_init_model_state = _gpu_model_runner.init_model_state

    def _init_model_state_with_capture(vllm_config, model, encoder_cache, device):
        state = _orig_init_model_state(vllm_config, model, encoder_cache, device)
        box = _PENDING_CAPTURE
        if box is not None:
            orig_custom = state.custom_sampler

            def custom_sampler(sampler):
                box["armed"] = True
                custom = orig_custom(sampler)
                base, rejection = custom if custom is not None else (sampler, None)
                deliver = lambda rows: box["deliver"](rows)  # noqa: E731
                return _CaptureSampler(base, deliver), rejection

            state.custom_sampler = custom_sampler
        return state

    _gpu_model_runner.init_model_state = _init_model_state_with_capture

    # The runner reads a generated position's input token from its per-request
    # last-sampled buffer, which the capture sampler leaves untouched. Appends
    # ride the SchedulerOutput; writing them here — after the stock request
    # updates, before input preparation — makes the forward consume the
    # appended token instead of a stale zero.
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
                states.last_sampled_tokens[
                    torch.tensor(idxs, dtype=torch.int64, device=device)
                ] = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(1)
            scheduler_output.fed_ack()

    _gpu_model_runner.GPUModelRunner.update_requests = _update_requests_with_appends

    class AsyncVirtualLM(AsyncLM):  # pragma: no cover
        """Batched logprobs server over resident vLLM requests.

        Concurrent ``next_token_logprobs`` calls collect in an autobatch
        window (await-0 drain: the window fires when a full event-loop pass
        adds no new asks) and land on the crank thread as one cohort. The
        crank reconciles each cohort against the residency table — an exact
        one-token extension of an idle resident appends that token; anything
        else births a request — then steps the engine until every owed row
        has been captured; cohorts arriving mid-crank ride the running
        frames. Identical contexts in one cohort share one row.
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

            # name -> LoRARequest; every forward selects its adapter per request
            # via ``lora_name`` (``None`` = base). Ids are monotonic: vLLM caches
            # adapter weights by int id, so an id must never be reused for
            # different weights (re-registering a name gets a fresh id). Asks
            # resolve their LoRARequest loop-side, so the crank never reads
            # this map.
            self._lora_requests = {}
            self._next_lora_id = 1

            # Loop-side window state.
            self._queries = []
            self._window_armed = False

            # Crank-owned state: the crank thread is the only reader and
            # writer of everything below (the in-process engine runs
            # schedule() and the capture shim on that same thread).
            # Batching breadcrumbs: ("cohort", n)/("steps", n) histograms plus
            # "appends"/"births"/"shared" totals. Crank-thread writes; snapshot
            # via take_stats().
            self.stats = Counter()

            self._requests = {}  # rid -> (tuple(ids), lora_name), recency order
            self._by_content = {}  # (tuple(ids), lora_name) -> rid
            self._pending = {}  # rid -> [(future, loop)]; row or exception, never silence
            self._served = 0  # rows delivered; the crank's progress signal
            self._next_rid = 0
            self._max_residents = 1 << 30  # tightened by from_name

            self._work = queue.SimpleQueue()
            self._crank = None  # started by from_name once the engine is wired
            self._sched = None
            self._core = None  # in-process engine-core client; the crank turns it
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
            box = {"armed": False, "deliver": None}
            _PENDING_CAPTURE = box
            try:
                llm = LLM(model=model_name, tokenizer=model_name, **engine_opts)
            finally:
                _PENDING_CAPTURE = None
            if not box["armed"]:
                raise RuntimeError(
                    "capture sampler was not installed: vLLM's "
                    "ModelState.custom_sampler hook has moved"
                )

            inst = cls(llm, **kwargs)
            box["deliver"] = inst._deliver

            # The crank turns the engine core directly: our requests emit no
            # tokens, so LLMEngine's output processor has nothing to do and
            # would only reject requests it never registered.
            inst._core = llm.llm_engine.engine_core
            engine_core = inst._core.engine_core
            sched = engine_core.scheduler
            if not isinstance(sched, BackendScheduler):
                raise RuntimeError(
                    f"engine scheduler is {type(sched).__name__}, not BackendScheduler"
                )
            inst._sched = sched
            inst._block_hasher = engine_core.request_block_hasher
            inst._max_residents = max(
                8, llm.llm_engine.vllm_config.scheduler_config.max_num_seqs - 8
            )
            inst._crank = threading.Thread(
                target=inst._crank_loop, name="crank", daemon=True
            )
            inst._crank.start()
            return inst

        @property
        def underlying_model(self):
            """Access the underlying model for advanced use cases."""
            engine_core = self.llm_engine.llm_engine.engine_core.engine_core
            return engine_core.model_executor.driver_worker.worker.model_runner.model

        # -- LoRA -------------------------------------------------------------

        def add_new_lora(self, lora_path, lora_name="lora_1"):
            """Register a LoRA adapter under ``lora_name``.

            Re-registering an existing name purges the name's requests (their
            KV came from the old weights) and binds ``lora_path`` under a
            fresh id — a training loop pushes updated weights with this one
            call. Forwards select the adapter per call via ``lora_name=``.
            The purge is queued before the rebind returns, so any later ask
            under the name births against the new weights.

            Args:
                lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
                lora_name (str): Name to assign to the loaded adapter.
            """
            if lora_name in self._lora_requests:
                del self._lora_requests[lora_name]
                self._work.put(("purge", lora_name, None, None))
            lid = self._next_lora_id
            self._next_lora_id += 1
            self._lora_requests[lora_name] = LoRARequest(lora_name, lid, lora_path)

        async def remove_lora(self, lora_name):
            """Unregister ``lora_name`` and evict its weights from the engine.
            The adapter's requests are reaped first — their KV would outlive
            the weights that produced it."""
            req = self._lora_requests.pop(lora_name)
            await self._run_on_crank("remove_lora", (lora_name, req.lora_int_id))

        def _purge_adapter(self, lora_name):
            """(crank) Drop every request under ``lora_name``: fail rows still
            owed (the weights are changing under them) and finish the engine
            requests."""
            stale = [
                rid for rid, (_, name) in self._requests.items() if name == lora_name
            ]
            exc = RuntimeError(
                f"adapter {lora_name!r} was rebound or removed mid-forward"
            )
            for rid in stale:
                waiters = self._pending.pop(rid, None)
                if waiters:
                    self._fail_waiters(waiters, exc)
                self._forget(rid)
            if stale:
                self._sched.finish_requests(stale, RequestStatus.FINISHED_ABORTED)

        # -- the window -------------------------------------------------------

        async def next_token_logprobs(self, token_ids, lora_name=None):
            """Request log probabilities of next token asynchronously with auto-batching.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.
                lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """
            rows = await self.batch_next_token_logprobs(
                [token_ids], lora_name=lora_name
            )
            return rows[0]

        async def batch_next_token_logprobs(self, token_ids_list, lora_name=None):
            """Batch request log probabilities for multiple token sequences.

            The whole batch enters the window as one set of asks; concurrent
            callers (batched or single) meet there and land on the crank as one
            cohort. The window is held open by its first caller until a full
            event-loop pass adds no new ask -- callers reach their asks at
            different depths of a `gather` tree, and each level is another
            scheduler turn.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists.
                lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

            Returns:
                (torch.Tensor): A ``[N, vocab]`` tensor of normalized log probabilities.
            """
            if self._crank is None:
                raise RuntimeError(
                    "engine crank not running; construct via from_name()"
                )
            if lora_name is None:
                lora_request = None
            elif lora_name in self._lora_requests:
                lora_request = self._lora_requests[lora_name]
            else:
                raise ValueError(f"unknown LoRA adapter: {lora_name!r}")
            if any(not token_ids for token_ids in token_ids_list):
                raise ValueError("token_ids must not be empty")

            loop = asyncio.get_running_loop()
            futures = []
            for token_ids in token_ids_list:
                future = loop.create_future()
                futures.append(future)
                self._queries.append(
                    ((tuple(token_ids), lora_name), lora_request, future, loop)
                )
            if not self._window_armed:
                self._window_armed = True
                try:
                    while True:
                        n = len(self._queries)
                        await asyncio.sleep(0)
                        if len(self._queries) == n:
                            break
                    cohort, self._queries = self._queries, []
                finally:
                    self._window_armed = False
                self._work.put(("asks", cohort, None, None))
            return torch.stack(await asyncio.gather(*futures))

        # -- the crank ----------------------------------------------------------

        def _crank_loop(self):
            """Single-flight by construction: the one thread that touches the
            engine. Handle a work item, step until no row is owed (handling
            items that arrive mid-crank between steps), sleep on the queue.
            Owns the pending contract — on any failure every owed future gets
            the exception; the thread survives for the next item."""
            while True:
                item = self._work.get()
                if item is _STOP:
                    self._fail_owed(RuntimeError("backend was shut down"))
                    return
                stopping = False
                try:
                    self._handle(item)
                    stalled = 0
                    steps = 0
                    while self._pending:
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
                                    "engine made no progress on owed rows: "
                                    f"{list(self._pending)}"
                                )
                        else:
                            stalled = 0
                    if steps:
                        self.stats[("steps", steps)] += 1
                except BaseException as exc:
                    self._fail_owed(exc)
                if stopping:
                    self._fail_owed(RuntimeError("backend was shut down"))
                    return

        def _handle(self, item):
            """(crank) Execute one work item. Barrier items resolve their own
            future — with the result or the failure; ask cohorts leave their
            futures in ``_pending`` for delivery or ``_fail_owed``."""
            kind, arg, future, loop = item
            if kind == "asks":
                self._reconcile(arg)
                return
            try:
                if kind == "purge":
                    self._purge_adapter(arg)
                elif kind == "release":
                    self._reap_idle(len(self._requests))
                elif kind == "remove_lora":
                    lora_name, lora_int_id = arg
                    self._purge_adapter(lora_name)
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

        def _reconcile(self, cohort):
            """(crank) One cohort against the residency table. Shortest first:
            a context is reconciled before any extension of it, so a chain
            (c, c+1, c+2) in one cohort resolves parent-first. A failure
            mid-cohort fails every ask in it — no future is left behind."""
            try:
                self._reconcile_inner(cohort)
            except BaseException as exc:
                for _, _, future, loop in cohort:
                    self._resolve(loop, future, exc=exc)
                raise

        def _reconcile_inner(self, cohort):
            self._reap_under_pressure()
            grouped = {}
            for key, lora_request, future, loop in cohort:
                entry = grouped.setdefault(key, (lora_request, []))
                entry[1].append((future, loop))
            self.stats[("cohort", len(cohort))] += 1
            self.stats["shared"] += len(cohort) - len(grouped)

            for key in sorted(grouped, key=lambda k: len(k[0])):
                lora_request, waiters = grouped[key]
                ids, lora_name = key
                rid = None
                # A resident is extendable only while it owes no row: a cohort
                # can hold both a context and its extension (a critic leaf
                # trails the draw leaf) and one request cannot serve both.
                cand = self._by_content.get((ids[:-1], lora_name))
                if cand is not None and cand not in self._pending:
                    request = self._sched.requests.get(cand)
                    if request is None:
                        self._forget(cand)  # engine dropped it (e.g. preempt races)
                    else:
                        rid = cand
                        self._sched.feed_token(request, ids[-1])
                        self._move(rid, key)
                        self.stats["appends"] += 1
                if rid is None:
                    rid = self._mint()
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
                    self.stats["births"] += 1
                self._pending[rid] = waiters

            self._reap_over_cap()

        def _mint(self):
            self._next_rid += 1
            return f"{_REQ_PREFIX}{self._next_rid}"

        # -- delivery -----------------------------------------------------------

        def _deliver(self, rows):
            """Capture-shim callback (crank thread, inside the engine step):
            resolve the frame's owed futures, one callback per event loop, so
            a whole population becomes runnable in the same loop pass."""
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

        def _fail_owed(self, exc):
            owed, self._pending = self._pending, {}
            for waiters in owed.values():
                self._fail_waiters(waiters, exc)

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

        async def _run_on_crank(self, kind, arg=None):
            if self._crank is None:
                raise RuntimeError(
                    "engine crank not running; construct via from_name()"
                )
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._work.put((kind, arg, future, loop))
            return await future

        # -- the residency table ----------------------------------------------

        def _admit(self, rid, key):
            """Register a live request at ``key``. A newcomer takes the content
            index; a displaced incumbent stays in the table as an idle row."""
            self._requests[rid] = key
            self._by_content[key] = rid

        def _move(self, rid, key):
            """Re-key an extended request (re-insertion keeps recency order)."""
            self._forget(rid)
            self._admit(rid, key)

        def _forget(self, rid):
            """Drop ``rid`` from the table and, if it still holds it, the index."""
            key = self._requests.pop(rid, None)
            if key is not None and self._by_content.get(key) == rid:
                del self._by_content[key]

        def _reap_idle(self, n):
            """(crank) Finish up to ``n`` oldest requests owing no row. Freed
            blocks keep their prefix-cache hashes, so a reaped path that
            returns rebirths with a cache-hit prefill."""
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
            return victims

        def _reap_over_cap(self):
            over = len(self._requests) - self._max_residents
            if over > 0:
                self._reap_idle(over)

        def _reap_under_pressure(self):
            """An idle request is CPU-free (the scheduler skips it) but pins
            its KV blocks; shed the oldest when the block pool runs hot."""
            if self._sched.kv_cache_manager.usage >= 0.9:
                self._reap_idle(max(1, len(self._requests) // 8))

        async def release_all(self):
            """Reap every idle request (end of an inference run)."""
            await self._run_on_crank("release")

        def take_stats(self):
            """Snapshot and reset the batching breadcrumbs."""
            stats, self.stats = self.stats, Counter()
            return stats

        # -- sync paths -----------------------------------------------------

        def next_token_logprobs_sync(self, token_ids, lora_name=None):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.
                lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """
            return asyncio.run(self.next_token_logprobs(token_ids, lora_name=lora_name))

        def batch_next_token_logprobs_sync(self, token_ids_list, lora_name=None):
            """Batch request log probabilities for multiple token sequences synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists.
                lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

            Returns:
                (torch.Tensor): A tensor of log probability tensors.
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
            """Stop the crank and tear down vLLM's distributed state. Runs from
            both :meth:`cleanup` and :meth:`__del__`, so it must be idempotent
            and survive interpreter shutdown: ``ImportError``/``AttributeError``
            arise when ``__del__`` runs after ``sys.meta_path`` is torn down,
            ``AssertionError`` when vLLM's teardown is called twice, and
            ``RuntimeError`` when CUDA is tearing down at the same time."""
            if getattr(self, "_engine_cleaned", False):
                return
            self._engine_cleaned = True
            crank = getattr(self, "_crank", None)
            if crank is not None and crank.is_alive():
                self._work.put(_STOP)
                crank.join(timeout=10)
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except (ImportError, AttributeError, AssertionError, RuntimeError):
                pass
