"""vLLM backend: a batched next-token-logprobs server over resident engine
requests.

The public surface is ``next_token_logprobs`` / ``batch_next_token_logprobs``;
concurrent calls meet in an autobatch window and execute as one batch. Behind
the window, each distinct growing context holds a resident engine request:
extending a context by one token appends that token to its request (a request
whose next token hasn't arrived is skipped by the scheduler — resident and
free), and the full-vocabulary row for every step leaves through a capture
shim in the model runner's sampler slot. The engine never waits on a caller;
callers only ever await rows.
"""

import os
import sys
import asyncio
import warnings
import threading
import torch
import logging

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache


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
    os.environ["VLLM_USE_V1"] = "1"
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

    _REQ_PREFIX = "genlm-"

    class _CaptureSampler:  # pragma: no cover
        """Sits in the model runner's sampler slot. For genlm requests it
        captures the full-vocabulary log-probability row (device-resident) and
        reports zero sampled tokens — the scheduler then appends nothing and
        the request idles until its next token arrives from the caller. All
        other requests defer verbatim to the wrapped stock sampler."""

        def __init__(self, base, deliver):
            self._base = base
            self._deliver = deliver  # (req_id, row) -> None, thread-safe

        def __getattr__(self, name):
            return getattr(self._base, name)

        def __call__(self, logits, input_batch):
            req_ids = input_batch.req_ids
            n = input_batch.num_reqs
            cu = input_batch.cu_num_logits_np
            ours = []
            for i in range(n):
                if req_ids[i].startswith(_REQ_PREFIX):
                    ours.append(i)
                    lo, hi = int(cu[i]), int(cu[i + 1])
                    if hi > lo:
                        # The last position's row is the next-token distribution
                        # at the request's current context (mid prefill chunks
                        # produce no logits and are skipped).
                        row = torch.log_softmax(logits[hi - 1].float(), dim=-1)
                        self._deliver(req_ids[i], row)
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

    class GenlmScheduler(AsyncScheduler):  # pragma: no cover
        """Owns genlm requests' life and death directly — no engine front door.

        Callers submit (births, appends, reaps) through a thread-safe inbox
        drained at the top of ``schedule()``, so all queue mutation happens on
        the engine's own thread. An appended token makes a request schedulable
        for exactly one step; unappended requests are skipped by the stock
        accounting (``num_new_tokens == 0``), resident and free. Placeholder
        discipline: genlm tokens never materialize from the sampler, so async
        scheduling's output placeholders are zeroed for them after every
        schedule — otherwise the run-ahead accounting would corrupt.
        """

        GENLM_PARAMS = (
            None  # set lazily; SamplingParams at class-def time races env setup
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._genlm_inbox = []
            self._genlm_lock = threading.Lock()
            self._genlm_ids = set()
            self.genlm_block_hasher = None  # bound by AsyncVirtualLM.from_name
            if GenlmScheduler.GENLM_PARAMS is None:
                GenlmScheduler.GENLM_PARAMS = SamplingParams(
                    n=1, max_tokens=1 << 20, ignore_eos=True, detokenize=False
                )

        def genlm_submit(self, births, appends, reaps):
            """births: (req_id, prompt_ids, lora_request); appends: (req_id,
            token); reaps: req_id. Thread-safe; applied before the next
            schedule pass."""
            with self._genlm_lock:
                self._genlm_inbox.append((births, appends, reaps))

        def has_requests(self):
            # The engine tests this before calling schedule(), so a submission
            # still in the inbox has to count as work or it would never drain.
            with self._genlm_lock:
                if self._genlm_inbox:
                    return True
            return super().has_requests()

        def schedule(self, *args, **kwargs):
            self._genlm_drain()
            return super().schedule(*args, **kwargs)

        def _genlm_drain(self):
            with self._genlm_lock:
                batch, self._genlm_inbox = self._genlm_inbox, []
            for births, appends, reaps in batch:
                for req_id, token in appends:
                    request = self.requests.get(req_id)
                    if request is not None:
                        request.append_output_token_ids(int(token))
                for req_id, prompt_ids, lora_request in births:
                    request = Request(
                        request_id=req_id,
                        prompt_token_ids=list(prompt_ids),
                        sampling_params=self.GENLM_PARAMS,
                        pooling_params=None,
                        lora_request=lora_request,
                        block_hasher=self.genlm_block_hasher,
                    )
                    self._genlm_ids.add(req_id)
                    self.add_request(request)
                if reaps:
                    live = [r for r in reaps if r in self.requests]
                    if live:
                        self.finish_requests(live, RequestStatus.FINISHED_ABORTED)
                    self._genlm_ids.difference_update(reaps)

        def _update_after_schedule(self, scheduler_output):
            super()._update_after_schedule(scheduler_output)
            for req_id in scheduler_output.num_scheduled_tokens:
                if req_id in self._genlm_ids:
                    request = self.requests.get(req_id)
                    if request is not None:
                        request.num_output_placeholders = 0

        def genlm_apply_now(self):
            """Apply queued submissions without waiting for a schedule pass. Only
            safe while no engine step is in flight (the caller owns the crank)."""
            self._genlm_drain()

        def genlm_kv_usage(self):
            """Fraction of the KV block pool in use (thread-safe int reads)."""
            pool = self.kv_cache_manager.block_pool
            return 1.0 - pool.get_num_free_blocks() / pool.num_gpu_blocks

    # from_name needs the shim installed while ``LLM(...)`` constructs the
    # model runner; the runner reaches its ModelState through this module
    # function, wrapped once here.
    _PENDING_DELIVER = None
    _orig_init_model_state = _gpu_model_runner.init_model_state

    def _init_model_state_with_capture(vllm_config, model, encoder_cache, device):
        state = _orig_init_model_state(vllm_config, model, encoder_cache, device)
        deliver = _PENDING_DELIVER
        if deliver is not None:
            orig_custom = state.custom_sampler

            def custom_sampler(sampler):
                custom = orig_custom(sampler)
                base, rejection = custom if custom is not None else (sampler, None)
                return _CaptureSampler(base, deliver), rejection

            state.custom_sampler = custom_sampler
        return state

    _gpu_model_runner.init_model_state = _init_model_state_with_capture

    class AsyncVirtualLM(AsyncLM):  # pragma: no cover
        """Batched logprobs server over resident vLLM requests.

        Concurrent ``next_token_logprobs`` calls collect in an autobatch
        window (await-0 drain: the window fires when a full event-loop pass
        adds no new asks; ``timeout`` adds a cooperative linger only while no
        crank is running). The executor diffs each context against its
        resident request — an exact one-token extension appends that token; a
        miss births a request — and cranks ``engine.step()`` on a worker
        thread until every asked row has been captured. Identical contexts in
        one window share one row.
        """

        def __init__(
            self,
            llm_engine,
            cache_size=0,
            cache_opts=None,
            timeout=0.0,
        ):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                llm_engine (LLM): The vLLM engine instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to None (no extra options).
                timeout (float, optional): Cooperative linger in seconds spent once per window while no engine crank is running, letting late concurrent callers join the batch. Defaults to 0.
            """
            self.llm_engine = llm_engine
            self.tokenizer = llm_engine.get_tokenizer()
            self.timeout = timeout
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )

            # name -> LoRARequest; every forward selects its adapter per request
            # via ``lora_name`` (``None`` = base). Ids are monotonic: vLLM caches
            # adapter weights by int id, so an id must never be reused for
            # different weights (re-registering a name gets a fresh id).
            self._lora_requests = {}
            self._next_lora_id = 1

            # (tuple(context ids), lora_name) -> resident request id, in
            # recency order (re-inserted on every touch) for LRU eviction.
            self._residents = {}
            self._extra_rids = []  # duplicate-content residents, reaped at release
            self._evictions = []
            self._max_residents = 1 << 30  # tightened by from_name
            self._next_rid = 0
            self._pending = {}  # req_id -> [asyncio futures]
            self._pending_lock = threading.Lock()
            self._loop = None
            self._queries = []
            self._window_armed = False
            self._cranking = False
            self._sched = None  # bound by from_name
            self._core = None  # in-process engine-core client; the crank turns it

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
                "scheduler_cls": GenlmScheduler,
                **(engine_opts or {}),
            }

            global _PENDING_DELIVER
            deliver_box = {}
            _PENDING_DELIVER = lambda req_id, row: deliver_box["fn"](req_id, row)  # noqa: E731
            try:
                llm = LLM(model=model_name, tokenizer=model_name, **engine_opts)
            finally:
                _PENDING_DELIVER = None

            inst = cls(llm, **kwargs)
            deliver_box["fn"] = inst._deliver

            # The crank turns the engine core directly: our requests emit no
            # tokens, so LLMEngine's output processor has nothing to do and
            # would only reject requests it never registered.
            inst._core = llm.llm_engine.engine_core
            engine_core = inst._core.engine_core
            sched = engine_core.scheduler
            if not isinstance(sched, GenlmScheduler):
                raise RuntimeError(
                    f"engine scheduler is {type(sched).__name__}, not GenlmScheduler"
                )
            sched.genlm_block_hasher = engine_core.request_block_hasher
            inst._sched = sched
            inst._max_residents = max(
                8, llm.llm_engine.vllm_config.scheduler_config.max_num_seqs - 8
            )
            return inst

        @property
        def underlying_model(self):
            """Access the underlying model for advanced use cases."""
            engine_core = self.llm_engine.llm_engine.engine_core.engine_core
            return engine_core.model_executor.driver_worker.worker.model_runner.model

        def add_new_lora(self, lora_path, lora_name="lora_1"):
            """Register a LoRA adapter under ``lora_name``.

            Re-registering an existing name evicts the old weights and binds the
            name to ``lora_path`` under a fresh id — a training loop pushes updated
            weights with this one call. Forwards select the adapter per call via
            ``lora_name=`` (or ``lora_view``).

            Args:
                lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
                lora_name (str): Name to assign to the loaded adapter.
            """
            if lora_name in self._lora_requests:
                self.remove_lora(lora_name)
            lid = self._next_lora_id
            self._next_lora_id += 1
            self._lora_requests[lora_name] = LoRARequest(lora_name, lid, lora_path)

        def remove_lora(self, lora_name):
            """Unregister ``lora_name`` and evict its weights from the engine.
            Residents under the adapter are reaped first: their KV would outlive
            the weights that produced it."""
            req = self._lora_requests.pop(lora_name)
            stale = [
                rid for (_, name), rid in self._residents.items() if name == lora_name
            ]
            for key in [k for k in self._residents if k[1] == lora_name]:
                del self._residents[key]
            if stale:
                self._sched.genlm_submit([], [], stale)
                self._sched.genlm_apply_now()
                self._core.get_output()
            self.llm_engine.llm_engine.remove_lora(req.lora_int_id)
            # OutputCache keys are (ids, lora_name); a later re-register under this
            # name must not serve the old adapter's logprobs.
            if self.cache is not None:
                self.cache.clear()

        def _lora_request_for(self, lora_name):
            """Per-request LoRARequest for ``lora_name`` (``None`` = base, LoRA off)."""
            return None if lora_name is None else self._lora_requests[lora_name]

        # -- the window -----------------------------------------------------

        async def next_token_logprobs(self, token_ids, lora_name=None):
            """Request log probabilities of next token asynchronously with auto-batching.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.
                lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """
            key = (tuple(token_ids), lora_name)
            if self.cache is not None and key in self.cache:
                return self.cache[key]

            self._loop = asyncio.get_running_loop()
            future = self._loop.create_future()
            self._queries.append((key, future))
            if not self._window_armed:
                self._window_armed = True
                try:
                    await self._collect()
                finally:
                    self._window_armed = False
                queries, self._queries = self._queries, []
                self._execute(queries)
            result = await future

            if self.cache is not None:
                self.cache[key] = result
            return result

        async def _collect(self):
            """Hold the window open: return once a full event-loop pass adds no
            new ask, after one cooperative linger (skipped while a crank is
            already turning — its frames do the batching).

            Run by the window's first caller, never a background task: window
            state must not outlive the loop the callers are on."""
            lingered = self._cranking or not self.timeout
            while True:
                n = len(self._queries)
                await asyncio.sleep(0)
                if len(self._queries) > n:
                    continue
                if lingered:
                    return
                lingered = True
                await asyncio.sleep(self.timeout)

        def _execute(self, queries):
            """Reconcile one window's contexts into the scheduler and make sure
            a crank is turning."""
            self._evict_under_pressure()
            grouped = {}
            for key, future in queries:
                grouped.setdefault(key, []).append(future)

            births, appends = [], []
            claimed = set()  # requests this window already speaks for
            for (ids, lora_name), futures in sorted(
                grouped.items(), key=lambda kv: len(kv[0][0])
            ):
                rid = None
                if ids:
                    # A resident is extendable once, and only when it owes no row:
                    # a window can hold both a context and its extension (a critic
                    # leaf trails the draw leaf), and one request cannot serve both.
                    cand = self._residents.get((ids[:-1], lora_name))
                    if (
                        cand is not None
                        and cand not in claimed
                        and cand not in self._pending
                    ):
                        del self._residents[(ids[:-1], lora_name)]
                        rid = cand
                        appends.append((rid, ids[-1]))
                        self._residents[(ids, lora_name)] = rid
                if rid is None:
                    rid = self._mint()
                    births.append((rid, ids, self._lora_request_for(lora_name)))
                    if (ids, lora_name) in self._residents:
                        self._extra_rids.append(rid)  # duplicate content
                    else:
                        self._residents[(ids, lora_name)] = rid
                        self._evict_over_cap()
                claimed.add(rid)
                with self._pending_lock:
                    self._pending[rid] = futures

            reaps = self._take_evictions()
            self._sched.genlm_submit(births, appends, reaps)
            if not self._cranking:
                asyncio.create_task(self._crank())

        def _mint(self):
            self._next_rid += 1
            return f"{_REQ_PREFIX}{self._next_rid}"

        def _evict_over_cap(self):
            while len(self._residents) > self._max_residents:
                if not self._evict_oldest_idle():
                    return

        def _evict_under_pressure(self):
            """Reap the oldest-recency residents when the KV block pool runs
            hot: an idle resident is CPU-free but pins its blocks. Freed blocks
            keep their prefix-cache hashes, so a reaped path that returns
            rebirths with a cache-hit prefill."""
            if self._sched is None or self._sched.genlm_kv_usage() < 0.9:
                return
            for _ in range(max(1, len(self._residents) // 8)):
                if not self._evict_oldest_idle():
                    return

        def _evict_oldest_idle(self):
            """Evict the oldest resident with no row in flight; False if none."""
            for key, rid in self._residents.items():
                if rid not in self._pending:
                    del self._residents[key]
                    self._evictions.append(rid)
                    return True
            return False

        def _take_evictions(self):
            evictions, self._evictions = self._evictions, []
            return evictions

        async def _crank(self):
            """Single-flight: step the engine off-loop until every pending row
            has been captured. Later windows piggyback on the running crank."""
            if self._cranking:
                return
            self._cranking = True
            try:
                while self._pending:
                    if not self._sched.has_requests():
                        # Rows are owed but the engine holds nothing for them:
                        # a request was lost, and stepping would spin forever.
                        with self._pending_lock:
                            owed = list(self._pending)
                            self._pending.clear()
                        raise RuntimeError(f"engine dropped requests {owed}")
                    await asyncio.to_thread(self._core.get_output)
            finally:
                self._cranking = False
            if self._pending:
                asyncio.create_task(self._crank())

        def _deliver(self, req_id, row):
            """Capture-shim callback (engine thread): hand ``req_id``'s row to
            its awaiting futures on the event loop."""
            with self._pending_lock:
                futures = self._pending.pop(req_id, None)
            if not futures or self._loop is None:
                return

            def _resolve():
                for i, f in enumerate(futures):
                    if not f.done():
                        f.set_result(row if i == 0 else row.clone())

            try:
                self._loop.call_soon_threadsafe(_resolve)
            except RuntimeError:
                pass  # the run's loop closed; the rows have no reader

        async def release_all(self):
            """Reap every resident request (end of an inference run). Waits out
            any turning crank so the reap lands on a quiet engine."""
            reaps = list(self._residents.values()) + self._extra_rids
            self._residents.clear()
            self._extra_rids = []
            if not reaps or self._sched is None:
                return
            self._sched.genlm_submit([], [], reaps)
            while self._cranking:
                await asyncio.sleep(0.001)
            self._sched.genlm_apply_now()
            self._core.get_output()  # carry the freed rows to the model runner

        def reset_async_queries(self):
            """Clear any pending queries from the queue.

            Use this method when an exception prevented an inference algorithm
            from executing to completion.
            """
            self._queries = []
            with self._pending_lock:
                self._pending.clear()

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

        def clear_cache(self):
            """Clear output cache."""
            if self.cache:
                self.cache.clear()

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
              tearing down at the same time.
            """
            if getattr(self, "_engine_cleaned", False):
                return
            self._engine_cleaned = True
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except (ImportError, AttributeError, AssertionError, RuntimeError):
                pass
