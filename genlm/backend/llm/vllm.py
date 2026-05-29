import os
import sys
import asyncio
import contextlib
import warnings
import torch
import logging
import threading
import hashlib
from collections import defaultdict

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache

try:
    # Enable vLLM v1 with in-process mode (no multiprocessing). These env vars
    # must be set BEFORE vllm is imported for the first time in this process;
    # once vllm is imported the values have already been captured and cannot be
    # changed by rewriting os.environ. We hard-set (rather than setdefault) so
    # that pre-existing values from the user's environment do not silently
    # switch us back to v0 or re-enable multiprocessing.
    if "vllm" in sys.modules:
        warnings.warn(
            "vllm was imported before genlm.backend.llm.vllm; "
            "VLLM_USE_V1=1 / VLLM_ENABLE_V1_MULTIPROCESSING=0 may not take "
            "effect and AsyncVirtualLM may fail to capture logprobs.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.inputs import TokensPrompt
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm.v1.sample.logits_processor import LogitsProcessor
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.outputs import SamplerOutput, LogprobsTensors

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

    class GlobalLogprobsCapture(LogitsProcessor):  # pragma: no cover
        """A global logits processor that captures full vocabulary logprobs.

        This processor is injected once into the vLLM v1 engine and records
        the log probabilities for *the most recent* sampling step, as a
        single ``[batch_size, vocab_size]`` tensor.

        Semantics:

        * :meth:`apply` is invoked by the v1 sampler exactly once per decode
          step across a batch of prompts, so the captured tensor always
          reflects the final token-position logprobs for every prompt in
          that batch.
        * It does NOT retain history. Each :meth:`apply` call overwrites
          ``_captured_batch``. This is intentional: for the
          ``next_token_logprobs`` paths in :class:`AsyncVirtualLM`, every
          ``generate`` is issued with ``max_tokens=1`` and preceded by
          :meth:`clear`, so exactly one decode step runs and the overwrite
          never hides information. For sampling paths (:meth:`sample`,
          :meth:`batch_sample`) ``max_tokens > 1``, :meth:`apply` fires
          once per step, and the final-step capture is correct but earlier
          steps are discarded - callers of those methods don't read
          ``_captured_batch`` anyway.
        * Concurrent reads/writes are serialized by ``_lock``, so a
          consumer thread calling :meth:`get_logprobs` never observes a
          half-written tensor.
        """

        def __init__(self):
            self._captured_batch = None  # [batch_size, vocab_size] tensor
            self._lock = threading.Lock()

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            """Capture logprobs and pass through logits unchanged.

            Overwrites any previously captured batch; see class docstring.
            """
            # Do the clone outside the critical section so readers aren't blocked
            # on the full [batch, vocab] copy.
            captured = torch.log_softmax(logits, dim=-1, dtype=logits.dtype).clone()
            with self._lock:
                self._captured_batch = captured
            return logits

        def is_argmax_invariant(self) -> bool:
            """Return True since we don't modify logits."""
            return True

        def update_state(self, batch_update) -> None:
            """No state updates needed."""
            pass

        def get_logprobs(self, batch_index=0):
            """Get captured logprobs for a batch index."""
            with self._lock:
                if self._captured_batch is None:
                    return None
                if batch_index >= self._captured_batch.shape[0]:
                    return None
                return self._captured_batch[batch_index].clone()

        def get_all_logprobs(self):
            """Get all captured logprobs as a batch tensor."""
            with self._lock:
                if self._captured_batch is None:
                    return None
                return self._captured_batch.clone()

        def clear(self):
            """Clear captured logprobs."""
            with self._lock:
                self._captured_batch = None

    class ControlSampler(Sampler):  # pragma: no cover
        """A ``Sampler`` whose decode step is driven by an SMC control object (an ``EngineControl``).

        This is a thin engine *arm*: it owns no SMC logic. When a control object
        (:class:`~genlm.backend.llm.engine_control.EngineControl`) is attached
        for the current burst, :meth:`forward` reproduces the stock sampler's
        logits-shaping pipeline and then hands off to it:

        1. compute raw logprobs if requested (stock behavior),
        2. cast logits to float32 (stock behavior),
        3. ``apply_logits_processors(...)`` -- the non-argmax-invariant
           processors and penalties (stock behavior),
        4. apply the argmax-invariant processors -- this is where
           :class:`GlobalLogprobsCapture` lives, so full-vocab logprob capture
           keeps working and reflects the post-processor logits,
        5. ``control.draw(logits, rows)`` -- the control forms its own proposal from the
           logits and draws one token per row,
        6. package the drawn ids into a :class:`SamplerOutput`.

        Pop-out is out-of-band: ``run_burst`` aborts the rows the control names in
        :meth:`EngineControl.drain_aborts` after each step (not via an EOS draw).

        When no control is attached, :meth:`forward` defers entirely to
        ``super().forward`` so normal generation and the ``next_token_logprobs``
        paths are byte-for-byte unaffected.

        Row -> request identity is read from ``model_runner.input_batch.req_ids``
        (see :mod:`genlm.backend.llm.engine_control`). The sampler is constructed
        with a reference to its ``model_runner`` for exactly this.
        """

        def __init__(self, logprobs_mode, model_runner):
            super().__init__(logprobs_mode=logprobs_mode)
            self._model_runner = model_runner
            self._control = None

        def attach(self, control):
            """Bind the control object that drives the current burst (or ``None``)."""
            self._control = control

        def detach(self):
            """Unbind any control; subsequent steps behave like the stock sampler."""
            self._control = None

        def _row_handles(self, num_rows):
            """Row -> control handle (int); strips vLLM's ``{ext}-{8char}`` internal id."""
            req_ids = self._model_runner.input_batch.req_ids
            return [int(r.rsplit("-", 1)[0]) for r in req_ids[:num_rows]]

        def forward(
            self,
            logits,
            sampling_metadata,
            predict_bonus_token=False,
            logprobs_mode_override=None,
        ):
            control = self._control
            if control is None:
                # No burst active: identical to the stock sampler.
                return super().forward(
                    logits,
                    sampling_metadata,
                    predict_bonus_token=predict_bonus_token,
                    logprobs_mode_override=logprobs_mode_override,
                )

            logprobs_mode = logprobs_mode_override or self.logprobs_mode
            num_logprobs = sampling_metadata.max_num_logprobs
            raw_logprobs = None
            if num_logprobs is not None:
                if logprobs_mode == "raw_logprobs":
                    raw_logprobs = self.compute_logprobs(logits)
                elif logprobs_mode == "raw_logits":
                    raw_logprobs = (
                        logits.clone()
                        if logits.dtype == torch.float32
                        else logits.to(torch.float32)
                    )

            logits = logits.to(torch.float32)

            # Stock non-argmax-invariant processors + penalties.
            logits = self.apply_logits_processors(
                logits, sampling_metadata, predict_bonus_token
            )
            # Stock argmax-invariant processors (GlobalLogprobsCapture lives
            # here). These normally run inside ``sample`` after temperature; the
            # control owns temperature/draw, so we run them here on the post-processor
            # logits to keep capture working.
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            rows = self._row_handles(logits.shape[0])

            # Hand the draw to the control (it forms its own proposal from the logits).
            sampled = control.draw(logits, rows)

            if not isinstance(sampled, torch.Tensor):
                sampled = torch.tensor(sampled, dtype=torch.int64, device=logits.device)
            sampled = sampled.to(logits.device).long().view(-1)

            logprobs_tensors = None
            if num_logprobs is not None and raw_logprobs is not None:
                if num_logprobs == -1:
                    logprobs_tensors = LogprobsTensors(
                        torch.empty(0), raw_logprobs, torch.empty(0)
                    )
                else:
                    logprobs_tensors = self.gather_logprobs(
                        raw_logprobs, num_logprobs, token_ids=sampled
                    )

            return SamplerOutput(
                sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
                logprobs_tensors=logprobs_tensors,
            )

    class AsyncVirtualLM(AsyncLM):  # pragma: no cover
        """Async language model using vLLM v1 with global logits processor.

        This implementation uses vLLM v1's in-process mode with a global
        logits processor to efficiently capture full vocabulary log probabilities.
        """

        supports_burst = True  # has run_burst; drives the engine-native burst lane

        default_params = {
            "max_tokens": 1,
            "n": 1,
            "detokenize": False,
            "stop": None,
            "ignore_eos": True,
        }

        def __init__(
            self,
            llm_engine,
            logprobs_capture,
            cache_size=0,
            cache_opts=None,
            batch_size=20,
            timeout=0.02,
        ):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                llm_engine (LLM): The vLLM engine instance.
                logprobs_capture (GlobalLogprobsCapture): The global logprobs capture processor.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to None (no extra options).
                batch_size (int, optional): Maximum queries to process in one batch during auto-batching. Defaults to 20.
                timeout (float, optional): Seconds to wait after the first queued query before processing the current batch. The batch also fires immediately when ``batch_size`` is reached. Defaults to 0.02.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
                ``batch_next_token_logprobs_sync`` bypasses this cache and always re-evaluates; the other three logprobs methods consult it.
            """
            self.llm_engine = llm_engine
            self.logprobs_capture = logprobs_capture
            # The engine-native sampler, swapped in once by from_name (None until
            # then, and on any non-engine construction path). Detached except for
            # the duration of a run_burst call.
            self._control_sampler = None
            self.tokenizer = llm_engine.get_tokenizer()
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
            self.lora_request = None
            self.lora_name_to_ids = {}
            # name -> LoRARequest, for per-request LoRA in run_burst (multi-view).
            self._lora_requests = {}

            self.queries = []
            self.batch_size = batch_size
            self.timeout = timeout
            self.timer = None

            self.sample_queries = []
            self.sample_timer = None

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
                **(engine_opts or {}),
            }

            llm = LLM(model=model_name, tokenizer=model_name, **engine_opts)

            logprobs_capture = GlobalLogprobsCapture()
            model_runner = cls._get_model_runner(llm)
            model_runner.input_batch.logitsprocs.argmax_invariant.append(
                logprobs_capture
            )

            # Install the engine-native sampler ONCE. Detached (the default) it
            # defers verbatim to the stock sampler, so normal generation and the
            # next_token_logprobs paths are byte-unaffected; run_burst attaches a
            # control object only for a burst's duration.
            control_sampler = ControlSampler(
                logprobs_mode=model_runner.model_config.logprobs_mode,
                model_runner=model_runner,
            )
            model_runner.sampler = control_sampler

            inst = cls(llm, logprobs_capture, **kwargs)
            inst._control_sampler = control_sampler
            return inst

        @staticmethod
        def _get_model_runner(llm):
            """Walk the vLLM v1 internals to reach the driver worker's model runner.

            This path is brittle against vLLM refactors, so it lives in one
            place and is reused by ``from_name`` (to inject the logits
            processor) and ``underlying_model``.
            """
            engine_core = llm.llm_engine.engine_core.engine_core
            return engine_core.model_executor.driver_worker.worker.model_runner

        @property
        def underlying_model(self):
            """Access the underlying model for advanced use cases."""
            return self._get_model_runner(self.llm_engine).model

        def clear_lora(self):
            """
            Disable any active LoRA adapter for the vLLM engine.
            """
            self.lora_request = None

        def add_new_lora(self, lora_path, lora_name="lora_1"):
            """Load a LoRA adapter into the base model by creating a unique id for it.

            Args:
                lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
                lora_name (str): Name to assign to the loaded adapter.

            Notes:
                This does not activate the adapter immediately. Call `set_lora()` to enable the adapter.
            """
            lid = self.hash_to_int(lora_name)
            self.lora_name_to_ids[lora_name] = lid
            self._lora_requests[lora_name] = LoRARequest(lora_name, lid, lora_path)

        def _lora_request_for(self, lora_name):
            """Per-request LoRARequest for ``lora_name`` (``None`` = base, LoRA off)."""
            return None if lora_name is None else self._lora_requests[lora_name]

        def hash_to_int(self, value):
            """Generates a deterministic unique id for a LoRA adapter from its name.

            Args:
                value (str): The name of the LoRA adapter to hash.

            Returns:
                An integer ID corresponding to the LoRA adapter, in the range 0–255.
            """
            hash_bytes = hashlib.shake_128(value.encode("utf-8")).digest(1)
            return int.from_bytes(hash_bytes, "big")

        def set_lora(self, lora_path, lora_name="lora_1"):
            """Configure a LoRA adapter request for the vLLM engine.

            Args:
                lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
                lora_name (str): Identifier name to associate with this LoRA adapter within vLLM.
                lora_id (int): Globally unique ID for the adapter.
            """
            if lora_name not in self.lora_name_to_ids:
                raise ValueError(
                    f"A LoRA adapter named '{lora_name}' has not been loaded yet. Please call add_new_lora() first to load and name your LoRA adapters."
                )
            self.lora_request = LoRARequest(
                lora_name, self.lora_name_to_ids[lora_name], lora_path
            )

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with auto-batching.

            Concurrent calls to this method are automatically batched into a single
            ``LLM.generate()`` call for efficiency. Use with ``await``.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """
            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            future = asyncio.get_running_loop().create_future()
            self._add_query(token_ids, future)
            result = await future

            if self.cache is not None:
                self.cache[key] = result

            return result

        def _add_query(self, token_ids, future):
            """Add a query to be evaluated in the next batch.

            The timeout is measured from the *first* queued query, not the most
            recent one: we only arm the timer when the queue transitions from
            empty to non-empty. This prevents starvation when queries trickle in
            faster than ``self.timeout`` but never fill a batch.

            Args:
                token_ids (list[int]): Token IDs representing the query prompt.
                future (asyncio.Future): Future to store the result in.
            """
            self.queries.append((token_ids, future))

            if len(self.queries) >= self.batch_size:
                if self.timer:
                    self.timer.cancel()
                    self.timer = None
                self._batch_evaluate()
            elif self.timer is None:
                self.timer = asyncio.get_running_loop().call_later(
                    self.timeout, self._batch_evaluate
                )

        def _batch_evaluate(self):
            """Process all queued queries in a single batched ``generate()`` call."""
            queries, self.queries = self.queries, []
            if not queries:
                return

            if self.timer:
                self.timer.cancel()
                self.timer = None

            if self.logprobs_capture is None:
                exc = RuntimeError("Cannot use model after cleanup() has been called")
                for _, future in queries:
                    future.set_exception(exc)
                return

            # Deduplicate: group futures by identical prompts
            query_groups = defaultdict(list)
            for token_ids, future in queries:
                query_groups[tuple(token_ids)].append(future)

            unique_token_ids = list(query_groups.keys())

            self.logprobs_capture.clear()

            prompts = [
                TokensPrompt(prompt_token_ids=list(token_ids))
                for token_ids in unique_token_ids
            ]

            try:
                self.llm_engine.generate(
                    prompts=prompts,
                    sampling_params=SamplingParams(**self.default_params),
                    lora_request=self.lora_request,
                    use_tqdm=False,
                )

                all_logprobs = self.logprobs_capture.get_all_logprobs()
                assert all_logprobs is not None, "Logprobs should be captured"
                assert all_logprobs.shape[0] == len(unique_token_ids), (
                    f"Expected {len(unique_token_ids)} logprobs, got {all_logprobs.shape[0]}"
                )

                for i, key in enumerate(unique_token_ids):
                    logprobs = all_logprobs[i]
                    futures = query_groups[key]
                    if len(futures) == 1:
                        futures[0].set_result(logprobs)
                    else:
                        for future in futures:
                            future.set_result(logprobs.clone())
            except Exception as exc:
                for futures in query_groups.values():
                    for future in futures:
                        if not future.done():
                            future.set_exception(exc)

        def reset_async_queries(self):
            """Clear any pending queries from the queue.

            Use this method when an exception prevented an inference algorithm
            from executing to completion.
            """
            self.queries = []
            if self.timer:
                self.timer.cancel()
                self.timer = None

            self.sample_queries = []
            if self.sample_timer:
                self.sample_timer.cancel()
                self.sample_timer = None

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Does not support auto-batching. For batched sync calls, use
            ``batch_next_token_logprobs_sync`` instead.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            if self.logprobs_capture is None:
                raise RuntimeError("Cannot use model after cleanup() has been called")

            self.logprobs_capture.clear()

            self.llm_engine.generate(
                prompts=TokensPrompt(prompt_token_ids=list(token_ids)),
                sampling_params=SamplingParams(**self.default_params),
                lora_request=self.lora_request,
                use_tqdm=False,
            )

            result = self.logprobs_capture.get_logprobs(batch_index=0)
            assert result is not None, "Logprobs should be captured by global processor"

            if self.cache is not None:
                self.cache[key] = result

            return result

        def batch_next_token_logprobs_sync(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.

            Note:
                This method does not consult the output cache (unlike the async batch path,
                which delegates to the cached ``next_token_logprobs``). Every prompt is
                re-evaluated.
            """
            if self.logprobs_capture is None:
                raise RuntimeError("Cannot use model after cleanup() has been called")
            # Clear any stale captured logprobs
            self.logprobs_capture.clear()

            # Create prompts for batch
            prompts = [
                TokensPrompt(prompt_token_ids=list(token_ids))
                for token_ids in token_ids_list
            ]

            # Generate one token for each prompt
            self.llm_engine.generate(
                prompts=prompts,
                sampling_params=SamplingParams(**self.default_params),
                lora_request=self.lora_request,
                use_tqdm=False,
            )

            # Get all captured logprobs at once (optimized - single clone)
            all_logprobs = self.logprobs_capture.get_all_logprobs()
            assert all_logprobs is not None, "Logprobs should be captured"
            assert all_logprobs.shape[0] == len(token_ids_list), (
                f"Expected {len(token_ids_list)} logprobs, got {all_logprobs.shape[0]}"
            )

            return all_logprobs

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
              already being torn down.

            Anything else is re-raised so real bugs are not swallowed.
            """
            try:
                # ``import gc`` can itself raise ImportError when ``__del__`` is
                # invoked after ``sys.meta_path`` has been torn down at
                # interpreter shutdown, so it lives inside the try block.
                import gc

                # Clear our references
                if hasattr(self, "logprobs_capture"):
                    if self.logprobs_capture is not None:
                        self.logprobs_capture.clear()
                    self.logprobs_capture = None

                # Delete the engine to free GPU memory
                if hasattr(self, "llm_engine") and self.llm_engine is not None:
                    del self.llm_engine
                    self.llm_engine = None

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Clean up distributed state
                destroy_model_parallel()
                destroy_distributed_environment()
            except (
                ImportError,
                AttributeError,
                AssertionError,
                RuntimeError,
            ) as e:
                # Best-effort log; during interpreter shutdown logging itself
                # may already be torn down, in which case silently drop.
                with contextlib.suppress(Exception):
                    logging.getLogger(__name__).debug(
                        "AsyncVirtualLM cleanup raised %s: %s",
                        type(e).__name__,
                        e,
                    )

        def _add_sample_query(self, prompt_token_ids, sampling_params, future):
            """Enqueue a ``sample()`` request; mirrors ``_add_query`` for the logprobs path."""
            self.sample_queries.append((prompt_token_ids, sampling_params, future))
            if len(self.sample_queries) >= self.batch_size:
                if self.sample_timer:
                    self.sample_timer.cancel()
                    self.sample_timer = None
                self._batch_sample_evaluate()
            elif self.sample_timer is None:
                self.sample_timer = asyncio.get_running_loop().call_later(
                    self.timeout, self._batch_sample_evaluate
                )

        def _batch_sample_evaluate(self):
            """Dispatch queued ``sample()`` requests in one batched ``generate()`` call."""
            queries, self.sample_queries = self.sample_queries, []
            if not queries:
                return
            if self.sample_timer:
                self.sample_timer.cancel()
                self.sample_timer = None
            if self.logprobs_capture is None:
                exc = RuntimeError("Cannot use model after cleanup() has been called")
                for _, _, future in queries:
                    future.set_exception(exc)
                return
            try:
                outputs = self.llm_engine.generate(
                    prompts=[TokensPrompt(prompt_token_ids=t) for t, _, _ in queries],
                    sampling_params=[sp for _, sp, _ in queries],
                    lora_request=self.lora_request,
                    use_tqdm=False,
                )
                assert len(outputs) == len(queries)
                for output, (_, _, future) in zip(outputs, queries):
                    future.set_result(list(output.outputs[0].token_ids))
            except Exception as exc:
                for _, _, future in queries:
                    if not future.done():
                        future.set_exception(exc)

        async def sample(
            self,
            prompt_token_ids,
            max_tokens,
            eos_token_ids,
            temperature=1.0,
            seed=None,
        ):
            """Sample from the language model.

            Concurrent calls are auto-batched into a single ``LLM.generate()``
            so vLLM continuous-batches the decode steps. Use with ``await``.

            Args:
                prompt_token_ids (list[int]): The token IDs of the prompt.
                eos_token_ids (list[int]): The token IDs of the end-of-sequence tokens.
                temperature (float, optional): The temperature to use to rescale the logits. Defaults to 1.0.
                max_tokens (int): The maximum number of tokens to generate.
                seed (int, optional): The seed for the random number generator. Defaults to None.

            Returns:
                (list[int]): The sampled token IDs.
            """
            future = asyncio.get_running_loop().create_future()
            self._add_sample_query(
                list(prompt_token_ids),
                SamplingParams(
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    stop=[self.byte_vocab[i].decode() for i in eos_token_ids],
                ),
                future,
            )
            token_ids = await future
            if token_ids and token_ids[-1] in eos_token_ids:
                token_ids = token_ids[:-1]
            return token_ids

        def run_burst(
            self,
            requests,
            control,
            max_steps,
        ):  # pragma: no cover
            """Run one engine-native decode burst driven by an SMC ``control`` (an ``EngineControl``).

            Submits one engine request per substream, attaches ``control`` to the
            persistent :class:`ControlSampler` for the burst, and drives the engine's decode
            loop for up to ``max_steps`` steps. Each step the control draws the next
            token for every live request (via :meth:`EngineControl.draw`); after the step
            the control's :meth:`EngineControl.drain_aborts` names the rows to drop, which
            we ``abort_request`` -- the out-of-band pop-out (a particle terminating,
            or all rows at the ESS crossing). No EOS stop tokens, no discard forward.

            This method owns NO SMC logic -- no ESS, no resampling, no weights.
            Those all live in ``control``.

            Args:
                requests (list[tuple]): ``(ext_id, prompt_token_ids, lora_name)`` per
                    substream. ``ext_id`` is the control's request handle; ``lora_name``
                    is ``None`` for the base model. A particle owns K substreams
                    (multi-view); the control maps each ``ext_id`` back to its
                    (particle, view).
                control (EngineControl): the SMC control object.
                max_steps (int): maximum decode steps for the burst.

            Returns:
                (list[list[int]]): empty lists -- committed tokens are tracked
                control-side, so the return is unused by the caller.
            """
            from vllm.sampling_params import RequestOutputKind

            sampler = self._control_sampler
            # self.llm_engine is the vLLM LLM; .llm_engine is the inner LLMEngine.
            # Bind it once rather than hopping through both attrs at every call.
            engine = self.llm_engine.llm_engine

            sampling_params = SamplingParams(
                n=1,
                max_tokens=max_steps,
                detokenize=False,
                # Pop-out is the control's explicit abort_request, NOT an EOS stop
                # token -- the control draws in its own vocab and ends rows out-of-band.
                ignore_eos=True,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            # External request id is the control's ext_id (str); drain_aborts() returns
            # these same ids. Attach the control object to the persistent ControlSampler
            # for this burst only; the finally below detaches it so normal
            # generation stays stock even if the burst raises.
            sampler.attach(control)
            gone = set()
            added = set()  # every request id ever added (initial + in-place re-adds)
            try:
                for ext_id, prompt, lora_name in requests:
                    rid = str(ext_id)
                    engine.add_request(
                        rid,
                        TokensPrompt(prompt_token_ids=list(prompt)),
                        sampling_params,
                        lora_request=self._lora_request_for(lora_name),
                    )
                    added.add(rid)
                # Drive the decode loop. Each engine.step() runs the forward + the
                # control's draw (which banks the SMC step and flags rows to drop via
                # abort_rows); after the step we abort the flagged requests. The burst
                # ends when no request remains -- each row is aborted at its particle's
                # termination, or all at once when the ESS test crosses (the pop-out).
                # No SMC logic (ESS/resample/weights) lives here -- only abort plumbing.
                while engine.has_unfinished_requests():
                    step_outputs = engine.step()
                    for output in step_outputs:
                        if output.finished:
                            gone.add(output.request_id)
                    aborts = [
                        str(i) for i in control.drain_aborts() if str(i) not in gone
                    ]
                    if aborts:
                        engine.abort_request(aborts)
                        gone.update(aborts)
                    # In-place per-group resample (Plan B): the control re-adds forked
                    # rows mid-burst so a resampled group rejoins the live batch
                    # WITHOUT draining the engine -- the other groups keep decoding.
                    # Each is a FRESH request id (the control maps it back to its
                    # population slot), so there's no abort/add id race. Empty for the
                    # pop+relaunch path and for controls without ``drain_adds``.
                    for ext_id, prompt, lora_name in getattr(
                        control, "drain_adds", lambda: []
                    )():
                        rid = str(ext_id)
                        engine.add_request(
                            rid,
                            TokensPrompt(prompt_token_ids=list(prompt)),
                            sampling_params,
                            lora_request=self._lora_request_for(lora_name),
                        )
                        added.add(rid)
                        gone.discard(rid)
            finally:
                sampler.detach()
                # Safety net: drop anything still running (an exception mid-burst, or
                # a row that hit the max_steps length cap before the control aborted).
                remaining = [r for r in added if r not in gone]
                if remaining and engine.has_unfinished_requests():
                    with contextlib.suppress(Exception):
                        engine.abort_request(remaining)

            # The committed tokens are tracked control-side (the control banks each draw
            # into its particle contexts), so run_burst returns nothing useful.
            return [[] for _ in requests]
