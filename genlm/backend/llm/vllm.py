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
    from vllm.v1.sample.sampler import Sampler

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

    class CaptureSampler(Sampler):  # pragma: no cover
        """Sampler subclass that captures per-row logprobs keyed by request_id.

        Replaces the old ``GlobalLogprobsCapture`` LogitsProcessor, which could
        not distinguish a real submission from a vLLM-internal warmup forward
        (no request_id is visible at the LogitsProcessor layer). Here capture
        lives at the sampler layer where ``model_runner.input_batch.req_ids``
        is the authoritative row -> request_id mapping; ``take(ids)`` reads
        back only the rows for the request_ids the caller submitted, so a
        warmup row lands in a dict slot we never read.
        """

        def __init__(self, logprobs_mode, model_runner):
            super().__init__(logprobs_mode=logprobs_mode)
            self._model_runner = model_runner
            self._captured = {}

        def forward(self, logits, sampling_metadata, **kwargs):
            req_ids = list(self._model_runner.input_batch.req_ids[: logits.shape[0]])
            lp = torch.log_softmax(logits.float(), dim=-1).clone()
            for i, rid in enumerate(req_ids):
                self._captured[rid] = lp[i]
            if os.environ.get("GENLM_CAPTURE_DIAG"):
                print(
                    f"[CAPTURE-DIAG] req_ids[:3]={req_ids[:3]} "
                    f"N={len(req_ids)} shape={tuple(lp.shape)}",
                    flush=True,
                )
            return super().forward(logits, sampling_metadata, **kwargs)

        def take(self, request_ids):
            return torch.stack([self._captured.pop(rid) for rid in request_ids])

        def clear(self):
            self._captured.clear()

    class AsyncVirtualLM(AsyncLM):  # pragma: no cover
        """Async language model using vLLM v1 with global logits processor.

        This implementation uses vLLM v1's in-process mode with a global
        logits processor to efficiently capture full vocabulary log probabilities.
        """

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
            capture_sampler,
            cache_size=0,
            cache_opts=None,
            batch_size=20,
            timeout=0.02,
        ):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                llm_engine (LLM): The vLLM engine instance.
                capture_sampler (CaptureSampler): The sampler subclass that records per-row logprobs keyed by request_id.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to None (no extra options).
                batch_size (int, optional): Maximum queries to process in one batch during auto-batching. Defaults to 20.
                timeout (float, optional): Seconds to wait after the first queued query before processing the current batch. The batch also fires immediately when ``batch_size`` is reached. Defaults to 0.02.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
                ``batch_next_token_logprobs_sync`` bypasses this cache and always re-evaluates; the other three logprobs methods consult it.
            """
            self.llm_engine = llm_engine
            self.capture_sampler = capture_sampler
            self.tokenizer = llm_engine.get_tokenizer()
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
            self.lora_request = None
            self.lora_name_to_ids = {}

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

            model_runner = cls._get_model_runner(llm)
            capture_sampler = CaptureSampler(
                logprobs_mode=model_runner.model_config.logprobs_mode,
                model_runner=model_runner,
            )
            model_runner.sampler = capture_sampler

            return cls(llm, capture_sampler, **kwargs)

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
            self.lora_name_to_ids[lora_name] = self.hash_to_int(lora_name)

        def hash_to_int(self, value):
            """Generates a deterministic unique id for a LoRA adapter from its name.

            Args:
                value (str): The name of the LoRA adapter to hash.

            Returns:
                An integer ID corresponding to the LoRA adapter, in the range [1, 2^31 - 1].
            """
            hash_bytes = hashlib.shake_128(value.encode("utf-8")).digest(4)
            return (int.from_bytes(hash_bytes, "big") % (2**31 - 2)) + 1

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

            if self.capture_sampler is None:
                exc = RuntimeError("Cannot use model after cleanup() has been called")
                for _, future in queries:
                    future.set_exception(exc)
                return

            # Deduplicate: group futures by identical prompts
            query_groups = defaultdict(list)
            for token_ids, future in queries:
                query_groups[tuple(token_ids)].append(future)

            unique_token_ids = list(query_groups.keys())

            self.capture_sampler.clear()

            prompts = [
                TokensPrompt(prompt_token_ids=list(token_ids))
                for token_ids in unique_token_ids
            ]

            try:
                outputs = self.llm_engine.generate(
                    prompts=prompts,
                    sampling_params=SamplingParams(**self.default_params),
                    lora_request=self.lora_request,
                    use_tqdm=False,
                )
                all_logprobs = self.capture_sampler.take(
                    [o.request_id for o in outputs]
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

            if self.capture_sampler is None:
                raise RuntimeError("Cannot use model after cleanup() has been called")

            self.capture_sampler.clear()

            outputs = self.llm_engine.generate(
                prompts=TokensPrompt(prompt_token_ids=list(token_ids)),
                sampling_params=SamplingParams(**self.default_params),
                lora_request=self.lora_request,
                use_tqdm=False,
            )

            result = self.capture_sampler.take([outputs[0].request_id])[0]

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
            if self.capture_sampler is None:
                raise RuntimeError("Cannot use model after cleanup() has been called")
            self.capture_sampler.clear()

            prompts = [
                TokensPrompt(prompt_token_ids=list(token_ids))
                for token_ids in token_ids_list
            ]

            outputs = self.llm_engine.generate(
                prompts=prompts,
                sampling_params=SamplingParams(**self.default_params),
                lora_request=self.lora_request,
                use_tqdm=False,
            )

            return self.capture_sampler.take([o.request_id for o in outputs])

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
                if hasattr(self, "capture_sampler"):
                    if self.capture_sampler is not None:
                        self.capture_sampler.clear()
                    self.capture_sampler = None

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
            if self.capture_sampler is None:
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
