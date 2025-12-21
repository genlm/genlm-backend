import os
import torch
import logging
import threading

# Enable vLLM v1 with in-process mode (no multiprocessing)
# This must be set BEFORE importing vllm
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate

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

    class GlobalLogprobsCapture(LogitsProcessor):
        """A global logits processor that captures full vocabulary logprobs.

        This processor is injected into the vLLM v1 engine and captures
        log probabilities for all tokens in the vocabulary during inference.
        """

        def __init__(self, device):
            self.device = device
            self._captured_batch = None  # [batch_size, vocab_size] tensor
            self._lock = threading.Lock()

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            """Capture logprobs and pass through logits unchanged."""
            logprobs = torch.log_softmax(logits, dim=-1, dtype=logits.dtype)
            with self._lock:
                # Single clone of entire batch - O(1) instead of O(batch_size)
                self._captured_batch = logprobs.clone()
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

    class AsyncVirtualLM(AsyncLM):
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

        def __init__(self, llm_engine, logprobs_capture, cache_size=0, cache_opts={}):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                llm_engine (LLM): The vLLM engine instance.
                logprobs_capture (GlobalLogprobsCapture): The global logprobs capture processor.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to {}.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
            """
            self.llm_engine = llm_engine
            self.logprobs_capture = logprobs_capture
            self.tokenizer = llm_engine.get_tokenizer()
            self._request_id = 0
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            super().__init__(tokenizer=self.tokenizer)

        def _next_request_id(self):
            """Generate a unique request ID."""
            self._request_id += 1
            return str(self._request_id)

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
                "gpu_memory_utilization": 0.5,
                **(engine_opts or {}),
            }

            # Create the vLLM engine
            llm = LLM(model=model_name, tokenizer=model_name, **engine_opts)

            # Access the model runner to inject our global logprobs capture processor
            engine_core = llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            input_batch = model_runner.input_batch

            # Create and inject the logprobs capture processor
            logprobs_capture = GlobalLogprobsCapture(device=input_batch.device)
            input_batch.logitsprocs.argmax_invariant.append(logprobs_capture)

            return cls(llm, logprobs_capture, **kwargs)

        @property
        def underlying_model(self):
            """Access the underlying model for advanced use cases."""
            engine_core = self.llm_engine.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            return model_runner.model

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with output caching.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.

            Note:
                This method is async for API compatibility but uses synchronous vLLM internally.
            """
            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            result = self._next_token_logprobs_impl(token_ids)

            if self.cache is not None:
                self.cache[key] = result

            return result

        def _next_token_logprobs_impl(self, token_ids):
            """Internal implementation for getting next token logprobs."""
            if self.logprobs_capture is None:
                raise RuntimeError("Cannot use model after cleanup() has been called")
            # Clear any stale captured logprobs
            self.logprobs_capture.clear()

            # Generate one token
            self.llm_engine.generate(
                prompts=TokensPrompt(prompt_token_ids=list(token_ids)),
                sampling_params=SamplingParams(**self.default_params),
                use_tqdm=False,
            )

            # Get captured logprobs
            logprobs = self.logprobs_capture.get_logprobs(batch_index=0)
            assert logprobs is not None, "Logprobs should be captured by global processor"

            return logprobs

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            result = self._next_token_logprobs_impl(token_ids)

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
            """Clean up the vLLM engine and associated resources."""
            import gc
            try:
                # Clear our references
                if hasattr(self, 'logprobs_capture'):
                    self.logprobs_capture.clear()
                    self.logprobs_capture = None
                
                # Delete the engine to free GPU memory
                if hasattr(self, 'llm_engine') and self.llm_engine is not None:
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
            except Exception:
                pass  # Ignore cleanup errors

        async def sample(
            self,
            prompt_token_ids,
            max_tokens,
            eos_token_ids,
            temperature=1.0,
            seed=None,
        ):
            """Sample from the language model.

            Args:
                prompt_token_ids (list[int]): The token IDs of the prompt.
                eos_token_ids (list[int]): The token IDs of the end-of-sequence tokens.
                temperature (float, optional): The temperature to use to rescale the logits. Defaults to 1.0.
                max_tokens (int): The maximum number of tokens to generate.
                seed (int, optional): The seed for the random number generator. Defaults to None.

            Returns:
                (list[int]): The sampled token IDs.
            """
            outputs = self.llm_engine.generate(
                prompts=TokensPrompt(prompt_token_ids=list(prompt_token_ids)),
                sampling_params=SamplingParams(
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    stop=[self.byte_vocab[i].decode() for i in eos_token_ids],
                ),
                use_tqdm=False,
            )

            assert len(outputs) == 1, "Expected exactly one output"
            assert len(outputs[0].outputs) == 1, "Expected exactly one sequence"

            token_ids = list(outputs[0].outputs[0].token_ids)
            if token_ids and token_ids[-1] in eos_token_ids:
                token_ids = token_ids[:-1]

            return token_ids
