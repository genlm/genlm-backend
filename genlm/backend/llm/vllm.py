import asyncio
import torch
import logging
import warnings

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache

try:
    from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.utils.counter import Counter
    from vllm.inputs import TokensPrompt

    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )

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
    logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

    class AsyncVirtualLM(AsyncLM):

        def __init__(self, async_llm_engine, cache_size=0, cache_opts={}):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                async_llm_engine (AsyncLLMEngine): The async vLLM engine instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to {}.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
            """
            self.async_llm_engine = async_llm_engine
            self.tokenizer = async_llm_engine.tokenizer
            self.request_counter = Counter()
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            super().__init__(tokenizer=self.tokenizer)

            # Store vocab size for logprobs requests
            self._vocab_size = len(self.tokenizer)

            # Default sampling params for logprobs - request full vocab logprobs
            self._logprobs_params = SamplingParams(
                max_tokens=1,
                n=1,
                detokenize=False,
                stop=None,
                ignore_eos=True,
                logprobs=self._vocab_size,
            )

        @classmethod
        def from_name(cls, model_name, engine_opts=None, **kwargs):
            """Create a `AsyncVirtualLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load.
                engine_opts (dict): Additional options to pass to the `AsyncLLMEngine`. The engine will be
                    configured with prefix caching enabled by default.
                **kwargs: Additional arguments passed to `AsyncVirtualLM` constructor.

            Returns:
                (AsyncVirtualLM): An `AsyncVirtualLM` instance.
            """
            if not HAS_VLLM:
                raise ImportError(  # pragma: no cover
                    "vLLM not available. Install vLLM or use AsyncTransformer instead."
                )

            if engine_opts is not None and "enable_chunked_prefill" in engine_opts:
                if engine_opts["enable_chunked_prefill"]:
                    warnings.warn(  # pragma: no cover
                        "Setting enable_chunked_prefill to True may interfere with AsyncVirtualLM's "
                        "custom sampling functionality."
                    )

            # Get vocab size to set max_logprobs - we need to load tokenizer first
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            vocab_size = len(tokenizer)

            engine_opts = {
                "enable_prefix_caching": True,
                "max_logprobs": vocab_size,  # Enable full vocab logprobs
                **(engine_opts or {}),
            }

            engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(model=model_name, tokenizer=model_name, **engine_opts)
            )

            return cls(engine, **kwargs)

        @property
        def underlying_model(self):
            raise NotImplementedError(
                "underlying_model is not available with vLLM V1 engine. "
                "The V1 engine does not expose direct model access."
            )

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with output caching.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.

            Warning:
                Do not use `asyncio.run(next_token_logprobs())` as it may interfere with vLLM's background loop.
                For synchronous usage, use the `next_token_logprobs_sync()` method instead.
            """
            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            result = await self._next_token_logprobs(key)

            if self.cache is not None:
                self.cache[key] = result

            return result

        async def _next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            req_id = str(next(self.request_counter))
            prompt = TokensPrompt(prompt_token_ids=token_ids)

            async for output in self.async_llm_engine.generate(
                prompt=prompt,
                sampling_params=self._logprobs_params,
                request_id=req_id,
            ):
                if output.finished:
                    # Extract logprobs from output
                    # output.outputs[0].logprobs is a list of dicts (one per generated token)
                    # Each dict maps token_id -> LogProb object with .logprob attribute
                    logprobs_dict = output.outputs[0].logprobs[0]

                    # Convert to tensor - logprobs_dict contains LogProb objects
                    # We need to create a full vocab tensor
                    log_probs = torch.full(
                        (self._vocab_size,), float("-inf"), dtype=torch.float32
                    )
                    for token_id, logprob_obj in logprobs_dict.items():
                        log_probs[token_id] = logprob_obj.logprob

                    return log_probs

            raise RuntimeError("No output received from vLLM engine")

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            return self.batch_next_token_logprobs_sync([token_ids])[0]

        def batch_next_token_logprobs_sync(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """

            async def _batch_async():
                results = []
                for token_ids in token_ids_list:
                    result = await self._next_token_logprobs(tuple(token_ids))
                    results.append(result)
                return torch.stack(results)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # We're already in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _batch_async())
                    return future.result()
            else:
                return asyncio.run(_batch_async())

        def clear_cache(self):
            """Clear output cache."""
            if self.cache:
                self.cache.clear()

        def __del__(self):
            """Clean up resources on deletion."""
            self._cleanup_engine()

        def _cleanup_engine(self):
            """Clean up the vLLM engine and associated resources."""
            if async_engine := getattr(self, "async_llm_engine", None):
                async_engine.shutdown()
                destroy_model_parallel()
                destroy_distributed_environment()

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
            async for output in self.async_llm_engine.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=SamplingParams(
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    stop=[self.byte_vocab[i].decode() for i in eos_token_ids],
                ),
                request_id=str(next(self.request_counter)),
            ):
                if output.finished:
                    assert len(output.outputs) == 1, (
                        "Expected exactly one sequence group"
                    )
                    token_ids = list(output.outputs[0].token_ids)
                    if token_ids[-1] in eos_token_ids:
                        token_ids = token_ids[:-1]
                    return token_ids
