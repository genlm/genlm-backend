import torch
import logging
import warnings

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache
import mlx_lm
from mlx_lm.generate import generate_step
import mlx.core as mx
import asyncio
from mlx_lm.sample_utils import make_sampler


""" try:
    from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.utils import Counter
    from vllm.inputs import TokensPrompt

    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    HAS_VLLM = True
except ImportError:  # pragma: no cover
    HAS_VLLM = False  # pragma: no cover """


if False:
    pass

else:
    #logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

    #class PassThroughLogitsProcessor:
    #    """A logits processor that stores the logprobs and passes the logits through."""
#
    #    def __init__(self):
    #        self.log_probs = None
#
    #    def __call__(self, past_token_ids, logits):
    #        assert self.log_probs is None, (
    #            "Log probs already set. This should never happen."
    #        )
    #        self.log_probs = torch.log_softmax(logits, dim=-1, dtype=logits.dtype)
    #        return logits

    class AsyncMlxLM(AsyncLM):
        default_params = {
            "max_tokens": 1,
            "n": 1,
            "detokenize": False,
            "stop": None,
            "ignore_eos": True,
        }

        def __init__(self, mlx_lm_model, tokenizer, cache_size=0, cache_opts={}):
            """Initialize an `AsyncMlxLM` instance.

            Args:
                mlx_lm_model (Model): The async MLX LM model instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to {}.

            """
            
            self.mlx_lm_model = mlx_lm_model
            self.tokenizer = tokenizer
            #self.request_counter = Counter()
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            #async_llm_engine.engine.log_stats = False

            super().__init__(tokenizer=self.tokenizer)

        #@classmethod
        #def from_name(cls, model_name, engine_opts=None, **kwargs):
        #    """Create a `AsyncVirtualLM` instance from a model name.
#
        #    Args:
        #        model_name (str): Name of the model to load.
        #        engine_opts (dict): Additional options to pass to the `AsyncLLMEngine`. The engine will be
        #            configured with prefix caching enabled and async output processing disabled by default.
        #        **kwargs: Additional arguments passed to `AsyncVirtualLM` constructor.
#
        #    Returns:
        #        (AsyncVirtualLM): An `AsyncVirtualLM` instance.
        #    """
        #    if not HAS_VLLM:
        #        raise ImportError(  # pragma: no cover
        #            "vLLM not available. Install vLLM or use AsyncTransformer instead."
        #        )
#
        #    if engine_opts is not None and "enable_chunked_prefill" in engine_opts:
        #        if engine_opts["enable_chunked_prefill"]:
        #            warnings.warn(  # pragma: no cover
        #                "Setting enable_chunked_prefill to True may interfere with AsyncVirtualLM's "
        #                "custom sampling functionality."
        #            )
#
        #    engine_opts = {
        #        "enable_prefix_caching": True,
        #        "disable_log_requests": True,
        #        "disable_async_output_proc": True,  # This parameter forces vLLM to use v0, which is currently what we want to do.
        #        **(engine_opts or {}),
        #    }
#
        #    engine = AsyncLLMEngine.from_engine_args(
        #        AsyncEngineArgs(model=model_name, tokenizer=model_name, **engine_opts)
        #    )
#
        #    return cls(engine, **kwargs)

        #@property
        #def underlying_model(self):
        #    return self.async_llm_engine.engine.model_executor.driver_worker.model_runner.model


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
            return self.next_token_logprobs_sync(token_ids)


        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            token_ids_array = mx.array(token_ids)
            tok_gen = generate_step(token_ids_array, self.mlx_lm_model, max_tokens=1)
            _, logprobs = next(tok_gen)

            if self.cache is not None:
                self.cache[key] = torch.log_softmax(torch.tensor(logprobs), dim=-1)


            return torch.log_softmax(torch.tensor(logprobs), dim=-1)


        def batch_next_token_logprobs_sync(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            outputs = []
            for token_ids in token_ids_list:
                outputs.append(self.next_token_logprobs_sync(token_ids))
            return torch.stack(outputs)


        def clear_cache(self):
            """Clear output cache."""
            if self.cache:
                self.cache.clear()

        def __del__(self):
            """Clean up resources on deletion."""
            self._cleanup_engine()

        def _cleanup_engine(self):
            """Clean up the MLX LM engine and associated resources."""
            pass
            

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

            if seed is not None:
                mx.random.seed(seed)

            """ async for output in self.async_llm_engine.generate(
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
                    return token_ids """

            sampler = make_sampler(temp=temperature)
            prompt_token_ids_array = mx.array(prompt_token_ids)
            token_generator = generate_step(prompt_token_ids_array, self.mlx_lm_model, max_tokens=max_tokens, sampler=sampler)
            generated_token_ids = []
            for sampled, _ in token_generator:
                if sampled in eos_token_ids:
                    break
                generated_token_ids.append(sampled)
            return generated_token_ids