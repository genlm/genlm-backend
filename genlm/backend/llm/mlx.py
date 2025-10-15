from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputMLXCache
import torch

from typing import (
    Any,
    Optional,
)

try:
    import mlx_lm
    from mlx_lm.generate import generate_step
    import mlx.core as mx
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_sampler

    HAS_MLX = True
except ImportError:  # pragma: no cover
    HAS_MLX = False  # pragma: no cover


if not HAS_MLX:

    class AsyncMlxLM:  # pragma: no cover
        """Placeholder class when MLX is not installed."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "To use the MLX-based AsyncLM model, "
                "install the package with 'pip install genlm-backend[mlx]'"
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "To use the MLX-based AsyncLM model, "
                "install the package with 'pip install genlm-backend[mlx]'"
            )

else:

    class AsyncMlxLM(AsyncLM):
        def __init__(self, mlx_lm_model, tokenizer, cache_size=0, cache_opts={}):
            """Initialize an `AsyncMlxLM` instance.

            Args:
                mlx_lm_model (Model): The async MLX LM model instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputMLXCache`][genlm.backend.cache.OutputMLXCache] constructor. Defaults to {}.

            """

            self.mlx_lm_model = mlx_lm_model
            self.tokenizer = tokenizer
            self.cache = (
                OutputMLXCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )
            self.generation_stream = mx.new_stream(mx.default_device())

            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_name, **kwargs):
            """Create a `AsyncMlxLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load. Could be a Hugging Face model name.
                engine_opts (dict): Additional options to pass to the `AsyncLLMEngine`. The engine will be
                    configured with prefix caching enabled and async output processing disabled by default.
                **kwargs: Additional arguments passed to `AsyncMlxLM` constructor.

            Returns:
                (AsyncMlxLM): An `AsyncMlxLM` instance.
            """

            model, tokenizer = mlx_lm.load(model_name)
            return cls(model, tokenizer, **kwargs)

        def clear_cache(self):
            """Clear output cache."""
            if self.cache is not None:
                self.cache.clear()

        def _generate_step_custom(
            self,
            prompt: mx.array,
            max_kv_size: Optional[int] = None,
            prompt_cache: Optional[Any] = None,
            prefill_step_size: int = 2048,
        ) -> mx.array:
            """
            Produce log probabilities for the next token from the model.
            Args:
                prompt (mx.array): The input prompt.
                max_kv_size (int, optional): Maximum size of the prompt cache. Old
                entries will be overwritten.
                prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
                provided, the cache will be updated in place.
                prefill_step_size (int): Step size for processing the prompt.
            Returns:
                Tuple[mx.array, mx.array]: A vector of log probabilities.
            """
            if len(prompt) == 0:
                raise ValueError("Prompt must be provided.")
            # Create the prompt cache for generation
            if prompt_cache is None:
                prompt_cache = cache.make_prompt_cache(
                    self.mlx_lm_model,
                    max_kv_size=max_kv_size,
                )

            def _model_call(input_tokens: mx.array):
                return self.mlx_lm_model(input_tokens, cache=prompt_cache)

            def _step(input_tokens: mx.array):
                with mx.stream(self.generation_stream):
                    logits = _model_call(
                        input_tokens=input_tokens[None],
                    )
                    logits = logits[:, -1, :]
                    logprobs = logits - mx.logsumexp(logits, keepdims=True)
                    return logprobs.squeeze(0)

            with mx.stream(self.generation_stream):
                total_prompt_tokens = len(prompt)
                prompt_processed_tokens = 0
                while total_prompt_tokens - prompt_processed_tokens > 1:
                    n_to_process = min(prefill_step_size, prompt.size - 1)
                    _model_call(
                        input_tokens=prompt[:n_to_process][None],
                    )
                    mx.eval([c.state for c in prompt_cache])
                    prompt_processed_tokens += n_to_process
                    prompt = prompt[n_to_process:]
                    mx.clear_cache()
                logprobs = _step(input_tokens=prompt)
            mx.async_eval(logprobs)
            return logprobs

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with output caching.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.

            Warning:
                Do not use `asyncio.run(next_token_logprobs())` as it may interfere with MLX's background loop.
                For synchronous usage, use the `next_token_logprobs_sync()` method instead.
            """
            return self.next_token_logprobs_sync(token_ids)

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

            token_ids_array = mx.array(token_ids)
            logprobs = self._generate_step_custom(token_ids_array)
            logprobs = torch.tensor(logprobs)
            if self.cache is not None:
                self.cache[key] = logprobs
            return logprobs

        async def batch_next_token_logprobs(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch asynchronously.
            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.
            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            return self.batch_next_token_logprobs_sync(token_ids_list)

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

            sampler = make_sampler(temp=temperature)
            prompt_token_ids_array = mx.array(prompt_token_ids)
            token_generator = generate_step(
                prompt_token_ids_array,
                self.mlx_lm_model,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            generated_token_ids = []
            for sampled, _ in token_generator:
                if sampled in eos_token_ids:
                    break
                generated_token_ids.append(sampled)
            return generated_token_ids
