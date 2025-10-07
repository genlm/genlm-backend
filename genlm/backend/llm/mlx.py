import torch

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache, TokenKVTrie


from typing import (
    Any,
    Optional,
    Tuple,
)

import mlx.core as mx
import mlx.nn as nn
import functools

from mlx_lm.models import cache


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


try:
    import mlx_lm
    from mlx_lm.generate import generate_step
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    HAS_MLX = True
except ImportError:  # pragma: no cover
    HAS_MLX = False  # pragma: no cover


if not HAS_MLX:

    class AsyncMlxLM:  # pragma: no cover
        """Placeholder class when MLX is not installed."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "MLX is not installed. Please install it with 'pip install mlx-lm' "
                "to use the MLX-based AsyncLM model."
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "MLX is not installed. Please install it with 'pip install mlx-lm' "
                "to use the MLX-based AsyncLM model."
            )

else:

    class Query:
        """A query to a language model, waiting to be batched."""

        def __init__(self, prompt, future, past=None):
            self.prompt = prompt
            self.future = future
            self.past = past

            if self.past is not None:
                self.past_len = past[
                    0
                ][
                    0
                ].shape[
                    2
                ]  # layers, key or value, batch size, num heads, num tokens, head repr length
            else:
                self.past_len = 0

        """ @torch.no_grad()
        def past_padded(self, layer, j, to_length, dtype, device, past_shape):
            if self.past is not None:
                return torch.cat(
                    (
                        self.past[layer][j],
                        torch.zeros(
                            1,
                            past_shape[1],
                            to_length - self.past_len,
                            past_shape[3],
                            dtype=dtype,
                            device=device,
                        ),
                    ),
                    dim=2,
                )
            else:
                return torch.zeros(
                    1, past_shape[1], to_length, past_shape[3], dtype=dtype, device=device
                ) """

        def prompt_padded(self, pad_token, to_length):
            return [
                *self.prompt,
                *[pad_token for _ in range(to_length - len(self.prompt))],
            ]

        def attention_mask(self, total_past_length, total_seq_length):
            return [
                *[1 for _ in range(self.past_len)],
                *[0 for _ in range(total_past_length - self.past_len)],
                *[1 for _ in range(len(self.prompt))],
                *[0 for _ in range(total_seq_length - len(self.prompt))],
            ]

        def position_ids(self, total_past_length, total_seq_length):
            return [
                *range(self.past_len, self.past_len + len(self.prompt)),
                *[0 for _ in range(total_seq_length - len(self.prompt))],
            ]

    # logging.getLogger("mlx.engine.async_llm_engine").setLevel(logging.WARNING)
    def _generate_step_custom(
        prompt: mx.array,
        model: nn.Module,
        max_kv_size: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
        prefill_step_size: int = 2048,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """
        A generator producing token ids based on the given prompt from the model.
        Args:
            prompt (mx.array): The input prompt.
            model (nn.Module): The model to use for generation.
            max_kv_size (int, optional): Maximum size of the key-value cache. Old
            entries (except the first 4 tokens) will be overwritten.
            prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
            provided, the cache will be updated in place.
            prefill_step_size (int): Step size for processing the prompt.
            kv_bits (int, optional): Number of bits to use for KV cache quantization.
            None implies no cache quantization. Default: ``None``.
            kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
            quantized_kv_start (int): Step to begin using a quantized KV cache.
            when ``kv_bits`` is non-None. Default: ``0``.
        Yields:
            Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
        """
        if len(prompt) == 0:
            raise ValueError("Either prompt (or both) must be provided.")
        # Create the KV cache for generation
        if prompt_cache is None:
            prompt_cache = cache.make_prompt_cache(
                model,
                max_kv_size=max_kv_size,
            )
        quantize_cache_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

        def _model_call(input_tokens: mx.array):
            return model(input_tokens, cache=prompt_cache)

        def _step(input_tokens: mx.array):
            with mx.stream(generation_stream):
                logits = _model_call(
                    input_tokens=input_tokens[None],
                )
                logits = logits[:, -1, :]
                quantize_cache_fn(prompt_cache)
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                return logprobs.squeeze(0)

        with mx.stream(generation_stream):
            total_prompt_tokens = len(prompt)
            prompt_processed_tokens = 0
            while total_prompt_tokens - prompt_processed_tokens > 1:
                n_to_process = min(prefill_step_size, prompt.size - 1)
                _model_call(
                    input_tokens=prompt[:n_to_process][None],
                )
                quantize_cache_fn(prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                prompt_processed_tokens += n_to_process
                prompt = prompt[n_to_process:]
                mx.clear_cache()
            logprobs = _step(input_tokens=prompt)
        mx.async_eval(logprobs)
        return logprobs

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
            """ self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            ) """
            self.kv_cache = TokenKVTrie()
            self.cache_size = cache_size
            self.cache_opts = cache_opts
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

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

        @property
        def underlying_model(self):
            return self.mlx_lm_model

        def clear_kv_cache(self):
            self.kv_cache.clear_kv_cache()

        def clear_cache(self):
            """Clear output cache."""
            self.kv_cache = TokenKVTrie()
            self.cache = (
                OutputCache(maxsize=self.cache_size, **self.cache_opts)
                if self.cache_size > 0
                else None
            )

        def walk_cache(self, token_ids):
            """Walk the cache tree to find the deepest node matching a sequence of tokens.

            Args:
                token_ids (list[int]): Sequence of token IDs to follow in the cache tree

            Returns:
                tuple:
                    - CacheNode: The deepest node in the cache tree that matches the token sequence
                    - int: Number of tokens matched from the start of token_ids
                    - list[tuple[torch.Tensor]]|None: Past key/value states from the deepest cached node,
                        or None if no cached states were found
                    - int: Base index indicating where the past states start in token_ids
            """
            node = self.kv_cache
            base_node = self.kv_cache
            next_token_index = 0
            path_kvs = []
            collecting = True

            while next_token_index < len(token_ids) and node.has_token(
                token_ids[next_token_index]
            ):
                node = node.get_token(token_ids[next_token_index])
                next_token_index += 1

                if collecting:
                    if node.key_values is not None:
                        path_kvs.append(node.key_values)
                        base_node = node
                    else:
                        collecting = False

            base = len(path_kvs)

            if path_kvs:
                keys = mx.stack([kv[0] for kv in path_kvs])
                values = mx.stack([kv[1] for kv in path_kvs])
                stacked_kvs = (keys, values)
            else:
                stacked_kvs = None

            return node, base_node, next_token_index, stacked_kvs, base

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
            logprobs = _generate_step_custom(token_ids_array, self.mlx_lm_model)
            logprobs = torch.tensor(logprobs)

            if self.cache is not None:
                self.cache[key] = logprobs
            return logprobs

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
