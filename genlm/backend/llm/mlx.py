import asyncio
from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputMLXCache
from collections import defaultdict
import torch

from typing import (
    Any,
    Optional,
)


try:
    import mlx_lm
    from mlx_lm.generate import generate_step, BatchGenerator, wired_limit
    import mlx.core as mx
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        KVCache,
        RotatingKVCache,
    )

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

    def _to_torch(logprobs):
        """Converts MLX array into torch tensors."""
        if isinstance(logprobs, mx.array):
            if logprobs.dtype in [mx.bfloat16]:
                logprobs = logprobs.astype(mx.float32)
            return torch.tensor(logprobs)
        elif isinstance(logprobs, (list, tuple)):
            return [_to_torch(lp) for lp in logprobs]
        return logprobs

    def _has_bf16(mlx_lm_model):
        def check(x):
            if isinstance(x, dict):
                return any(check(v) for v in x.values())
            return getattr(x, "dtype", None) == mx.bfloat16

        return any(
            check(param)
            for layer in mlx_lm_model.layers
            for param in layer.parameters().values()
        )

    def _cache_batchable(mlx_lm_model):
        if not hasattr(mlx_lm_model, "make_cache"):
            return True

        cache = mlx_lm_model.make_cache()
        batchable = (CacheList, KVCache, ArraysCache)
        return all(
            isinstance(c, batchable) or (isinstance(c, RotatingKVCache) and c.keep == 0)
            for c in cache
        )

    def _supports_batching(mlx_lm_model):
        """Return True only if MLX-LM has batching cache support for the model, and does not have bfloat16 parameters."""
        return _cache_batchable(mlx_lm_model) and not _has_bf16(mlx_lm_model)

    class BatchGeneratorCustom(BatchGenerator):
        """A custom batch generator optimzed for logprobs computation."""

        def _next(self):
            prompt_processing = False
            batch = self.active_batch
            num_active = len(batch) if batch else 0
            num_to_add = self.completion_batch_size - num_active
            while num_to_add >= self.prefill_batch_size:
                prompts = self.unprocessed_prompts[: self.prefill_batch_size]
                # Finish processing the last examples of the last batch
                if len(prompts) == 0 and num_active > 0:
                    break
                # No more prompts and no more completions, all done
                elif len(prompts) == 0:
                    self.active_batch = None
                    return []
                # Process prompts
                if batch is not None and not prompt_processing:
                    # Finish any active completion tokens
                    mx.eval(batch.y, batch.logprobs)
                batch = self._process_prompts(prompts)
                self.unprocessed_prompts = self.unprocessed_prompts[
                    self.prefill_batch_size :
                ]
                prompt_processing = True
                # If there was no active batch, set it
                if self.active_batch is None:
                    self.active_batch = batch
                else:
                    self.active_batch.extend(batch)

                num_active = len(self.active_batch)
                num_to_add -= len(batch)

            batch = self.active_batch
            y, logprobs = batch.y, batch.logprobs
            batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
            mx.async_eval(batch.y, batch.logprobs)
            return logprobs, batch

    class Query:
        """A query to a language model, waiting to be batched."""

        def __init__(self, prompt, future):
            self.prompt = prompt
            self.future = future

    class AsyncMlxLM(AsyncLM):
        def __init__(
            self,
            mlx_lm_model,
            tokenizer,
            cache_size=0,
            cache_opts={},
            batch_size=5,
            timeout=0.02,
            **batch_opts,
        ):
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
            self.queries = []
            self.batch_size = batch_size
            self.timeout = timeout
            self.timer = None
            self.batching = _supports_batching(self.mlx_lm_model) and batch_size > 1
            self.batch_opts = batch_opts

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
            mx.clear_cache()
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

        def reset_async_queries(self):
            """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
            to completion."""
            self.queries = []

        def _batch_logits_custom(
            self,
            prompts,
        ):
            """
            Compute next-token logits for each prompt in a batch using BatchGenerator.

            Args:
                model (nn.Module): The language model.
                prompts (List[List[int]]): Each inner list is a prompt of token IDs.
                verbose (bool): If True, prints progress info.
                kwargs: Passed through to BatchGenerator.

            Returns:
                Tuple[List[mx.array], Stats]: A list of logits arrays (one per prompt),
                and BatchGenerator statistics.
            """
            gen = BatchGeneratorCustom(
                self.mlx_lm_model, stop_tokens=[], **self.batch_opts
            )
            with wired_limit(self.mlx_lm_model, [self.generation_stream]):
                _ = gen.insert(prompts, 1)
                logprobs, batch = gen.next()
                self.gen = batch
            mx.clear_cache()
            return logprobs

        def batch_evaluate_queries(self):
            """
            Process a batch of queued language model queries.

            This method is called internally when the `batch_size` has been met or the `timeout` has expired.
            """

            queries, self.queries = self.queries, []
            if len(queries) == 0:
                return

            query_groups = defaultdict(list)
            for query in queries:
                key = tuple(query.prompt)
                query_groups[key].append(query)

            # Use one representative query from each group
            unique_queries = [group[0] for group in query_groups.values()]

            input_prompts = [q.prompt for q in unique_queries]
            if self.batching:
                results = self._batch_logits_custom(
                    input_prompts,
                )
            else:
                results = [
                    self.next_token_logprobs_sync(q.prompt) for q in unique_queries
                ]

            assert len(results) == len(unique_queries)

            results = _to_torch(results)
            for i, q in enumerate(unique_queries):
                for dup_query in query_groups[tuple(q.prompt)]:
                    dup_query.future.set_result(results[i])

        def add_query(self, query, future):
            """Add a query to be evaluated in the next batch.

            This method is called internally when a `next_token_logprobs` request is made.

            Args:
                query (list[int]): Token IDs representing the query prompt
                future (asyncio.Future): Future to store the result in
            """
            self.queries.append(Query(query, future))

            if self.timer:
                self.timer.cancel()
                self.timer = None
            if len(self.queries) >= self.batch_size:
                self.batch_evaluate_queries()
            else:
                self.timer = asyncio.get_running_loop().call_later(
                    self.timeout, lambda: self.batch_evaluate_queries()
                )

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token. This version is asynchronous because it automatically batches concurrent requests; use with `await`.

            Args:
                token_ids (list[int]): a list of token ids, representing a prompt to the language model.

            Returns:
                logprobs (torch.Tensor): a tensor of with the language model's log (normalized) probabilities for the next token following the prompt.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")

            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            future = asyncio.get_running_loop().create_future()
            self.add_query(token_ids, future)
            logprobs = await future
            if self.cache is not None:
                self.cache[key] = logprobs
            return logprobs

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")

            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            token_ids_array = mx.array(token_ids)
            logprobs = _to_torch(self._generate_step_custom(token_ids_array))
            if self.cache is not None:
                self.cache[key] = logprobs
            return logprobs

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
