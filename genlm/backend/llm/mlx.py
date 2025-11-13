import asyncio
import warnings
from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import DynamicTokenTrie
from collections import defaultdict
import torch
from dataclasses import dataclass


try:
    import mlx_lm
    from mlx_lm.generate import (
        generate_step,
        wired_limit,
        _left_pad_prompts,
        _make_cache,
    )
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import (
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

    @dataclass
    class Query:
        """A query to a language model, waiting to be batched.

        Attributes:
            prompt (list[int]): Token IDs representing the input prompt.
            future (asyncio.Future): Future object to store the result when
                the query is processed.
            past (mx.array, optional): Past key-value cache states from
                previous computations. Defaults to None.
            node (DynamicTokenTrie, optional): The cache node where this query
                should be stored. Defaults to None.
            next_token_index (int, optional): The index in the prompt where
                new tokens start (after cached prefix). Defaults to None.
        """

        prompt: list[int]
        future: asyncio.Future
        past: mx.array | None = None
        node: DynamicTokenTrie | None = None
        next_token_index: int | None = None

    class AsyncMlxLM(AsyncLM):
        """Asynchronous MLX-based language model wrapper.

        This class provides an async interface to MLX language models with
        automatic batching, caching, and KV cache management. It extends
        AsyncLM to provide efficient batched inference with prefix caching.

        The model automatically batches concurrent requests and uses a trie-based
        cache to store computed log probabilities and KV states for reuse.
        """

        def __init__(
            self,
            mlx_lm_model,
            tokenizer,
            batch_size=5,
            timeout=0.001,
            prefill_step_size=2048,
            cache_size=400,
        ):
            """Initialize an `AsyncMlxLM` instance.

            Args:
                mlx_lm_model: The MLX language model instance.
                tokenizer: The tokenizer for encoding/decoding text.
                batch_size (int, optional): Maximum number of queries to batch
                    together.
                timeout (float, optional): Maximum time in seconds to wait
                    before processing a batch, even if batch_size is not met.
                prefill_step_size (int, optional): Number of tokens to process
                    per step during prompt prefilling.
                cache_size (int, optional): Maximum number of KV cache entries
                    to keep in memory.
            """
            self.mlx_lm_model = mlx_lm_model
            self.tokenizer = tokenizer
            self.cache = DynamicTokenTrie()
            self.generation_stream = mx.new_stream(mx.default_device())
            self.queries = []
            self.timeout = timeout
            self.timer = None
            self.prefill_step_size = prefill_step_size
            self.cache_size = cache_size

            self.batch_size = batch_size
            self.kv_cachable = self._kv_cachable(self.mlx_lm_model)
            if not self.kv_cachable:
                warnings.warn(
                    f"Model {type(self.mlx_lm_model).__name__} does not support KV caching; "
                    f"prefix caching will be disabled.",
                    UserWarning,
                    stacklevel=2,
                )
            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_name, **kwargs):
            """Create an `AsyncMlxLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load. Can be a Hugging Face
                    model identifier or local path.
                **kwargs: Additional arguments passed to `AsyncMlxLM` constructor,
                    such as `batch_size`, `timeout`, `prefill_step_size`, `cache_size`.

            Returns:
                AsyncMlxLM: An `AsyncMlxLM` instance with the loaded model and tokenizer.
            """

            model, tokenizer = mlx_lm.load(model_name)
            return cls(model, tokenizer, **kwargs)

        @staticmethod
        def _to_torch(logprobs):
            """Convert MLX arrays into PyTorch tensors."""
            if logprobs.dtype is mx.bfloat16:
                logprobs = logprobs.astype(mx.float16)
            return torch.tensor(logprobs)

        @staticmethod
        def _kv_cachable(mlx_lm_model):
            """Check if an MLX model supports KV cache storage.

            A model is KV-cacheable if all its cache layers are KVCache or
            RotatingKVCache with keep=0.
            """
            if not hasattr(mlx_lm_model, "make_cache"):
                return True
            cache = mlx_lm_model.make_cache()
            return all(
                isinstance(c, KVCache)
                or (isinstance(c, RotatingKVCache) and c.keep == 0)
                for c in cache
            )

        def clear_cache(self):
            """Clear the output cache and MLX device cache.

            This method resets the internal token trie cache and clears
            any cached arrays on the MLX device to free memory.
            """
            if self.cache is not None:
                self.cache = DynamicTokenTrie()
            mx.clear_cache()

        def walk_cache(self, token_ids):
            """Walk the cache tree to find the deepest node matching a sequence of tokens.

            Args:
                token_ids (list[int]): Sequence of token IDs to follow in the cache tree

            Returns:
                tuple: A 5-tuple containing:
                    - node: The deepest node in the cache tree that matches
                        the token sequence, irregardless of whether its kv is cached or not
                    - next_token_index: Number of tokens matched from the start of token_ids
                    - past_kvs: Past key/value states concatenated from cached nodes, or None if no cached states were found
                    - kv_node: The cache node where KV states start
                    - kv_next_token_index: Number of tokens matched from the start of token_ids for the KV states
            """
            # Walk while tokens can be found
            node = self.cache
            kv_next_token_index = 0
            kv_node = node
            collecting = True
            next_token_index = 0
            past_kvs = []

            while next_token_index < len(token_ids):
                if node.past_key_values is not None and collecting:
                    past_kvs.append(node.past_key_values)
                    kv_node = node
                    kv_next_token_index = next_token_index
                elif next_token_index > 0:
                    collecting = False
                if node.has_token(token_ids[next_token_index]):
                    node = node.get_token(token_ids[next_token_index])
                    next_token_index += 1
                else:
                    break

            past_kvs = None if len(past_kvs) == 0 else mx.concatenate(past_kvs, axis=3)

            return node, next_token_index, past_kvs, kv_node, kv_next_token_index

        def cache_kv(self, token_ids):
            """Pre-compute and cache KV states for a given token sequence."""
            query = Query(token_ids, None, None, self.cache, 0)
            self._batch_logits_custom([query])

        def reset_async_queries(self):
            """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
            to completion."""
            self.queries = []

        def add_to_cache(self, queries, prompt_cache=None, logprobs=None):
            """Add computed log probabilities and KV states to the cache tree."""
            left_paddings = prompt_cache[0].left_padding.tolist()
            for i, query in enumerate(queries):
                token_ids, node, next_token_index = (
                    query.prompt,
                    query.node,
                    query.next_token_index,
                )
                if node is None or next_token_index is None:
                    node = self.cache
                    next_token_index = 0
                lp = left_paddings[i]
                if prompt_cache is not None and self.kv_cachable:
                    keys = [
                        c.keys[i, :, lp + next_token_index : lp + len(token_ids), :]
                        for c in prompt_cache
                    ]
                    values = [
                        c.values[i, :, lp + next_token_index : lp + len(token_ids), :]
                        for c in prompt_cache
                    ]
                    keys = mx.stack(keys, axis=0)
                    values = mx.stack(values, axis=0)
                    keys_values = mx.stack([keys, values], axis=0)
                    node.extend_cache(
                        next_token_index, token_ids, logprobs[i], keys_values
                    )
                else:
                    node.extend_cache(next_token_index, token_ids, logprobs[i])

            self.cache.evict_lru_kv(self.cache_size)

        def _process_kv(self, left_paddings, prompt_cache, pasts=None, step_size=256):
            """Process and integrate past KV cache states into prompt cache.

            This method takes past key-value cache states from the cache tree
            and integrates them into the prompt cache for efficient prefix
            reuse. It handles padding and alignment of cache states across
            different query lengths.

            Args:
                left_paddings (list[int]): Left padding amounts for each query
                    in the batch.
                prompt_cache (list): List of cache objects to update with
                    past states.
                pasts (list[mx.array], optional): List of past KV cache states,
                    one per query.
                step_size (int, optional): Step size for cache size alignment.

            Returns:
                tuple: A 2-tuple containing:
                    - list: Updated prompt_cache objects
                    - cached_len: Number of tokens that were cached
            """
            if pasts is None or all(past is None for past in pasts):
                return prompt_cache, 0
            max_match_lengths = [0 if past is None else past.shape[3] for past in pasts]
            min_pos_cached = min(
                ml + lp for ml, lp in zip(max_match_lengths, left_paddings)
            )
            cache_grabs = [max(min_pos_cached - lp, 0) for lp in left_paddings]
            non_zero_index = next(
                (i for i, grab in enumerate(cache_grabs) if grab), None
            )
            if non_zero_index is None:
                return prompt_cache, 0
            _, num_layers, N, _, D = pasts[non_zero_index].shape
            cache_size = (step_size + min_pos_cached - 1) // step_size * step_size
            right_paddings = [
                max(cache_size - lp - max_len, 0)
                for lp, max_len in zip(left_paddings, max_match_lengths)
            ]
            padded_pasts = []
            for past, lp, rp in zip(pasts, left_paddings, right_paddings):
                if past is None:
                    padded_pasts.append(mx.zeros((2, num_layers, N, cache_size, D)))
                else:
                    padded_pasts.append(
                        mx.pad(
                            past[:, :, :, : cache_size - lp, :],
                            ((0, 0), (0, 0), (0, 0), (lp, rp), (0, 0)),
                        )
                    )

            padded_pasts = mx.stack(padded_pasts, axis=2)
            for i, cache in enumerate(prompt_cache):
                cache.keys = padded_pasts[0, i]
                cache.values = padded_pasts[1, i]
                cache.offset += min_pos_cached
                cache._idx += min_pos_cached
            return prompt_cache, min_pos_cached

        def _process_prompts(self, queries):
            """Process a batch of prompts and compute next-token log probabilities."""
            inputs = [q.prompt for q in queries]
            pasts = [q.past for q in queries]
            lengths = [len(p) for p in inputs]
            max_length = max(lengths)
            left_padding = [max_length - length for length in lengths]
            prompt_cache = _make_cache(self.mlx_lm_model, left_padding)
            inputs_padded = _left_pad_prompts(inputs, max_length=max_length)

            if self.kv_cachable:
                prompt_cache, cached_len = self._process_kv(
                    left_padding, prompt_cache, pasts
                )
            else:
                cached_len = 0
            inputs_padded = inputs_padded[:, cached_len:]

            while inputs_padded.shape[1] > 1:
                n_to_process = min(self.prefill_step_size, inputs_padded.shape[1] - 1)
                self.mlx_lm_model(inputs_padded[:, :n_to_process], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                inputs_padded = inputs_padded[:, n_to_process:]

            logits = self.mlx_lm_model(inputs_padded, cache=prompt_cache)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            mx.async_eval(logprobs)

            return logprobs, prompt_cache

        def _batch_logits_custom(
            self,
            queries,
        ):
            """Compute next-token log probabilities for each query in a batch and add to cache.
            Args:
                queries (list[Query]): List of query objects to process.
            Returns:
                logprobs (list[torch.Tensor]): List of normalized log probability tensors."""
            with wired_limit(self.mlx_lm_model, [self.generation_stream]):
                logprobs, prompt_cache = self._process_prompts(queries)
                logprobs = AsyncMlxLM._to_torch(logprobs)
            mx.clear_cache()
            self.add_to_cache(queries, prompt_cache, logprobs)
            return logprobs

        def batch_evaluate_queries(self):
            """Process a batch of queued language model queries."""

            queries, self.queries = self.queries, []
            if len(queries) == 0:
                return

            query_groups = defaultdict(list)
            for query in queries:
                key = tuple(query.prompt)
                query_groups[key].append(query)

            # Use one representative query from each group
            unique_queries = [group[0] for group in query_groups.values()]

            results = self._batch_logits_custom(unique_queries)

            assert len(results) == len(unique_queries)

            results = results
            for i, q in enumerate(unique_queries):
                for dup_query in query_groups[tuple(q.prompt)]:
                    dup_query.future.set_result(results[i])

        def add_query(self, query):
            """Add a query to be evaluated in the next batch and reset the timeout."""
            self.queries.append(query)

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

            node, next_token_index, past, kv_node, kv_next_token_index = (
                self.walk_cache(token_ids)
            )
            if next_token_index == len(token_ids) and node.logprobs is not None:
                return node.logprobs

            future = asyncio.get_running_loop().create_future()
            query = Query(token_ids, future, past, kv_node, kv_next_token_index)
            self.add_query(query)
            logprobs = await future

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

            node, next_token_index, past, kv_node, kv_next_token_index = (
                self.walk_cache(token_ids)
            )
            if next_token_index == len(token_ids) and node.logprobs is not None:
                return node.logprobs

            query = Query(token_ids, None, past, kv_node, kv_next_token_index)
            logprobs = self._batch_logits_custom([query])[0]

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
                prompt_token_ids (list[int]): The token IDs of the prompt to
                    start generation from.
                max_tokens (int): The maximum number of tokens to generate.
                eos_token_ids (list[int]): The token IDs that signal
                    end-of-sequence. Generation stops when one of these is
                    sampled.
                temperature (float, optional): The temperature to use for
                    sampling. Higher values make the distribution more uniform,
                    lower values make it more peaked. Defaults to 1.0.
                seed (int, optional): The seed for the random number generator.
                    If provided, sets the random seed before sampling.
                    Defaults to None.

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
