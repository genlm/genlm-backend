import asyncio
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

from genlm_backend.cache import TokenTrie
from genlm_backend.llm.base import AsyncLM


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

    @torch.no_grad()
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
            )

    def prompt_padded(self, pad_token, to_length):
        return [*self.prompt, *[pad_token for _ in range(to_length - len(self.prompt))]]

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


class AsyncTransformer(AsyncLM):
    """Asynchronous wrapper around a HuggingFace causal language model with caching support.

    This class provides an asynchronous interface to HuggingFace language models with automatic batching
    and caching (output and KV) for improved efficiency.
    """

    @classmethod
    def from_name(cls, model_id, bitsandbytes_opts=None, hf_opts=None, **kwargs):
        """Create an AsyncTransformer instance from a pretrained HuggingFace model.

        Args:
            model_id (str): Model identifier in HuggingFace's model hub.
            bitsandbytes_opts (dict, optional): Additional configuration options for bitsandbytes quantization.
                Defaults to None.
            hf_opts (dict, optional): Additional configuration options for loading the HuggingFace model.
                Defaults to None.
            **kwargs: Additional arguments passed to the `AsyncTransformer` constructor

        Returns:
            (AsyncTransformer): An initialized `AsyncTransformer` instance.
        """
        if bitsandbytes_opts:
            bnb_config = BitsAndBytesConfig(**bitsandbytes_opts)
        else:
            bnb_config = None

        _hf_opts = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        if hf_opts:
            _hf_opts.update(hf_opts)

        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, **_hf_opts
        )

        return cls(mod, tok, **kwargs)

    @torch.no_grad()
    def __init__(self, hf_model, hf_tokenizer, batch_size=20, timeout=0.02):
        """Initialize an AsyncTransformer instance.

        Args:
            hf_model: A HuggingFace CausalLM model instance.
            hf_tokenizer: A HuggingFace Tokenizer.
            batch_size (int, optional): Maximum queries to process in one batch during auto-batching.
                Defaults to 20.
            timeout (float, optional): Seconds to wait since last query before processing current batch.
                Defaults to 0.02.
        """
        self.model = hf_model
        self.tokenizer = hf_tokenizer
        self.device = hf_model.device
        self.cache = TokenTrie()

        # Queries to be batched. Each query is a sequence of tokens,
        # and a Future to be called when the query is resolved.
        self.queries = []
        self.batch_size = batch_size
        self.timeout = timeout
        self.timer = None

        self.model.eval()

        super().__init__(tokenizer=self.tokenizer)

    def clear_cache(self):
        """Clear the cache of log probabilities and key/value pairs."""
        self.cache = TokenTrie(None, self.cache.logprobs)

    def clear_kv_cache(self):
        """Clear any key and value vectors from the cache."""
        self.cache.clear_kv_cache()

    def reset_async_queries(self):
        """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
        to completion."""
        self.queries = []

    @torch.no_grad()
    def cache_kv(self, prompt_tokens):
        """Cache the key and value vectors for a prompt. Future queries that have this prompt as a prefix will only run the LLM on new tokens.

        Args:
            prompt_tokens (list[int]): token ids for the prompt to cache.
        """
        result = self.model(torch.tensor([prompt_tokens]).to(self.device))
        node = self.cache.extend_cache(1, prompt_tokens, result.logits[0], 0)
        node.past_key_values = result.past_key_values

    @torch.no_grad()
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
            key = tuple(query.prompt)  # XXX: cache based on past_len too?
            query_groups[key].append(query)

        # Use one representative query from each group
        unique_queries = [group[0] for group in query_groups.values()]

        past_example = next((q.past for q in unique_queries if q.past), False)
        max_past_length = max(q.past_len for q in unique_queries)
        max_query_length = max(len(q.prompt) for q in unique_queries)

        padding_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        input_ids = torch.tensor(
            [
                q.prompt_padded(padding_token_id, max_query_length)
                for q in unique_queries
            ]
        ).to(self.device)
        attn_masks = torch.tensor(
            [
                q.attention_mask(max_past_length, max_query_length)
                for q in unique_queries
            ]
        ).to(self.device)
        posn_ids = torch.tensor(
            [q.position_ids(max_past_length, max_query_length) for q in unique_queries]
        ).to(self.device)
        if past_example:
            pasts = [
                [
                    torch.cat(
                        (
                            *(
                                q.past_padded(
                                    layer,
                                    j,
                                    max_past_length,
                                    past_example[0][0].dtype,
                                    self.device,
                                    past_example[0][0].shape,
                                )
                                for q in unique_queries
                            ),
                        ),
                        dim=0,
                    )
                    for j in range(2)
                ]
                for layer in range(len(past_example))
            ]
        else:
            pasts = None

        results = self.model(
            input_ids,
            attention_mask=attn_masks,
            position_ids=posn_ids,
            past_key_values=pasts,
            use_cache=pasts is not None,
        )

        assert len(results.logits) == len(unique_queries)

        for i, q in enumerate(unique_queries):
            result = results.logits[i]
            for dup_query in query_groups[tuple(q.prompt)]:
                dup_query.future.set_result(result)

    @torch.no_grad()
    def add_query(self, query, future, past):
        """Add a query to be evaluated in the next batch.

        This method is called internally when a `next_token_logprobs` request is made.

        Args:
            query (list[int]): Token IDs representing the query prompt
            future (asyncio.Future): Future to store the result in
            past (list[tuple[torch.Tensor]]|None): Past key/value states from previous evaluation,
                or None if this is a new query
        """
        self.queries.append(Query(query, future, past))

        if self.timer:
            self.timer.cancel()
            self.timer = None
        if len(self.queries) >= self.batch_size:
            self.batch_evaluate_queries()
        else:
            self.timer = asyncio.get_running_loop().call_later(
                self.timeout, lambda: self.batch_evaluate_queries()
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
        # Walk while tokens can be found
        node = self.cache
        next_token_index = 0

        past = None
        base = 0
        while next_token_index < len(token_ids):
            if node.past_key_values is not None:
                past = node.past_key_values
                base = next_token_index
            if node.has_token(token_ids[next_token_index]):
                node = node.get_token(token_ids[next_token_index])
                next_token_index += 1
            else:
                break

        return node, next_token_index, past, base

    @torch.no_grad()
    async def next_token_logprobs(self, token_ids):
        """Request log probabilities of next token. This version is asynchronous because it automatically batches concurrent requests; use with `await`.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.

        Returns:
            logprobs (torch.Tensor): a tensor of with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        if not token_ids:
            raise ValueError("Token ids must not be empty")

        node, next_token_index, past, base = self.walk_cache(token_ids)

        # If we processed all tokens, then we're done.
        if next_token_index == len(token_ids):
            return node.logprobs

        # Create a future with the prompt
        future = asyncio.get_running_loop().create_future()
        self.add_query(token_ids[base:], future, past)
        logits = await future

        # Create new nodes
        node = node.extend_cache(next_token_index, token_ids, logits, base)

        return node.logprobs

    @torch.no_grad()
    def next_token_logprobs_sync(self, token_ids):
        """Request log probabilities of next token. Not asynchronous, and does not support auto-batching.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.

        Returns:
            logprobs (torch.Tensor): a tensor with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        if not token_ids:
            raise ValueError("Token ids must not be empty")

        # Walk while tokens can be found
        node, next_token_index, past, base = self.walk_cache(token_ids)

        if next_token_index == len(token_ids):
            return node.logprobs

        logits = self.model(
            torch.tensor([token_ids[base:]]).to(self.device),
            past_key_values=node.past_key_values,
            use_cache=node.past_key_values is not None,
        ).logits[0]

        node = node.extend_cache(next_token_index, token_ids, logits, base)

        return node.logprobs

    def next_token_logprobs_uncached(self, token_ids):
        """Request log probabilities of next token. No KV or output caching, and does not support auto-batching.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.

        Returns:
            logprobs (torch.Tensor): a tensor with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        if not token_ids:
            raise ValueError("Token ids must not be empty")

        with torch.no_grad():
            logits = self.model(
                torch.tensor([token_ids]).to(self.device),
                past_key_values=None,
                use_cache=False,
            ).logits[0]
            return torch.log_softmax(logits[-1], dim=0)
