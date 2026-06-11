import asyncio
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import DynamicCache

from genlm.backend.cache import TokenTrie
from genlm.backend.llm.base import AsyncLM


class Query:
    """A query to a language model, waiting to be batched."""

    def __init__(self, prompt, future, past=None, lora_name=None):
        self.prompt = prompt
        self.future = future
        self.past = past
        self.lora_name = lora_name

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
        # One logprob/KV trie per LoRA adapter (None = base): cached activations
        # are adapter-dependent, so the tries must never mix.
        self.caches = defaultdict(TokenTrie)

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
        self.caches = defaultdict(TokenTrie)

    def clear_kv_cache(self):
        """Clear any key and value vectors from the cache."""
        for trie in self.caches.values():
            trie.clear_kv_cache()

    def reset_async_queries(self):
        """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
        to completion."""
        self.queries = []

    @torch.no_grad()
    def cache_kv(self, prompt_tokens, lora_name=None):
        """Cache the key and value vectors for a prompt. Future queries that have this prompt as a prefix will only run the LLM on new tokens.

        Args:
            prompt_tokens (list[int]): token ids for the prompt to cache.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).
        """
        self._activate(lora_name)
        result = self.model(torch.tensor([prompt_tokens]).to(self.device))
        node = self.caches[lora_name].extend_cache(
            0, prompt_tokens, result.logits[0], 0
        )
        node.past_key_values = result.past_key_values

    def _has_adapter(self, lora_name):
        return (
            getattr(self.model, "_hf_peft_config_loaded", False)
            and lora_name in self.model.peft_config
        )

    def add_new_lora(self, lora_path, lora_name="lora_1"):
        """Register a LoRA adapter under ``lora_name``.

        Re-registering an existing name evicts the old weights and binds the
        name to ``lora_path`` — a training loop pushes updated weights with this
        one call. Forwards select the adapter per call via ``lora_name=``
        (or ``lora_view``).

        Args:
            lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
            lora_name (str): Name to assign to the loaded adapter.
        """
        if self._has_adapter(lora_name):
            self.remove_lora(lora_name)
        self.model.load_adapter(lora_path, lora_name)

    def remove_lora(self, lora_name):
        """Unregister ``lora_name``: drop its weights and its cache trie."""
        self.model.delete_adapter(lora_name)
        self.caches.pop(lora_name, None)

    def _activate(self, lora_name):
        """Set the model's peft state for a forward under ``lora_name`` (``None`` = base).

        Only manages adapters registered via ``add_new_lora``; a model whose adapters
        live outside the transformers peft mixin (e.g. a peft training wrapper sharing
        the weights) is left untouched in the ``None`` case.
        """
        loaded = getattr(self.model, "_hf_peft_config_loaded", False)
        if lora_name is None:
            if loaded:
                self.model.disable_adapters()
        else:
            if not loaded or lora_name not in self.model.peft_config:
                raise ValueError(
                    f"A LoRA adapter named '{lora_name}' has not been loaded yet. "
                    "Call add_new_lora() first."
                )
            self.model.set_adapter(lora_name)
            self.model.enable_adapters()

    @torch.no_grad()
    def batch_evaluate_queries(self):
        """
        Process a batch of queued language model queries, one forward per
        distinct LoRA adapter.

        This method is called internally when the `batch_size` has been met or the `timeout` has expired.
        """

        queries, self.queries = self.queries, []
        if len(queries) == 0:
            return

        by_lora = defaultdict(list)
        for query in queries:
            by_lora[query.lora_name].append(query)
        for lora_name, group in by_lora.items():
            # Trap per group and fail its futures: an exception escaping a timer
            # callback would leave the awaiting callers hung forever.
            try:
                self._activate(lora_name)
                self._evaluate_queries(group)
            except Exception as exc:
                for query in group:
                    if not query.future.done():
                        query.future.set_exception(exc)

    @torch.no_grad()
    def _evaluate_queries(self, queries):
        """One padded forward over ``queries`` (all under the currently active adapter)."""
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

        pasts = DynamicCache.from_legacy_cache(pasts)

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
    def add_query(self, query, future, past, lora_name=None):
        """Add a query to be evaluated in the next batch.

        This method is called internally when a `next_token_logprobs` request is made.

        Args:
            query (list[int]): Token IDs representing the query prompt
            future (asyncio.Future): Future to store the result in
            past (list[tuple[torch.Tensor]]|None): Past key/value states from previous evaluation,
                or None if this is a new query
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).
        """
        self.queries.append(Query(query, future, past, lora_name))

        if self.timer:
            self.timer.cancel()
            self.timer = None
        if len(self.queries) >= self.batch_size:
            self.batch_evaluate_queries()
        else:
            self.timer = asyncio.get_running_loop().call_later(
                self.timeout, lambda: self.batch_evaluate_queries()
            )

    def walk_cache(self, token_ids, lora_name=None):
        """Walk the cache tree to find the deepest node matching a sequence of tokens.

        Args:
            token_ids (list[int]): Sequence of token IDs to follow in the cache tree
            lora_name (str, optional): Which adapter's cache tree to walk (``None`` = base).

        Returns:
            tuple:
                - CacheNode: The deepest node in the cache tree that matches the token sequence
                - int: Number of tokens matched from the start of token_ids
                - list[tuple[torch.Tensor]]|None: Past key/value states from the deepest cached node,
                    or None if no cached states were found
                - int: Base index indicating where the past states start in token_ids
        """
        # Walk while tokens can be found
        node = self.caches[lora_name]
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
    async def next_token_logprobs(self, token_ids, lora_name=None):
        """Request log probabilities of next token. This version is asynchronous because it automatically batches concurrent requests; use with `await`.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            logprobs (torch.Tensor): a tensor of with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        if not token_ids:
            raise ValueError("Token ids must not be empty")

        node, next_token_index, past, base = self.walk_cache(token_ids, lora_name)

        # If we processed all tokens, then we're done.
        if next_token_index == len(token_ids):
            return node.logprobs

        # Create a future with the prompt
        future = asyncio.get_running_loop().create_future()
        self.add_query(token_ids[base:], future, past, lora_name)
        logits = await future

        # Create new nodes
        node = node.extend_cache(next_token_index, token_ids, logits, base)

        return node.logprobs

    @torch.no_grad()
    def next_token_logprobs_sync(self, token_ids, lora_name=None):
        """Request log probabilities of next token. Not asynchronous, and does not support auto-batching.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            logprobs (torch.Tensor): a tensor with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        if not token_ids:
            raise ValueError("Token ids must not be empty")

        # Walk while tokens can be found
        node, next_token_index, past, base = self.walk_cache(token_ids, lora_name)

        if next_token_index == len(token_ids):
            return node.logprobs

        self._activate(lora_name)
        logits = self.model(
            torch.tensor([token_ids[base:]]).to(self.device),
            past_key_values=node.past_key_values,
            use_cache=node.past_key_values is not None,
        ).logits[0]

        node = node.extend_cache(next_token_index, token_ids, logits, base)

        return node.logprobs

    def next_token_logprobs_uncached(self, token_ids, lora_name=None):
        """Request log probabilities of next token. No KV or output caching, and does not support auto-batching.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            logprobs (torch.Tensor): a tensor with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        if not token_ids:
            raise ValueError("Token ids must not be empty")

        with torch.no_grad():
            self._activate(lora_name)
            logits = self.model(
                torch.tensor([token_ids]).to(self.device),
                past_key_values=None,
                use_cache=False,
            ).logits[0]
            return torch.log_softmax(logits[-1], dim=0)
