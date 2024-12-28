import asyncio
import string
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

from async_llm.cache import TokenTrie
from async_llm.vocabulary import decode_vocab

class Query:
    """A query to a language model, waiting to be batched."""

    def __init__(self, prompt, future, past=None):
        self.prompt = prompt
        self.future = future
        self.past = past

        if self.past is not None:
            self.past_len = past[0][0].shape[
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

class AsyncTransformer:
    """Asynchronous wrapper around a HuggingFace causal language model with caching support.

    This class provides an asynchronous interface to HuggingFace language models with automatic batching
    and caching of previous computations for improved efficiency.

    Attributes:
        model: The underlying HuggingFace causal language model.
        tokenizer: The underlying HuggingFace tokenizer.
        device (str): PyTorch device identifier (e.g. "cpu" or "cuda:0") where the model is loaded.
        cache (TokenTrie): Cache storing previously evaluated log probabilities and key/value vectors.
        str_vocab (list[str]): List mapping token IDs to string tokens.
        byte_vocab (list[bytes]): List mapping token IDs to byte sequences.
        batch_size (int): Maximum number of queries to process in one batch during auto-batching.
        timeout (float): Seconds to wait since last query before processing current batch, even if not full.
        timer: Timer for tracking batch timeouts.
    """

    @classmethod 
    def from_name(cls, model_id, load_in_8bit=True, bitsandbytes_opts=None, hf_opts=None, **kwargs):
        """Create an AsyncTransformer instance from a pretrained HuggingFace model.

        Args:
            model_id (str): Model identifier in HuggingFace's model hub.
            load_in_8bit (bool): Whether to load model in 8-bit quantized form using bitsandbytes.
                Defaults to True.
            bitsandbytes_opts (dict, optional): Additional configuration options for bitsandbytes quantization.
                Defaults to None.
            hf_opts (dict, optional): Additional configuration options for loading the HuggingFace model.
                Defaults to None.
            **kwargs: Additional arguments passed to AsyncTransformer constructor

        Returns:
            AsyncTransformer: An initialized instance interfacing with the specified model.
        """
        bnb_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, **(bitsandbytes_opts or {}))
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", 
            quantization_config=bnb_config,
            **(hf_opts or {})
        )
        return cls(mod, tok, **kwargs)

    @torch.no_grad()
    def __init__(self, hf_model, hf_tokenizer, batch_size=20, timeout=0.02):
        """Initialize an AsyncTransformer instance.

        Args:
            hf_model: A HuggingFace CausalLM model instance.
            hf_tokenizer: A HuggingFace Tokenizer instance matching the model.
            batch_size (int, optional): Maximum queries to process in one batch during auto-batching.
                Defaults to 20.
            timeout (float, optional): Seconds to wait since last query before processing current batch.
                Defaults to 0.02.
        """
        self.model = hf_model
        self.tokenizer = hf_tokenizer
        self.device = hf_model.device

        self.cache = TokenTrie()

        self.str_vocab, self.byte_vocab = decode_vocab(self.tokenizer)

        # Queries to be batched. Each query is a sequence of tokens,
        # and a Future to be called when the query is resolved.
        self.queries = []
        self.batch_size = batch_size
        self.timeout = timeout
        self.timer = None

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

        queries, self.queries = self.queries, []
        if len(queries) == 0:
            return

        past_example = next((q.past for q in queries if q.past), False)
        max_past_length = max(q.past_len for q in queries)
        max_query_length = max(len(q.prompt) for q in queries)

        padding_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        input_ids = torch.tensor(
            [q.prompt_padded(padding_token_id, max_query_length) for q in queries]
        ).to(self.device)
        attn_masks = torch.tensor(
            [q.attention_mask(max_past_length, max_query_length) for q in queries]
        ).to(self.device)
        posn_ids = torch.tensor(
            [q.position_ids(max_past_length, max_query_length) for q in queries]
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
                                for q in queries
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

        for i, q in enumerate(queries):
            q.future.set_result(results.logits[i])

    @torch.no_grad()
    def add_query(self, query, future, past):
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
        # Walk while tokens can be found
        node = self.cache
        next_token_index = 1

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
            logprobs (torch.Tensor): a tensor of `len(vocab)`, with the language model's log (normalized) probabilities for the next token following the prompt.
        """

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
            logprobs (torch.Tensor): a tensor of `len(vocab)`, with the language model's log (normalized) probabilities for the next token following the prompt.
        """

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

    async def batch_next_token_logprobs(self, token_ids_list):
        """Batch request log probabilities for multiple token sequences asynchronously.
        
        Args:
            token_ids_list (List[List[int]]): A list of token ID lists, each representing a prompt to the language model.
                
        Returns:
            (torch.Tensor): A tensor of log probability tensors, one for each input sequence.
        """
        return torch.stack(await asyncio.gather(
            *[self.next_token_logprobs(token_ids) for token_ids in token_ids_list]
        ))
