import torch
import asyncio
import numpy as np
from abc import ABC, abstractmethod

from genlm_backend.vocabulary import decode_vocab

class AsyncLM(ABC):
    def __init__(self, tokenizer, eos_token=None):
        self.tokenizer = tokenizer
        self.byte_vocab, self.str_vocab = decode_vocab(self.tokenizer)
        self.eos_token = eos_token if eos_token else tokenizer.eos_token

    @abstractmethod
    async def next_token_logprobs(self, token_ids):
        """Request log probabilities of next token asynchronously.
        
        Args:
            token_ids (List[int]): A list of token IDs representing the prompt.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        pass

    @abstractmethod
    def next_token_logprobs_sync(self, token_ids):
        """Request log probabilities of next token synchronously.
        
        Args:
            token_ids (List[int]): A list of token IDs representing the prompt.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        pass

    async def batch_next_token_logprobs(self, token_ids_list):
        """Batch request log probabilities for multiple token sequences asynchronously.
        
        Args:
            token_ids_list (List[List[int]]): A list of token ID lists.
                
        Returns:
            torch.Tensor: A tensor of log probability tensors.
        """
        return torch.stack(await asyncio.gather(
            *[self.next_token_logprobs(token_ids) for token_ids in token_ids_list]
        ))


class MockAsyncLM(AsyncLM):
    def __init__(self, tokenizer, eos_token=None):
        super().__init__(tokenizer, eos_token)
        self._rng = np.random.RandomState(42)

    @classmethod
    def from_name(cls, model_name, **kwargs):
        from transformers import AutoTokenizer
        return cls(AutoTokenizer.from_pretrained(model_name, **kwargs))
        
    def _get_logits(self, token_ids):
        # Use token_ids to seed the random generator
        # This ensures same token_ids always produce same logits
        seed = sum([(i + 1) * t for i, t in enumerate(token_ids)])
        self._rng.seed(seed)
        logits = torch.from_numpy(self._rng.rand(len(self.tokenizer)).astype(np.float32))
        return torch.softmax(logits, dim=-1)

    async def next_token_logprobs(self, token_ids):
        return self._get_logits(token_ids)

    def next_token_logprobs_sync(self, token_ids):
        return self._get_logits(token_ids)
