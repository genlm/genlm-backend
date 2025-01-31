import torch
import asyncio
import numpy as np
from abc import ABC, abstractmethod

from genlm_backend.tokenization import decode_vocab


class AsyncLM(ABC):
    """Abstract base class for asynchronous language models.

    This class provides an interface for language models that can generate token probabilities
    asynchronously. It handles tokenization and vocabulary management.

    Args:
        tokenizer: A Hugging Face tokenizer instance compatible with the language model
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.byte_vocab, self.str_vocab = decode_vocab(self.tokenizer)

    @abstractmethod
    async def next_token_logprobs(self, token_ids):
        """Request log probabilities of next token asynchronously.

        Args:
            token_ids (list[int]): A list of token IDs representing the prompt.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        pass

    @abstractmethod
    def next_token_logprobs_sync(self, token_ids):
        """Request log probabilities of next token synchronously.

        Args:
            token_ids (list[int]): A list of token IDs representing the prompt.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        pass

    async def batch_next_token_logprobs(self, token_ids_list):
        """Batch request log probabilities for multiple token sequences asynchronously.

        Args:
            token_ids_list (list[list[int]]): A list of token ID lists.

        Returns:
            (torch.Tensor): A tensor of log probability tensors.
        """
        logprobs = await asyncio.gather(
            *[self.next_token_logprobs(token_ids) for token_ids in token_ids_list]
        )

        return torch.stack(logprobs)

    def batch_next_token_logprobs_sync(self, token_ids_list):
        """Batch request log probabilities for multiple token sequences synchronously.

        Args:
            token_ids_list (list[list[int]]): A list of token ID lists.

        Returns:
            (torch.Tensor): A tensor of log probability tensors.
        """
        return torch.stack(
            [self.next_token_logprobs_sync(token_ids) for token_ids in token_ids_list]
        )

    def clear_cache(self):
        """Clear any caches used by the language model. No-op in base class."""
        pass


class MockAsyncLM(AsyncLM):
    """Mock implementation of AsyncLM used for testing."""

    def __init__(self, tokenizer):
        """Initialize a `MockAsyncLM` instance.

        Args:
            tokenizer: Hugging Face tokenizer instance
        """
        super().__init__(tokenizer)
        self._rng = np.random.RandomState(42)

    @classmethod
    def from_name(cls, model_name, **kwargs):
        """Create a MockAsyncLM instance over the vocabulary of the model's tokenizer.

        Args:
            model_name (str): Name of pretrained model to load tokenizer from
            **kwargs: Additional arguments passed to `MockAsyncLM` constructor

        Returns:
            (MockAsyncLM): `MockAsyncLM` instance initialized with tokenizer from `model_name`
        """
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_name), **kwargs)

    async def next_token_logprobs(self, token_ids):
        """Get next token log probabilities asynchronously.

        Args:
            token_ids (list[int]): Input token IDs.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        return self._get_logprobs(token_ids)

    def next_token_logprobs_sync(self, token_ids):
        """Get next token log probabilities synchronously.

        Args:
            token_ids (list[int]): Input token IDs.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        return self._get_logprobs(token_ids)

    def _get_logprobs(self, token_ids):
        """Generate random but deterministic log probabilities for given tokens.

        Uses token_ids to seed the random generator, ensuring same inputs produce same outputs.

        Args:
            token_ids (list[int]): Input token IDs.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        seed = sum([(i + 1) * t for i, t in enumerate(token_ids)])
        self._rng.seed(seed)
        logits = torch.from_numpy(
            self._rng.rand(len(self.tokenizer)).astype(np.float32)
        )
        return torch.log_softmax(logits, dim=-1)
