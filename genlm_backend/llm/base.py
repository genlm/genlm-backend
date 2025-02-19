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

    async def sequence_logprob(self, token_ids, prompt_ids=None):
        """Calculate log probability of a token sequence given a prompt asynchronously.

        Args:
            token_ids (list[int]): A list of token IDs representing the sequence.
            prompt_ids (list[int]): A list of token IDs representing the prefix context.

        Returns:
            (torch.Tensor): The log probability of the sequence.
        """
        return await self.batch_sequence_logprob([token_ids], prompt_ids)[0]

    async def batch_sequence_logprob(self, token_ids_list, prompt_ids=None):
        """Calculate log probabilities for multiple token sequences given a prompt asynchronously.

        Args:
            token_ids_list (list[list[int]]): A list of token ID lists.
            prompt_ids (list[int], optional): A list of token IDs representing the prefix context.
                Defaults to None.

        Returns:
            (torch.Tensor): The log probability of each sequence in token_ids_list.
        """
        prepared = self._prepare_sequence_inputs(token_ids_list, prompt_ids)
        if prepared is None:
            return torch.zeros(len(token_ids_list))

        input_ids, target_ids, split_sizes = prepared
        log_ps = await self.batch_next_token_logprobs(input_ids)
        return self._gather_scores(log_ps, target_ids, split_sizes)

    def sequence_logprob_sync(self, token_ids, prompt_ids=None):
        """Calculate log probability of a token sequence given a prompt synchronously."""
        return self.batch_sequence_logprob_sync([token_ids], prompt_ids)[0]

    def batch_sequence_logprob_sync(self, token_ids_list, prompt_ids=None):
        """Calculate log probabilities for multiple token sequences given a prompt synchronously."""
        prepared = self._prepare_sequence_inputs(token_ids_list, prompt_ids)
        if prepared is None:
            return torch.zeros(len(token_ids_list))

        input_ids, target_ids, split_sizes = prepared
        log_ps = self.batch_next_token_logprobs_sync(input_ids)
        return self._gather_scores(log_ps, target_ids, split_sizes)

    def _prepare_sequence_inputs(self, token_ids_list, prompt_ids):
        """Prepare input-target pairs for sequence log likelihood computation.

        Args:
            token_ids_list (list[list[int]]): List of token sequences to score
            prompt_ids (list[int]): Prefix context tokens

        Returns:
            tuple: (input_ids, target_ids, split_sizes) or None if empty input
        """
        if not token_ids_list:
            raise ValueError("token_ids_list cannot be empty")

        prompt_ids = prompt_ids or []

        input_target_pairs = [
            (prompt_ids + ids[:i], ids[i])
            for ids in token_ids_list
            for i in range(len(ids))
        ]

        if not input_target_pairs:
            return None

        input_ids, target_ids = zip(*input_target_pairs)
        split_sizes = [len(ids) for ids in token_ids_list]

        return input_ids, target_ids, split_sizes

    def _gather_scores(self, log_ps, target_ids, split_sizes):
        """Sum the log probabilities from the log_ps tensor for each sequence.

        Args:
            log_ps (torch.Tensor): Token log probabilities
            target_ids (list[int]): Target token ids
            split_sizes (list[int]): Sizes of each sequence

        Returns:
            torch.Tensor: Log probability for each sequence
        """
        target_tensor = torch.tensor(target_ids, device=log_ps.device).unsqueeze(1)
        token_logprobs = log_ps.gather(1, target_tensor).squeeze(1)
        return torch.stack([chunk.sum() for chunk in token_logprobs.split(split_sizes)])


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
