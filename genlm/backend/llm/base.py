import torch
import asyncio
import numpy as np
from abc import ABC, abstractmethod

from genlm.backend.tokenization import decode_vocab


class _LoRABoundLM:
    """Forward handle that runs every logprobs call under a fixed LoRA adapter.
    Returned by ``AsyncLM.lora_view(name)``; duck-typed to the forward subset
    of ``AsyncLM`` that ``PromptedLLM`` uses on the slow lane."""

    def __init__(self, lm, lora_name):
        self._lm = lm
        self._lora_name = lora_name

    async def next_token_logprobs(self, token_ids):
        return await self._lm.next_token_logprobs(token_ids, lora_name=self._lora_name)

    def next_token_logprobs_sync(self, token_ids):
        return self._lm.next_token_logprobs_sync(token_ids, lora_name=self._lora_name)

    async def batch_next_token_logprobs(self, token_ids_list):
        return torch.stack(
            await asyncio.gather(*[self.next_token_logprobs(t) for t in token_ids_list])
        )


class AsyncLM(ABC):
    """Abstract base class for asynchronous language models.

    This class provides an interface for language models that can generate token probabilities
    asynchronously. It handles tokenization and vocabulary management.

    Args:
        tokenizer: A Hugging Face tokenizer instance compatible with the language model
    """

    supports_burst = False  # overridden True by backends with run_burst

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

    async def batch_next_token_logprobs(self, token_ids_list, lora_name=None):
        """Batch request log probabilities for multiple token sequences asynchronously.

        Args:
            token_ids_list (list[list[int]]): A list of token ID lists.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            (torch.Tensor): A tensor of log probability tensors.
        """
        fwd = self.lora_view(lora_name)
        logprobs = await asyncio.gather(
            *[fwd.next_token_logprobs(token_ids) for token_ids in token_ids_list]
        )

        return torch.stack(logprobs)

    def batch_next_token_logprobs_sync(self, token_ids_list, lora_name=None):
        """Batch request log probabilities for multiple token sequences synchronously.

        Args:
            token_ids_list (list[list[int]]): A list of token ID lists.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            (torch.Tensor): A tensor of log probability tensors.
        """
        fwd = self.lora_view(lora_name)
        return torch.stack(
            [fwd.next_token_logprobs_sync(token_ids) for token_ids in token_ids_list]
        )

    def add_new_lora(self, lora_path, lora_name):
        """Register a LoRA adapter under ``lora_name``; re-registering an existing
        name rebinds it to the new weights. Forwards select the adapter per call
        via ``lora_name=`` (or ``lora_view``).

        Args:
            lora_path (str): Path to the adapter weights directory or identifier in HuggingFace's model hub.
            lora_name (str): Name to assign to the loaded adapter.

        """
        raise NotImplementedError(
            "add_new_lora must be implemented by subclasses"
        )  # pragma: no cover

    def remove_lora(self, lora_name):
        """Unregister ``lora_name`` and evict its weights.

        Args:
            lora_name (str): Name of the adapter to remove.

        """
        raise NotImplementedError(
            "remove_lora must be implemented by subclasses"
        )  # pragma: no cover

    def lora_view(self, lora_name):
        """A forward handle whose logprobs calls run under LoRA adapter ``lora_name``
        (``None`` = base/self). The view forwards with ``lora_name=``, so a backend
        without LoRA support fails at the first forward."""
        return self if lora_name is None else _LoRABoundLM(self, lora_name)

    def set_lora(self, lora_path, lora_name):
        """Removed: adapter selection is per-request now."""
        raise RuntimeError(
            "set_lora() was removed: there is no active-adapter global anymore. "
            "Pass lora_name= per call (next_token_logprobs/sample/...) or bind a "
            "view with lora_view(name)."
        )

    def clear_lora(self):
        """Removed: ``lora_name=None`` (the default) is the base model."""
        raise RuntimeError(
            "clear_lora() was removed: omit lora_name (None = base model)."
        )

    def clear_cache(self):
        """Clear any caches used by the language model. No-op in base class."""
        pass  # pragma: no cover

    async def sample(
        self,
        prompt_token_ids,
        max_tokens,
        eos_token_ids,
        temperature=1.0,
        seed=None,
        lora_name=None,
    ):
        """Sample from the language model.

        Args:
            prompt_token_ids (list[int]): The token IDs of the prompt.
            eos_token_ids (list[int]): The token IDs of the end-of-sequence tokens.
            temperature (float, optional): The temperature to use to rescale the logits. Defaults to 1.0.
            max_tokens (int): The maximum number of tokens to generate.
            seed (int, optional): The seed for the random number generator. Defaults to None.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            (list[int]): The sampled token IDs.
        """
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        fwd = self.lora_view(lora_name)
        generated_token_ids = []
        for _ in range(max_tokens):
            logprobs = await fwd.next_token_logprobs(
                prompt_token_ids + generated_token_ids
            )
            probs = torch.softmax(logprobs / temperature, dim=-1)
            next_token_id = torch.multinomial(
                probs.cpu() if seed is not None else probs,
                num_samples=1,
                generator=generator,
            ).item()
            if next_token_id in eos_token_ids:
                break
            generated_token_ids.append(next_token_id)

        return generated_token_ids

    async def batch_sample(
        self,
        prompt_token_ids_list,
        max_tokens,
        eos_token_ids,
        temperature=1.0,
        seed=None,
        lora_name=None,
    ):
        """Batch sample from the language model.

        Args:
            prompt_token_ids_list (list[list[int]]): The token IDs of the prompts.
            max_tokens (int): The maximum number of tokens to generate.
            eos_token_ids (list[int]): The token IDs of the end-of-sequence token.
            temperature (float): The temperature to use for the logits.
            seed (int, optional): The seed for the random number generator. Defaults to None.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            (list[list[int]]): The sampled token IDs.
        """
        return await asyncio.gather(
            *[
                self.sample(
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=max_tokens,
                    eos_token_ids=eos_token_ids,
                    temperature=temperature,
                    seed=seed,
                    lora_name=lora_name,
                )
                for prompt_token_ids in prompt_token_ids_list
            ]
        )


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
            self._rng.rand(len(self.byte_vocab)).astype(np.float32)
        )
        return torch.log_softmax(logits, dim=-1)
