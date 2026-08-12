import torch
import asyncio
import numpy as np
from abc import ABC, abstractmethod

from genlm.backend.tokenization import decode_vocab


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
        # Batch-window state; concurrent asks collect in ``_queries``.
        self._queries = []
        self._window_armed = False

    async def _window_cohort(self, entries, *, linger=0.0):
        """Meet concurrent asks in a batch window.

        ``entries`` are appended before any yield, so a whole batch enters as
        one set of asks. The first caller to arm the window holds it open
        until a full event-loop pass adds no new ask (preceded by one
        cooperative ``linger`` sleep for late callers, when nonzero) and
        receives the drained cohort; every other caller receives ``None``.
        The holding caller must evaluate the cohort in its own coroutine and
        never in a background task: window state must not outlive the loop
        its callers are on.

        Args:
            entries (list): Asks to add to the current window.
            linger (float, optional): Seconds to sleep once for late callers.
                Defaults to 0.0, which skips the sleep.

        Returns:
            (list | None): The drained cohort for the caller holding the
                window, ``None`` for every other caller.
        """
        self._queries.extend(entries)
        if self._window_armed:
            return None
        self._window_armed = True
        try:
            lingered = not linger
            while True:
                n = len(self._queries)
                await asyncio.sleep(0)
                if len(self._queries) > n:
                    continue
                if lingered:
                    break
                lingered = True
                await asyncio.sleep(linger)
            cohort, self._queries = self._queries, []
            return cohort
        finally:
            self._window_armed = False

    @abstractmethod
    async def next_token_logprobs(self, token_ids, lora_name=None):
        """Request log probabilities of next token asynchronously.

        Args:
            token_ids (list[int]): A list of token IDs representing the prompt.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        pass

    @abstractmethod
    def next_token_logprobs_sync(self, token_ids, lora_name=None):
        """Request log probabilities of next token synchronously.

        Args:
            token_ids (list[int]): A list of token IDs representing the prompt.
            lora_name (str, optional): LoRA adapter to forward under (``None`` = base).

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
        logprobs = await asyncio.gather(
            *[
                self.next_token_logprobs(token_ids, lora_name=lora_name)
                for token_ids in token_ids_list
            ]
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
        return torch.stack(
            [
                self.next_token_logprobs_sync(token_ids, lora_name=lora_name)
                for token_ids in token_ids_list
            ]
        )

    def add_new_lora(self, lora_path, lora_name):
        """Register a LoRA adapter under ``lora_name``.

        Re-registering an existing name rebinds it to the weights at
        ``lora_path``. Forwards select the adapter per call via ``lora_name=``.

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

    def set_lora(self, lora_path, lora_name):
        """Unsupported: adapter selection is per-request, via ``lora_name=``."""
        raise RuntimeError(
            "set_lora() was removed: there is no active-adapter global anymore. "
            "Pass lora_name= per call (next_token_logprobs/sample/...)."
        )

    def clear_lora(self):
        """Unsupported: ``lora_name=None`` (the default) is the base model."""
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

        generated_token_ids = []
        for _ in range(max_tokens):
            logprobs = await self.next_token_logprobs(
                prompt_token_ids + generated_token_ids, lora_name=lora_name
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

    async def next_token_logprobs(self, token_ids, lora_name=None):
        """Get next token log probabilities asynchronously.

        Args:
            token_ids (list[int]): Input token IDs.
            lora_name (str, optional): Must be ``None``; the mock has no adapters.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        if lora_name is not None:
            raise ValueError(f"MockAsyncLM has no adapter named {lora_name!r}")
        return self._get_logprobs(token_ids)

    def next_token_logprobs_sync(self, token_ids, lora_name=None):
        """Get next token log probabilities synchronously.

        Args:
            token_ids (list[int]): Input token IDs.
            lora_name (str, optional): Must be ``None``; the mock has no adapters.

        Returns:
            (torch.Tensor): Normalized log probability tensor.
        """
        if lora_name is not None:
            raise ValueError(f"MockAsyncLM has no adapter named {lora_name!r}")
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
