import torch
import asyncio
import logging
import numpy as np
from dataclasses import dataclass

from genlm_backend.trie.base import TokenCharacterTrie
from genlm_backend.trie.parallel import ParallelTokenCharacterTrie

logger = logging.getLogger(__name__)


class AsyncTokenCharacterTrie:
    """An asynchronous wrapper for `TokenCharacterTrie` implementations.

    This class provides asynchronous access to mass sum calculations, with automatic batching of concurrent requests.
    It maintains a background task that processes queued requests.
    """

    def __init__(self, trie):
        """Initialize an `AsyncTokenCharacterTrie`.

        Args:
            trie (TokenCharacterTrie|ParallelTokenCharacterTrie): The underlying `TokenCharacterTrie` or `ParallelTokenCharacterTrie` instance
        """
        self.trie = trie
        self._queue = asyncio.Queue()
        self._task = None

    @classmethod
    def from_vocab(cls, byte_vocab, backend="parallel", **kwargs):
        """Creates an `AsyncTokenCharacterTrie` from a byte vocabulary.

        Args:
            byte_vocab (list[byte]): The byte vocabulary over which the trie will be defined.
            backend (str, optional): The trie implementation to use - either 'sequential' or 'parallel'.
                    Defaults to 'parallel' which uses GPU acceleration when available.
            **kwargs: Additional arguments passed to the trie constructor

        Returns:
            (AsyncTokenCharacterTrie): The initialized asynchronous trie instance.
        """
        if backend == "sequential":
            trie = TokenCharacterTrie(decode=byte_vocab, **kwargs)
        elif backend == "parallel":
            trie = ParallelTokenCharacterTrie(decode=byte_vocab, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Must be one of ['sequential', 'parallel']"
            )
        return cls(trie)

    async def mass_sum(self, p_llm):
        """Asynchronously computes the mass at each node of the trie.

        This method queues the mass calculation to be processed in a background task.
        Multiple concurrent requests are automatically batched together.

        Args:
            p_llm (torch.Tensor): Probability distribution over the trie's vocabulary of length `len(trie.decode)`.

        Returns:
            (float): The calculated mass sum for the given distribution.
        """
        if not self._task:
            self.start()

        future = asyncio.Future()
        await self._queue.put((p_llm, future))
        return await future

    def start(self):
        """Start the background processing task if not already running."""
        if not self._task:
            self._task = asyncio.create_task(self._background_loop())

    async def _do_mass_sums(self, p_llms):
        """Compute mass sums for a batch of distributions.

        Args:
            p_llms (list[torch.Tensor]): List of distributions over trie vocabulary.

        Returns:
            (torch.Tensor): Batch of computed mass sums
        """
        return self.trie.batch_mass_sum(torch.stack(p_llms))  # XXX handle device

    async def _background_loop(self):
        """Background task that processes queued mass sum requests.

        Continuously monitors the queue for new requests and processes them using the underlying trie implementation.

        Raises:
            Exception: If any error occurs during processing, it is propagated to all
                      pending futures in the current batch.
        """
        while True:
            try:
                requests = []
                futures = []

                request, future = await self._queue.get()
                requests.append(request)
                futures.append(future)

                while not self._queue.empty():
                    request, future = await self._queue.get()
                    requests.append(request)
                    futures.append(future)

                logger.debug(f"Processing batch of {len(requests)} requests.")
                results = await self._do_mass_sums(requests)

                for future, result in zip(futures, results):
                    future.set_result(result)

            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                raise

    def shutdown(self):
        """Stop the background processing task and cleanup resources."""
        if self._task:
            self._task.cancel()
            self._task = None

    def __del__(self):
        self.shutdown()
