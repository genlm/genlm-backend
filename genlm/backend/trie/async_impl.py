import asyncio
import logging
from collections import defaultdict
from genlm.backend.trie.base import TokenCharacterTrie
from genlm.backend.trie.parallel import ParallelTokenCharacterTrie

logger = logging.getLogger(__name__)


class AsyncTokenCharacterTrie:
    """An asynchronous wrapper for TokenCharacterTrie implementations that provides automatic request batching."""

    def __init__(self, trie):
        """Initialize an `AsyncTokenCharacterTrie`.

        Args:
            trie (TokenCharacterTrie|ParallelTokenCharacterTrie): The underlying `TokenCharacterTrie` or `ParallelTokenCharacterTrie` instance
        """
        self.trie = trie
        self._queue = None
        self._task = None

    @classmethod
    def from_vocab(cls, vocab, backend="parallel", **kwargs):
        """Creates an `AsyncTokenCharacterTrie` from a vocabulary.

        Args:
            vocab (list): The vocabulary over which the trie will be defined.
            backend (str, optional): The trie implementation to use - either 'sequential' or 'parallel'.
                    Defaults to 'parallel' which uses GPU acceleration when available.
            **kwargs: Additional arguments passed to the trie constructor

        Returns:
            (AsyncTokenCharacterTrie): The initialized asynchronous trie instance.
        """
        if backend == "sequential":
            trie = TokenCharacterTrie(decode=vocab, **kwargs)
        elif backend == "parallel":
            trie = ParallelTokenCharacterTrie(decode=vocab, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Must be one of ['sequential', 'parallel']"
            )
        return cls(trie)

    async def _queue_request(self, request, op):
        if not self._task or self._task.done():
            self.start()

        future = asyncio.Future()
        await self._queue.put((request, future, op))
        return future

    async def weight_sum(self, ws):
        """Queue a `weight_sum` request. Multiple concurrent calls will be automatically batched
        together.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.trie.decode)`,).

        Returns:
            (np.ndarray): The calculated mass sums for the given distribution.
        """
        future = await self._queue_request(ws, "sum")
        result = await future
        return result

    async def weight_max(self, ws):
        """Queue a `weight_max` request. Multiple concurrent calls will be automatically batched
        together.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.trie.decode)`,).

        Returns:
            (np.ndarray): The calculated max weights for the given distribution.
        """
        future = await self._queue_request(ws, "max")
        result = await future
        return result

    def start(self):
        """Start the background processing task if not already running."""
        if not self._task or self._task.done():
            self._queue = (
                asyncio.Queue()
            )  # Create a new queue so that it is bound to the current event loop
            self._task = asyncio.create_task(self._background_loop())

    def _do_weight_sums(self, batch_weights):
        return self.trie.batch_weight_sum(batch_weights)

    def _do_weight_maxs(self, batch_weights):
        return self.trie.batch_weight_max(batch_weights)

    async def _background_loop(self):
        """Background task that processes queued weight sum and max requests.

        Continuously monitors the queue for new requests and processes them in batches
        using the underlying trie implementation.

        Raises:
            Exception: If any error occurs during processing, it is propagated to all
                      pending futures in the current batch.
        """
        while True:
            try:
                op_groups = defaultdict(list)

                request, future, op = await self._queue.get()
                op_groups[op].append((request, future))

                while not self._queue.empty():
                    request, future, op = await self._queue.get()
                    op_groups[op].append((request, future))

                for op, group in op_groups.items():
                    requests, futures = zip(*group)

                    if op == "sum":
                        logger.debug(f"processing {len(requests)} sum requests")
                        results = self._do_weight_sums(requests)
                    elif op == "max":
                        logger.debug(f"processing {len(requests)} max requests")
                        results = self._do_weight_maxs(requests)
                    else:
                        raise ValueError(f"Unknown operation: {op}")

                    for future, result in zip(futures, results):
                        future.set_result(result)

            except Exception as e:
                for group in op_groups.values():
                    for _, future in group:
                        if not future.done():
                            future.set_exception(e)
                raise

    async def cleanup(self):
        """Async cleanup - preferred method"""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def shutdown(self):
        """Stop the background processing task and cleanup resources."""
        if self._task is not None:
            try:
                self._task.cancel()
            except RuntimeError:
                # Ignore runtime errors that might occur if event loop is closed
                pass
            self._task = None

    def __del__(self):
        self.shutdown()
