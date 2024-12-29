import torch
import asyncio
import logging
import numpy as np
from dataclasses import dataclass

from genlm_backend.trie.base import TokenCharacterTrie
from genlm_backend.trie.parallel import ParallelTokenCharacterTrie

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class NextTokenTrie:
    mass: np.ndarray
    root: int
    children: list
    old_eos: str | bytes
    new_eos: str | bytes

    @classmethod
    def from_trie(cls, trie, mass):
        return cls(
            mass=mass,
            root=trie.root,
            children=trie.children,
            old_eos=trie.old_eos,
            new_eos=trie.new_eos,
        )


class AsyncTokenCharacterTrie:
    def __init__(self, trie):
        self.trie = trie
        self._queue = asyncio.Queue()
        self._pending = {}
        self._task = None 

    @classmethod
    def from_llm(cls, async_llm, backend='parallel', **kwargs):
        """Creates an AsyncTokenCharacterTrie from a language model.

        Args:
            async_llm (AsyncLM): The asynchronous language model to use
            backend (str, optional): The trie implementation to use - either 'sequential' or 'parallel'.
                    Defaults to 'parallel' which uses GPU acceleration when available.
            **kwargs: Additional arguments passed to the trie constructor
        """
        if backend == 'sequential':
            trie = TokenCharacterTrie(decode=async_llm.byte_vocab, **kwargs)
        elif backend == 'parallel':
            trie = ParallelTokenCharacterTrie(decode=async_llm.byte_vocab, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Must be one of ['sequential', 'parallel']")
        return cls(trie)

    async def mass_sum(self, p_llm):
        if not self._task:
            self.start()
            
        future = asyncio.Future()
        await self._queue.put((p_llm, future))
        return await future

    async def do_mass_sums(self, p_llms):
        return self.trie.batch_mass_sum(torch.stack(p_llms)) # XXX handle device

    def start(self):
        if not self._task:
            self._task = asyncio.create_task(self._background_loop())

    async def _background_loop(self):
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

                logger.debug(f'Processing batch of {len(requests)} requests.')
                results = await self.do_mass_sums(requests)
                
                for future, result in zip(futures, results):
                    future.set_result(result)
                    
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                raise

    def shutdown(self):
        if self._task:
            self._task.cancel()
            self._task = None

    def __del__(self):
        self.shutdown()