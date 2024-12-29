import torch
import asyncio
import logging
import numpy as np
from dataclasses import dataclass

from genlm_backend.util import resolve_device
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
    """An asynchronous worker that builds character tries over next tokens.

    Args:
        llm (AsyncLM): The language model to use for generating token probabilities
        new_eos (bytes): The new end-of-sequence token to replace the model's default EOS token
        device (str, optional): The device to use ('cuda' or 'cpu'). If None, automatically selects
            'cuda' if available, else 'cpu'.
    """
    def __init__(self, async_llm, new_eos, device=None):
        self.async_llm = async_llm
        self.device = resolve_device(device)

        self._queue = asyncio.Queue()
        self._pending = {}
        self._task = None        

        self.trie = ParallelTokenCharacterTrie(
            decode=self.async_llm.byte_vocab,
            new_eos=new_eos,
            old_eos=self.async_llm.eos_token,
            device=self.device
        ) if self.device == 'cuda' else TokenCharacterTrie(
            decode=self.async_llm.byte_vocab,
            new_eos=new_eos,
            old_eos=self.async_llm.eos_token
        )

    def start(self):
        if not self._task:
            self._task = asyncio.create_task(self._background_loop())

    async def next_token_trie(self, token_ids):
        if not self._task:
            self.start()
            
        future = asyncio.Future()
        await self._queue.put((token_ids, future))
        return await future

    async def batch_next_token_trie(self, all_token_ids):
        p_llms = torch.exp(
            await self.async_llm.batch_next_token_logprobs(all_token_ids)
        )
        masses = self.trie.batch_mass_sum(p_llms) # TODO: Cache mass sum results.
        tries = [NextTokenTrie.from_trie(self.trie, mass) for mass in masses]
        return tries

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
                results = await self.batch_next_token_trie(requests)
                
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