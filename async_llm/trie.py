import numba
import torch
import asyncio
import numpy as np
from numba.typed import List
from dataclasses import dataclass

from async_llm.async_worker import AsyncWorker

class TokenCharacterTrie:
    def __init__(self, decode, old_eos, new_eos):
        use_bytes = isinstance(decode[0], bytes)
        if use_bytes:
            if not isinstance(old_eos, bytes):
                old_eos = old_eos.encode('utf-8')
            if not isinstance(new_eos, bytes):
                new_eos = new_eos.encode('utf-8')

        self.old_eos = old_eos
        self.old_eos_id = decode.index(old_eos)
        assert self.old_eos_id is not None
        self.new_eos = new_eos

        word2leaf = {}
        children = {}
        root = 0
        children = [{}]

        token_id_to_leaf = []

        for token_id, word in enumerate(decode):
            # coerce old eos to new eos
            _word = word
            if word == self.old_eos:
                word = self.new_eos

            curr = root
            for letter in word:
                if letter not in children[curr]:
                    children[curr][letter] = len(children)
                    children.append({})
                curr = children[curr][letter]

            # Reuse existing leaf node if the word was seen before.
            # This can happen when decode contains duplicates, which is unfortunately
            # the case for the string vocabularies of certain tokenizers.
            if word in word2leaf:
                last = word2leaf[word]
            else:
                children[curr][None] = last = len(children)
                children.append({})
                word2leaf[word] = last

            token_id_to_leaf.append((token_id, last))

        self.token_id_to_leaf = token_id_to_leaf
        self.root = root
        self.children = children
        self.word2leaf = word2leaf
        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))
        self.jump = List([np.array(sorted(x.values()), dtype=np.int32) for x in children])
        self.ordering = np.array(list(self._order(self.root)), np.int32)

        # Renumber the states of the trie so that they are named by a contiguous
        # range of integers and those integers respect the are topologically
        # ordering of the trie topology.  This improves the efficiency of the
        # updating the trie as it improves memory locality.
        ordering = {}
        for i, x in enumerate(self._order_full(self.root)):
            ordering[x] = i
        self.rename(f=lambda x: ordering[x])

        node2prefix = {self.root: b'' if use_bytes else ''}
        for x in reversed(range(len(self.children))):
            for letter, y in self.children[x].items():
                if isinstance(letter, int):
                    letter = bytes([letter])
                if letter is None:
                    node2prefix[y] = node2prefix[x]
                else:
                    node2prefix[y] = node2prefix[x] + letter
        self.node2prefix = node2prefix

    def rename(self, f):
        N = len(self.children)

        new_children = [{} for _ in range(N)]
        nodes = range(N)

        for x in nodes:
            for letter, y in self.children[x].items():
                new_children[f(x)][letter] = f(y)

        self.root = f(self.root)
        self.children = new_children
        self.word2leaf = {w: f(x) for w, x in self.word2leaf.items()}
        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))

        self.token_id_to_leaf = np.array(
            [(i, f(x)) for i, x in self.token_id_to_leaf], dtype=np.int32
        )

        self.ordering = np.array([f(x) for x in self.ordering])
        self.jump = List(
            [np.array(sorted(x.values()), dtype=np.int32) for x in new_children]
        )

    def alloc_mass(self):
        return np.zeros(len(self.children), dtype=np.float64)

    def mass_sum(self, p_llm):
        if isinstance(p_llm, torch.Tensor):
            if p_llm.device.type != 'cpu':
                p_llm = p_llm.cpu()
            p_llm = p_llm.numpy()
        mass = self.alloc_mass()
        # convert llm.eos to guide.eos
        mass[self.word2leaf[self.new_eos]] = p_llm[self.old_eos_id]
        _update_trie_numba(
            mass=mass,
            _p=p_llm,
            token_id_to_leaf=self.token_id_to_leaf,
            jump=self.jump,
            ordering=self.ordering,
        )
        return mass

    def batch_mass_sum(self, p_llms):
        return np.array([self.mass_sum(p_llm) for p_llm in p_llms])

    def _order(self, node):
        "Topological ordering of nodes beneath `node`."
        for a in self.children[node]:
            if a is None:
                pass
            else:
                yield from self._order(self.children[node][a])
        yield node

    def _order_full(self, node):
        "Topological ordering of nodes beneath `node`."
        for a in self.children[node]:
            yield from self._order_full(self.children[node][a])
        yield node


@numba.jit(nopython=True)
def _update_trie_numba(
    mass: numba.float64[:],
    _p: numba.float64[:],
    jump: List[numba.int32[:]],
    token_id_to_leaf: numba.int32[:, :],
    ordering: numba.int32[:],
):  # pragma: no cover
    # update leaves
    M = token_id_to_leaf.shape[0]
    for k in range(M):
        i = token_id_to_leaf[k, 0]
        x = token_id_to_leaf[k, 1]
        mass[x] = _p[i]

    # update internal nodes
    N = ordering.shape[0]
    for i in range(N):
        node = ordering[i]
        total_mass = 0
        for child in jump[node]:
            total_mass += mass[child]
        mass[node] = total_mass


class ParallelTokenCharacterTrie(TokenCharacterTrie):
    """A GPU-optimized version of TokenCharacterTrie that performs mass_sum in parallel.

    The mass at leaf nodes is propagated to their ancestors
    through sparse matrix multiplication with a reachability matrix.

    The reachability matrix M is a num_leafs × num_nodes matrix
    where M[i,j] = 1 if:
        - leaf_indices[i] == j (self connection) or
        - j is an ancestor of leaf_indices[i] in the trie

    Example:
        Trie:          M:
              0           [[1, 1, 0, 1],
             / \           [1, 0, 1, 0]]
            1   2 (1)
            |
            3 (0)

    The matrix is stored as a sparse tensor in CSR (Compressed Sparse Row) format,
    built from COO (Coordinate) format. For example,
        rows = [1, 0, 1, 0, 0] (index of leaf node)
        cols = [2, 3, 0, 1, 0] (connections)
        vals = [1, 1, 1, 1, 1] (connection weights)

    When computing masses (batch_size × num_leafs) @ M, each leaf node's mass
    flows up to all its ancestors.
    """
    def __init__(self, decode, old_eos, new_eos, device=None):
        super().__init__(decode, old_eos, new_eos)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.M = self._build_reachability_matrix()
        self.token_ids = torch.tensor(
            self.token_id_to_leaf[:, 0], dtype=torch.long, device=self.device
        )

    def _build_parent_map(self):
        parent = {}
        for node in range(len(self.children)):
            for child in self.jump[node]:
                parent[child] = node
        return parent

    def _build_reachability_matrix(self):
        rows, cols = [], []
        leaf_indices = self.token_id_to_leaf[:, 1]

        # add self connections
        for i, node in enumerate(leaf_indices):
            rows.append(i)
            cols.append(node)

        # add all ancestor connections
        parent = self._build_parent_map()
        for i, node in enumerate(leaf_indices):
            current = node
            while current in parent:        # Walk up to root
                ancestor = parent[current]
                rows.append(i)
                cols.append(ancestor)
                current = ancestor

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(rows), device=self.device)
        M = torch.sparse_coo_tensor(
            indices, values, (len(leaf_indices), len(self.children))
        ).to_sparse_csr()

        return M

    def batch_mass_sum(self, p_llms):
        if p_llms.device != self.device:
            p_llms = p_llms.to(self.device)
        masses = torch.sparse.mm(p_llms[:, self.token_ids], self.M)
        return masses.cpu().numpy()


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


class AsyncTokenCharacterTrie(AsyncWorker):
    """An asynchronous worker that builds character tries over next tokens.

    This class combines the functionality of TokenCharacterTrie/GPUTokenCharacterTrie with AsyncWorker
    to enable efficient batch processing of trie construction. It automatically selects
    between CPU and GPU implementations based on device availability.

    Args:
        llm: The language model to use for generating token probabilities
        new_eos: The new end-of-sequence token to replace the model's default EOS token
        vocab (str, optional): The vocabulary type to use ('bytes' or 'str'). Defaults to 'bytes'.
        device (str, optional): The device to use ('cuda' or 'cpu'). If None, automatically selects
            'cuda' if available, else 'cpu'.
    """
    def __init__(self, async_llm, new_eos, vocab='byte', device=None):
        # TODO: Cache mass sum results.
        super().__init__()

        self.async_llm = async_llm

        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cpu' or device == 'cuda':
            self.device = device
        else:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda' or None to automatically select it.")

        if vocab == 'byte':
            decode = async_llm.byte_vocab
        elif vocab == 'str': 
            decode = async_llm.str_vocab
        else:
            raise ValueError(f"Invalid vocab type: {vocab}. Must be 'byte' or 'str'.")
        
        old_eos = async_llm.tokenizer.eos_token 

        if self.device == 'cuda':
            self.trie = ParallelTokenCharacterTrie(
                decode=decode, new_eos=new_eos, old_eos=old_eos, device=self.device
            )
        else:
            self.trie = TokenCharacterTrie(
                decode=decode, new_eos=new_eos, old_eos=old_eos
            )

    async def next_token_trie(self, token_ids):
        request_id = str(next(self.request_counter))
        # Queue the request and wait for it to complete.
        return await self.add_request(request_id, token_ids)

    async def batch_next_token_trie(self, all_token_ids):
        return await asyncio.gather(*[
            self.next_token_trie(token_ids) for token_ids in all_token_ids
        ])

    async def batch_process_requests(self, all_token_ids):
        logp_llms = await self.async_llm.batch_next_token_logprobs(all_token_ids)
        p_llms = torch.exp(logp_llms)
        masses = self.trie.batch_mass_sum(p_llms)
        tries = [NextTokenTrie.from_trie(self.trie, mass) for mass in masses]
        return tries