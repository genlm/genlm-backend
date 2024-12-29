import torch
import numba
import numpy as np
from numba.typed import List

class TokenCharacterTrie:
    def __init__(self, decode, old_eos, new_eos):
        if not all(isinstance(x, bytes) for x in decode):
            raise ValueError("All elements in decode must be byte strings")
        
        if not isinstance(old_eos, bytes):
            old_eos = old_eos.encode('utf-8')
        if not isinstance(new_eos, bytes):
            new_eos = new_eos.encode('utf-8')

        self.old_eos = old_eos
        try:
            self.old_eos_id = decode.index(old_eos)
        except ValueError:
            raise ValueError(f"Could not find old_eos token {old_eos} in vocabulary")
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

            children[curr][None] = last = len(children)
            children.append({})
            assert word not in word2leaf
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

        node2prefix = {self.root: b''}
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