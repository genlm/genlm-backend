import torch
from collections import OrderedDict
from time import time


class OutputCache:
    """A cache for storing tensor outputs with optional CPU offloading.

    This cache stores tensors along with their original devices and can optionally
    move tensors to CPU to save GPU memory. When retrieving tensors, they are
    moved back to their original device.

    Args:
        maxsize (int): Maximum number of items to store in the cache
        move_to_cpu (bool): If True, tensors will be moved to CPU when cached
    """

    def __init__(self, maxsize, move_to_cpu=False):
        self.maxsize = maxsize
        self.move_to_cpu = move_to_cpu
        self.cache = OrderedDict()  # stores (device, tensor) tuples

    def __getitem__(self, key):
        if key in self.cache:
            device, value = self.cache.pop(key)
            self.cache[key] = (device, value)
            return value.to(device) if self.move_to_cpu else value
        raise KeyError(key)

    def __setitem__(self, key, value):
        if len(self.cache) >= self.maxsize:
            old_key, (_, old_tensor) = self.cache.popitem(last=False)
            del old_tensor

        self.cache[key] = (value.device, value.cpu() if self.move_to_cpu else value)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def clear(self):
        self.cache.clear()


class TokenTrie:
    """Class used internally to cache language model results.

    The TokenTrie maintains a tree of token sequences, storing logits and key-value
    states for each path.
    """

    # maybe TODO: Implement eviction policy

    # Trie of tokens.

    def __init__(self, parent=None, logprobs=None):
        self.children = {}  # maps token ID to child
        self.logprobs = logprobs  # for next token
        self.past_key_values = None

    def __repr__(self):
        return (
            f"{'*' if self.past_key_values is not None else ''}["
            + ", ".join(
                [
                    f"{node_id}: {node.__repr__()}"
                    for (node_id, node) in self.children.items()
                ]
            )
            + "]"
        )

    def clear_kv_cache(self):
        self.past_key_values = None
        for child, node in self.children.items():
            node.clear_kv_cache()

    def has_token(self, token_id):
        return token_id in self.children

    def get_token(self, token_id):
        return self.children[token_id]

    def add_token(self, token_id, logprobs=None):
        self.children[token_id] = TokenTrie(self, logprobs)
        return self.children[token_id]

    def extend_cache(self, next_token_index, token_ids, logits, base):
        node = self

        for j in range(next_token_index, len(token_ids)):
            token_id = token_ids[j]
            token_logits = logits[j - base]
            token_logprobs = torch.log_softmax(token_logits, 0)

            node = node.add_token(token_id, token_logprobs.cpu())

        return node


class DynamicTokenTrie(TokenTrie):
    def __init__(self, parent=None, logprobs=None, past_key_values=None):
        super().__init__(parent, logprobs)
        self.past_key_values = past_key_values
        self.last_access = time()
        self.kv_size = 0
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1

    def touch(self):
        """Update access timestamp recursively upward."""
        t = time()
        node = self
        while node:
            node.last_access = t
            node = node.parent

    def add_token(self, token_id, logprobs=None, past_key_values=None):
        if token_id in self.children:
            child = self.children[token_id]
            child.store_kv(past_key_values)
            if child.logprobs is None:
                child.logprobs = logprobs
        else:
            self.children[token_id] = DynamicTokenTrie(
                parent=self, logprobs=logprobs, past_key_values=past_key_values
            )
        self.children[token_id].touch()
        return self.children[token_id]

    def store_kv(self, past_key_values):
        """Store KV states on this node. KV should be list[(K, V), ...] without batch dimension."""
        if self.past_key_values is not None or past_key_values is None:
            return
        self.past_key_values = past_key_values

    def extend_cache(self, next_token_index, token_ids, logprobs=None, kv=None):
        node = self
        token_ids_current = token_ids[next_token_index:]
        if kv is None:
            kv = [None] * len(token_ids_current)
        else:
            kv = [kv[:, :, :, i : i + 1, :] for i in range(len(token_ids_current))]

        for i, token_id in enumerate(token_ids_current):
            node = node.add_token(token_id, None, kv[i])

        if node.logprobs is None:
            node.logprobs = logprobs

        return node

    def count_kv_size(self):
        """Recompute how many nodes currently store KVs."""
        total = 1 if self.past_key_values is not None else 0
        for c in self.children.values():
            total += c.count_kv_size()
        self.kv_size = total
        return total

    def collect_nodes_with_kv(self):
        """Collect nodes that have stored KVs (for eviction decisions)."""
        nodes = []
        if self.past_key_values is not None:
            nodes.append(self)
        for c in self.children.values():
            nodes.extend(c.collect_nodes_with_kv())
        return nodes

    def evict_lru_kv(self, max_kv):
        """Evict least recently used KV entries (and descendants) until under limit."""
        total = self.count_kv_size()
        if total <= max_kv:
            return
        nodes = self.collect_nodes_with_kv()
        nodes.sort(key=lambda n: (n.last_access, -n.depth))

        for node in nodes:
            if self.kv_size <= max_kv:
                break
            node._clear_kv_recursive()
            self.count_kv_size()

    def _clear_kv_recursive(self):
        """Remove KV from this node and all descendants."""
        if self.past_key_values is not None:
            self.past_key_values = None
        for c in self.children.values():
            c._clear_kv_recursive()
