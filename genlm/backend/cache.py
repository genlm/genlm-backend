import torch
from collections import OrderedDict


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


class OutputMLXCache(OutputCache):
    """A cache for storing tensor outputs with MLX.

    Since MLX uses unified memory, we don't need to move tensors between CPU and GPU.

    Args:
        maxsize (int): Maximum number of items to store in the cache
    """

    def __init__(self, maxsize):
        super().__init__(maxsize, move_to_cpu=False)

    def __getitem__(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        raise KeyError(key)

    def __setitem__(self, key, value):
        if len(self.cache) >= self.maxsize:
            _, old_tensor = self.cache.popitem(last=False)
            del old_tensor

        self.cache[key] = value


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
