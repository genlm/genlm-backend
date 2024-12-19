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
            del old_tensor # XXX 
            
        self.cache[key] = (value.device, value.cpu() if self.move_to_cpu else value)
    
    def __contains__(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()
        torch.cuda.empty_cache()