import torch
from conftest import cuda_only
from genlm_backend.cache import OutputCache


@cuda_only
def test_memory_freed_on_eviction():
    cache_size = 2
    cache = OutputCache(maxsize=cache_size, move_to_cpu=False)

    initial_memory = torch.cuda.memory_allocated()

    for i in range(cache_size):
        cache[f"tensor{i}"] = torch.rand(1000, 1000, device="cuda")

    memory_at_capacity = torch.cuda.memory_allocated()

    # sanity check
    assert initial_memory < memory_at_capacity

    # add more tensors (should trigger evictions)
    for i in range(cache_size * 2):
        cache[f"tensor_extra_{i}"] = torch.rand(1000, 1000, device="cuda")
        # memory shouldn't grow significantly beyond when at capacity
        assert torch.cuda.memory_allocated() <= memory_at_capacity * 1.2


@cuda_only
def test_memory_freed_on_clear():
    cache_size = 2
    cache = OutputCache(maxsize=cache_size, move_to_cpu=False)

    initial_memory = torch.cuda.memory_allocated()

    for i in range(cache_size):
        cache[f"tensor{i}"] = torch.rand(1000, 1000, device="cuda")

    # sanity check
    assert torch.cuda.memory_allocated() > initial_memory

    cache.clear()

    assert torch.cuda.memory_allocated() <= initial_memory * 1.1
