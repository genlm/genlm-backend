import pytest
import torch
from conftest import cuda_only
from genlm.backend.cache import OutputCache, OutputMLXCache


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_cache_size_limit(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cache = OutputCache(maxsize=2, move_to_cpu=False)

    # Add first tensor
    cache["tensor1"] = torch.rand(1000, 1000, device=device)
    assert len(cache) == 1

    # Add second tensor
    cache["tensor2"] = torch.rand(1000, 1000, device=device)
    assert len(cache) == 2

    # Add third tensor (should evict first)
    cache["tensor3"] = torch.rand(1000, 1000, device=device)
    assert len(cache) == 2
    assert "tensor1" not in cache
    assert "tensor2" in cache
    assert "tensor3" in cache


def test_invalid_key():
    cache = OutputCache(maxsize=2, move_to_cpu=False)

    with pytest.raises(KeyError):
        cache["invalid"]


def test_mlx_invalid_key():
    cache = OutputMLXCache(maxsize=2)

    with pytest.raises(KeyError):
        cache["invalid"]


def test_mlx_cache_size_limit():
    cache = OutputMLXCache(maxsize=2)

    # Add first tensor
    cache["tensor1"] = torch.rand(1000, 1000)
    assert len(cache) == 1

    # Add second tensor
    cache["tensor2"] = torch.rand(1000, 1000)
    assert len(cache) == 2

    # Add third tensor (should evict first)
    cache["tensor3"] = torch.rand(1000, 1000)
    assert len(cache) == 2
    assert "tensor1" not in cache
    assert "tensor2" in cache
    assert "tensor3" in cache


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


@cuda_only
def test_cache_cpu_transfer():
    cache = OutputCache(maxsize=2, move_to_cpu=True)

    # Test storing tensor
    cuda_tensor = torch.rand(100, 100, device="cuda")
    cache["test"] = cuda_tensor

    # Verify tensor was moved to CPU
    (_, value) = cache.cache["test"]
    assert value.device.type == "cpu"

    # Test moving back to CUDA when accessing
    retrieved_tensor = cache["test"]
    assert retrieved_tensor.device.type == "cuda"


@cuda_only
def test_cache_contains():
    cache = OutputCache(maxsize=2, move_to_cpu=False)

    # Test __contains__
    tensor = torch.rand(100, 100, device="cuda")
    cache["key"] = tensor
    assert "key" in cache
    assert "nonexistent" not in cache
