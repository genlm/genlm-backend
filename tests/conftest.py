import pytest
import torch

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda_only: mark test to run only when CUDA is available"
    )

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="test requires CUDA"
)