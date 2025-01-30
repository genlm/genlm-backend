import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="test requires CUDA"
)
