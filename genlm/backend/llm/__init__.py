from genlm.backend.llm.vllm import AsyncVirtualLM
from genlm.backend.llm.hf import AsyncTransformer
from genlm.backend.llm.base import AsyncLM, MockAsyncLM
from genlm.backend.llm.mlx import AsyncMlxLM

import torch


def load_model_by_name(name, backend=None, llm_opts=None):
    """Load a language model by name.

    Args:
        name (str): Hugging Face model name (e.g. "gpt2", "meta-llama/Llama-3.2-1B-Instruct")
        backend (str, optional): Backend to use for inference. Can be "vllm", "hf" or "mock".
            If None, defaults to "vllm" if CUDA is available, otherwise "hf".
        llm_opts (dict, optional): Additional options to pass to the backend constructor.
            See AsyncVirtualLM and AsyncTransformer documentation for details.

    Returns:
        (AsyncLM): An asynchronous language model.

    Raises:
        (ValueError): If an invalid backend is specified.
    """
    if backend is None:
        backend = "vllm" if torch.cuda.is_available() else "hf"

    if llm_opts is None:
        llm_opts = {}

    if backend == "vllm":
        return AsyncVirtualLM.from_name(name, **llm_opts)
    elif backend == "hf":
        return AsyncTransformer.from_name(name, **llm_opts)
    elif backend == "mock":
        return MockAsyncLM.from_name(name, **llm_opts)
    elif backend == "mlx":
        return AsyncMlxLM.from_name(name, **llm_opts)
    else:
        raise ValueError(f"Invalid backend: {backend}")


__all__ = [
    "load_model_by_name",
    "AsyncLM",
    "AsyncVirtualLM",
    "AsyncTransformer",
    "AsyncMlxLM",
    "MockAsyncLM",
]
