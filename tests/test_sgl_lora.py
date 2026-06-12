"""LoRA support tests for the SGLang backend.

Mirrors the structure of ``tests/test_vllm_lora.py`` but uses the HF backend
with the same adapter applied as the parity reference, since SGLang does not
have an in-process reference.
"""

import asyncio
import pytest
import torch
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name
from conftest import cuda_only


@pytest.fixture(scope="module")
def model_name():
    return "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def lora_path():
    return "vxef/smol_lora_toy"


@pytest.fixture(scope="module")
def async_llm(model_name):
    """SGLang backend with LoRA enabled at construction time."""
    llm_opts = {
        "cache_size": 100,
        "engine_opts": {
            "disable_cuda_graph": True,
            "attention_backend": "torch_native",
            "chunked_prefill_size": 200,
            "enable_lora": True,
            "max_lora_rank": 16,
            "max_loras_per_batch": 1,
            # Required by sglang when no initial --lora-paths is supplied:
            # both max_lora_rank and lora_target_modules must be set up
            # front so the LoRAManager can size its weight buffers.
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    }
    llm = load_model_by_name(model_name, backend="sgl", llm_opts=llm_opts)
    yield llm
    llm._cleanup_engine()


@pytest.fixture(scope="module")
def transformer_llm(model_name, lora_path):
    """HF + adapter as the parity reference."""
    llm = load_model_by_name(
        model_name,
        backend="hf",
        llm_opts={"hf_opts": {"torch_dtype": torch.float16}},
    )
    llm.add_new_lora(lora_path, "lora_1")
    llm.set_lora(lora_name="lora_1")
    yield llm
    llm.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong",
        "It's probably this or that",
        "with the language model code",
    ]
    return [async_llm.tokenizer.encode(p) for p in test_prompts]


@cuda_only
def test_set_lora_unknown_name_raises(async_llm):
    with pytest.raises(ValueError):
        async_llm.set_lora(lora_name="never_loaded")


@cuda_only
def test_logprobs_match_hf_with_adapter(
    async_llm, transformer_llm, token_ids_list, lora_path
):
    """SGL + adapter should match HF + adapter within parity tolerance."""
    async_llm.add_new_lora(lora_path)
    async_llm.set_lora(lora_path=lora_path)
    try:
        for token_ids in token_ids_list:
            have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
            want = transformer_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
            # 2e-2 mirrors the cross-backend tolerance used in test_sgl.py.
            assert compare(have, want).max_rel_err < 2e-2, token_ids
    finally:
        async_llm.clear_lora()


@cuda_only
def test_set_lora_does_not_leak_base_cache(async_llm, token_ids_list, lora_path):
    """Base-model cache entries must not be served to post-set_lora calls.

    ``set_lora`` wipes the output cache, so the post-switch lookup misses
    and runs a fresh forward under the adapter. Observable behavior: same
    prompt under the adapter returns different logprobs from the base.
    """
    async_llm.clear_lora()
    async_llm.clear_cache()

    base = asyncio.run(async_llm.next_token_logprobs(token_ids_list[0])).cpu().numpy()

    async_llm.add_new_lora(lora_path)
    async_llm.set_lora(lora_path=lora_path)
    try:
        adapted = (
            asyncio.run(async_llm.next_token_logprobs(token_ids_list[0])).cpu().numpy()
        )
        # Adapter is non-trivial; if the cache leaked the base entry,
        # max_rel_err would be ~0. Threshold is loose by design — we only
        # need to detect "exact same array."
        assert compare(base, adapted).max_rel_err > 1e-3, (
            "Cache appears to have returned base-model logprobs after set_lora"
        )
    finally:
        async_llm.clear_lora()


@cuda_only
def test_clear_lora_returns_to_base(async_llm, token_ids_list, lora_path):
    """After clear_lora, logprobs match the base-model logprobs."""
    async_llm.clear_lora()
    async_llm.clear_cache()
    base = asyncio.run(async_llm.next_token_logprobs(token_ids_list[0])).cpu().numpy()

    async_llm.add_new_lora(lora_path)
    async_llm.set_lora(lora_path=lora_path)
    _ = asyncio.run(async_llm.next_token_logprobs(token_ids_list[0]))
    async_llm.clear_lora()

    after_clear = (
        asyncio.run(async_llm.next_token_logprobs(token_ids_list[0])).cpu().numpy()
    )
    assert compare(base, after_clear).max_rel_err < 2e-2
