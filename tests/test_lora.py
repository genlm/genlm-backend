import torch
import pytest
import asyncio
from conftest import (
    v1_capable,
    ReferenceVirtualLM,
    logprobs,
    batch_logprobs,
    assert_rows_close,
)
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name
import numpy as np

LORA_NAME = "lora_1"


@pytest.fixture(scope="module")
def model_name():
    return "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def async_llm(model_name):
    return load_model_by_name(
        model_name,
        backend="vllm",
        llm_opts={
            "engine_opts": {
                "enable_lora": True,
                "dtype": "float16",
                "gpu_memory_utilization": 0.1,
                "max_model_len": 15,
            }
        },
    )


@pytest.fixture(scope="module")
def reference_llm(model_name):
    return ReferenceVirtualLM.from_name(
        model_name,
        llm_opts={
            "enable_lora": True,
            "dtype": "float16",
            "gpu_memory_utilization": 0.1,
            "max_model_len": 15,
        },
    )


@pytest.fixture(scope="module")
def lora_path():
    return "vxef/smol_lora_toy"


@pytest.fixture(scope="module")
def transformer_llm(model_name, lora_path):
    transformer_llm_base = load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float16}}
    )
    transformer_llm_base.add_new_lora(lora_path, LORA_NAME)
    return transformer_llm_base


@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong",
        "It's probably this or that",
        "with the language model code",
        "It's probably this or that",
    ]
    tokenizer = async_llm.tokenizer
    token_ids_list = [tokenizer.encode(p) for p in test_prompts]
    return token_ids_list


def test_unknown_lora_error(transformer_llm):
    with pytest.raises(ValueError):
        transformer_llm.next_token_logprobs_uncached([0], lora_name="lora_2")


@v1_capable
@pytest.mark.parametrize("entry", ["async", "sync"])
def test_next_token_logprobs(
    async_llm, reference_llm, token_ids_list, lora_path, entry
):
    async_llm.add_new_lora(lora_path)
    reference_llm.set_lora(lora_path)
    for token_ids in token_ids_list:
        have = logprobs(async_llm, token_ids, entry, lora_name=LORA_NAME)
        have = have.float().cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
        assert have.shape == want.shape, ["Unexpected vocab mismatch.", have, want]
        assert compare(have, want).max_rel_err < 1e-2, token_ids
    reference_llm.clear_lora()


@v1_capable
@pytest.mark.parametrize("entry", ["async", "sync"])
def test_batch_next_token_logprobs(
    async_llm, reference_llm, token_ids_list, lora_path, entry
):
    async_llm.add_new_lora(lora_path)
    reference_llm.set_lora(lora_path)
    have = batch_logprobs(async_llm, token_ids_list, entry, lora_name=LORA_NAME)
    have = have.float().cpu().numpy()
    want = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
    assert have.shape == want.shape, ["Unexpected vocab mismatch.", have, want]
    assert_rows_close(have, want, token_ids_list, rel=1e-2)
    reference_llm.clear_lora()


@v1_capable
def test_swapping_lora_requests(token_ids_list, async_llm, lora_path):
    """Interleaved base and adapter requests match contiguous runs of each."""
    async_llm.add_new_lora(lora_path)

    def row(token_ids, lora_name):
        return logprobs(async_llm, token_ids, lora_name=lora_name).float().cpu().numpy()

    logits_noswapped_nolora = [row(ids, None) for ids in token_ids_list]
    logits_noswapped_lora = [row(ids, LORA_NAME) for ids in token_ids_list]

    logits_swapped_nolora = []
    logits_swapped_lora = []
    for token_ids in token_ids_list:
        logits_swapped_nolora.append(row(token_ids, None))
        logits_swapped_lora.append(row(token_ids, LORA_NAME))

    assert_rows_close(
        logits_noswapped_lora, logits_swapped_lora, token_ids_list, rel=1e-3
    )
    assert_rows_close(
        logits_noswapped_nolora, logits_swapped_nolora, token_ids_list, rel=1e-3
    )


@v1_capable
def test_reregistration(async_llm, token_ids_list, lora_pair):
    """Re-registering a name rebinds it to the new weights."""
    identity_path, shifted_path = lora_pair
    ids = token_ids_list[0]

    base = asyncio.run(async_llm.next_token_logprobs(ids)).float().cpu().numpy()
    async_llm.add_new_lora(identity_path, "reg")
    lp_identity = (
        asyncio.run(async_llm.next_token_logprobs(ids, lora_name="reg"))
        .float()
        .cpu()
        .numpy()
    )
    finite = np.isfinite(lp_identity) & np.isfinite(base)
    assert np.abs(lp_identity[finite] - base[finite]).max() < 1e-2

    async_llm.add_new_lora(shifted_path, "reg")
    lp_shifted = (
        asyncio.run(async_llm.next_token_logprobs(ids, lora_name="reg"))
        .float()
        .cpu()
        .numpy()
    )
    finite = np.isfinite(lp_shifted) & np.isfinite(lp_identity)
    assert np.abs(lp_shifted[finite] - lp_identity[finite]).max() > 1e-2

    async_llm.remove_lora("reg")
    with pytest.raises(ValueError):
        asyncio.run(async_llm.next_token_logprobs(ids, lora_name="reg"))


@v1_capable
def test_next_token_logprobs_agreement(
    transformer_llm, async_llm, token_ids_list, lora_path
):
    async_llm.add_new_lora(lora_path)
    for token_ids in token_ids_list:
        have = (
            transformer_llm.next_token_logprobs_uncached(token_ids, lora_name=LORA_NAME)
            .cpu()
            .numpy()
        )
        want = (
            asyncio.run(async_llm.next_token_logprobs(token_ids, lora_name=LORA_NAME))
            .cpu()
            .numpy()
        )

        hf_vocab = have.shape[0]

        want_trimmed = want[:hf_vocab]
        comparison = compare(have, want_trimmed)
        assert comparison.max_rel_err < 0.03, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids,
        ]
        assert comparison.pearson > 0.99, ["corr", comparison.pearson, token_ids]


@v1_capable
def test_batch_next_token_logprobs_agreement(
    transformer_llm, async_llm, token_ids_list, lora_path
):
    async_llm.add_new_lora(lora_path)
    haves = (
        asyncio.run(
            transformer_llm.batch_next_token_logprobs(
                token_ids_list, lora_name=LORA_NAME
            )
        )
        .cpu()
        .numpy()
    )
    wants = (
        asyncio.run(
            async_llm.batch_next_token_logprobs(token_ids_list, lora_name=LORA_NAME)
        )
        .cpu()
        .numpy()
    )
    for i, (have, want) in enumerate(zip(haves, wants)):
        hf_vocab = have.shape[0]
        want_trimmed = want[:hf_vocab]
        comparison = compare(have, want_trimmed)
        assert comparison.max_rel_err < 0.04, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids_list[i],
        ]
        assert comparison.pearson > 0.99, [
            "corr",
            comparison.pearson,
            token_ids_list[i],
        ]
