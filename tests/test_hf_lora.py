import pytest
import asyncio
import numpy as np
import torch
from conftest import cuda_only
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name


@pytest.fixture(scope="module")
def model_name():
    return "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def merged_path():
    return "vxef/smol_merged_toy"


@pytest.fixture(scope="module")
def lora_path():
    return "vxef/smol_lora_toy"


@pytest.fixture(scope="module")
def transformer_merged_llm(merged_path):
    return load_model_by_name(
        merged_path, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
    )


@pytest.fixture(scope="module")
def transformer_llm(model_name):
    return load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
    )


@pytest.fixture(scope="module")
def transformer_llm_nolora(model_name):
    return load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
    )


@pytest.fixture(scope="module", autouse=True)
def load_lora(transformer_llm, lora_path):
    transformer_llm.add_new_lora(lora_path, "lora_1")


@pytest.fixture(scope="module")
def token_ids_list(transformer_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",
    ]
    return [transformer_llm.tokenizer.encode(p) for p in test_prompts]


def test_unknown_lora_error(transformer_llm):
    with pytest.raises(ValueError):
        transformer_llm.next_token_logprobs_uncached([0], lora_name="lora_2")
    # The auto-batched path must fail the future, not leave the caller hung.
    with pytest.raises(ValueError):
        asyncio.run(transformer_llm.next_token_logprobs([0], lora_name="lora_2"))


def test_reregistration(transformer_llm, token_ids_list, lora_pair):
    """Re-registering a name rebinds it to the new weights and purges its cache
    trie — the cached sync path must not serve the old adapter's logprobs."""
    identity_path, shifted_path = lora_pair
    ids = token_ids_list[0]

    base = transformer_llm.next_token_logprobs_sync(ids).cpu().numpy()
    transformer_llm.add_new_lora(identity_path, "reg")
    lp_identity = (
        transformer_llm.next_token_logprobs_sync(ids, lora_name="reg").cpu().numpy()
    )
    assert np.abs(lp_identity - base).max() < 1e-3

    transformer_llm.add_new_lora(shifted_path, "reg")
    lp_shifted = (
        transformer_llm.next_token_logprobs_sync(ids, lora_name="reg").cpu().numpy()
    )
    assert np.abs(lp_shifted - lp_identity).max() > 1e-2

    transformer_llm.remove_lora("reg")
    with pytest.raises(ValueError):
        transformer_llm.next_token_logprobs_sync(ids, lora_name="reg")


@cuda_only
def test_transformer_llm(transformer_llm):
    assert transformer_llm is not None


@cuda_only
def test_transformer_merged_llm(transformer_merged_llm):
    assert transformer_merged_llm is not None


@cuda_only
def test_next_token_logprobs_lora_uncached(
    transformer_llm, transformer_merged_llm, token_ids_list
):
    for token_ids in token_ids_list:
        unmerged_logprobs = (
            transformer_llm.next_token_logprobs_uncached(token_ids, lora_name="lora_1")
            .cpu()
            .numpy()
        )
        merged_logprobs = (
            transformer_merged_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
        )
        assert compare(unmerged_logprobs, merged_logprobs).max_rel_err < 1e-3, token_ids


@cuda_only
def test_next_token_logprobs_lora(
    transformer_llm, transformer_merged_llm, token_ids_list
):
    for token_ids in token_ids_list:
        unmerged_logprobs = (
            asyncio.run(
                transformer_llm.next_token_logprobs(token_ids, lora_name="lora_1")
            )
            .cpu()
            .numpy()
        )
        merged_logprobs = (
            asyncio.run(transformer_merged_llm.next_token_logprobs(token_ids))
            .cpu()
            .numpy()
        )
        assert compare(unmerged_logprobs, merged_logprobs).max_rel_err < 1e-3, token_ids


@cuda_only
def test_token_logprobs_lora_sync(
    transformer_llm, transformer_merged_llm, token_ids_list
):
    unmerged_logprobs = [
        transformer_llm.next_token_logprobs_sync(token_ids, lora_name="lora_1")
        .cpu()
        .numpy()
        for token_ids in token_ids_list
    ]
    merged_logprobs = [
        transformer_merged_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        for token_ids in token_ids_list
    ]

    for i, (unmerged_logprob, merged_logprob) in enumerate(
        zip(unmerged_logprobs, merged_logprobs)
    ):
        assert compare(unmerged_logprob, merged_logprob).max_rel_err < 1e-3, (
            token_ids_list[i]
        )


@cuda_only
def test_batch_token_logprobs_lora(
    transformer_llm, transformer_merged_llm, token_ids_list
):
    unmerged_logprobs = (
        asyncio.run(
            transformer_llm.batch_next_token_logprobs(
                token_ids_list, lora_name="lora_1"
            )
        )
        .cpu()
        .numpy()
    )
    merged_logprobs = (
        asyncio.run(transformer_merged_llm.batch_next_token_logprobs(token_ids_list))
        .cpu()
        .numpy()
    )
    for i, (unmerged_logprob, merged_logprob) in enumerate(
        zip(unmerged_logprobs, merged_logprobs)
    ):
        assert compare(unmerged_logprob, merged_logprob).max_rel_err < 1e-3, (
            token_ids_list[i]
        )


@cuda_only
def test_batch_token_logprobs_lora_sync(
    transformer_llm, transformer_merged_llm, token_ids_list
):
    unmerged_logprobs = (
        transformer_llm.batch_next_token_logprobs_sync(
            token_ids_list, lora_name="lora_1"
        )
        .cpu()
        .numpy()
    )
    merged_logprobs = (
        transformer_merged_llm.batch_next_token_logprobs_sync(token_ids_list)
        .cpu()
        .numpy()
    )
    for i, (unmerged_logprob, merged_logprob) in enumerate(
        zip(unmerged_logprobs, merged_logprobs)
    ):
        assert compare(unmerged_logprob, merged_logprob).max_rel_err < 1e-3, (
            token_ids_list[i]
        )


@cuda_only
def test_adapter_swap(transformer_llm, token_ids_list, transformer_llm_nolora):
    """Interleaving base and adapter requests on one model matches a dedicated
    base model and a contiguous adapter run."""
    lora_logprobs_noswapped = []
    nolora_reference = []
    for token_ids in token_ids_list:
        lora_logprobs_noswapped.append(
            asyncio.run(
                transformer_llm.next_token_logprobs(token_ids, lora_name="lora_1")
            )
            .cpu()
            .numpy()
        )
        nolora_reference.append(
            asyncio.run(transformer_llm_nolora.next_token_logprobs(token_ids))
            .cpu()
            .numpy()
        )

    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in token_ids_list:
        lora_logprobs_swapped.append(
            asyncio.run(
                transformer_llm.next_token_logprobs(token_ids, lora_name="lora_1")
            )
            .cpu()
            .numpy()
        )
        nolora_logprobs_swapped.append(
            asyncio.run(transformer_llm.next_token_logprobs(token_ids)).cpu().numpy()
        )

    for i, (noswapped, swapped) in enumerate(
        zip(lora_logprobs_noswapped, lora_logprobs_swapped)
    ):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (reference, swapped) in enumerate(
        zip(nolora_reference, nolora_logprobs_swapped)
    ):
        assert compare(reference, swapped).max_rel_err < 1e-3, token_ids_list[i]


@cuda_only
def test_adapter_swap_uncached(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = []
    nolora_reference = []
    for token_ids in token_ids_list:
        lora_logprobs_noswapped.append(
            transformer_llm.next_token_logprobs_uncached(token_ids, lora_name="lora_1")
            .cpu()
            .numpy()
        )
        nolora_reference.append(
            transformer_llm_nolora.next_token_logprobs_uncached(token_ids).cpu().numpy()
        )

    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in token_ids_list:
        lora_logprobs_swapped.append(
            transformer_llm.next_token_logprobs_uncached(token_ids, lora_name="lora_1")
            .cpu()
            .numpy()
        )
        nolora_logprobs_swapped.append(
            transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
        )

    for i, (noswapped, swapped) in enumerate(
        zip(lora_logprobs_noswapped, lora_logprobs_swapped)
    ):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (reference, swapped) in enumerate(
        zip(nolora_reference, nolora_logprobs_swapped)
    ):
        assert compare(reference, swapped).max_rel_err < 1e-3, token_ids_list[i]


@cuda_only
def test_adapter_swap_sync(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = [
        transformer_llm.next_token_logprobs_sync(token_ids, lora_name="lora_1")
        .cpu()
        .numpy()
        for token_ids in token_ids_list
    ]
    nolora_reference = [
        transformer_llm_nolora.next_token_logprobs_sync(token_ids).cpu().numpy()
        for token_ids in token_ids_list
    ]

    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in token_ids_list:
        lora_logprobs_swapped.append(
            transformer_llm.next_token_logprobs_sync(token_ids, lora_name="lora_1")
            .cpu()
            .numpy()
        )
        nolora_logprobs_swapped.append(
            transformer_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        )

    for i, (noswapped, swapped) in enumerate(
        zip(lora_logprobs_noswapped, lora_logprobs_swapped)
    ):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (reference, swapped) in enumerate(
        zip(nolora_reference, nolora_logprobs_swapped)
    ):
        assert compare(reference, swapped).max_rel_err < 1e-3, token_ids_list[i]


@cuda_only
def test_adapter_swap_mixed_batch(
    transformer_llm, token_ids_list, transformer_llm_nolora
):
    """One auto-batched dispatch containing BOTH base and adapter queries routes
    each query through its own adapter."""
    transformer_llm.clear_cache()

    async def mixed(token_ids_list):
        lora = asyncio.gather(
            *[
                transformer_llm.next_token_logprobs(t, lora_name="lora_1")
                for t in token_ids_list
            ]
        )
        base = asyncio.gather(
            *[transformer_llm.next_token_logprobs(t) for t in token_ids_list]
        )
        return await lora, await base

    lora_logprobs, base_logprobs = asyncio.run(mixed(token_ids_list))

    lora_reference = (
        asyncio.run(
            transformer_llm.batch_next_token_logprobs(
                token_ids_list, lora_name="lora_1"
            )
        )
        .cpu()
        .numpy()
    )
    base_reference = (
        asyncio.run(transformer_llm_nolora.batch_next_token_logprobs(token_ids_list))
        .cpu()
        .numpy()
    )

    for i, token_ids in enumerate(token_ids_list):
        assert (
            compare(lora_logprobs[i].cpu().numpy(), lora_reference[i]).max_rel_err
            < 1e-3
        ), token_ids
        assert (
            compare(base_logprobs[i].cpu().numpy(), base_reference[i]).max_rel_err
            < 1e-3
        ), token_ids
