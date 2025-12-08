import pytest
import asyncio
import torch
from conftest import cuda_only
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name

@pytest.fixture(scope="module")
def model_name():
    return "meta-llama/Llama-3.2-1B"

@pytest.fixture(scope="module")
def adapter_path():
    return "vxef/lora_adapter_merged_model"

@pytest.fixture(scope="module")
def lora_path():
    return "vxef/lora_adapter_toy"

@pytest.fixture(scope="module")
def transformer_merged_llm(adapter_path):
    return load_model_by_name(
        adapter_path, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
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
    transformer_llm.load_lora(lora_path, 'lora_1')
    transformer_llm.set_lora(lora_name='lora_1')


@pytest.fixture(scope="module")
def token_ids_list(transformer_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",  
    ]
    return [transformer_llm.tokenizer.encode(p) for p in test_prompts]

@cuda_only
def test_transformer_llm(transformer_llm):
    assert transformer_llm is not None

@cuda_only
def test_transformer_merged_llm(transformer_merged_llm):
    assert transformer_merged_llm is not None

@cuda_only
def test_next_token_logprobs_lora_uncached(transformer_llm, transformer_merged_llm, token_ids_list):
    for token_ids in token_ids_list:
        unmerged_logprobs = transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
        merged_logprobs = transformer_merged_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
        assert compare(unmerged_logprobs, merged_logprobs).max_rel_err < 1e-3, token_ids

@cuda_only
def test_next_token_logprobs_lora(transformer_llm, transformer_merged_llm, token_ids_list):
    for token_ids in token_ids_list:
        unmerged_logprobs = asyncio.run(transformer_llm.next_token_logprobs(token_ids)).cpu().numpy()
        merged_logprobs = asyncio.run(transformer_merged_llm.next_token_logprobs(token_ids)).cpu().numpy()
        assert compare(unmerged_logprobs, merged_logprobs).max_rel_err < 1e-3, token_ids

@cuda_only
def test_token_logprobs_lora_sync(transformer_llm, transformer_merged_llm, token_ids_list):
    unmerged_logprobs = [transformer_llm.next_token_logprobs_sync(token_ids).cpu().numpy() for token_ids in token_ids_list]
    merged_logprobs = [transformer_merged_llm.next_token_logprobs_sync(token_ids).cpu().numpy() for token_ids in token_ids_list]
    
    for i, (unmerged_logprob, merged_logprob) in enumerate(zip(unmerged_logprobs, merged_logprobs)):
        assert compare(unmerged_logprob, merged_logprob).max_rel_err < 1e-3, token_ids_list[i]
    
@cuda_only
def test_batch_token_logprobs_lora(transformer_llm, transformer_merged_llm, token_ids_list):
    unmerged_logprobs = (
        asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    merged_logprobs = (
        asyncio.run(transformer_merged_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    for i, (unmerged_logprob, merged_logprob) in enumerate(zip(unmerged_logprobs, merged_logprobs)):
        assert compare(unmerged_logprob, merged_logprob).max_rel_err < 1e-3, token_ids_list[i]

@cuda_only
def test_batch_token_logprobs_lora_sync(transformer_llm, transformer_merged_llm, token_ids_list):
    unmerged_logprobs = transformer_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    merged_logprobs = transformer_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    for i, (unmerged_logprob, merged_logprob) in enumerate(zip(unmerged_logprobs, merged_logprobs)):
        assert compare(unmerged_logprob, merged_logprob).max_rel_err < 1e-3, token_ids_list[i]

@cuda_only 
def test_set_disable_swap(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = []
    nolora_logprobs_noswapped = []
    for token_ids in token_ids_list:
        lora_logprobs_noswapped.append(asyncio.run(transformer_llm.next_token_logprobs(token_ids)).cpu().numpy())
        nolora_logprobs_noswapped.append(asyncio.run(transformer_llm_nolora.next_token_logprobs(token_ids)).cpu().numpy())
                    
    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in token_ids_list:
        lora_logprobs_swapped.append(asyncio.run(transformer_llm.next_token_logprobs(token_ids)).cpu().numpy())
        transformer_llm.clear_lora()
        nolora_logprobs_swapped.append(asyncio.run(transformer_llm.next_token_logprobs(token_ids)).cpu().numpy())
        transformer_llm.set_lora('lora_1')
        
    for i, (noswapped, swapped) in enumerate(zip(lora_logprobs_noswapped, lora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (noswapped, swapped) in enumerate(zip(nolora_logprobs_noswapped, nolora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]

@cuda_only 
def test_set_disable_swap_unchached(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = []
    nolora_logprobs_noswapped = []
    for token_ids in token_ids_list:
        lora_logprobs_noswapped.append(transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy())
        nolora_logprobs_noswapped.append(transformer_llm_nolora.next_token_logprobs_uncached(token_ids).cpu().numpy())
        
    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in token_ids_list:
        lora_logprobs_swapped.append(transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy())
        transformer_llm.clear_lora()
        nolora_logprobs_swapped.append(transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy())
        transformer_llm.set_lora('lora_1')
        
    for i, (noswapped, swapped) in enumerate(zip(lora_logprobs_noswapped, lora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (noswapped, swapped) in enumerate(zip(nolora_logprobs_noswapped, nolora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]


@cuda_only 
def test_set_disable_swap_sync(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = [transformer_llm.next_token_logprobs_sync(token_ids).cpu().numpy() for token_ids in token_ids_list]
    nolora_logprobs_noswapped = [transformer_llm_nolora.next_token_logprobs_sync(token_ids).cpu().numpy() for token_ids in token_ids_list]
        
    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in token_ids_list:
        lora_logprobs_swapped.append(transformer_llm.next_token_logprobs_sync(token_ids).cpu().numpy())
        transformer_llm.clear_lora()
        nolora_logprobs_swapped.append(transformer_llm.next_token_logprobs_sync(token_ids).cpu().numpy())
        transformer_llm.set_lora('lora_1')
        
    for i, (noswapped, swapped) in enumerate(zip(lora_logprobs_noswapped, lora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (noswapped, swapped) in enumerate(zip(nolora_logprobs_noswapped, nolora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]


@cuda_only 
def test_set_disable_swap_batch(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = (
        asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
        )
    nolora_logprobs_noswapped = (
        asyncio.run(transformer_llm_nolora.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
        )
    
    batches = [token_ids_list[i:i+2] for i in range(0, len(token_ids_list), 2)]

    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in batches:
        lora_logprobs_swapped.extend(asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids)).cpu().numpy())
        transformer_llm.clear_lora()
        nolora_logprobs_swapped.extend(asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids)).cpu().numpy())
        transformer_llm.set_lora('lora_1')
        
    for i, (noswapped, swapped) in enumerate(zip(lora_logprobs_noswapped, lora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (noswapped, swapped) in enumerate(zip(nolora_logprobs_noswapped, nolora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]

@cuda_only 
def test_set_disable_swap_batch_sync(transformer_llm, token_ids_list, transformer_llm_nolora):
    lora_logprobs_noswapped = transformer_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    nolora_logprobs_noswapped = transformer_llm_nolora.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    
    batches = [token_ids_list[i:i+2] for i in range(0, len(token_ids_list), 2)]

    lora_logprobs_swapped = []
    nolora_logprobs_swapped = []
    for token_ids in batches:
        lora_logprobs_swapped.extend(transformer_llm.batch_next_token_logprobs_sync(token_ids).cpu().numpy())
        transformer_llm.clear_lora()
        nolora_logprobs_swapped.extend(transformer_llm.batch_next_token_logprobs_sync(token_ids).cpu().numpy())
        transformer_llm.set_lora('lora_1')
        
    for i, (noswapped, swapped) in enumerate(zip(lora_logprobs_noswapped, lora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
    for i, (noswapped, swapped) in enumerate(zip(nolora_logprobs_noswapped, nolora_logprobs_swapped)):
        assert compare(noswapped, swapped).max_rel_err < 1e-3, token_ids_list[i]
