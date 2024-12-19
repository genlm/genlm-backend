import pytest
import asyncio
from arsenal.maths import compare
from async_llm.llm import AsyncLLM, ReferenceLLM

@pytest.fixture(scope="module")
def model_name(): 
    return 'gpt2'

@pytest.fixture(scope="module")
def reference_llm(model_name):
    return ReferenceLLM.from_name(model_name, llm_opts={'gpu_memory_utilization': 0.45})

@pytest.fixture(scope="module")
def async_llm(model_name):
    return AsyncLLM.from_name(model_name, engine_opts={'gpu_memory_utilization': 0.45})

@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
    ]
    tokenizer = async_llm.tokenizer
    token_ids_list = [tokenizer.encode(p) for p in test_prompts]
    return token_ids_list

def test_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids))
        have = have.cpu().numpy()
        want = reference_llm.next_token_logprobs(token_ids)
        assert compare(have, want).max_rel_err < 1e-5, token_ids

def test_batch_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    haves = asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list))
    haves = haves.cpu().numpy()
    wants = reference_llm.batch_next_token_logprobs(token_ids_list)
    for i, (have, want) in enumerate(zip(haves, wants)):          
        assert compare(have, want).max_rel_err < 1e-5, token_ids_list[i]