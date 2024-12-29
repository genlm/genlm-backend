import torch
import pytest
import asyncio
import numpy as np
from conftest import cuda_only
from arsenal.maths import compare
from transformers import AutoTokenizer

from genlm_backend.llm import MockAsyncLM
from genlm_backend.trie import TokenCharacterTrie, ParallelTokenCharacterTrie, AsyncTokenCharacterTrie

@pytest.fixture() 
def vocab():
    decode = [b'a', b'b', b'ab', b'<eos>']
    old_eos = b'<eos>'
    new_eos = b'.'
    return decode, old_eos, new_eos

@pytest.fixture(scope="module")
def mock_llm():
    return MockAsyncLM(AutoTokenizer.from_pretrained('gpt2'))

def test_sequential_mass_sum(vocab): 
    decode, old_eos, new_eos = vocab

    trie = TokenCharacterTrie(decode=decode, old_eos=old_eos, new_eos=new_eos)
    haves = trie.mass_sum(torch.tensor([0.1, 0.2, 0.2, 0.5]))

    leaf_wants = {b'a' : 0.1, b'b' : 0.2, b'ab' : 0.2, b'.' : 0.5}
    internal_wants = {b"" : 1, b'a' : 0.3, b'b' : 0.2, b'ab' : 0.2, b'.' : 0.5}

    for node, prefix in trie.node2prefix.items():
        have = haves[node]
        if node in trie.leaf2word:
            want = leaf_wants[prefix]
        else:
            want = internal_wants[prefix]
        assert np.isclose(have, want, rtol=1e-5, atol=1e-8), [have, want, prefix]

def _test_mass_sum_agreement(vocab, device):
    decode, old_eos, new_eos = vocab

    sequential_trie = TokenCharacterTrie(
        decode=decode,
        old_eos=old_eos,
        new_eos=new_eos
    )

    parallel_trie = ParallelTokenCharacterTrie(
        decode=decode,
        old_eos=old_eos,
        new_eos=new_eos,
        device=device
    )

    p_llms = torch.stack([
        torch.tensor([0.1, 0.2, 0.2, 0.5]),
        torch.tensor([0, 0.3, 0.6, 0.1]),
        torch.tensor([.99, 0.01, 0, 0])
    ]).to(device)

    parallel_masses = parallel_trie.batch_mass_sum(p_llms)
    sequential_masses = sequential_trie.batch_mass_sum(p_llms)

    assert len(parallel_masses) == len(sequential_masses)

    for have, want in zip(sequential_masses, parallel_masses):
        assert compare(have, want).max_rel_err <= 0.001

def test_mass_sum_agreement_cpu(vocab):
    _test_mass_sum_agreement(vocab, 'cpu')

@cuda_only
def test_mass_sum_agreement_gpu(vocab):
    _test_mass_sum_agreement(vocab, 'cuda')

def _test_async_trie(mock_llm, device):
    async_trie = AsyncTokenCharacterTrie(
        mock_llm, new_eos=mock_llm.tokenizer.eos_token, device=device,
    )
    all_token_ids = [[0,1,3], [10,20,30], [8,100]]
    next_token_tries = asyncio.run(async_trie.batch_next_token_trie(all_token_ids))
    haves = [trie.mass for trie in next_token_tries]

    logp_llms = asyncio.run(mock_llm.batch_next_token_logprobs(all_token_ids))
    wants = async_trie.trie.batch_mass_sum(np.exp(logp_llms))

    assert len(haves) == len(wants)

    for have, want in zip(haves, wants):
        assert compare(have, want).max_rel_err <= 0.001, [have, want]

def test_async_trie_cpu(mock_llm):
    _test_async_trie(mock_llm, 'cpu')

@cuda_only
def test_async_trie_gpu(mock_llm):
    _test_async_trie(mock_llm, 'cuda')