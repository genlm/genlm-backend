import torch
import pytest
import asyncio
import numpy as np
from arsenal.maths import compare
from transformers import AutoTokenizer

from async_llm.vocabulary import decode_vocab
from async_llm.trie import TokenCharacterTrie, ParallelTokenCharacterTrie, AsyncTokenCharacterTrie

# Run tests for both byte and str vocabularies.
@pytest.fixture(params=['byte', 'str']) 
def vocab(request):
    decode = ['a', 'b', 'ab', '<eos>']
    old_eos = '<eos>'
    new_eos = '.'
    vocab_type = request.param
    
    if vocab_type == "byte":
        decode = [bytes(v, 'utf-8') for v in decode]
        old_eos = bytes(old_eos, 'utf-8')
        new_eos = bytes(new_eos, 'utf-8')
        
    return decode, old_eos, new_eos

# Run tests for CPU and GPU (when available).
@pytest.fixture(
    params=[
        "cpu", 
        pytest.param(
            "cuda", 
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            )
        )
    ]
)
def device(request):
    return request.param

# Mock LLM backend for simplicity.
@pytest.fixture(scope="module")
def mock_llm():
    class MockAsyncLLM:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.byte_vocab, self.str_vocab = decode_vocab(tokenizer)
            self._rng = np.random.RandomState(42)
            
        def _get_logits(self, token_ids):
            # Use token_ids to seed the random generator
            # This ensures same token_ids always produce same logits
            seed = sum([(i + 1) * t for i, t in enumerate(token_ids)])
            self._rng.seed(seed)
            logits = torch.from_numpy(self._rng.rand(len(self.tokenizer)).astype(np.float32))
            return torch.softmax(logits, dim=-1)
            
        async def batch_next_token_logprobs(self, token_ids_list):
            return torch.stack([self._get_logits(token_ids) for token_ids in token_ids_list])
    
    return MockAsyncLLM(AutoTokenizer.from_pretrained('gpt2'))

def test_sequential_mass_sum(vocab): 
    decode, old_eos, new_eos = vocab

    trie = TokenCharacterTrie(decode=decode, old_eos=old_eos, new_eos=new_eos)
    haves = trie.mass_sum(torch.tensor([0.1, 0.2, 0.2, 0.5]))

    leaf_wants = {'a' : 0.1, 'b' : 0.2, 'ab' : 0.2, '.' : 0.5}
    internal_wants = {"" : 1, 'a' : 0.3, 'b' : 0.2, 'ab' : 0.2, '.' : 0.5}

    for node, prefix in trie.node2prefix.items():
        have = haves[node]
        prefix = prefix if isinstance(prefix, str) else prefix.decode('utf-8')
        if node in trie.leaf2word:
            want = leaf_wants[prefix]
        else:
            want = internal_wants[prefix]
        assert np.isclose(have, want, rtol=1e-5, atol=1e-8), [have, want, prefix]

def test_mass_sum_agreement(vocab, device):
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

@pytest.mark.parametrize("vocab_type", ['byte', 'str'])
def test_async_trie(mock_llm, vocab_type, device):
    async_trie = AsyncTokenCharacterTrie(
        mock_llm, new_eos=mock_llm.tokenizer.eos_token, device=device, vocab=vocab_type
    )
    all_token_ids = [[0,1,3], [10,20,30], [8,100]]
    next_token_tries = asyncio.run(async_trie.batch_next_token_trie(all_token_ids))
    haves = [trie.mass for trie in next_token_tries]

    logp_llms = asyncio.run(mock_llm.batch_next_token_logprobs(all_token_ids))
    wants = async_trie.trie.batch_mass_sum(np.exp(logp_llms))

    assert len(haves) == len(wants)

    for have, want in zip(haves, wants):
        assert compare(have, want).max_rel_err <= 0.001, [have, want]

    del async_trie