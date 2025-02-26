import torch
import pytest
import asyncio
import numpy as np
from transformers import AutoTokenizer

from genlm_backend.llm import MockAsyncLM
from genlm_backend.trie import (
    TokenCharacterTrie,
    ParallelTokenCharacterTrie,
    AsyncTokenCharacterTrie,
)


@pytest.fixture()
def decode():
    return [b"a", b"b", b"ab", b"<eos>"]


@pytest.fixture(scope="module")
def mock_llm():
    return MockAsyncLM(AutoTokenizer.from_pretrained("gpt2"))


def test_sequential_weight_sum(decode):
    trie = TokenCharacterTrie(decode=decode)
    haves = trie.weight_sum(torch.tensor([0.1, 0.2, 0.2, 0.5]))

    leaf_wants = {
        b"a": 0.1,
        b"b": 0.2,
        b"ab": 0.2,
        b"<eos>": 0.5,
    }
    internal_wants = {
        b"": 1,
        b"a": 0.3,
        b"b": 0.2,
        b"ab": 0.2,
        b"<": 0.5,
        b"<e": 0.5,
        b"<eo": 0.5,
        b"<eos": 0.5,
        b"<eos>": 0.5,
    }

    for node, prefix in trie.node2prefix.items():
        have = haves[node]
        if node in trie.leaf2word:
            want = leaf_wants[bytes(prefix)]
        else:
            want = internal_wants[bytes(prefix)]
        assert np.isclose(have, want, rtol=1e-5, atol=1e-8), [have, want, prefix]


def test_sequential_weight_max(decode):
    trie = TokenCharacterTrie(decode=decode)
    haves = trie.weight_max(torch.tensor([0.1, 0.2, 0.2, 0.5]))

    leaf_wants = {
        b"a": 0.1,
        b"b": 0.2,
        b"ab": 0.2,
        b"<eos>": 0.5,
    }
    internal_wants = {
        b"": 0.5,  # max of all nodes
        b"a": 0.2,  # max of "a" and "ab"
        b"b": 0.2,  # just "b"
        b"ab": 0.2,  # just "ab"
        b"<": 0.5,  # just "<eos>"
        b"<e": 0.5,  # just "<eos>"
        b"<eo": 0.5,  # just "<eos>"
        b"<eos": 0.5,  # just "<eos>"
        b"<eos>": 0.5,  # just "<eos>"
    }

    for node, prefix in trie.node2prefix.items():
        have = haves[node]
        if node in trie.leaf2word:
            want = leaf_wants[bytes(prefix)]
        else:
            want = internal_wants[bytes(prefix)]
        assert np.isclose(have, want, rtol=1e-5, atol=1e-8), [have, want, prefix]


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_single_agreement(decode, device):
    trie = TokenCharacterTrie(decode=decode)
    ws = torch.tensor([0.1, 0.2, 0.2, 0.5]).to(device)
    parallel_trie = ParallelTokenCharacterTrie(decode=decode, device=device)

    parallel_weights = parallel_trie.weight_sum(ws)
    sequential_weights = trie.weight_sum(ws)

    assert np.allclose(parallel_weights, sequential_weights, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_batch_agreement(decode, device):
    sequential_trie = TokenCharacterTrie(decode=decode)
    parallel_trie = ParallelTokenCharacterTrie(decode=decode, device=device)

    ws = torch.stack(
        [
            torch.tensor([0.1, 0.2, 0.2, 0.5]),
            torch.tensor([0, 0.3, 0.6, 0.1]),
            torch.tensor([0.99, 0.01, 0, 0]),
        ]
    ).to(device)

    parallel_weights = parallel_trie.batch_weight_sum(ws)
    sequential_weights = sequential_trie.batch_weight_sum(ws)

    assert len(parallel_weights) == len(sequential_weights)

    for have, want in zip(sequential_weights, parallel_weights):
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-8)

    parallel_weights = parallel_trie.batch_weight_max(ws)
    sequential_weights = sequential_trie.batch_weight_max(ws)

    assert len(parallel_weights) == len(sequential_weights)

    for have, want in zip(sequential_weights, parallel_weights):
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", ["sequential", "parallel"])
@pytest.mark.asyncio
async def test_async_trie(mock_llm, backend):
    async_trie = AsyncTokenCharacterTrie.from_vocab(
        mock_llm.byte_vocab, backend=backend
    )
    all_token_ids = [[0, 1, 3], [10, 20, 30], [8, 100]]
    all_weights = torch.exp(await mock_llm.batch_next_token_logprobs(all_token_ids))

    haves = await asyncio.gather(*[async_trie.weight_sum(ws) for ws in all_weights])
    wants = async_trie.trie.batch_weight_sum(all_weights)

    assert len(haves) == len(wants)

    for have, want in zip(haves, wants):
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-8)

    haves = await asyncio.gather(*[async_trie.weight_max(ws) for ws in all_weights])
    wants = async_trie.trie.batch_weight_max(all_weights)

    assert len(haves) == len(wants)

    for have, want in zip(haves, wants):
        np.testing.assert_allclose(have, want, rtol=1e-5, atol=1e-8)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sequential", "parallel"])
async def test_async_trie_cleanup(mock_llm, backend):
    async_trie = AsyncTokenCharacterTrie.from_vocab(
        mock_llm.byte_vocab, backend=backend
    )
    await async_trie.cleanup()
    assert async_trie._task is None


def test_sequential_preprocessing(decode):
    trie = TokenCharacterTrie(decode=decode)

    ws = torch.zeros(len(decode))
    processed = trie._preprocess_ws(ws)
    assert isinstance(processed, np.ndarray)

    ws = torch.tensor([0.1] * len(decode))
    processed = trie._preprocess_ws(ws)
    assert isinstance(processed, np.ndarray)

    if torch.cuda.is_available():
        ws = torch.tensor([0.1] * len(decode), device="cuda")
        processed = trie._preprocess_ws(ws)
        assert isinstance(processed, np.ndarray)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_parallel_preprocessing(decode, device):
    parallel_trie = ParallelTokenCharacterTrie(decode=decode, device=device)

    # Test numpy array input
    np_weights = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]])
    processed = parallel_trie._preprocess_ws(np_weights)
    assert isinstance(processed, torch.Tensor)
    assert processed.device.type == parallel_trie.device
    assert processed.dtype == torch.float32

    # Test list input
    list_weights = [[0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]]
    processed = parallel_trie._preprocess_ws(list_weights)
    assert isinstance(processed, torch.Tensor)
    assert processed.device.type == parallel_trie.device
    assert processed.dtype == torch.float32

    # Test tensor with wrong device
    if torch.cuda.is_available():
        wrong_device = "cuda" if parallel_trie.device == "cpu" else "cpu"
        tensor_weights = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]], device=wrong_device
        )
        processed = parallel_trie._preprocess_ws(tensor_weights)
        assert processed.device.type == parallel_trie.device
        assert processed.dtype == torch.float32


def test_visualize(decode):
    trie = TokenCharacterTrie(decode=decode)
    ws = torch.tensor([0.1] * len(trie.children))
    trie.visualize(ws)
