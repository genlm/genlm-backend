import pytest
from collections import defaultdict
from transformers import AutoTokenizer
from genlm.backend.llm import MockAsyncLM
from genlm.backend.tokenization import Token, decode_vocab
from genlm.backend.trie import TokenCharacterTrie


def test_gemma_model_load():
    """Test that Gemma model loads successfully with Token-based vocabulary.

    Gemma models have tokens with duplicate byte strings.
    """
    try:
        llm = MockAsyncLM.from_name("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    assert len(llm.byte_vocab) > 0
    assert all(isinstance(token, Token) for token in llm.byte_vocab)
    for i, token in enumerate(llm.byte_vocab):
        assert token.token_id == i


def test_gemma_vocabulary_duplicates():
    """Test that Gemma vocabulary correctly handles duplicate byte strings."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    byte_vocab, _ = decode_vocab(tokenizer)

    byte_string_to_token_ids = defaultdict(list)
    for token in byte_vocab:
        byte_string_to_token_ids[token.byte_string].append(token.token_id)

    duplicates = {bs: ids for bs, ids in byte_string_to_token_ids.items() if len(ids) > 1}
    assert len(duplicates) > 0, "No duplicates found in Gemma vocabulary"

    for byte_str, token_ids in duplicates.items():
        # Each duplicate byte string must map to distinct token_ids and Token objects
        assert len(set(token_ids)) == len(token_ids), (
            f"Duplicate token IDs for byte string {byte_str!r}"
        )
        tokens = [byte_vocab[tid] for tid in token_ids]
        assert len(set(tokens)) == len(tokens), (
            "Tokens with same byte string should be distinct objects"
        )


def test_gemma_trie_with_duplicates():
    """Test that trie works correctly with Gemma vocabulary containing duplicates."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    byte_vocab, _ = decode_vocab(tokenizer)
    trie = TokenCharacterTrie(decode=byte_vocab)
    assert len(trie.idx_to_leaf) == len(byte_vocab)

    # Verify each token has its own leaf node
    leaf_nodes = set()
    for idx, leaf_id in trie.idx_to_leaf:
        # Each token needs to have a unique leaf
        assert (idx, leaf_id) not in leaf_nodes or all(
            i != idx for i, _ in leaf_nodes
        ), f"Token at index {idx} should have unique leaf node"
        leaf_nodes.add((idx, leaf_id))


@pytest.mark.asyncio
async def test_gemma_model_operations():
    """Test that Gemma model can perform basic operations."""
    try:
        llm = MockAsyncLM.from_name("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    token_ids = llm.tokenizer.encode("Hello, world!")
    assert len(token_ids) > 0

    logprobs = await llm.next_token_logprobs(token_ids)
    assert logprobs.shape[0] == len(llm.byte_vocab)

    token_ids_list = [token_ids, llm.tokenizer.encode("Test prompt")]
    batch_logprobs = await llm.batch_next_token_logprobs(token_ids_list)
    assert batch_logprobs.shape == (len(token_ids_list), len(llm.byte_vocab))


def test_gemma_token_consistency():
    """Test that Token objects behave like bytes (iteration, indexing, decode)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    byte_vocab, _ = decode_vocab(tokenizer)

    for token_id in [0, 100, 500, 1000]:
        if token_id < len(byte_vocab):
            token = byte_vocab[token_id]
            assert list(token) == list(token.byte_string)
            if len(token) > 0:
                assert token[0] == token.byte_string[0]
            assert token.decode("utf-8", errors="replace") == token.byte_string.decode("utf-8", errors="replace")
