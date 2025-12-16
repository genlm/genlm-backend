"""Tests for Token class and handling of duplicate byte strings."""

import pytest
import torch
import numpy as np
from genlm.backend.tokenization import Token
from genlm.backend.trie import TokenCharacterTrie, ParallelTokenCharacterTrie


def test_token_creation():
    """Test basic Token creation and attributes."""
    token = Token(token_id=42, byte_string=b"hello")
    assert token.token_id == 42
    assert token.byte_string == b"hello"
    assert len(token) == 5
    assert repr(token) == "Token(token_id=42, byte_string=b'hello')"


def test_token_equality():
    """Test Token equality comparison.

    Equality is based on token_id only, since that's the unique identifier.
    Multiple tokens can have the same byte_string but different token_ids.
    """
    token1 = Token(token_id=1, byte_string=b"test")
    token2 = Token(token_id=1, byte_string=b"test")
    token3 = Token(token_id=2, byte_string=b"test")
    token4 = Token(token_id=1, byte_string=b"other")

    assert token1 == token2  # Same token_id
    assert token1 != token3  # Different token_id
    assert token1 == token4  # Same token_id (even though different byte_string)
    assert token3 != token4  # Different token_id


def test_token_hash():
    """Test Token hashing for use in sets and dicts."""
    token1 = Token(token_id=1, byte_string=b"test")
    token2 = Token(token_id=1, byte_string=b"test")
    token3 = Token(token_id=2, byte_string=b"test")

    # Same tokens should have same hash
    assert hash(token1) == hash(token2)

    # Can be used in sets
    token_set = {token1, token2, token3}
    assert len(token_set) == 2


def test_token_iteration():
    """Test Token iteration over bytes."""
    token = Token(token_id=1, byte_string=b"abc")
    assert list(token) == [97, 98, 99]  # ASCII values of 'a', 'b', 'c'
    assert token[0] == 97
    assert token[1] == 98
    assert token[2] == 99


def test_token_type_validation():
    """Test that Token validates input types."""
    Token(token_id=1, byte_string=b"test")
    with pytest.raises(TypeError):
        Token(token_id="1", byte_string=b"test")
    with pytest.raises(TypeError):
        Token(token_id=1, byte_string="test")


def test_trie_with_duplicate_byte_strings():
    """Test that trie correctly handles multiple tokens with same byte string."""
    vocab = [
        Token(token_id=0, byte_string=b"a"),
        Token(token_id=1, byte_string=b"hello"),
        Token(token_id=2, byte_string=b"hello"),
        Token(token_id=3, byte_string=b"world"),
    ]

    trie = TokenCharacterTrie(decode=vocab)

    # Verify that both tokens got their own leaf nodes
    assert len(trie.token_id_to_leaf) == 4

    # Get the leaf nodes for duplicate tokens
    leaf_1 = trie.token_id_to_leaf[1][1]
    leaf_2 = trie.token_id_to_leaf[2][1]

    # should have different leaf nodes
    assert leaf_1 != leaf_2, "Tokens with same byte_string should have different leaves"

    # Both should be valid leaf nodes
    assert leaf_1 in trie.leaf2word.keys()
    assert leaf_2 in trie.leaf2word.keys()


def test_trie_weight_sum_with_duplicates():
    """Test that weight sums work correctly with duplicate byte strings."""
    vocab = [
        Token(token_id=0, byte_string=b"a"),
        Token(token_id=1, byte_string=b"hello"),
        Token(token_id=2, byte_string=b"hello"),
        Token(token_id=3, byte_string=b"world"),
    ]

    trie = TokenCharacterTrie(decode=vocab)

    # Assign different weights to the duplicate tokens
    weights = torch.tensor([0.1, 0.3, 0.5, 0.1])

    node_weights = trie.weight_sum(weights)

    # Get the leaf weights for the duplicate tokens
    leaf_1 = trie.token_id_to_leaf[1][1]
    leaf_2 = trie.token_id_to_leaf[2][1]

    # Each leaf should have its own weight
    assert np.isclose(node_weights[leaf_1], 0.3, rtol=1e-5)
    assert np.isclose(node_weights[leaf_2], 0.5, rtol=1e-5)


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
def test_parallel_trie_with_duplicates(device):
    """Test that parallel trie correctly handles duplicate byte strings."""
    vocab = [
        Token(token_id=0, byte_string=b"a"),
        Token(token_id=1, byte_string=b"test"),
        Token(token_id=2, byte_string=b"test"),
        Token(token_id=3, byte_string=b"b"),
    ]

    trie = ParallelTokenCharacterTrie(decode=vocab, device=device)

    # Verify token_id_to_leaf mapping
    assert len(trie.token_id_to_leaf) == 4

    # Test weight sum
    weights = torch.tensor([0.1, 0.2, 0.7, 0.0], device=device)
    node_weights = trie.weight_sum(weights)

    # Verify the duplicate tokens have independent weights
    leaf_1 = trie.token_id_to_leaf[1][1]
    leaf_2 = trie.token_id_to_leaf[2][1]
    assert leaf_1 != leaf_2
    assert np.isclose(node_weights[leaf_1], 0.2, rtol=1e-5)
    assert np.isclose(node_weights[leaf_2], 0.7, rtol=1e-5)


def test_trie_requires_token_objects():
    """Test trie requires Token objects and rejects raw bytes."""
    with pytest.raises(TypeError, match="TokenCharacterTrie requires Token objects"):
        TokenCharacterTrie(decode=[b"a", b"b", b"c"])
    with pytest.raises(TypeError, match="TokenCharacterTrie requires Token objects"):
        TokenCharacterTrie(decode=["a", "b", "c"])
    with pytest.raises(TypeError, match="TokenCharacterTrie requires Token objects"):
        TokenCharacterTrie(decode=[Token(0, b"a"), b"b"])


def test_token_in_trie_word2leaf_key():
    """Test that word2leaf correctly uses (bytes, token_id) as key."""
    vocab = [
        Token(token_id=5, byte_string=b"test"),
        Token(token_id=10, byte_string=b"test"),
    ]

    trie = TokenCharacterTrie(decode=vocab)

    # Check that both tokens are in word2leaf with proper keys
    key1 = (b"test", 5)
    key2 = (b"test", 10)

    assert key1 in trie.word2leaf
    assert key2 in trie.word2leaf
    assert trie.word2leaf[key1] != trie.word2leaf[key2]
