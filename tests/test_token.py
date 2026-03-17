"""Tests for Token class and handling of duplicate byte strings."""

import pytest
import torch
import numpy as np
import copy
import pickle
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
    assert len(trie.idx_to_leaf) == 4

    # Get the leaf nodes for duplicate tokens
    leaf_1 = trie.idx_to_leaf[1][1]
    leaf_2 = trie.idx_to_leaf[2][1]

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
    leaf_1 = trie.idx_to_leaf[1][1]
    leaf_2 = trie.idx_to_leaf[2][1]

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

    # Verify idx_to_leaf mapping
    assert len(trie.idx_to_leaf) == 4

    # Test weight sum
    weights = torch.tensor([0.1, 0.2, 0.7, 0.0], device=device)
    node_weights = trie.weight_sum(weights)

    # Verify the duplicate tokens have independent weights
    leaf_1 = trie.idx_to_leaf[1][1]
    leaf_2 = trie.idx_to_leaf[2][1]
    assert leaf_1 != leaf_2
    assert np.isclose(node_weights[leaf_1], 0.2, rtol=1e-5)
    assert np.isclose(node_weights[leaf_2], 0.7, rtol=1e-5)


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


# ---------------------------------------------------------------------------
# Token as bytes subclass: deepcopy, pickle, join, ordering
# ---------------------------------------------------------------------------


def test_token_deepcopy():
    """deepcopy must preserve token_id (SMC resampling uses deepcopy)."""
    t = Token(42, b"hello")
    t2 = copy.deepcopy(t)
    assert t2.token_id == 42
    assert t2.byte_string == b"hello"
    assert t2 == t
    assert t2 is not t


def test_token_pickle_roundtrip():
    """Tokens must survive pickle (used by multiprocessing potentials)."""
    t = Token(42, b"hello")
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.token_id == 42
    assert t2.byte_string == b"hello"


def test_token_bytes_join():
    """b''.join must work on Token objects (bytes subclass)."""
    tokens = [Token(0, b"hello"), Token(1, b" "), Token(2, b"world")]
    assert b"".join(tokens) == b"hello world"


def test_token_ordering():
    """Tokens must sort by token_id, not byte content."""
    t_z = Token(1, b"z")
    t_a = Token(2, b"a")
    assert t_z < t_a  # 1 < 2, even though z > a
    assert t_a > t_z
    assert t_z <= t_z
    assert t_z >= t_z
    assert t_z <= t_a
    assert sorted([t_a, t_z]) == [t_z, t_a]


def test_token_ne():
    """!= between Tokens must use token_id, not byte content."""
    t1 = Token(0, b"same")
    t2 = Token(1, b"same")  # same bytes, different id
    t3 = Token(0, b"other")  # same id, different bytes
    assert t1 != t2
    assert not (t1 != t3)  # same id → not unequal


def test_token_byte_string_is_plain_bytes():
    """.byte_string must return plain bytes, not a Token."""
    t = Token(0, b"hello")
    bs = t.byte_string
    assert type(bs) is bytes
    assert not isinstance(bs, Token)


def test_token_cross_type_comparison():
    """Token compared to plain bytes falls back to bytes content comparison.

    This exercises the NotImplemented return paths in __eq__, __ne__,
    __lt__, __le__, __gt__, __ge__ (Token subclasses bytes, so Python
    falls back to bytes.__eq__ etc).
    """
    t = Token(0, b"hello")

    # eq/ne fall back to bytes content comparison
    assert t == b"hello"  # bytes.__eq__ compares content
    assert not (t != b"hello")  # bytes.__ne__
    assert t != b"other"
    assert not (t == b"other")

    # Ordering against plain bytes: Token returns NotImplemented,
    # Python falls back to bytes.__lt__ (content comparison).
    # This is a consequence of subclassing bytes.
    assert not (t < b"hello")  # same content
    assert t <= b"hello"
    assert not (t > b"hello")
    assert t >= b"hello"


# ---------------------------------------------------------------------------
# Trie: weight_max with duplicates (only weight_sum was tested)
# ---------------------------------------------------------------------------


def test_trie_weight_max_with_duplicates():
    """weight_max must propagate the correct max through shared trie paths."""
    vocab = [
        Token(0, b"ab"),
        Token(1, b"ab"),  # duplicate byte_string
        Token(2, b"ac"),
        Token(3, b"b"),
    ]
    trie = TokenCharacterTrie(decode=vocab)

    # Token 1 has the highest weight among the "ab" pair
    weights = torch.tensor([0.1, 0.9, 0.3, 0.2])
    node_ws = trie.weight_max(weights)

    # The "a" internal node should have max(0.1, 0.9, 0.3) = 0.9
    # (it's the ancestor of both "ab" tokens and the "ac" token)
    # Root should have max(0.9, 0.2) = 0.9
    assert np.isclose(node_ws[trie.root], 0.9, rtol=1e-5)

    # Individual leaves
    leaf_0 = trie.idx_to_leaf[0][1]
    leaf_1 = trie.idx_to_leaf[1][1]
    assert np.isclose(node_ws[leaf_0], 0.1, rtol=1e-5)
    assert np.isclose(node_ws[leaf_1], 0.9, rtol=1e-5)


def test_trie_internal_node_sums_both_duplicates():
    """An internal node shared by two duplicate tokens must sum both weights."""
    vocab = [
        Token(0, b"ab"),
        Token(1, b"ab"),  # shares the a→b path
        Token(2, b"c"),
    ]
    trie = TokenCharacterTrie(decode=vocab)

    weights = torch.tensor([0.3, 0.5, 0.2])
    node_ws = trie.weight_sum(weights)

    # Root = 0.3 + 0.5 + 0.2 = 1.0
    assert np.isclose(node_ws[trie.root], 1.0, rtol=1e-5)

    # The "a" internal node = 0.3 + 0.5 = 0.8
    # (both "ab" tokens descend from it)
    # Find "a" node: it's the parent of the "b" node
    leaf_0 = trie.idx_to_leaf[0][1]
    leaf_1 = trie.idx_to_leaf[1][1]
    # Both leaf weights should be independent
    assert np.isclose(node_ws[leaf_0], 0.3, rtol=1e-5)
    assert np.isclose(node_ws[leaf_1], 0.5, rtol=1e-5)
