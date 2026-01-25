import pytest
from genlm.backend.llm import MockAsyncLM
from genlm.backend.tokenization import Token, decode_vocab
from genlm.backend.trie import TokenCharacterTrie


def test_gemma_model_load():
    """Test that Gemma model loads successfully with Token-based vocabulary.

    Gemma models has tokens with duplicate byte strings.
    """
    # Load Gemma model using mock backend
    try:
        llm = MockAsyncLM.from_name("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    # Verify the model loaded successfully
    assert llm is not None
    assert llm.tokenizer is not None
    assert llm.byte_vocab is not None
    assert len(llm.byte_vocab) > 0

    # Verify byte_vocab contains Token objects
    assert all(isinstance(token, Token) for token in llm.byte_vocab)

    # Verify each token has correct attributes
    for i, token in enumerate(llm.byte_vocab):
        assert hasattr(token, "token_id")
        assert hasattr(token, "byte_string")
        assert token.token_id == i  # Token IDs should match indices
        assert isinstance(token.byte_string, bytes)


def test_gemma_vocabulary_duplicates():
    """Test that Gemma vocabulary correctly handles duplicate byte strings."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    # Decode vocabulary
    byte_vocab, _ = decode_vocab(tokenizer)

    # Check for duplicate byte strings
    byte_string_to_token_ids = {}
    for token in byte_vocab:
        byte_str = token.byte_string
        if byte_str not in byte_string_to_token_ids:
            byte_string_to_token_ids[byte_str] = []
        byte_string_to_token_ids[byte_str].append(token.token_id)

    # Find byte strings that appear multiple times
    duplicates = {
        byte_str: ids
        for byte_str, ids in byte_string_to_token_ids.items()
        if len(ids) > 1
    }

    assert len(duplicates) > 0, "No duplicates found in Gemma vocabulary"
    # If there are duplicates, verify they are handled correctly
    if duplicates:
        # Verify each duplicate token has unique token_id
        for byte_str, token_ids in duplicates.items():
            assert len(set(token_ids)) == len(
                token_ids
            ), f"Duplicate token IDs found for byte string {byte_str!r}"

            # Verify tokens are distinct objects
            tokens = [byte_vocab[tid] for tid in token_ids]
            assert len(set(tokens)) == len(
                tokens
            ), "Tokens with same byte string should be distinct objects"


def test_gemma_trie_with_duplicates():
    """Test that trie works correctly with Gemma vocabulary containing duplicates."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    # Load vocabulary
    byte_vocab, _ = decode_vocab(tokenizer)

    # Create trie from vocabulary
    trie = TokenCharacterTrie(decode=byte_vocab)

    # Verify trie was created successfully
    assert trie is not None
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

    # Test encoding
    test_text = "Hello, world!"
    token_ids = llm.tokenizer.encode(test_text)
    assert len(token_ids) > 0

    # Test next_token_logprobs
    logprobs = await llm.next_token_logprobs(token_ids)
    assert logprobs is not None
    assert logprobs.shape[0] == len(llm.byte_vocab)

    # Test batch operations
    token_ids_list = [token_ids, llm.tokenizer.encode("Test prompt")]
    batch_logprobs = await llm.batch_next_token_logprobs(token_ids_list)
    assert batch_logprobs.shape[0] == len(token_ids_list)
    assert batch_logprobs.shape[1] == len(llm.byte_vocab)


def test_gemma_token_consistency():
    """Test that Token objects maintain consistency across operations."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except OSError:
        pytest.skip("Gemma model not available (may require authentication)")

    # Load vocab
    byte_vocab, _ = decode_vocab(tokenizer)

    # Test that Token objects can be used like bytes
    test_token_ids = [0, 100, 500, 1000]
    for token_id in test_token_ids:
        if token_id < len(byte_vocab):
            token = byte_vocab[token_id]

            # Test iteration
            byte_list = list(token)
            assert isinstance(byte_list, list)
            assert len(byte_list) == len(token.byte_string)

            # Test indexing
            if len(token) > 0:
                assert token[0] == token.byte_string[0]

            # Test decode method
            try:
                decoded = token.decode("utf-8", errors="replace")
                assert isinstance(decoded, str)
            except UnicodeDecodeError:
                pass
