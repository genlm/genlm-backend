import pytest
from transformers import AutoTokenizer
from genlm.backend.tokenization import decode_vocab
from genlm.backend.tokenization.bytes import ByteDecoderError, check_byte_decoder
from conftest import assert_roundtrip_bytes
from hypothesis import given, strategies as st, settings


MAX_SIZE = 50
MAX_EXAMPLES = 10
tokenizer_cache = {}


def load_tokenizer(name, use_fast):
    if (name, use_fast) in tokenizer_cache:
        return tokenizer_cache[(name, use_fast)]
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    except OSError:
        pytest.skip(f"Skipping due to gated model access: {name}")
    tokenizer_cache[(name, use_fast)] = tokenizer
    return tokenizer


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_gpt2(text, is_fast):
    tokenizer = load_tokenizer("gpt2", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_llama3(text, is_fast):
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3-8B", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_codellama(text, is_fast):
    tokenizer = load_tokenizer("codellama/CodeLlama-7b-Instruct-hf", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_gemma(text, is_fast):
    tokenizer = load_tokenizer("google/gemma-7b", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_phi(text, is_fast):
    tokenizer = load_tokenizer("microsoft/phi-2", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_mistral(text, is_fast):
    tokenizer = load_tokenizer("mistralai/Mistral-7B-Instruct-v0.3", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(text=st.text(min_size=1, max_size=MAX_SIZE), is_fast=st.booleans())
def test_deepseek_r1_unsloth(text, is_fast):
    tokenizer = load_tokenizer("unsloth/DeepSeek-R1-Distill-Llama-8B", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


def test_byte2str_fallbacks():
    tokenizer = load_tokenizer("gpt2", False)

    byte_vocab1, str_vocab1 = decode_vocab(tokenizer, byte2str_fallback="latin1")
    assert all(
        hasattr(token, "token_id") and hasattr(token, "byte_string")
        for token in byte_vocab1
    )
    assert isinstance(str_vocab1, list)

    byte_vocab2, str_vocab2 = decode_vocab(tokenizer, byte2str_fallback="tokenizer")
    assert all(
        hasattr(token, "token_id") and hasattr(token, "byte_string")
        for token in byte_vocab2
    )
    assert isinstance(str_vocab2, list)

    byte_vocab3, str_vocab3 = decode_vocab(tokenizer, byte2str_fallback="replace")
    assert all(
        hasattr(token, "token_id") and hasattr(token, "byte_string")
        for token in byte_vocab3
    )
    assert isinstance(str_vocab3, list)

    with pytest.raises(ValueError):
        decode_vocab(tokenizer, byte2str_fallback="invalid")


def test_byte_decoder_error_handling():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test with invalid byte decoder
    invalid_byte_decoder = {"a": 999}  # Invalid byte value
    with pytest.raises(ByteDecoderError):
        check_byte_decoder(tokenizer, invalid_byte_decoder)

    # Test with missing bytes
    incomplete_byte_decoder = {}
    with pytest.raises(ByteDecoderError):
        check_byte_decoder(tokenizer, incomplete_byte_decoder)


def test_decode_vocab_failure_both_tokenizers():
    """Test that decode_vocab raises ValueError when both slow and fast tokenizers fail."""
    from unittest.mock import patch, MagicMock
    from genlm.backend.tokenization.bytes import ByteVocabError

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = False
    mock_tokenizer.name_or_path = "test-model"

    # Mock AutoTokenizer.from_pretrained to return our mock
    with patch(
        "genlm.backend.tokenization.vocab.AutoTokenizer.from_pretrained"
    ) as mock_from_pretrained:
        mock_from_pretrained.return_value = mock_tokenizer

        # Mock get_byte_vocab to always raise ByteVocabError
        with patch(
            "genlm.backend.tokenization.vocab.get_byte_vocab"
        ) as mock_get_byte_vocab:
            mock_get_byte_vocab.side_effect = ByteVocabError("Cannot decode vocabulary")

            # This should raise ValueError after both slow and fast tokenizers fail
            with pytest.raises(
                ValueError, match="Could not decode byte representation"
            ):
                decode_vocab(mock_tokenizer)
