"""Functions to get and check HuggingFace tokenizer vocabularies"""

import warnings
from contextlib import contextmanager
from transformers import AutoTokenizer

from genlm_backend.tokenization.bytes import ByteVocabError, get_byte_vocab


def decode_vocab(tokenizer, byte2str_fallback="tokenizer"):
    """Convert tokenizer vocabulary into byte and string representations.

    Warning:
        The byte representation is the canonical form. The string representation is provided for
        convenience but may not decode properly for all tokens, especially those containing invalid UTF-8 sequences.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte2str_fallback (str): Strategy for converting invalid UTF-8 bytes to strings. Options:\n
            - 'tokenizer': Use tokenizer's `convert_ids_to_tokens` (default)
            - 'latin1': Decode using latin1 encoding
            - 'replace': Use Unicode replacement character '�'

    Returns:
        (tuple): (byte_vocab, str_vocab)
    """
    if byte2str_fallback not in ["latin1", "tokenizer", "replace"]:
        raise ValueError(f"Unknown byte2str_fallback strategy: {byte2str_fallback}")

    if tokenizer.is_fast:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer.name_or_path, use_fast=False
        )

    # Try slow tokenizer.
    try:
        byte_vocab = get_byte_vocab(tokenizer)
    except ByteVocabError:
        # warnings.warn("Could not decode vocabulary from slow tokenizer. Trying using fast tokenizer.")

        # Try fast tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, use_fast=True)
        try:
            byte_vocab = get_byte_vocab(tokenizer)
        except ByteVocabError as e:
            raise ValueError(
                f"Could not decode byte representation of token vocabuary from tokenizer {tokenizer.name_or_path}"
            ) from e

    str_vocab = bytes_to_strs(tokenizer, byte_vocab, byte2str_fallback)

    return byte_vocab, str_vocab


def bytes_to_strs(tokenizer, byte_vocab, byte2str_fallback):
    """Convert byte representations to UTF-8 strings.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_vocab (list[bytes]): List of byte representations of tokens
        byte2str_fallback (str): Strategy for converting invalid UTF-8 bytes to strings:
            - 'tokenizer': Use tokenizer's convert_ids_to_tokens (default)
            - 'latin1': Decode using latin1 encoding
            - 'replace': Use Unicode replacement character '�'

    Returns:
        (list[str]): List of string representations of tokens

    Note:
        May produce duplicate strings for different token IDs. A warning is issued if duplicates are found.
    """
    str_vocab = []
    seen_tokens = {}
    for token_id, raw_token in enumerate(byte_vocab):
        try:
            token = raw_token.decode("utf-8")
        except UnicodeDecodeError:
            if byte2str_fallback == "latin1":
                try:
                    token = raw_token.decode("latin1")
                except UnicodeDecodeError:
                    token = tokenizer.convert_ids_to_tokens(token_id)
            elif byte2str_fallback == "tokenizer":
                token = tokenizer.convert_ids_to_tokens(token_id)
            elif byte2str_fallback == "replace":
                token = raw_token.decode("utf-8", errors="replace")

        if token in seen_tokens:
            seen_tokens[token].append(token_id)
        else:
            seen_tokens[token] = [token_id]

        str_vocab.append(token)

    duplicates = {
        token: indices for token, indices in seen_tokens.items() if len(indices) > 1
    }
    if duplicates:
        warnings.warn(
            "Duplicate tokens found in string vocabulary. "
            "This may lead to downstream issues with the string vocabulary; we recommend using the byte vocabulary."
        )

    return str_vocab


def assert_roundtrip_bytes(test_case, tokenizer, byte_vocab):
    """Assert that encoding and decoding a test case using byte vocabulary matches the tokenizer's output.

    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        byte_vocab (list): List of byte representations of tokens

    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
    """
    return assert_roundtrip(test_case, tokenizer, byte_vocab, vocab_type="byte")


def assert_roundtrip_strs(test_case, tokenizer, str_vocab):
    """Assert that encoding and decoding a test case using string vocabulary matches the tokenizer's output.

    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        str_vocab (list): List of string representations of tokens

    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
    """
    return assert_roundtrip(test_case, tokenizer, str_vocab, vocab_type="str")


def assert_roundtrip(test_case, tokenizer, vocab, vocab_type):
    """Assert that encoding and decoding a test case matches the tokenizer's output.

    A unified function that handles both string and byte vocabularies.

    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        vocab (list): List of token representations (either strings or bytes)
        vocab_type (str): Type of vocabulary - either 'str' or 'byte'

    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
        ValueError: If vocab_type is not 'str' or 'byte'
    """
    with turn_off_space_cleaning(tokenizer):
        encd = tokenizer.encode(test_case)

        if vocab_type == "str":
            have = "".join([vocab[i] for i in encd])
        elif vocab_type == "byte":
            have = b"".join([vocab[i] for i in encd]).decode("utf-8")
        else:
            raise ValueError(
                f"Invalid vocab_type: {vocab_type}. Must be 'str' or 'byte'."
            )

        want = tokenizer.decode(encd)

        if have != want:
            pos = next(
                (i for i in range(min(len(have), len(want))) if have[i] != want[i]),
                min(len(have), len(want)),
            )
            context = 20

            error_msg = (
                f"\nRoundtrip assertion failed for {vocab_type} vocabulary:"
                f"\nMismatch at position {pos}"
                f"\nHave: ...{repr(have[max(0, pos - context) : pos + context])}..."
                f"\nWant: ...{repr(want[max(0, pos - context) : pos + context])}..."
            )

            raise AssertionError(error_msg)


@contextmanager
def turn_off_space_cleaning(tokenizer):
    original_value = tokenizer.clean_up_tokenization_spaces
    try:
        tokenizer.clean_up_tokenization_spaces = False
        yield
    finally:
        tokenizer.clean_up_tokenization_spaces = original_value
