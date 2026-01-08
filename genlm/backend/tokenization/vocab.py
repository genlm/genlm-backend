"""Functions to get and check HuggingFace tokenizer vocabularies"""

from transformers import AutoTokenizer

from genlm.backend.tokenization.bytes import ByteVocabError, get_byte_vocab
from genlm.backend.tokenization.token import Token


def decode_vocab(tokenizer, byte2str_fallback="tokenizer"):
    """Convert tokenizer vocabulary into byte and string representations.

    Warning:
        The byte representation is the canonical form. Each element in byte_vocab is a Token object that 
        contains both the token_id and byte_string. The string representation is provided for convenience 
        but may not decode properly for all tokens, especially those containing invalid UTF-8 sequences.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte2str_fallback (str): Strategy for converting invalid UTF-8 bytes to strings. Options:\n
            - 'tokenizer': Use tokenizer's `convert_ids_to_tokens` (default)
            - 'latin1': Decode using latin1 encoding
            - 'replace': Use Unicode replacement character '�'

    Returns:
        (tuple): (byte_vocab, str_vocab) where byte_vocab is a list of Token objects
            and str_vocab is a list of strings
    """
    if byte2str_fallback not in ["latin1", "tokenizer", "replace"]:
        raise ValueError(f"Unknown byte2str_fallback strategy: {byte2str_fallback}")

    if tokenizer.is_fast:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer.name_or_path, use_fast=False
        )

    # Try slow tokenizer.
    try:
        raw_byte_vocab = get_byte_vocab(tokenizer)
    except ByteVocabError:
        # warnings.warn("Could not decode vocabulary from slow tokenizer. Trying using fast tokenizer.")

        # Try fast tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, use_fast=True)
        try:
            raw_byte_vocab = get_byte_vocab(tokenizer)
        except ByteVocabError as e:
            raise ValueError(
                f"Could not decode byte representation of token vocabuary from tokenizer {tokenizer.name_or_path}"
            ) from e

    # Create Token objects for byte_vocab
    byte_vocab = [
        Token(token_id=i, byte_string=raw_byte_vocab[i])
        for i in range(len(raw_byte_vocab))
    ]
    str_vocab = bytes_to_strs(tokenizer, raw_byte_vocab, byte2str_fallback)

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

    return str_vocab
