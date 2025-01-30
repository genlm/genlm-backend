"""Functions to get the byte vocabulary from a HuggingFace tokenizer"""

import re
from transformers import AutoTokenizer


class ByteDecoderError(Exception):
    pass


class ByteVocabError(Exception):
    pass


def get_byte_vocab(tokenizer):
    """Extract byte vocabulary from a tokenizer using various methods.

    This function attempts to extract the byte representation of each token in the vocabulary
    using multiple methods, trying each in sequence until one succeeds:

    1. If the tokenizer has a byte_decoder attribute, attempt to use that directly
    2. If the tokenizer has an sp_model (SentencePiece) attribute, use that
    3. Try encoding the token strings directly
    4. Fall back to using the default GPT2 byte decoder

    Args:
        tokenizer: A Hugging Face tokenizer instance.

    Returns:
        (list[byte]): List of byte representations of tokens.

    Raises:
        ByteVocabError: If vocabulary cannot be decoded using any of the available methods.
    """
    # Try byte decoder.
    if hasattr(tokenizer, "byte_decoder"):
        try:
            byte_decoder = tokenizer.byte_decoder
            check_byte_decoder(tokenizer, byte_decoder)
            return get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder)
        except ByteDecoderError as e:
            pass
            # warnings.warn(f"Could not decode vocabulary using byte_decoder: {e!r}")

    # Try SentencePiece model.
    if hasattr(tokenizer, "sp_model"):
        return get_byte_tokens_from_sp(tokenizer)

    # Try through token encoding.
    try:
        return get_byte_tokens_by_encoding_token_strings(tokenizer)
    except Exception as e:
        # warnings.warn(f"Could not decode vocabulary through string encoding: {e!r}")
        pass

    # Try using GPT2 byte decoder.
    try:
        byte_decoder = _get_default_byte_decoder()
        check_byte_decoder(tokenizer, byte_decoder)
        return get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder)
    except ByteDecoderError as e:
        raise ByteVocabError(
            "Could not decode vocabulary by falling back to GPT2 byte decoder."
        ) from e


def get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder):
    """Convert tokens to bytes using a byte decoder mapping.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder (dict): Dictionary mapping characters to bytes

    Returns:
        byte_tokens (list[byte]): List of byte representations for each token
    """
    byte_tokens = [
        bytes([byte_decoder[b] for b in tokenizer.convert_ids_to_tokens(i)])
        for i in range(len(tokenizer))
    ]
    return byte_tokens


def get_byte_tokens_by_encoding_token_strings(tokenizer):
    """Convert tokens to bytes by encoding token strings directly.

    This function attempts to convert each token in the vocabulary to its byte representation
    by directly encoding the token strings. It handles special tokens separately and has
    multiple fallback strategies for encoding regular tokens:

    1. For special tokens, uses the string representation from the tokenizer's added vocab
    2. For regular tokens:
        a. If the token is already bytes, uses it directly
        b. If the token is a string and the tokenizer has convert_tokens_to_string:
            - Converts single token to string
            - Verifies roundtrip encoding matches original token ID
            - Falls back to byte decoder if roundtrip fails
        c. If the token is a string without convert_tokens_to_string:
            - Directly encodes the token string

    Args:
        tokenizer: A Hugging Face tokenizer instance.

    Returns:
        byte_tokens (list[byte]): List of byte representations for each token in the vocabulary.

    Raises:
        ValueError: If token encoding fails (roundtrip produces multiple tokens), or if
                   a token has an unexpected type (not str or bytes).
    """
    byte_tokens = [b""] * len(tokenizer)
    special_tokens_map = {
        id: token for token, id in tokenizer.get_added_vocab().items()
    }
    byte_encoder = _bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    for i in range(len(tokenizer)):
        if i in special_tokens_map:
            byte_coded = special_tokens_map[i].encode()
        else:
            token = tokenizer.convert_ids_to_tokens(i)
            if isinstance(token, bytes):
                byte_coded = token
            elif isinstance(token, str):
                if hasattr(tokenizer, "convert_tokens_to_string"):
                    token_str = tokenizer.convert_tokens_to_string([token])
                    encoded_str = tokenizer.encode(token_str)
                    if len(encoded_str) != 1:
                        raise ValueError(
                            f"Round-trip encoding of tokens [{token}] failed! Got {encoded_str}"
                        )
                    roundtrip_id = encoded_str[0]
                    if roundtrip_id == i:
                        byte_coded = token_str.encode()
                    else:
                        byte_coded = bytes([byte_decoder[c] for c in token])
                else:
                    byte_coded = token.encode()
            else:
                raise ValueError(f"Unexpected token type: {type(token)}")
        byte_tokens[i] = byte_coded

    return byte_tokens


def get_byte_tokens_from_sp(tokenizer):
    """Convert tokens to their byte representations using a SentencePiece model.

    Uses the SentencePiece model's id_to_piece method to get the raw byte representation
    of each token, handling special tokens separately. Converts any hex-encoded bytes
    (in <0xXX> format) to their actual byte values and replaces the SentencePiece
    prefix space marker with a regular space.

    Args:
        tokenizer: A Hugging Face tokenizer instance with a SentencePiece model

    Returns:
        byte_tokens (list[byte]): List of byte representations for each token in the vocabulary

    Note:
        Special tokens are handled by directly encoding their string representation,
        while normal tokens go through the SentencePiece conversion process.
    """
    special_tokens_map = {
        token_id: token for token, token_id in tokenizer.get_added_vocab().items()
    }
    byte_tokens = [b""] * len(tokenizer)
    prefix_space = "▁".encode()
    for i in range(len(tokenizer)):
        if i in special_tokens_map:
            byte_coded = special_tokens_map[i].encode()
        else:
            byte_coded = re.sub(
                rb"<0x(..)>",
                lambda x: bytes.fromhex(x[1].decode()),
                tokenizer.sp_model.id_to_piece(i).encode(),
            )
        byte_tokens[i] = byte_coded.replace(prefix_space, b" ")
    return byte_tokens


def check_byte_decoder(tokenizer, byte_decoder):
    """Verify that a byte decoder can properly handle all tokens.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder (dict): Dictionary mapping characters to bytes

    Raises:
        ByteDecoderError: If byte decoder fails validation checks
    """
    _check_byte_decoder_has_all_bytes(tokenizer, byte_decoder)
    _check_complex_roundtrip(tokenizer, byte_decoder)


def _check_byte_decoder_has_all_bytes(tokenizer, byte_decoder):
    """Verify byte decoder contains mappings for all bytes in vocabulary.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder (dict): Dictionary mapping characters to bytes

    Raises:
        ByteDecoderError: If byte decoder is missing required bytes
    """
    all_bytes = set()
    for x in tokenizer.get_vocab().keys():
        for y in x:
            all_bytes.add(y)
    if not set(byte_decoder.keys()) >= all_bytes:
        raise ByteDecoderError(
            f"Byte decoder is missing bytes: {all_bytes - set(byte_decoder.keys())}"
        )


def _check_complex_roundtrip(tokenizer, byte_decoder):
    """Test byte decoder by round-trip encoding/decoding complex characters.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder (dict): Dictionary mapping characters to bytes

    Raises:
        ByteDecoderError: If round-trip conversion fails
    """
    s = "’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨"
    reconstructed = b""
    try:
        input_ids = tokenizer(s)["input_ids"]
        for i in input_ids:
            nxt_bytes = []
            token_str = tokenizer.convert_ids_to_tokens(i)
            for c in token_str:
                nxt_bytes.append(byte_decoder[c])
            reconstructed += bytes(nxt_bytes)

        if (
            hasattr(tokenizer, "bos_token")
            and tokenizer.bos_token
            and reconstructed.startswith(tokenizer.bos_token.encode())
        ):
            reconstructed = reconstructed[len(tokenizer.bos_token) :]
    except Exception as e:
        raise ByteDecoderError(
            f"The tokenizer being used is unable to convert a special character in {s}."
        ) from e

    if reconstructed.decode() != s:
        raise ByteDecoderError(
            f"Failed to reconstruct the string {s} from the tokenizer's byte_decoder: {reconstructed.decode()!r} != {s!r}"
        )


def _bytes_to_unicode():
    """Create a mapping from bytes to Unicode characters.

    Returns:
        (dict): Mapping from byte values to Unicode characters
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _get_default_byte_decoder():
    """Get the default GPT-2 byte decoder with additional special character mappings.

    Returns:
        (dict): Mapping from characters to bytes including special characters
    """
    byte_decoder = AutoTokenizer.from_pretrained("gpt2", use_fast=False).byte_decoder
    byte_decoder.update(
        {
            " ": 32,
            "\n": 10,
            "\r": 13,
            "\t": 9,
            "▁": 32,
        }
    )
    return byte_decoder
