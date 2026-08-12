"""Functions to get the byte vocabulary from a HuggingFace tokenizer"""

import json
import re


class ByteDecoderError(Exception):
    pass


class ByteVocabError(Exception):
    pass


def get_byte_vocab(tokenizer):
    """Byte representation of every token, indexed by token id.

    The tokenizer's own decoder names the scheme: ``ByteLevel`` is the GPT-2 byte
    alphabet, ``ByteFallback``/``Metaspace`` is SentencePiece's ``<0xXX>`` escapes and
    ``▁`` space marker. A tokenizer declaring neither is rejected.

    Args:
        tokenizer: A Hugging Face tokenizer instance.

    Returns:
        (list[bytes]): Byte representation of each token.

    Raises:
        ByteVocabError: If the tokenizer's byte encoding cannot be determined, or the
            scheme it declares fails to round-trip.
    """
    kinds = _decoder_kinds(tokenizer)

    if kinds & {"ByteFallback", "Metaspace"}:
        return get_byte_tokens_from_pieces(tokenizer)

    if "ByteLevel" in kinds:
        byte_decoder = _get_default_byte_decoder()
        try:
            check_byte_decoder(tokenizer, byte_decoder)
        except ByteDecoderError as e:
            raise ByteVocabError(
                "Tokenizer declares a ByteLevel decoder but does not round-trip under "
                "the GPT-2 byte alphabet."
            ) from e
        return get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder)

    if kinds:
        raise ByteVocabError(
            f"Cannot determine the byte encoding of {tokenizer.name_or_path!r}: its "
            f"decoder is {sorted(kinds)}, which is neither ByteLevel nor SentencePiece."
        )

    # A tokenizer with no backend declares no scheme; `sp_model` marks SentencePiece,
    # whose pieces are already `<0xXX>`-escaped.
    if hasattr(tokenizer, "sp_model"):
        return get_byte_tokens_from_pieces(tokenizer)

    byte_decoder = (
        getattr(tokenizer, "byte_decoder", None) or _get_default_byte_decoder()
    )
    try:
        check_byte_decoder(tokenizer, byte_decoder)
    except ByteDecoderError as e:
        raise ByteVocabError(
            f"Tokenizer {tokenizer.name_or_path!r} exposes no decoder to inspect and "
            f"does not round-trip under the GPT-2 byte alphabet."
        ) from e
    return get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder)


def _decoder_kinds(tokenizer):
    """Decoder component names the tokenizer's backend declares.

    Args:
        tokenizer: A Hugging Face tokenizer instance

    Returns:
        (set): Component type names, e.g. ``{"Sequence", "Replace", "ByteFallback",
            "Fuse"}``. Empty when the tokenizer has no backend.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return set()
    decoder = json.loads(backend.to_str()).get("decoder")
    kinds = set()

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("type"), str):
                kinds.add(node["type"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(decoder)
    return kinds


def get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder):
    """Convert tokens to bytes using a byte decoder mapping.

    Special tokens are handled by directly encoding their string representation.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder (dict): Dictionary mapping characters to bytes

    Returns:
        byte_tokens (list[byte]): List of byte representations for each token
    """
    special_tokens_map = {v: k for k, v in tokenizer.get_added_vocab().items()}
    byte_tokens = [
        bytes([byte_decoder[b] for b in tokenizer.convert_ids_to_tokens(i)])
        if i not in special_tokens_map
        else special_tokens_map[i].encode()
        for i in range(len(tokenizer))
    ]
    return byte_tokens


def get_byte_tokens_from_pieces(tokenizer):
    """Convert SentencePiece token strings to bytes.

    ``<0xXX>`` escapes become the byte they name and ``▁`` becomes a space; special
    tokens are encoded directly.

    Args:
        tokenizer: A Hugging Face tokenizer instance

    Returns:
        byte_tokens (list[bytes]): Byte representation of each token
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
                tokenizer.convert_ids_to_tokens(i).encode(),
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
    """Verify byte decoder contains mappings for all bytes in vocabulary,
    excluding special tokens.

    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder (dict): Dictionary mapping characters to bytes

    Raises:
        ByteDecoderError: If byte decoder is missing required bytes
    """
    special_tokens = tokenizer.get_added_vocab().keys()
    all_bytes = set()
    for x in tokenizer.get_vocab().keys():
        if x in special_tokens:
            continue
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
        input_ids = tokenizer(s, add_special_tokens=False)["input_ids"]
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
    """The canonical GPT-2 byte->unicode map (transformers' ``bytes_to_unicode``)."""
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

    Built from ``_bytes_to_unicode``; transformers 5.x exposes no tokenizer attribute
    to read the same map off.

    Returns:
        (dict): Mapping from characters to bytes including special characters
    """
    byte_decoder = {ch: b for b, ch in _bytes_to_unicode().items()}
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
