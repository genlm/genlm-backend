import re
import warnings
from contextlib import contextmanager
from transformers import AutoTokenizer

# Byte decoding functionality taken from https://github.com/guidance-ai/guidance/blob/567528a5437a13783aab4681c4b6a99d1612948d/guidance/models/transformers/_transformers.py

class ByteDecoderError(Exception):
    pass

class ByteVocabError(Exception):
    pass

def decode_vocab(tokenizer, byte2str_fallback='tokenizer'):
    """Convert tokenizer vocabulary into byte and string representations.
    
    The byte representation is the canonical form. The string representation is provided for 
    convenience but may not decode properly for all tokens, especially those containing invalid UTF-8 sequences.
    
    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte2str_fallback: Strategy for converting invalid UTF-8 bytes to strings. Options:
            - 'tokenizer': Use tokenizer's convert_ids_to_tokens (default)
            - 'latin1': Decode using latin1 encoding
            - 'replace': Use Unicode replacement character '�'
    
    Returns:
        tuple: (byte_vocab, str_vocab)
    """
    if byte2str_fallback not in ['latin1', 'tokenizer', 'replace']:
        raise ValueError(f"Unknown byte2str_fallback strategy: {byte2str_fallback}")

    if tokenizer.is_fast:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, use_fast=False)

    # Try slow tokenizer.
    try:
        byte_vocab = get_byte_vocab(tokenizer)
    except ByteVocabError:
        warnings.warn("Could not decode vocabulary from slow tokenizer. Trying using fast tokenizer.")
        
        # Try fast tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, use_fast=True)
        try:
            byte_vocab = get_byte_vocab(tokenizer)
        except ByteVocabError as e:
            raise ValueError(f"Could not decode byte representation of token vocabuary from tokenizer {tokenizer.name_or_path}") from e
            
    str_vocab = bytes_to_strs(tokenizer, byte_vocab, byte2str_fallback)

    return byte_vocab, str_vocab

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
        list: List of byte representations of tokens.
        
    Raises:
        ByteVocabError: If vocabulary cannot be decoded using any of the available methods.
    """
    # Try byte decoder.
    if hasattr(tokenizer, 'byte_decoder'):
        try:
            byte_decoder = tokenizer.byte_decoder 
            check_byte_decoder(tokenizer, byte_decoder)
            return get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder)
        except ByteDecoderError as e:
            warnings.warn(f"Could not decode vocabulary using byte_decoder: {e!r}")

    # Try SentencePiece model.
    if hasattr(tokenizer, 'sp_model'):
        return get_byte_tokens_from_sp(tokenizer)

    # Try through token encoding.
    try:
        return get_byte_tokens_by_encoding_token_strings(tokenizer)
    except Exception as e:
        warnings.warn(f"Could not decode vocabulary through string encoding: {e!r}")

    # Try using GPT2 byte decoder.
    try:
        byte_decoder = _get_default_byte_decoder()
        check_byte_decoder(tokenizer, byte_decoder)
        return get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder)
    except ByteDecoderError as e:
        raise ByteVocabError('Could not decode vocabulary by falling back to GPT2 byte decoder.') from e

def get_byte_tokens_from_byte_decoder(tokenizer, byte_decoder):
    """Convert tokens to bytes using a byte decoder mapping.
    
    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder: Dictionary mapping characters to bytes
        
    Returns:
        list: List of byte representations for each token
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
        list: List of byte representations for each token in the vocabulary.
        
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
                        raise ValueError(f"Round-trip encoding of tokens [{token}] failed! Got {encoded_str}")
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
        list: List of byte representations for each token in the vocabulary
        
    Note:
        Special tokens are handled by directly encoding their string representation,
        while normal tokens go through the SentencePiece conversion process.
    """
    special_tokens_map = {
        token_id: token 
        for token, token_id in tokenizer.get_added_vocab().items()
    }
    byte_tokens = [b''] * len(tokenizer)
    prefix_space = '▁'.encode()
    for i in range(len(tokenizer)):
        if i in special_tokens_map:
            byte_coded = special_tokens_map[i].encode()
        else:
            byte_coded = re.sub(
                rb'<0x(..)>',
                lambda x: bytes.fromhex(x[1].decode()),
                tokenizer.sp_model.id_to_piece(i).encode(),
            )
        byte_tokens[i] = byte_coded.replace(prefix_space, b' ')
    return byte_tokens

def check_byte_decoder(tokenizer, byte_decoder):
    """Verify that a byte decoder can properly handle all tokens.
    
    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder: Dictionary mapping characters to bytes
        
    Raises:
        ByteDecoderError: If byte decoder fails validation checks
    """
    _check_byte_decoder_has_all_bytes(tokenizer, byte_decoder)
    _check_complex_roundtrip(tokenizer, byte_decoder)

def _check_byte_decoder_has_all_bytes(tokenizer, byte_decoder):
    """Verify byte decoder contains mappings for all bytes in vocabulary.
    
    Args:
        tokenizer: A Hugging Face tokenizer instance
        byte_decoder: Dictionary mapping characters to bytes
        
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
        byte_decoder: Dictionary mapping characters to bytes
        
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

        if hasattr(tokenizer, "bos_token") and tokenizer.bos_token and reconstructed.startswith(
            tokenizer.bos_token.encode()
        ):
            reconstructed = reconstructed[len(tokenizer.bos_token):]
    except Exception as e:
        raise ByteDecoderError(f'The tokenizer being used is unable to convert a special character in {s}.') from e

    if reconstructed.decode() != s:
        raise ByteDecoderError(
            f"Failed to reconstruct the string {s} from the tokenizer's byte_decoder: {reconstructed.decode()!r} != {s!r}"
        )

def _bytes_to_unicode():
    """Create a mapping from bytes to Unicode characters.
    
    Returns:
        dict: Mapping from byte values to Unicode characters
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
        dict: Mapping from characters to bytes including special characters
    """
    byte_decoder = AutoTokenizer.from_pretrained('gpt2', use_fast=False).byte_decoder
    byte_decoder.update({
        ' ': 32,  
        '\n': 10, 
        '\r': 13,
        '\t': 9, 
        '▁': 32,
    })
    return byte_decoder

def bytes_to_strs(tokenizer, byte_vocab, byte2str_fallback):
    """Convert byte representations to UTF-8 strings."""
    str_vocab = []
    for token_id, raw_token in enumerate(byte_vocab):
        try:
            str_vocab.append(raw_token.decode('utf-8'))
        except UnicodeDecodeError:
            if byte2str_fallback == 'latin1':
                try:
                    fallback = raw_token.decode('latin1')
                except UnicodeDecodeError:
                    fallback = tokenizer.convert_ids_to_tokens(token_id)
            elif byte2str_fallback == 'tokenizer':
                fallback = tokenizer.convert_ids_to_tokens(token_id)
            elif byte2str_fallback == 'replace':
                fallback = raw_token.decode('utf-8', errors='replace')
            str_vocab.append(fallback)
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
    return assert_roundtrip(test_case, tokenizer, byte_vocab, vocab_type='byte')

def assert_roundtrip_strs(test_case, tokenizer, str_vocab):
    """Assert that encoding and decoding a test case using string vocabulary matches the tokenizer's output.
    
    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        str_vocab (list): List of string representations of tokens
    
    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
    """
    return assert_roundtrip(test_case, tokenizer, str_vocab, vocab_type='str')

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

        if vocab_type == 'str':
            have = ''.join([vocab[i] for i in encd])
        elif vocab_type == 'byte':
            have = b''.join([vocab[i] for i in encd]).decode('utf-8')
        else:
            raise ValueError(f"Invalid vocab_type: {vocab_type}. Must be 'str' or 'byte'.")

        want = tokenizer.decode(encd)
        
        if have != want:
            pos = next((i for i in range(min(len(have), len(want))) if have[i] != want[i]), min(len(have), len(want)))
            context = 20 
            
            error_msg = (
                f"\nRoundtrip assertion failed for {vocab_type} vocabulary:"
                f"\nMismatch at position {pos}"
                f"\nHave: ...{repr(have[max(0, pos-context):pos + context])}..."
                f"\nWant: ...{repr(want[max(0, pos-context):pos + context])}..."
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