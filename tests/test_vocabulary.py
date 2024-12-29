import pytest
import warnings
from functools import wraps
from datasets import load_dataset
from transformers import AutoTokenizer

from genlm_backend.vocabulary import decode_vocab, assert_roundtrip_bytes

def skip_if_gated(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except OSError as e:
            pytest.skip(f"Skipping due to gated model access: {e}")
    return wrapper

@pytest.fixture
def test_text():
    text = '\n'.join(load_dataset("wikitext", "wikitext-2-raw-v1")['test']['text'])
    return text

@skip_if_gated
def test_gpt2(test_text):
    # Uses byte decoder
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

@skip_if_gated
def test_llama3(test_text):
    # Uses GPT2 byte decoder
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=False)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

@skip_if_gated
def test_codellama(test_text):
    # Uses SentencePiece method
    tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf', use_fast=True)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

    tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf', use_fast=False)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

@skip_if_gated
def test_gemma(test_text):
    # Uses SentencePiece method
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b', use_fast=True)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

    tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b', use_fast=False)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)
    
@skip_if_gated
def _test_phi(test_text): # Currently fails.
    # Has a byte decoder, but it is missing bytes. Uses GPT2 byte decoder.
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', use_fast=True)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', use_fast=False)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

@skip_if_gated
def test_mistral(test_text):
    # Uses SentencePiece method
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', use_fast=True)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)

    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', use_fast=False)
    (byte_vocab, str_vocab) = decode_vocab(tokenizer)
    assert_roundtrip_bytes(test_text, tokenizer, byte_vocab)