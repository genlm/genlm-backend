import pytest
from functools import wraps
from transformers import AutoTokenizer

from genlm_backend.tokenization import decode_vocab
from genlm_backend.tokenization.vocab import assert_roundtrip_bytes
from hypothesis import given, strategies as st, settings


def skip_if_gated(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except OSError as e:
            pytest.skip(f"Skipping due to gated model access: {e}")

    return wrapper


tokenizer_cache = {}


def load_tokenizer(name, use_fast):
    if (name, use_fast) in tokenizer_cache:
        return tokenizer_cache[(name, use_fast)]
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    tokenizer_cache[(name, use_fast)] = tokenizer
    return tokenizer


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_gpt2(text, is_fast):
    tokenizer = load_tokenizer("gpt2", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_llama3(text, is_fast):
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3-8B", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_codellama(text, is_fast):
    tokenizer = load_tokenizer("codellama/CodeLlama-7b-Instruct-hf", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_gemma(text, is_fast):
    tokenizer = load_tokenizer("google/gemma-7b", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_phi(text, is_fast):
    tokenizer = load_tokenizer("microsoft/phi-2", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_mistral(text, is_fast):
    tokenizer = load_tokenizer("mistralai/Mistral-7B-Instruct-v0.3", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)


@skip_if_gated
@settings(deadline=None)
@given(text=st.text(min_size=1, max_size=500), is_fast=st.booleans())
def test_deepseek_r1_unsloth(text, is_fast):
    tokenizer = load_tokenizer("unsloth/DeepSeek-R1-Distill-Llama-8B", is_fast)
    byte_vocab, _ = decode_vocab(tokenizer)
    assert_roundtrip_bytes(text, tokenizer, byte_vocab)
