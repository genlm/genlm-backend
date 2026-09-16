"""The row contract: float32, normalized, exactly ``len(str_vocab)`` columns."""

import pytest
import torch

from genlm.backend.llm import MockAsyncLM


@pytest.fixture(scope="module")
def llm():
    return MockAsyncLM.from_name("gpt2")


def test_wider_logits_are_trimmed_then_normalized(llm):
    vocab = len(llm.str_vocab)
    logits = torch.randn(3, vocab + 100, dtype=torch.float16)
    logits[:, vocab:] = 50.0  # padding columns carry mass that a row must not keep
    rows = llm._normalize(logits)
    assert rows.shape == (3, vocab) and rows.dtype == torch.float32
    assert torch.allclose(rows.exp().sum(-1), torch.ones(3), atol=1e-5)
    want = torch.log_softmax(logits[:, :vocab].float(), -1)
    assert torch.allclose(rows, want, atol=1e-5)


def test_narrower_logits_are_padded_with_neg_inf(llm):
    vocab = len(llm.str_vocab)
    rows = llm._normalize(torch.randn(vocab - 2))
    assert rows.shape == (vocab,)
    assert rows[-2:].isneginf().all()
    assert torch.allclose(rows.exp().sum(), torch.tensor(1.0), atol=1e-5)


def test_mock_rows_follow_the_contract(llm):
    row = llm.next_token_logprobs_sync([1, 2, 3])
    assert row.shape == (len(llm.str_vocab),) and row.dtype == torch.float32
    assert row.device == llm.device
