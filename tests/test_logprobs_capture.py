"""Unit tests for ``GlobalLogprobsCapture`` accumulation semantics.

Regression coverage for the ``Expected N logprobs, got 1`` crash: when the
vLLM scheduler splits a submitted batch across multiple decode steps (waves),
``apply`` fires once per wave and the capture must *accumulate* rather than
overwrite. See ``docs/vllm_v1_logprobs_architecture.md`` §6/§10.

``GlobalLogprobsCapture`` subclasses vLLM's ``LogitsProcessor`` and is only
defined when vllm is importable, so these tests skip on environments without
vllm (e.g. CPU-only dev boxes).
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from genlm.backend.llm.vllm import GlobalLogprobsCapture  # noqa: E402


def _logsoftmax(t):
    return torch.log_softmax(t, dim=-1, dtype=torch.float32)


def test_accumulates_across_waves():
    """Two apply() calls (a split batch) -> rows concatenated in arrival order."""
    cap = GlobalLogprobsCapture()
    cap.clear()
    wave1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # prompts 0,1
    wave2 = torch.tensor([[0.0, 0.0, 1.0]])                    # prompt 2
    cap.apply(wave1)
    cap.apply(wave2)
    out = cap.get_all_logprobs()
    assert out.shape == (3, 3)
    expected = _logsoftmax(torch.cat([wave1, wave2], dim=0))
    assert torch.allclose(out, expected)


def test_single_wave_unchanged():
    """The common single-decode-step case still returns the one block."""
    cap = GlobalLogprobsCapture()
    cap.clear()
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    cap.apply(logits)
    out = cap.get_all_logprobs()
    assert out.shape == (2, 2)
    assert torch.allclose(out, _logsoftmax(logits))


def test_get_logprobs_indexes_across_waves():
    """get_logprobs(batch_index) indexes into the concatenated rows."""
    cap = GlobalLogprobsCapture()
    cap.clear()
    cap.apply(torch.tensor([[1.0, 0.0]]))   # row 0
    cap.apply(torch.tensor([[0.0, 1.0]]))   # row 1
    row1 = cap.get_logprobs(batch_index=1)
    assert torch.allclose(row1, _logsoftmax(torch.tensor([[0.0, 1.0]]))[0])


def test_clear_resets_accumulation():
    cap = GlobalLogprobsCapture()
    cap.apply(torch.tensor([[1.0, 0.0]]))
    cap.clear()
    assert cap.get_all_logprobs() is None
    assert cap.get_logprobs(0) is None


def test_out_of_range_index_returns_none():
    cap = GlobalLogprobsCapture()
    cap.clear()
    cap.apply(torch.tensor([[1.0, 0.0]]))
    assert cap.get_logprobs(batch_index=5) is None
