"""Sub-batching invariant for _batch_evaluate's one-prefill-step logprob capture.

GlobalLogprobsCapture captures a single prefill step, so each generate() sub-batch
must stay <= max_num_batched_tokens AND <= max_num_seqs (else rows are dropped ->
"Expected N logprobs, got M"). _sub_batches must enforce that while preserving all
prompts in order. Pure logic; no GPU.
"""
import pytest

pytest.importorskip("vllm")
from genlm.backend.llm.vllm import AsyncVirtualLM  # noqa: E402

sub_batches = AsyncVirtualLM._sub_batches


def _flatten(batches):
    return [t for b in batches for t in b]


def test_respects_max_seqs():
    prompts = [(1, 2)] * 8
    batches = list(sub_batches(prompts, max_tokens=10_000, max_seqs=3))
    assert all(len(b) <= 3 for b in batches)
    assert _flatten(batches) == prompts  # nothing lost, order preserved


def test_respects_max_tokens():
    prompts = [tuple(range(4))] * 5  # 4 tokens each
    batches = list(sub_batches(prompts, max_tokens=10, max_seqs=1000))
    assert all(sum(len(t) for t in b) <= 10 for b in batches)
    assert _flatten(batches) == prompts


def test_oversized_prompt_is_alone():
    big = tuple(range(50))
    prompts = [(1,), big, (2,)]
    batches = list(sub_batches(prompts, max_tokens=10, max_seqs=1000))
    assert [big] in batches  # the oversized prompt gets its own sub-batch
    assert _flatten(batches) == prompts


def test_single_batch_when_under_limits():
    prompts = [(1, 2), (3, 4)]
    assert list(sub_batches(prompts, max_tokens=10_000, max_seqs=128)) == [prompts]
