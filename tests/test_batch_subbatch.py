"""Regression tests for the batched-``next_token_logprobs`` multi-prefill-step bug.

``GlobalLogprobsCapture`` keeps only the most recent prefill step's rows, so a
single ``generate()`` whose prompts exceed ``max_num_batched_tokens`` (or
``max_num_seqs``) is chunked across steps and ``get_all_logprobs()`` returns fewer
rows than prompts -> ``AssertionError: Expected N logprobs, got M``.

``_batch_evaluate`` fixes this by splitting the queued prompts into one-prefill-step
sub-batches via ``_sub_batches``. These tests cover the packing logic directly and
exercise ``_batch_evaluate`` against a fake engine that *models* the chunked-prefill
overwrite, so they fail if the sub-batching is removed. No GPU / model load needed.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from genlm.backend.llm.vllm import HAS_VLLM

pytestmark = pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")

if HAS_VLLM:
    from genlm.backend.llm.vllm import AsyncVirtualLM


# --------------------------------------------------------------------------- #
# _sub_batches packing logic                                                  #
# --------------------------------------------------------------------------- #
def test_sub_batches_packs_by_token_budget():
    # 3 + 2 = 5 fits; adding 1 more (=6) exceeds max_tokens=5 -> new sub-batch.
    subs = list(AsyncVirtualLM._sub_batches([(1, 2, 3), (4, 5), (6,)], 5, 100))
    assert subs == [[(1, 2, 3), (4, 5)], [(6,)]]


def test_sub_batches_respects_max_seqs():
    subs = list(AsyncVirtualLM._sub_batches([(1,), (2,), (3,)], 1000, 2))
    assert subs == [[(1,), (2,)], [(3,)]]


def test_sub_batches_overlong_prompt_is_singleton():
    # A single prompt longer than max_tokens still forms one sub-batch on its own.
    subs = list(AsyncVirtualLM._sub_batches([(1, 2, 3, 4, 5, 6)], 3, 10))
    assert subs == [[(1, 2, 3, 4, 5, 6)]]


def test_sub_batches_small_batch_is_single_call():
    subs = list(AsyncVirtualLM._sub_batches([(1,), (2,)], 1000, 100))
    assert subs == [[(1,), (2,)]]  # no-op for small batches


def test_sub_batches_empty():
    assert list(AsyncVirtualLM._sub_batches([], 10, 10)) == []


def test_sub_batches_total_tokens_never_exceed_limit():
    rng = random.Random(0)
    prompts = [tuple(range(rng.randint(1, 50))) for _ in range(200)]
    for sub in AsyncVirtualLM._sub_batches(prompts, max_tokens=128, max_seqs=16):
        assert len(sub) <= 16
        # token budget may only be exceeded by a lone over-long prompt (singleton)
        assert sum(len(t) for t in sub) <= 128 or len(sub) == 1


# --------------------------------------------------------------------------- #
# _batch_evaluate against a fake chunked-prefill engine                       #
# --------------------------------------------------------------------------- #
class _Fut:
    """Minimal stand-in for an asyncio.Future (no event loop needed)."""

    def __init__(self):
        self.result = None
        self.exc = None
        self._done = False

    def set_result(self, r):
        self.result, self._done = r, True

    def set_exception(self, e):
        self.exc, self._done = e, True

    def done(self):
        return self._done


class _FakeCapture:
    """GlobalLogprobsCapture stand-in: a single buffer, overwritten each step.

    Each row is filled with the prompt's first token id (repeated across the vocab)
    so tests can assert which prompt a captured row corresponds to."""

    VOCAB = 4

    def __init__(self):
        self._buf = None

    def clear(self):
        self._buf = None

    def set_rows(self, first_tokens):
        if not first_tokens:
            self._buf = torch.empty(0, self.VOCAB)
        else:
            col = torch.tensor(first_tokens, dtype=torch.float32).reshape(-1, 1)
            self._buf = col.repeat(1, self.VOCAB)

    def get_all_logprobs(self):
        return self._buf


class _FakeEngine:
    """Models vLLM chunked prefill + the capture overwrite: a ``generate()`` whose
    prompts exceed the step budget is split across steps, and only the LAST step's
    rows survive in the (overwritten) capture -- exactly the original bug."""

    def __init__(self, capture, max_tokens, max_seqs, fail_on_call=None):
        self.capture = capture
        self.mt, self.ms = max_tokens, max_seqs
        self.calls = []
        self.fail_on_call = fail_on_call

    def generate(self, prompts, **kwargs):
        self.calls.append(len(prompts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated engine failure")
        toks = [len(p["prompt_token_ids"]) for p in prompts]
        firsts = [p["prompt_token_ids"][0] for p in prompts]
        # walk the scheduler steps; the capture keeps ONLY the last step's rows
        last_step, tot = [], 0
        for j, k in enumerate(toks):
            if last_step and (tot + k > self.mt or len(last_step) >= self.ms):
                last_step, tot = [], 0
            last_step.append(j)
            tot += k
        self.capture.set_rows([float(firsts[j]) for j in last_step])


def _make_lm(capture, engine, limits):
    lm = AsyncVirtualLM.__new__(AsyncVirtualLM)
    lm.queries = []
    lm.timer = None
    lm.logprobs_capture = capture
    lm.llm_engine = engine
    lm.default_params = {}
    lm.lora_request = None
    lm.byte_vocab = [b""] * _FakeCapture.VOCAB
    lm._cached_prefill_limits = limits
    return lm


def test_batch_evaluate_splits_oversized_batch_and_resolves_all():
    limits = (6, 100)  # 6-token step budget
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits)
    lm = _make_lm(cap, eng, limits)

    prompts = [
        (i, i, i) for i in range(5)
    ]  # 5 distinct prompts, 3 tokens each (15 > 6)
    futs = [_Fut() for _ in prompts]
    lm.queries = list(zip(prompts, futs))

    lm._batch_evaluate()

    # split into 3 one-step sub-batches (2 + 2 + 1), not one oversized generate()
    assert eng.calls == [2, 2, 1]
    assert all(f.done() and f.exc is None and f.result is not None for f in futs)


def test_batch_evaluate_dedups_identical_prompts():
    limits = (1000, 100)
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits)
    lm = _make_lm(cap, eng, limits)

    p = (7, 7, 7)
    f1, f2, f3 = _Fut(), _Fut(), _Fut()
    lm.queries = [(p, f1), (p, f2), ((8, 8), f3)]  # p duplicated

    lm._batch_evaluate()

    assert eng.calls == [2]  # 2 unique prompts in one sub-batch
    assert all(f.done() and f.exc is None for f in (f1, f2, f3))
    assert torch.equal(f1.result, f2.result)  # duplicates share the same logprobs


def test_batch_evaluate_failure_is_atomic():
    limits = (6, 100)
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits, fail_on_call=2)  # 2nd sub-batch blows up
    lm = _make_lm(cap, eng, limits)

    prompts = [(i, i, i) for i in range(5)]
    futs = [_Fut() for _ in prompts]
    lm.queries = list(zip(prompts, futs))

    lm._batch_evaluate()

    # collect-first/resolve-last: a mid-batch failure must fail ALL futures,
    # never half-fulfill (no future gets a result).
    assert all(f.exc is not None for f in futs)
    assert all(f.result is None for f in futs)


def test_batch_evaluate_skips_already_done_futures():
    # A future resolved/cancelled before _batch_evaluate must be skipped, not re-set
    # (real asyncio.Future.set_result raises on a done future).
    limits = (1000, 100)
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits)
    lm = _make_lm(cap, eng, limits)

    p = (5, 6, 7)
    done, live = _Fut(), _Fut()
    done.set_result("already")  # pre-resolved
    lm.queries = [(p, done), (p, live)]

    lm._batch_evaluate()

    assert done.result == "already"  # untouched
    assert live.result is not None and int(live.result[0]) == 5


# --------------------------------------------------------------------------- #
# batch_next_token_logprobs_sync (sync path, no dedup, ordered tensor)        #
# --------------------------------------------------------------------------- #
def test_batch_sync_splits_oversized_and_preserves_order():
    limits = (6, 100)
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits)
    lm = _make_lm(cap, eng, limits)

    # distinct first tokens, 3 tokens each (15 > 6 step budget)
    inputs = [(10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0), (50, 0, 0)]
    out = lm.batch_next_token_logprobs_sync(inputs)

    assert eng.calls == [2, 2, 1]  # split into one-step sub-batches
    assert out.shape == (
        len(inputs),
        _FakeCapture.VOCAB,
    )  # one full-vocab row per prompt
    # rows are returned in input order (first-token id encodes identity)
    assert out[:, 0].tolist() == [10, 20, 30, 40, 50]


def test_batch_sync_does_not_dedup():
    limits = (1000, 100)  # everything in one sub-batch
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits)
    lm = _make_lm(cap, eng, limits)

    inputs = [(10, 0, 0), (10, 0, 0), (20, 0, 0)]  # duplicate prompt kept
    out = lm.batch_next_token_logprobs_sync(inputs)

    assert eng.calls == [3]  # 3 prompts, no dedup
    assert out.shape[0] == 3
    assert out[:, 0].tolist() == [10, 10, 20]


def test_batch_sync_empty_returns_empty():
    lm = _make_lm(_FakeCapture(), _FakeEngine(_FakeCapture(), 10, 10), (10, 10))
    out = lm.batch_next_token_logprobs_sync([])
    # 2-D empty, same rank as the populated [N, V] return (not a 1-D torch.empty(0))
    assert out.shape == (0, _FakeCapture.VOCAB)


def test_batch_sync_overlong_prompt_through_helper():
    # A lone prompt over the token budget routes through the helper as its own
    # sub-batch and returns exactly one full-vocab row.
    limits = (3, 100)
    cap = _FakeCapture()
    eng = _FakeEngine(cap, *limits)
    lm = _make_lm(cap, eng, limits)

    out = lm.batch_next_token_logprobs_sync([(9, 9, 9, 9, 9, 9)])
    assert eng.calls == [1]
    assert out.shape == (1, _FakeCapture.VOCAB)
    assert out[0, 0] == 9


# --------------------------------------------------------------------------- #
# next_token_logprobs_sync (single-prompt path + cache)                       #
# --------------------------------------------------------------------------- #
def _make_lm_with_cache(capture, engine, limits, cache):
    lm = _make_lm(capture, engine, limits)
    lm.cache = cache
    return lm


def test_next_token_sync_returns_row_for_its_prompt():
    cap = _FakeCapture()
    eng = _FakeEngine(cap, 1000, 100)
    lm = _make_lm_with_cache(cap, eng, (1000, 100), cache={})

    out = lm.next_token_logprobs_sync([10, 11, 12])
    assert out.shape == (_FakeCapture.VOCAB,)  # one 1-D full-vocab row
    assert out[0] == 10  # first-token id encodes identity


def test_next_token_sync_hits_cache_on_repeat():
    cap = _FakeCapture()
    eng = _FakeEngine(cap, 1000, 100)
    lm = _make_lm_with_cache(cap, eng, (1000, 100), cache={})

    first = lm.next_token_logprobs_sync([10, 11, 12])
    n_calls = len(eng.calls)
    second = lm.next_token_logprobs_sync([10, 11, 12])
    assert len(eng.calls) == n_calls  # served from cache, no new generate()
    assert torch.equal(first, second)


def test_next_token_sync_after_cleanup_raises():
    cap = _FakeCapture()
    eng = _FakeEngine(cap, 1000, 100)
    lm = _make_lm_with_cache(cap, eng, (1000, 100), cache=None)
    lm.logprobs_capture = None
    with pytest.raises(RuntimeError, match="cleanup"):
        lm.next_token_logprobs_sync([1, 2, 3])


# --------------------------------------------------------------------------- #
# _prefill_step_limits (scheduler read + conservative fallback)               #
# --------------------------------------------------------------------------- #
class _FakeSchedulerEngine:
    """Exposes ``llm_engine.vllm_config.scheduler_config`` like vLLM; ``broken``
    makes the attribute walk raise so the fallback path is exercised."""

    def __init__(self, max_tokens=None, max_seqs=None, broken=False):
        if broken:
            self.llm_engine = None
            return
        cfg = SimpleNamespace(max_num_batched_tokens=max_tokens, max_num_seqs=max_seqs)
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(scheduler_config=cfg)
        )


def _bare_lm(engine):
    lm = AsyncVirtualLM.__new__(
        AsyncVirtualLM
    )  # no _cached_prefill_limits -> read path
    lm.llm_engine = engine
    return lm


def test_prefill_limits_reads_scheduler_config():
    lm = _bare_lm(_FakeSchedulerEngine(max_tokens=4096, max_seqs=64))
    assert lm._prefill_step_limits() == (4096, 64)
    assert lm._cached_prefill_limits == (4096, 64)  # cached for reuse


def test_prefill_limits_conservative_fallback_on_error():
    lm = _bare_lm(_FakeSchedulerEngine(broken=True))
    # conservative: small enough to only over-split on any real budget, never drop rows
    assert lm._prefill_step_limits() == (512, 128)
    assert lm._cached_prefill_limits == (512, 128)
