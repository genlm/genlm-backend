import asyncio
from collections import Counter

import numpy as np
import pytest
import torch

from genlm.backend.batching import BatchAbandoned, take_batch_stats
from genlm.backend.draw import (
    DRAW_METHODS,
    draw_from,
    logp_at,
    logsumexp_from,
    set_draw_method,
)


@pytest.fixture(autouse=True)
def _clean_batch_stats():
    # Drain any residue from a previous test/loop before and after, so a
    # test's ``take_batch_stats()`` snapshot is its own asks, nothing else.
    take_batch_stats()
    yield
    take_batch_stats()


def _rows(n, v=4):
    return [torch.log_softmax(torch.randn(v), -1) for _ in range(n)]


@pytest.mark.asyncio
async def test_cohort_formation():
    await asyncio.gather(*[draw_from(r) for r in _rows(6)])
    assert take_batch_stats()[("draw", 6)] == 1


@pytest.mark.asyncio
async def test_nested_gather_depths_land_in_one_cohort():
    # Peel ``depth`` extra layers of asyncio.gather before the actual draw_from
    # call -- each layer defers that call by one event-loop tick, staggering
    # arrivals at the window without opening a gap wide enough to flush early.
    async def nested(row, depth):
        if depth == 0:
            return await draw_from(row)
        return (await asyncio.gather(nested(row, depth - 1)))[0]

    rows = _rows(9)
    depths = [0, 1, 2] * 3
    await asyncio.gather(*[nested(r, d) for r, d in zip(rows, depths)])
    assert take_batch_stats()[("draw", 9)] == 1


@pytest.mark.asyncio
async def test_kinds_shapes_and_backends_split_into_separate_groups():
    torch_rows = _rows(4)
    np_rows = [np.log(np.full(3, 1 / 3)) for _ in range(3)]

    await asyncio.gather(
        *[draw_from(r) for r in torch_rows],
        *[draw_from(r) for r in np_rows],
        *[logsumexp_from(r) for r in torch_rows[:2]],
        logp_at(torch_rows[0], 1),
    )
    stats = take_batch_stats()
    assert stats[("draw", 4)] == 1  # the torch/V=4 draws
    assert stats[("draw", 3)] == 1  # the numpy/V=3 draws, not merged into the above
    assert stats[("logsumexp", 2)] == 1
    assert stats[("at", 1)] == 1
    assert sum(stats.values()) == 4


@pytest.mark.asyncio
async def test_importance_draw_matches_hand_computed_logw():
    proposal = torch.log_softmax(torch.tensor([5.0, 0.0, 0.0]), -1)  # peaked at 0
    target = torch.log_softmax(torch.tensor([1.0, 2.0, 5.0]), -1)  # peaked at 2

    out = await asyncio.gather(
        draw_from(proposal, target=target), *[draw_from(r) for r in _rows(2, 3)]
    )
    assert take_batch_stats()[("draw", 3)] == 1  # importance + plain, one cohort

    idx, logw, logp = out[0]
    expected_logp = (proposal[idx] - torch.logsumexp(proposal, 0)).item()
    assert logp == pytest.approx(expected_logp, abs=1e-5)
    assert logw == pytest.approx(target[idx].item() - expected_logp, abs=1e-5)
    for _, plain_logw, _ in out[1:]:
        assert abs(plain_logw) < 1e-5  # plain draw: logZ of an already-softmaxed row


@pytest.mark.asyncio
async def test_logsumexp_and_logp_at_read_back_the_row():
    rows = [torch.randn(4) * 3 for _ in range(4)]
    rows.append(torch.full((4,), float("-inf")))  # empty support: -inf, never NaN

    logZs = await asyncio.gather(*[logsumexp_from(r) for r in rows])
    for row, logZ in zip(rows, logZs):
        assert logZ == pytest.approx(torch.logsumexp(row, -1).item(), abs=1e-5)
    assert logZs[-1] == float("-inf")

    vals = await asyncio.gather(*[logp_at(r, i) for i, r in enumerate(rows[:4])])
    for i, (row, val) in enumerate(zip(rows, vals)):
        assert val == pytest.approx(row[i].item(), abs=1e-6)


@pytest.mark.asyncio
async def test_exception_fails_its_group_without_poisoning_others():
    good_rows = _rows(3)
    bad_rows = [
        torch.log_softmax(torch.randn(3), -1),
        torch.zeros(2, 3),  # wrong shape: torch.stack over this group raises
    ]

    good_task = asyncio.gather(*[draw_from(r) for r in good_rows])
    bad_task = asyncio.gather(*[draw_from(r) for r in bad_rows], return_exceptions=True)
    good_out, bad_out = await asyncio.gather(good_task, bad_task)

    stats = take_batch_stats()
    assert stats[("draw", 3)] == 1  # good group flushed
    assert stats[("draw", 2)] == 1  # bad group flushed too, same cohort

    for idx, _, _ in good_out:
        assert 0 <= idx < 4  # unaffected by the other group's failure

    assert len(bad_out) == 2
    assert all(isinstance(e, RuntimeError) for e in bad_out)
    assert bad_out[0] is bad_out[1]  # one exception instance across the group


@pytest.mark.asyncio
async def test_distribution_matches_known_categorical():
    probs = torch.tensor([0.2, 0.3, 0.5])
    row = probs.log()
    n = 2000

    torch.manual_seed(0)
    out = await asyncio.gather(*[draw_from(row) for _ in range(n)])
    assert take_batch_stats()[("draw", n)] == 1  # one reduction for the cohort

    counts = Counter(idx for idx, _, _ in out)
    for idx, p in enumerate(probs.tolist()):
        assert abs(counts[idx] / n - p) < 0.04


@pytest.mark.asyncio
async def test_set_draw_method_round_trip():
    row = torch.log_softmax(torch.tensor([1.0, 2.0, 3.0, 0.5]), -1)
    try:
        for name in DRAW_METHODS:
            set_draw_method(name)
            out = await asyncio.gather(*[draw_from(row) for _ in range(5)])
            for idx, logw, _ in out:
                assert 0 <= idx < 4
                assert logw == pytest.approx(0.0, abs=1e-5)
    finally:
        set_draw_method("gumbel_max")  # never leak a picker across test order


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(DRAW_METHODS))
async def test_empty_support_row_does_not_poison_its_cohort(name):
    rows = _rows(3)
    rows.append(torch.full((4,), float("-inf")))

    try:
        set_draw_method(name)
        out = await asyncio.gather(*[draw_from(r) for r in rows])
    finally:
        set_draw_method("gumbel_max")

    for idx, logw, logp in out[:-1]:
        assert 0 <= idx < 4
        assert logw == pytest.approx(0.0, abs=1e-5)
        assert logp <= 0
    idx, logw, logp = out[-1]
    assert 0 <= idx < 4
    assert logw == float("-inf")
    assert logp == float("-inf")

    # Importance draw off an empty proposal: dead, never an infinite weight.
    _, logw, logp = await draw_from(rows[-1], target=_rows(1)[0])
    assert logw == float("-inf")
    assert logp == float("-inf")


@pytest.mark.asyncio
async def test_other_kinds_consume_no_draw_noise():
    """A logsumexp or logp_at ask in the cohort never touches the picker, so the
    draws around it keep the noise they would have had on their own."""
    rows = _rows(3)
    extra = torch.randn(4)

    async def run(with_others):
        torch.manual_seed(0)
        tasks = [draw_from(r) for r in rows]
        if with_others:
            tasks.insert(1, logsumexp_from(extra))
            tasks.insert(3, logp_at(extra, 2))
        return await asyncio.gather(*tasks)

    plain = await run(False)
    mixed = await run(True)
    assert mixed[1] == pytest.approx(torch.logsumexp(extra, -1).item(), abs=1e-5)
    assert mixed[3] == pytest.approx(extra[2].item(), abs=1e-6)
    assert [d[0] for d in plain] == [mixed[0][0], mixed[2][0], mixed[4][0]]


@pytest.mark.asyncio
async def test_abandoned_batch_fails_co_callers():
    """A cancelled batch holder must fail its co-callers, not orphan them.

    The failure is never the holder's own `CancelledError`: handed to a caller who
    never asked for one, that leaves their task cancelled and skips their
    `except Exception`.
    """
    tasks = [asyncio.ensure_future(draw_from(r)) for r in _rows(3)]
    await asyncio.sleep(0)  # everyone is queued; tasks[0] holds the batch
    tasks[0].cancel()

    results = await asyncio.gather(*tasks[1:], return_exceptions=True)
    for result in results:
        assert isinstance(result, BatchAbandoned)
        assert not isinstance(result, asyncio.CancelledError)
    assert all(not t.cancelled() for t in tasks[1:])
    assert tasks[0].cancelled()


@pytest.mark.asyncio
async def test_readback_keeps_the_row_precision():
    row = torch.log(torch.tensor([0.3, 0.4, 0.2, 0.1], dtype=torch.float64))
    idx, logw, logp = await draw_from(row)
    assert logw == pytest.approx(0.0, abs=1e-12)
    assert logp == pytest.approx(row[idx].item(), abs=1e-12)
    assert await logsumexp_from(row) == pytest.approx(0.0, abs=1e-12)
    assert await logp_at(row, 2) == pytest.approx(row[2].item(), abs=1e-15)
