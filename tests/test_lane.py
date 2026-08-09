"""Lane-contract tests against the FakeLaneServer: state machine, feed barrier,
close-instead-of-feed, ownership, park/revive, one-shot coexistence."""

import asyncio

import numpy as np
import pytest

from genlm.backend.llm.lane import FakeLaneServer, LaneError

V = 8


def logits_fn(context, lora_name):
    # Deterministic, context-dependent row: argmax = (len + last token) % V.
    row = np.zeros(V)
    row[(len(context) + (context[-1] if context else 0)) % V] = 1.0
    return row


@pytest.fixture
def server():
    s = FakeLaneServer(logits_fn)
    s.start()
    yield s
    s.stop()
    assert s.error is None


async def drive(lane, n):
    out = []
    for _ in range(n):
        row = await lane.next()
        tok = int(np.argmax(row))
        lane.feed(tok)
        out.append(tok)
    return out


@pytest.mark.asyncio
async def test_two_lanes_advance_distinct(server):
    a = server.open_lane([1, 2, 3])
    b = server.open_lane([4])
    ta, tb = await asyncio.gather(drive(a, 6), drive(b, 6))
    assert len(ta) == len(tb) == 6
    assert a.context == [1, 2, 3] + ta
    assert b.context == [4] + tb
    assert ta != tb
    a.close()
    b.close()


@pytest.mark.asyncio
async def test_idempotent_next_until_feed(server):
    lane = server.open_lane([1])
    r1 = await lane.next()
    r2 = await lane.next()
    assert r1 is r2
    lane.feed(0)
    lane.close()


@pytest.mark.asyncio
async def test_feed_without_warm_raises(server):
    lane = server.open_lane([1])
    await lane.next()
    lane.feed(0)
    with pytest.raises(LaneError, match="without a warm"):
        lane.feed(1)
    lane.close()


@pytest.mark.asyncio
async def test_closed_lane_raises_and_close_idempotent(server):
    lane = server.open_lane([1])
    await lane.next()
    lane.close()
    lane.close()  # idempotent
    with pytest.raises(LaneError, match="closed"):
        await lane.next()
    with pytest.raises(LaneError, match="closed"):
        lane.feed(0)


@pytest.mark.asyncio
async def test_close_instead_of_feed_releases_barrier(server):
    a = server.open_lane([1])
    b = server.open_lane([2])
    await asyncio.gather(a.next(), b.next())
    a.close()  # answers the step without feeding
    b.feed(0)
    row = await b.next()  # engine steps again with only b resident
    assert row is not None
    b.close()


@pytest.mark.asyncio
async def test_verify_context(server):
    lane = server.open_lane([1, 2])
    lane.verify([1, 2])
    with pytest.raises(LaneError, match="mismatch"):
        lane.verify([1])
    lane.close()


@pytest.mark.asyncio
async def test_park_on_empty_then_revive(server):
    a = server.open_lane([1])
    await drive(a, 2)
    a.close()
    await asyncio.sleep(0.15)  # engine parks on empty
    steps = server.steps
    b = server.open_lane([5])
    toks = await drive(b, 3)
    assert len(toks) == 3
    assert server.steps > steps
    b.close()


@pytest.mark.asyncio
async def test_oneshot_while_resident(server):
    lane = server.open_lane([1])
    await lane.next()
    row = await server.score([9, 9, 9])
    assert int(np.argmax(row)) == (3 + 9) % V
    lane.feed(0)
    lane.close()


@pytest.mark.asyncio
async def test_owner_task_death_auto_closes(server):
    lane = server.open_lane([1])
    peer = server.open_lane([2])

    async def crasher():
        async with lane:
            await lane.next()
            raise RuntimeError("row died")

    async def survivor():
        return await drive(peer, 3)

    crashed, toks = await asyncio.gather(crasher(), survivor(), return_exceptions=True)
    assert isinstance(crashed, RuntimeError)
    assert not isinstance(toks, BaseException) and len(toks) == 3
    assert lane.closed
    peer.close()


@pytest.mark.asyncio
async def test_cancelled_add_never_reaches_engine(server):
    lane = server.open_lane([1])
    resident = server.open_lane([3])
    await resident.next()
    lane.close()  # may cancel the queued add before the engine drains it
    resident.feed(0)
    await drive(resident, 2)
    assert lane.rid not in server.live
    resident.close()


@pytest.mark.asyncio
async def test_restall_swaps_rid_same_lane():
    # Ledger-level: restall is the engine's own move (vLLM partial-schedule
    # stall), exercised here without a running engine thread.
    from genlm.backend.llm.lane import LaneLedger

    ledger = LaneLedger()
    lane = ledger.open_lane([1, 2])
    ledger.drain()  # the add reached the engine
    old = lane.rid
    ledger.restall(lane.row)
    assert lane.rid != old and ledger.lanes[lane.rid] is lane
    assert lane.context == [1, 2]  # the Lane object survives untouched
    adds, aborts = ledger.drain()
    assert adds == [lane] and aborts == [old]  # abort old rid, re-add at context


@pytest.mark.asyncio
async def test_row_handle_groups_lanes(server):
    row = server.ledger.row_handle()
    k0 = server.open_lane([1], row=row)
    k1 = server.open_lane([1], lora_name="q", row=row)
    assert row.lanes == [k0, k1]
    r0, r1 = await asyncio.gather(k0.next(), k1.next())
    assert r0 is not None and r1 is not None
    k0.feed(0)
    k1.feed(0)
    k0.close()
    k1.close()
    assert row.lanes == []
