"""Resident-lane primitives shared by the engine servers.

A ``Lane`` is one resident decode request: open it with a prompt, then per step
``await lane.next()`` for the post-processor logits row and ``lane.feed(token)``
to commit — or ``lane.close()`` instead of feeding when the row will not read the
next warm. The engine steps once every published lane has fed or closed.

The ``LaneLedger`` is the backend-internal half: id minting, add/abort queues the
engine thread drains between steps, and the feed barrier. Data crosses the
engine-thread/event-loop boundary (rows out in one callback, feeds in through the
barrier); control flow never does.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Optional


class LaneError(RuntimeError):
    """A lane-contract violation: read/feed on a closed lane, feed without a
    warm, a context-verification mismatch, or a feed-barrier timeout."""


class RowHandle:
    """Atomicity domain: the K lanes of one particle. The backend never publishes
    a step to a strict subset of a handle's open lanes."""

    __slots__ = ("uid", "lanes")

    def __init__(self, uid: int):
        self.uid = uid
        self.lanes: list[Lane] = []

    def __repr__(self):
        return f"RowHandle({self.uid}, lanes={len(self.lanes)})"


class Lane:
    """One resident decode request. States: OPEN (no warm), WARM (row published,
    idempotently readable until fed), CLOSED. ``close`` is idempotent; every
    other call on a closed lane raises."""

    def __init__(
        self, ledger: "LaneLedger", rid: int, prompt_ids, lora_name, row, pool_key=None
    ):
        self._ledger = ledger
        self.rid = rid  # current engine rid; the ledger may swap it on a stall
        self.context: list[int] = list(prompt_ids)
        self.lora_name = lora_name
        self.row = row
        # Cohort tag: lanes sharing a pool_key step together (an engine that owns
        # its scheduler steps one cohort's pools when all its lanes have fed).
        self.pool_key = pool_key
        self.closed = False
        self._warm_value: Optional[Any] = None
        self._waiter: Optional[asyncio.Future] = None

    @property
    def warm(self) -> bool:
        return self._warm_value is not None

    async def next(self) -> Any:
        if self.closed:
            raise LaneError(f"read on closed lane {self.rid}")
        if self._warm_value is not None:
            return self._warm_value
        if self._waiter is None:
            self._waiter = asyncio.get_running_loop().create_future()
        return await asyncio.shield(self._waiter)

    def feed(self, token_id: int) -> None:
        if self.closed:
            raise LaneError(f"feed on closed lane {self.rid}")
        if self._warm_value is None:
            raise LaneError(f"feed without a warm on lane {self.rid}")
        self._warm_value = None
        self.context.append(int(token_id))
        self._ledger._answered(self, int(token_id))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._warm_value = None
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_exception(LaneError(f"lane {self.rid} closed"))
        self._waiter = None
        self._ledger._retire(self)

    def verify(self, context) -> None:
        """Raise unless ``context`` is exactly this lane's context."""
        if list(context) != self.context:
            raise LaneError(
                f"lane {self.rid} context mismatch: caller has {len(list(context))} "
                f"items, lane has {len(self.context)}"
            )

    def _deliver(self, row: Any) -> None:
        # Event-loop thread only, via the ledger's single publish callback.
        if self.closed:
            return
        self._warm_value = row
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(row)
        self._waiter = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()

    def __repr__(self):
        state = "CLOSED" if self.closed else ("WARM" if self.warm else "OPEN")
        return f"Lane({self.rid}, {state}, len={len(self.context)})"


class LaneLedger:
    """Backend half of the lane contract: rid minting (monotonic, never reused),
    the add/abort queues an engine thread drains between steps, and the feed
    barrier (`publish_and_wait`)."""

    def __init__(self, barrier_timeout: float = 300.0):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._next_rid = 0
        self._next_row = 0
        self.lanes: dict[int, Lane] = {}  # open lanes by rid
        self._cv = threading.Condition()
        self._adds: list[Lane] = []  # queued for the engine, drained under _cv
        self._aborts: list[int] = []  # rids queued for engine abort
        self._owed: set[int] = set()
        self._fed: dict[int, int] = {}
        self._timeout = barrier_timeout
        self.submitted = threading.Event()  # wakes a parked engine thread

    # -- event-loop side -----------------------------------------------------

    def open_lane(
        self,
        prompt_ids,
        *,
        lora_name=None,
        row: Optional[RowHandle] = None,
        pool_key=None,
    ) -> Lane:
        # Rebind per open: the server outlives event loops (one asyncio.run per
        # inference), and publishing to a dead loop would strand every warm.
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
        if row is None:
            row = self.row_handle()
        lane = Lane(self, self._mint(), prompt_ids, lora_name, row, pool_key)
        row.lanes.append(lane)
        self.lanes[lane.rid] = lane
        with self._cv:
            self._adds.append(lane)
        self.submitted.set()
        return lane

    def row_handle(self) -> RowHandle:
        self._next_row += 1
        return RowHandle(self._next_row)

    def _mint(self) -> int:
        self._next_rid += 1
        return self._next_rid

    def _answered(self, lane: Lane, token_id: int) -> None:
        with self._cv:
            if lane.rid in self._owed:
                self._fed[lane.rid] = token_id
                self._cv.notify_all()
        # A feed may complete a cohort: wake an engine parked on readiness.
        self.submitted.set()

    def _retire(self, lane: Lane) -> None:
        self.lanes.pop(lane.rid, None)
        lane.row.lanes.remove(lane)
        with self._cv:
            if lane in self._adds:  # never reached the engine: cancel the add
                self._adds.remove(lane)
            else:
                self._aborts.append(lane.rid)
            if lane.rid in self._owed:
                self._owed.discard(lane.rid)
                self._cv.notify_all()
        self.submitted.set()

    # -- engine-thread side ----------------------------------------------------

    def drain(self) -> tuple[list[Lane], list[int]]:
        """Queued (adds, aborts), cleared on read. Aborts first: a swapped rid
        must be gone before its replacement mints one."""
        with self._cv:
            adds, self._adds = self._adds, []
            aborts, self._aborts = self._aborts, []
        return adds, aborts

    def publish_and_wait(self, rows: dict[int, Any]) -> dict[int, int]:
        """Deliver one step's rows (rid -> logits row) in a single loop callback,
        then block until every published lane has fed or closed. Returns the fed
        tokens by rid; closed lanes are simply absent."""
        if not rows:
            return {}
        assert self._loop is not None, "publish before any open_lane"
        with self._cv:
            self._owed = set(rows)
            self._fed = {}
        self._loop.call_soon_threadsafe(self._deliver_step, dict(rows))
        with self._cv:
            while self._owed - set(self._fed):
                if not self._cv.wait(self._timeout):
                    owing = sorted(self._owed - set(self._fed))
                    raise LaneError(f"feed barrier timeout; lanes owing: {owing}")
            fed, self._fed, self._owed = self._fed, {}, set()
        return fed

    def publish(self, rows: dict[int, Any]) -> None:
        """Deliver one cohort's rows in a single loop callback WITHOUT waiting for
        feeds — for an engine that owns its scheduler and steps a cohort only once
        every one of its lanes has fed (no cross-cohort barrier)."""
        if rows:
            assert self._loop is not None, "publish before any open_lane"
            self._loop.call_soon_threadsafe(self._deliver_step, dict(rows))

    def _deliver_step(self, rows: dict[int, Any]) -> None:
        for rid, row in rows.items():
            lane = self.lanes.get(rid)
            if lane is not None:
                lane._deliver(row)

    def restall(self, row: RowHandle) -> list[tuple[int, Lane]]:
        """Mint fresh rids for a handle's lanes (engine re-adds them at their
        committed contexts) and return ``(old_rid, lane)`` pairs to abort. The
        ``Lane`` objects survive; only the engine binding changes."""
        swapped = []
        for lane in row.lanes:
            old = lane.rid
            self.lanes.pop(old, None)
            lane.rid = self._mint()
            self.lanes[lane.rid] = lane
            swapped.append((old, lane))
        return swapped


class FakeLaneServer:
    """A scripted engine for lane-contract tests: a real worker thread drives the
    same ledger the production servers use, with ``logits_fn(context, lora_name)``
    in place of a model. Steps whenever every open lane is fed; parks when empty."""

    def __init__(self, logits_fn: Callable[[list[int], Optional[str]], Any]):
        self.ledger = LaneLedger(barrier_timeout=5.0)
        self._logits_fn = logits_fn
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.steps = 0
        self.error: Optional[BaseException] = None
        self.live: dict[int, Lane] = {}  # engine-side residency, rid -> lane

    def open_lane(self, prompt_ids, **kw) -> Lane:
        return self.ledger.open_lane(prompt_ids, **kw)

    async def score(self, ids) -> Any:
        """One-shot: runs between decode steps, lanes resident or not."""
        return self._logits_fn(list(ids), None)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.ledger.submitted.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                adds, aborts = self.ledger.drain()
                for rid in aborts:
                    self.live.pop(rid, None)
                for lane in adds:
                    self.live[lane.rid] = lane
                if not self.live:
                    self.ledger.submitted.wait(timeout=0.05)
                    self.ledger.submitted.clear()
                    continue
                rows = {
                    rid: self._logits_fn(list(lane.context), lane.lora_name)
                    for rid, lane in self.live.items()
                }
                self.ledger.publish_and_wait(rows)
                self.steps += 1
        except BaseException as e:  # surfaced by tests
            self.error = e
