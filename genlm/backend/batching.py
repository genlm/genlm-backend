"""Per-event-loop batch windows for concurrent asks.

Concurrent callers on one event loop append their entries to a window; the first
caller holds it open until a full scheduler pass adds nothing, then drains it and
serves the whole cohort. There is no background task: the holder resolves every
entry in its own coroutine, or fails them all if it dies first.
"""

import asyncio
from collections import Counter


class BatchAbandoned(RuntimeError):
    """The caller holding a batch died before it could be dispatched."""


def batch_abandoned(exc):
    """The failure handed to callers whose batch holder died.

    Never the cause itself: a ``CancelledError`` given to a caller who never asked
    for one leaves their task cancelled and skips their ``except Exception``.
    """
    abandoned = BatchAbandoned(f"batch holder did not survive it: {exc!r}")
    abandoned.__cause__ = exc
    return abandoned


def fail_futures(entries, exc):
    """Resolve each entry's future -- its last element -- with ``exc``."""
    for entry in entries:
        future = entry[-1]
        if not future.done():
            future.set_exception(exc)


class _Batch:
    """Per-event-loop request meeting point; must not outlive its loop."""

    __slots__ = ("queue", "armed")

    def __init__(self):
        self.queue = []
        self.armed = False


async def join_batch(store, entries, *, linger=0.0):
    """Join the batch of concurrent callers on this event loop.

    Appends ``entries`` before any yield, so they enter as one set of asks. If
    nobody holds the batch yet, this caller holds it open until a full
    event-loop pass adds nothing -- preceded, when ``linger`` is nonzero, by one
    cooperative sleep for late callers -- and receives the drained batch; every
    other caller receives ``None``. The holder must serve the batch in its own
    coroutine, never a background task: batch state must not outlive its loop.

    An entry is a tuple ending in its future. A holder that dies before handing
    the batch off fails every other queued future rather than orphaning it, so
    an entry is always resolved exactly once.

    Args:
        store (weakref.WeakKeyDictionary): Event loop to ``_Batch`` map, owned by
            the call site. One store per batch.
        entries (list): Requests to add, each ending in its future.
        linger (float, optional): Seconds to sleep once for late callers.
            Defaults to 0.0, which skips the sleep.

    Returns:
        (list | None): The drained batch for the holding caller, ``None`` for
            everyone else.
    """
    loop = asyncio.get_running_loop()
    batch = store.get(loop)
    if batch is None:
        batch = store[loop] = _Batch()
    batch.queue.extend(entries)
    if batch.armed:
        return None
    batch.armed = True
    try:
        # Callers reach the batch at different depths of a `gather` tree, and each
        # level is another scheduler turn; yield until a turn adds nothing.
        lingered = not linger
        while True:
            n = len(batch.queue)
            await asyncio.sleep(0)
            if len(batch.queue) > n:
                continue
            if lingered:
                break
            lingered = True
            await asyncio.sleep(linger)
        queue, batch.queue = batch.queue, []
        return queue
    except BaseException as exc:
        queue, batch.queue = batch.queue, []
        # Not this caller's own entries: it is unwinding past its ``await``, so an
        # exception set there is only ever logged as never retrieved.
        mine = {id(e) for e in entries}
        fail_futures([e for e in queue if id(e) not in mine], batch_abandoned(exc))
        raise
    finally:
        batch.armed = False


# Batching counters: (site, ...) -> count, e.g. ("draw", cohort_size). Read and
# clear via `take_batch_stats`.
batch_stats = Counter()


def take_batch_stats():
    """Snapshot and reset the batching counters. ``batch_stats`` keeps its identity,
    so importers of the counter itself stay attached to it.

    Returns:
        (collections.Counter): The counts since the last call.
    """
    stats = Counter(batch_stats)
    batch_stats.clear()
    return stats
