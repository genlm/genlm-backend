"""Per-target memoization and query accounting for deterministic embedding scores."""

import asyncio
from typing import Protocol


class Scorer(Protocol):
    """Return deterministic cosine scores aligned with the requested strings."""

    async def similarities(self, texts: tuple[str, ...]) -> tuple[float, ...]: ...


class MemoizedScorer:
    """Reuse scores for identical decoded text under one fixed target embedding."""

    def __init__(self, delegate: Scorer):
        self._delegate = delegate
        self._cache: dict[str, float] = {}
        self._requested_queries = 0
        self._lock = asyncio.Lock()

    @property
    def requested_queries(self) -> int:
        """Count requested text scores, including repeats."""
        return self._requested_queries

    @property
    def evaluated_queries(self) -> int:
        """Count distinct strings with completed score evaluations."""
        return len(self._cache)

    async def similarities(self, texts: tuple[str, ...]) -> tuple[float, ...]:
        """Evaluate missing strings once across concurrent batches and restore request order."""
        self._requested_queries += len(texts)
        async with self._lock:
            missing = tuple(
                text for text in dict.fromkeys(texts) if text not in self._cache
            )
            if missing:
                scores = await self._delegate.similarities(missing)
                self._cache.update(zip(missing, scores))
        return tuple(self._cache[text] for text in texts)
