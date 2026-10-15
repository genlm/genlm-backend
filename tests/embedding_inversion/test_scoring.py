import asyncio

from hypothesis import given
from hypothesis import strategies as st

from examples.embedding_inversion._algorithms.scoring import MemoizedScorer


def _score(text: str) -> float:
    return (sum(map(ord, text)) % 201 - 100) / 100


@given(
    st.lists(
        st.lists(st.text(alphabet="abc ", max_size=12), max_size=8),
        min_size=1,
        max_size=4,
    )
)
def test_memoized_scoring_preserves_order_and_evaluates_each_distinct_text_once(
    batches,
):
    class EncoderModel:
        def __init__(self):
            self.batches = []

        async def similarities(self, texts):
            self.batches.append(texts)
            await asyncio.sleep(0)
            return tuple(_score(text) for text in texts)

    encoder = EncoderModel()
    scorer = MemoizedScorer(encoder)

    async def run():
        return tuple(
            await asyncio.gather(
                *(scorer.similarities(tuple(batch)) for batch in batches)
            )
        )

    values = asyncio.run(run())
    requested = tuple(text for batch in batches for text in batch)
    evaluated = tuple(text for batch in encoder.batches for text in batch)
    assert values == tuple(tuple(_score(text) for text in batch) for batch in batches)
    assert set(evaluated) == set(requested)
    assert len(evaluated) == len(set(evaluated)) == scorer.evaluated_queries
    assert scorer.requested_queries == len(requested)
    assert all(encoder.batches)
