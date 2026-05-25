"""Engine-native SMC seams.

This module defines the *only* contract the SMC "hub" (which lives entirely in
genlm-control) needs to implement in order to drive a vLLM engine step-locked
from inside the sampler. The backend owns NO algorithm logic: there is no
driver, no ESS, no resampling, no weight bookkeeping here. The engine exposes
two callbacks -- :meth:`EngineControl.shape` and :meth:`EngineControl.draw` --
plus a way to learn which particle each batch row corresponds to.

The arm that consumes this contract is :class:`genlm.backend.llm.vllm.ControlSampler`,
a swapped-in ``vllm.v1.sample.sampler.Sampler`` whose ``forward`` calls back into
the hub. The hub is held by direct in-process reference; nothing is smuggled
through a registry or ``extra_args``.

Row -> request identity
-----------------------
Inside ``Sampler.forward`` the logits arrive as a ``[num_rows, vocab]`` tensor.
For the plain decode path vLLM uses (one sampled token per request, no
speculative decoding) row ``i`` corresponds to the request at
``model_runner.input_batch.req_ids[i]``. That list is exactly what the model
runner itself uses to scatter sampled tokens back to requests
(``req_ids_output_copy = self.input_batch.req_ids.copy()`` in
``gpu_model_runner._bookkeeping_sync``), so it is the authoritative source.

The request ids in ``input_batch.req_ids`` are vLLM's *internal* ids, which are
the externally-supplied id with 8 random characters appended
(``f"{external}-{random_uuid():.8}"`` in
``vllm.v1.engine.input_processor.assign_request_id``). ``ControlSampler`` therefore
passes those internal ids straight through to the hub; the hub maps them to
particle indices using the table it built from the ids ``add_request`` returned.

:class:`RowRequestTracker` is an optional helper that maintains the same kind of
row -> id table from ``BatchUpdate`` events (the ``update_state`` LogitsProcessor
seam), for callers who prefer not to reach into ``input_batch`` directly. Note
that ``BatchUpdate.added`` tuples are ``(index, SamplingParams, prompt_ids,
output_ids)`` and do NOT carry the request id, so a tracker driven purely by
``BatchUpdate`` can only report *positions*, not ids -- which is why
``ControlSampler`` reads ``input_batch.req_ids`` instead. The tracker is provided
for completeness and parity with the documented seam.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EngineControl(Protocol):
    """The seam the SMC hub implements to drive an in-engine decode window.

    Implementations live in genlm-control. The backend depends on nothing in
    genlm-control; this Protocol is the entire interface.
    """

    def shape(self, logits, request_ids: Sequence[str]) -> None:
        """Shape one decode step's logits into the proposal log-distribution.

        Called once per decode step, after vLLM has applied its own logits
        processors (allowed-token masks, penalties, bad-words, and the
        ``GlobalLogprobsCapture`` capture hook stay intact). The hub mutates
        ``logits[i]`` *in place* so that it becomes the (possibly unnormalized)
        log-distribution of the proposal for the particle identified by
        ``request_ids[i]``.

        Args:
            logits: ``[num_rows, vocab]`` float tensor on the engine device.
                Mutated in place.
            request_ids: length-``num_rows`` sequence of vLLM internal request
                ids; ``request_ids[i]`` owns ``logits[i]``.
        """
        ...

    def draw(self, logits, request_ids: Sequence[str], sampling_metadata):
        """Draw one token id per row from the shaped logits.

        Called immediately after :meth:`shape`. The hub returns one sampled
        token id per row (a 1-D tensor or a list of ints, length ``num_rows``).
        Returning an end-of-sequence token id for a row signals that the
        particle should pop out of the window: vLLM will finish that request and
        ``run_burst`` will stop emitting tokens for it.

        Args:
            logits: ``[num_rows, vocab]`` float tensor (already shaped).
            request_ids: length-``num_rows`` sequence of vLLM internal request
                ids; ``request_ids[i]`` owns row ``i``.
            sampling_metadata: vLLM ``SamplingMetadata`` for this step, in case
                the hub wants temperature / generators / etc.

        Returns:
            A length-``num_rows`` 1-D ``torch.Tensor`` (or list) of token ids.
        """
        ...


class RowRequestTracker:
    """Maintain a row-index -> request-id table from ``BatchUpdate`` events.

    Optional helper for hubs that want to track batch makeup through the
    LogitsProcessor ``update_state(BatchUpdate)`` seam rather than reading
    ``model_runner.input_batch.req_ids`` directly. Because ``BatchUpdate`` does
    not carry request ids, callers must supply the id for each *added* request
    via :meth:`note_added_id`, keyed by batch index, before/at the step the add
    is observed.

    The semantics follow vLLM's documented ordering: a ``BatchUpdate`` is applied
    as removed, then added, then moved.
    """

    def __init__(self):
        # row index -> request id (None for a hole that condense() will fill)
        self._rows: list[str | None] = []
        # batch index -> request id, staged by the caller for the next add
        self._pending_ids: dict[int, str] = {}

    def note_added_id(self, index: int, request_id: str) -> None:
        """Stage the request id that the next ``added`` tuple at ``index`` owns."""
        self._pending_ids[index] = request_id

    def update_state(self, batch_update) -> None:
        """Apply a ``BatchUpdate`` (or ``None``) to the row table."""
        if batch_update is None:
            return

        size = batch_update.batch_size
        if len(self._rows) < size:
            self._rows.extend([None] * (size - len(self._rows)))

        # Order matters: removed, then added, then moved.
        for index in batch_update.removed:
            if index < len(self._rows):
                self._rows[index] = None

        for added in batch_update.added:
            index = added[0]
            if index >= len(self._rows):
                self._rows.extend([None] * (index + 1 - len(self._rows)))
            self._rows[index] = self._pending_ids.pop(index, None)

        for moved in batch_update.moved:
            a, b = moved[0], moved[1]
            direct = moved[2]
            # MoveDirectionality.SWAP swaps a<->b; UNIDIRECTIONAL moves a->b.
            if getattr(direct, "name", str(direct)).upper().endswith("SWAP"):
                self._rows[a], self._rows[b] = self._rows[b], self._rows[a]
            else:
                self._rows[b] = self._rows[a]
                self._rows[a] = None

        del self._rows[size:]

    def request_ids(self, num_rows: int | None = None) -> list[str | None]:
        """Return the current row -> request-id table."""
        if num_rows is None:
            return list(self._rows)
        return list(self._rows[:num_rows])
