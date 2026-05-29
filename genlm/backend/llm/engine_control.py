"""Engine-native SMC seams.

This module defines the *only* contract the SMC "controller" (which lives entirely in
genlm-control) needs to implement in order to drive a vLLM engine step-locked
from inside the sampler. The backend owns NO algorithm logic: there is no
driver, no ESS, no resampling, no weight bookkeeping here. The engine exposes
one callback -- :meth:`EngineControl.draw` -- plus a way to learn which particle
each batch row corresponds to.

The arm that consumes this contract is :class:`genlm.backend.llm.vllm.ControlSampler`,
a swapped-in ``vllm.v1.sample.sampler.Sampler`` whose ``forward`` calls back into
the controller. The controller is held by direct in-process reference; nothing is smuggled
through a registry or ``extra_args``.

Row -> request identity
-----------------------
Row ``i``'s request is ``model_runner.input_batch.req_ids[i]`` (the authoritative
row->id source the model runner itself scatters tokens by). Those are vLLM's
*internal* ids -- the external id with 8 random chars appended
(``f"{external}-{random_uuid():.8}"``). ``ControlSampler`` strips that suffix and
hands the control its own external handle (int), so the control never parses the
backend's id format.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EngineControl(Protocol):
    """The seam the SMC controller implements to drive an in-engine decode window.

    Implementations live in genlm-control. The backend depends on nothing in
    genlm-control; this Protocol is the entire interface.
    """

    def draw(self, logits, handles: Sequence[int]):
        """Draw one token id per row from the step's ``[num_rows, vocab]`` logits.

        ``handles[i]`` is the control's own request handle owning row ``i`` (the
        backend strips its internal id format). Returns a length-``num_rows`` tensor
        or list of token ids. Pop-out is out-of-band via :meth:`drain_aborts`, not
        the drawn token. ``SamplingMetadata`` is not passed: the control draws in its
        own vocab/temperature.
        """
        ...

    def drain_aborts(self) -> Sequence[int]:
        """External request indices (the ``str(i)`` ``run_burst`` submitted, as
        ints) to abort, accumulated since the last call and cleared on read.

        ``run_burst`` calls this after every ``engine.step()`` and issues
        ``abort_request`` for the returned rows -- the out-of-band pop-out that
        replaces an EOS-stop draw. The controller flags a row when its particle
        terminates (staggered) or, all live rows at once, when its ESS test crosses.
        """
        ...
