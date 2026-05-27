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
passes those internal ids straight through to the controller; the controller maps them to
particle indices using the table it built from the ids ``add_request`` returned.
(``BatchUpdate.added`` tuples carry only ``(index, SamplingParams, prompt_ids,
output_ids)`` -- no request id -- so the ``input_batch.req_ids`` read is the
authoritative row -> id source, not the LogitsProcessor ``update_state`` seam.)
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EngineControl(Protocol):
    """The seam the SMC controller implements to drive an in-engine decode window.

    Implementations live in genlm-control. The backend depends on nothing in
    genlm-control; this Protocol is the entire interface.
    """

    def draw(self, logits, request_ids: Sequence[str]):
        """Draw one token id per row from the decode step's logits.

        Called once per decode step, after vLLM has applied its own logits
        processors (allowed-token masks, penalties, bad-words, and the
        ``GlobalLogprobsCapture`` capture hook stay intact). The controller returns
        one sampled token id per row (a 1-D tensor or a list of ints, length
        ``num_rows``); it forms its own proposal from the raw engine logits.
        Pop-out is out-of-band, not via the drawn token: the controller records the
        rows to drop and surfaces them from :meth:`drain_aborts`, which ``run_burst``
        calls after the step to ``abort_request`` them.

        The engine's ``SamplingMetadata`` is intentionally not passed: the
        controller draws in its own control-side vocabulary (its own temperature,
        no top-k/p), so the engine's per-row sampling params play no role.

        Args:
            logits: ``[num_rows, vocab]`` float tensor on the engine device.
            request_ids: length-``num_rows`` sequence of vLLM internal request
                ids; ``request_ids[i]`` owns row ``i``.

        Returns:
            A length-``num_rows`` 1-D ``torch.Tensor`` (or list) of token ids.
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
