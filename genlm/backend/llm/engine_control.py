"""The contract an SMC controller (in genlm-control) implements to drive a vLLM
engine step-locked from inside the sampler. The backend owns no algorithm logic.

The unit is a **group**: one logical sequence decoded under K weight-views, run as
K engine requests the backend keeps in lockstep — every group handed to
:meth:`EngineControl.draw` has all K views' logits, and the returned token is
committed to all K. Partial scheduling never reaches the control.

The consuming arm is :class:`genlm.backend.llm.vllm.ControlSampler`, a swapped-in
``vllm.v1.sample.sampler.Sampler`` holding the control by direct in-process
reference.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EngineControl(Protocol):
    """The seam the SMC controller implements to drive an in-engine decode window.

    Implementations live in genlm-control; this Protocol is the entire interface.
    """

    def draw(self, logits, groups: Sequence[int]):
        """Draw one token id per group from ``[G, K, vocab]`` logits; returns a
        length-``G`` tensor or list. ``logits[g, k]`` is view ``k`` of group
        ``groups[g]``; every group present is complete. A group already flagged
        for abort may appear until the abort drains — return any valid id for it.
        Pop-out is out-of-band via :meth:`drain_aborts`, never the drawn token.
        """
        ...

    def drain_aborts(self) -> Sequence[int]:
        """Group handles to abort, accumulated since the last call, cleared on
        read. Called after every ``engine.step()``; all K of a group's requests
        are aborted.
        """
        ...

    def drain_adds(self):
        """``(handle, prompts, lora_names)`` groups to (re-)add, accumulated since
        the last call, cleared on read. ``prompts``/``lora_names`` are K-long.
        ``handle`` must be fresh (no abort/add id race). Called once before the
        loop and after every ``engine.step()``.
        """
        ...

    def on_burst_end(self):
        """Settle work the control deferred past the last decode step, called once
        after the loop drains and before the final drains.
        """
        ...
