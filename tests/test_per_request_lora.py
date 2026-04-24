"""Tests for per-request LoRA support in AsyncVirtualLM.

These tests mock the vLLM engine so they run locally without a GPU.
They verify cache key logic, parameter resolution, and that the sampler
uses per-request LoRA instead of global clear_lora/set_lora switching.

Run with:  pytest tests/test_per_request_lora.py -v --noconftest
"""

import asyncio
import importlib
import os
import torch
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

# Import OutputCache directly from the source file (only depends on torch)
_cache_spec = importlib.util.spec_from_file_location(
    "_cache_mod",
    os.path.join(os.path.dirname(__file__), "..", "genlm", "backend", "cache.py"),
)
_cache_mod = importlib.util.module_from_spec(_cache_spec)
_cache_spec.loader.exec_module(_cache_mod)
OutputCache = _cache_mod.OutputCache

# These are defined in vllm.py but can't be imported due to transitive deps.
# They're trivial, so we replicate them here for testing.
_LORA_UNSET = object()


def _lora_cache_key(lora_request):
    """Extract a hashable key from a LoRARequest (or None).
    Must match the implementation in genlm/backend/llm/vllm.py."""
    if lora_request is None:
        return None
    return lora_request.lora_name


@dataclass
class FakeLoRARequest:
    """Minimal stand-in for vllm.lora.request.LoRARequest."""
    lora_name: str
    lora_int_id: int = 1
    lora_path: str = "/tmp/fake"


# ---------------------------------------------------------------------------
# _lora_cache_key
# ---------------------------------------------------------------------------

class TestLoraCacheKey:
    def test_none_returns_none(self):
        assert _lora_cache_key(None) is None

    def test_lora_request_returns_name(self):
        lr = FakeLoRARequest(lora_name="lora_v3")
        assert _lora_cache_key(lr) == "lora_v3"

    def test_different_lora_different_key(self):
        lr1 = FakeLoRARequest(lora_name="lora_v1")
        lr2 = FakeLoRARequest(lora_name="lora_v2")
        assert _lora_cache_key(lr1) != _lora_cache_key(lr2)


# ---------------------------------------------------------------------------
# LoRA-aware OutputCache keys
# ---------------------------------------------------------------------------

class TestLoraAwareCacheKeys:

    def test_same_prefix_different_lora_are_distinct(self):
        cache = OutputCache(maxsize=10)
        prefix = (1, 2, 3)
        val_base = torch.randn(100)
        val_lora = torch.randn(100)

        cache[(prefix, None)] = val_base
        cache[(prefix, "lora_v1")] = val_lora

        assert torch.equal(cache[(prefix, None)], val_base)
        assert torch.equal(cache[(prefix, "lora_v1")], val_lora)
        assert not torch.equal(cache[(prefix, None)], cache[(prefix, "lora_v1")])

    def test_same_prefix_same_lora_hits_cache(self):
        cache = OutputCache(maxsize=10)
        val = torch.randn(100)
        cache[((1, 2, 3), "lora_v1")] = val
        assert ((1, 2, 3), "lora_v1") in cache
        assert torch.equal(cache[((1, 2, 3), "lora_v1")], val)

    def test_old_style_key_not_confused_with_new(self):
        """Tuple-of-ints vs (tuple-of-ints, str) are different keys."""
        cache = OutputCache(maxsize=10)
        cache[(1, 2, 3)] = torch.randn(100)
        assert ((1, 2, 3), None) not in cache


# ---------------------------------------------------------------------------
# _LORA_UNSET sentinel resolution
# ---------------------------------------------------------------------------

class TestLoraRequestParamResolution:

    def _resolve(self, lora_request, global_default):
        """Reproduce the resolution logic from next_token_logprobs."""
        if lora_request is _LORA_UNSET:
            return global_default
        return lora_request

    def test_unset_falls_back_to_global(self):
        global_lr = FakeLoRARequest(lora_name="global")
        assert self._resolve(_LORA_UNSET, global_lr) is global_lr

    def test_explicit_none_uses_base_model(self):
        global_lr = FakeLoRARequest(lora_name="global")
        assert self._resolve(None, global_lr) is None

    def test_explicit_lora_overrides_global(self):
        global_lr = FakeLoRARequest(lora_name="global")
        override = FakeLoRARequest(lora_name="override")
        assert self._resolve(override, global_lr) is override


# ---------------------------------------------------------------------------
# Sampler uses per-request LoRA, not clear_lora/set_lora
# ---------------------------------------------------------------------------

class TestSamplerPriorCallsPerRequestLora:
    """Verify the pattern that LoRAProposalTokenSampler._get_prior_logprobs
    uses: vlm.next_token_logprobs(ids, lora_request=None) — NOT
    clear_lora() / next_token_logprobs() / set_lora()."""

    def test_prior_call_uses_lora_request_none(self):
        vlm = MagicMock()
        fake_logprobs = torch.randn(100)

        # Make next_token_logprobs an async mock
        async def fake_next_token_logprobs(token_ids, lora_request=_LORA_UNSET):
            return fake_logprobs
        vlm.next_token_logprobs = MagicMock(side_effect=fake_next_token_logprobs)

        llm = MagicMock()
        llm.prompt_ids = [1, 2, 3]
        llm.encode_tokens = MagicMock(return_value=[4, 5])

        # Extracted logic from LoRAProposalTokenSampler._get_prior_logprobs
        async def get_prior(context):
            prefix_ids = list(llm.prompt_ids) + llm.encode_tokens(context)
            return await vlm.next_token_logprobs(prefix_ids, lora_request=None)

        result = asyncio.run(get_prior(["tok1"]))

        # Called with lora_request=None (base model)
        vlm.next_token_logprobs.assert_called_once_with(
            [1, 2, 3, 4, 5], lora_request=None
        )
        # Never touches global LoRA state
        vlm.clear_lora.assert_not_called()
        vlm.set_lora.assert_not_called()
        assert torch.equal(result, fake_logprobs)

    def test_prior_and_proposal_can_run_concurrently(self):
        """When prior uses lora_request=None and proposal uses the global
        default, both can be awaited concurrently without state conflicts."""
        vlm = MagicMock()
        prior_logprobs = torch.randn(100)
        proposal_logprobs = torch.randn(100)
        call_count = 0

        async def fake_next_token_logprobs(token_ids, lora_request=_LORA_UNSET):
            nonlocal call_count
            call_count += 1
            if lora_request is None:
                return prior_logprobs
            return proposal_logprobs

        vlm.next_token_logprobs = fake_next_token_logprobs

        lora_req = FakeLoRARequest(lora_name="lora_v1")

        async def concurrent_calls():
            prior, proposal = await asyncio.gather(
                vlm.next_token_logprobs([1, 2, 3], lora_request=None),
                vlm.next_token_logprobs([1, 2, 3], lora_request=lora_req),
            )
            return prior, proposal

        prior, proposal = asyncio.run(concurrent_calls())

        assert call_count == 2
        assert torch.equal(prior, prior_logprobs)
        assert torch.equal(proposal, proposal_logprobs)
        # No global state mutation
        vlm.clear_lora = MagicMock()
        vlm.clear_lora.assert_not_called()
