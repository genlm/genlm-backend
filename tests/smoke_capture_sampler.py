"""Smoke tests for CaptureSampler against the LoRA-push phantom-apply bug.

Run one test per invocation:
  python tests/smoke_capture_sampler.py --test t1 --adapter PATH
  python tests/smoke_capture_sampler.py --test t5     # no adapter needed
"""
import argparse
import os
import sys

import torch


MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def make_llm(enable_lora=False):
    from genlm.backend.llm.vllm import AsyncVirtualLM

    opts = {"max_model_len": 2048, "gpu_memory_utilization": 0.6}
    if enable_lora:
        opts.update({"enable_lora": True, "max_loras": 4, "max_lora_rank": 64})
    return AsyncVirtualLM.from_name(MODEL, engine_opts=opts)


def prompts(n, seed=0):
    return [[151643, 1000 + seed * 10000 + i, 2000 + i, 3000 + i] for i in range(n)]


def t1_phantom_probe(adapter):
    """Probe what req_ids look like across forwards, especially across set_lora cycles."""
    os.environ["GENLM_CAPTURE_DIAG"] = "1"
    llm = make_llm(enable_lora=True)
    llm.add_new_lora(adapter, lora_name="t1")

    print("=== no LoRA, bs=8 ===", flush=True)
    lp = llm.batch_next_token_logprobs_sync(prompts(8, seed=1))
    print(f"shape={tuple(lp.shape)}", flush=True)

    print("=== set_lora, bs=8 (first set_lora triggers warmup) ===", flush=True)
    llm.set_lora(adapter, lora_name="t1")
    lp = llm.batch_next_token_logprobs_sync(prompts(8, seed=2))
    print(f"shape={tuple(lp.shape)}", flush=True)

    print("=== second set_lora (same adapter), bs=20 ===", flush=True)
    llm.set_lora(adapter, lora_name="t1")
    lp = llm.batch_next_token_logprobs_sync(prompts(20, seed=3))
    print(f"shape={tuple(lp.shape)}", flush=True)

    print(
        f"\n=== _captured residue after taking everything: "
        f"{len(llm.capture_sampler._captured)} orphans ===",
        flush=True,
    )
    print("T1 OK", flush=True)


def t2_assertion(adapter):
    """The pre-fix crash scenario: bs=20 with LoRA push between every call."""
    llm = make_llm(enable_lora=True)
    llm.add_new_lora(adapter, lora_name="t2")
    llm.set_lora(adapter, lora_name="t2")
    for cycle in range(20):
        lp = llm.batch_next_token_logprobs_sync(prompts(20, seed=cycle))
        assert lp.shape[0] == 20, f"cycle {cycle}: got {tuple(lp.shape)}"
        llm.set_lora(adapter, lora_name="t2")
    print("T2 OK (20 cycles, bs=20, no shape mismatch)", flush=True)


def t3_correctness(adapter):
    """No-LoRA captures must be invariant across set_lora cycles when we revert."""
    llm = make_llm(enable_lora=True)
    llm.add_new_lora(adapter, lora_name="t3")
    p = prompts(8, seed=42)

    lp_a = llm.batch_next_token_logprobs_sync(p)
    llm.set_lora(adapter, lora_name="t3")  # invalidate graph
    llm.clear_lora()                       # revert to no-LoRA
    lp_b = llm.batch_next_token_logprobs_sync(p)

    diff = (lp_a - lp_b).abs().max().item()
    print(f"max|Δlogprob| no-LoRA vs no-LoRA-after-cycle = {diff:.3e}", flush=True)
    assert diff < 1e-3, f"capture corrupted by set_lora cycle (max diff {diff})"

    # Distinct LoRA passes should also be self-consistent
    llm.set_lora(adapter, lora_name="t3")
    lp_c = llm.batch_next_token_logprobs_sync(p)
    lp_d = llm.batch_next_token_logprobs_sync(p)
    diff_cd = (lp_c - lp_d).abs().max().item()
    print(f"max|Δlogprob| LoRA call-1 vs call-2 = {diff_cd:.3e}", flush=True)
    assert diff_cd < 1e-3, f"LoRA captures inconsistent (max diff {diff_cd})"
    print("T3 OK", flush=True)


def t5_sync_paths():
    """Regression: single + batch sync paths, no LoRA, probs sum to 1, agree."""
    llm = make_llm(enable_lora=False)
    p = prompts(8, seed=7)
    lp_single = llm.next_token_logprobs_sync(p[0])
    assert lp_single.dim() == 1
    s = lp_single.exp().sum().item()
    assert abs(s - 1.0) < 1e-2, f"single prob sum {s}"
    lp_batch = llm.batch_next_token_logprobs_sync(p)
    assert lp_batch.shape[0] == 8
    sums = lp_batch.exp().sum(-1)
    assert (sums - 1.0).abs().max().item() < 1e-2, f"batch prob sums {sums}"
    diff = (lp_single - lp_batch[0]).abs().max().item()
    assert diff < 1e-3, f"single vs batch[0] differ by {diff}"
    print(f"T5 OK (prob_sum≈{s:.4f}, single==batch[0] within {diff:.2e})", flush=True)


def t6_batch_sweep(adapter):
    """Batch sizes across the spectrum with LoRA cycle between each."""
    llm = make_llm(enable_lora=True)
    llm.add_new_lora(adapter, lora_name="t6")
    llm.set_lora(adapter, lora_name="t6")
    for N in [1, 8, 20, 64, 128]:
        lp = llm.batch_next_token_logprobs_sync(prompts(N, seed=N))
        assert lp.shape[0] == N, f"N={N}: got {tuple(lp.shape)}"
        llm.set_lora(adapter, lora_name="t6")
    print("T6 OK (N ∈ {1,8,20,64,128})", flush=True)


def t7_memory(adapter):
    """500 LoRA-push cycles, watch _captured dict size."""
    llm = make_llm(enable_lora=True)
    llm.add_new_lora(adapter, lora_name="t7")
    llm.set_lora(adapter, lora_name="t7")
    sizes = []
    for step in range(1, 501):
        llm.batch_next_token_logprobs_sync(prompts(8, seed=step))
        llm.set_lora(adapter, lora_name="t7")
        if step % 50 == 0:
            n = len(llm.capture_sampler._captured)
            sizes.append((step, n))
            print(f"step {step}: _captured size = {n}", flush=True)
    final = sizes[-1][1]
    assert final < 200, f"_captured grew unboundedly: final size {final}"
    print(f"T7 OK (final _captured size {final})", flush=True)


TESTS = {
    "t1": (t1_phantom_probe, True),
    "t2": (t2_assertion, True),
    "t3": (t3_correctness, True),
    "t5": (t5_sync_paths, False),
    "t6": (t6_batch_sweep, True),
    "t7": (t7_memory, True),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=TESTS.keys())
    ap.add_argument("--adapter", default="")
    args = ap.parse_args()
    fn, needs_adapter = TESTS[args.test]
    if needs_adapter and not args.adapter:
        print(f"test {args.test} requires --adapter PATH", file=sys.stderr)
        sys.exit(2)
    if needs_adapter:
        fn(args.adapter)
    else:
        fn()
