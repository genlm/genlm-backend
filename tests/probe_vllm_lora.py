"""Standalone GPU probe for per-request LoRA on the vLLM server (run manually).

Exercises the adapter surface at row grain, no genlm-control involved: the
adapter actually changing the distribution, base and adapter asks sharing one
cohort, residency split per (context, lora), rows against an independent HF
fp32 arm carrying the same adapter, rebind purging the name's residents, a rebind
landing mid-batch binding the incoming weights, and removal evicting them.

    python tests/probe_vllm_lora.py [adapter_id]
"""

import asyncio
import json
import os
import sys

import torch

ADAPTER = (
    sys.argv[1] if len(sys.argv) > 1 else "farpluto/SmolLM-135M-Instruct-Finetune-LoRA"
)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)
    if not cond:
        raise SystemExit(f"probe failed at: {name}")


def tv(a, b):
    return 0.5 * (a.float().exp().cpu() - b.float().exp().cpu()).abs().sum().item()


async def main():
    from huggingface_hub import snapshot_download

    from genlm.backend.llm.vllm import AsyncVirtualLM

    path = snapshot_download(ADAPTER)
    base = json.load(open(os.path.join(path, "adapter_config.json")))[
        "base_model_name_or_path"
    ]
    lm = AsyncVirtualLM.from_name(
        base,
        engine_opts={
            "enable_lora": True,
            "max_lora_rank": 16,
            "max_loras": 2,
            "gpu_memory_utilization": 0.4,
            "max_model_len": 512,
        },
    )
    lm.add_new_lora(path, "vk")
    prompt = lm.tokenizer.encode("The capital of France is")

    # -- 1. the adapter is actually applied per request -----------------------
    row_base, row_vk = await asyncio.gather(
        lm.next_token_logprobs(prompt),
        lm.next_token_logprobs(prompt, lora_name="vk"),
    )
    d = tv(row_base, row_vk)
    check("adapter.applied", d > 1e-3, f"TV(base, vk)={d:.3e}")

    # -- 2. mixed cohort: base and adapter asks in one gather ------------------
    # Same context under both arms; the base row must be unaffected by sharing
    # the cohort with an adapter ask. Bound it against this engine's own
    # re-ask noise floor (same context asked alone twice: duplicate birth,
    # APC-hit re-prefill -- the identical numeric class, no adapter in sight)
    # and against the adapter's effect: contamination lands near TV(base, vk).
    row_base2 = await lm.next_token_logprobs(prompt)
    alt = lm.tokenizer.encode("Ships sail the winter sea at night")
    ref1 = await lm.next_token_logprobs(alt)
    ref2 = await lm.next_token_logprobs(alt)
    noise = tv(ref1, ref2)
    d2 = tv(row_base, row_base2)
    check(
        "mixed.cohort",
        d2 < max(5 * noise, 2e-2) and d2 < d / 3,
        f"TV(base, base')={d2:.3e} noise_floor={noise:.3e} adapter={d:.3e}",
    )

    # -- 3. residency splits per (context, lora) -------------------------------
    contexts = {None: list(prompt), "vk": list(prompt)}
    for _ in range(4):
        rows = await asyncio.gather(
            *[lm.next_token_logprobs(c, lora_name=n) for n, c in contexts.items()]
        )
        for (name, ctx), row in zip(contexts.items(), rows):
            ctx.append(int(torch.argmax(row)))
    names = sorted({name for (_, name) in lm._requests.values()}, key=str)
    check("residency.split", names == [None, "vk"], f"loras in table: {names}")

    # -- 4. adapter rows vs an independent backend (HF fp32 + peft) ------------
    from genlm.backend.llm.hf import AsyncTransformer

    hf = AsyncTransformer.from_name(base, hf_opts={"torch_dtype": torch.float32})
    hf.add_new_lora(path, "vk")
    ctx = list(prompt)
    agree = 0
    for step in range(6):
        row = await lm.next_token_logprobs(ctx, lora_name="vk")
        hf_row = await hf.next_token_logprobs(ctx, lora_name="vk")
        step_tv = tv(row, hf_row)
        same = int(row.argmax()) == int(hf_row.argmax())
        agree += same
        check(
            f"hfcross.vk.{step}", step_tv < 0.25, f"TV={step_tv:.3e} argmax_same={same}"
        )
        ctx.append(int(hf_row.argmax()))
    check("hfcross.argmax", agree >= 5, f"{agree}/6 argmax agree")

    # -- 5. rebinding a name purges its residents ------------------------------
    lm.add_new_lora(path, "vk")
    await lm.next_token_logprobs(prompt, lora_name="vk")  # serves under new binding
    stale = [rid for rid, (_, name) in lm._requests.items() if name == "vk"]
    check(
        "rebind.serves", len(stale) == 1, f"vk residents after rebind+ask: {len(stale)}"
    )

    # -- 6. a rebind mid-batch binds the incoming weights ----------------------
    # The ask sits in the batch while the rebind lands. Bound at ask time it would
    # birth under the outgoing id and every extension would ride old-weight KV.
    fresh = list(prompt) + [lm.tokenizer.encode(" very")[-1]]
    task = asyncio.ensure_future(lm.next_token_logprobs(fresh, lora_name="vk"))
    await asyncio.sleep(0)  # the ask is queued and holding the batch
    lm.add_new_lora(path, "vk")
    want_id = lm._lora_requests["vk"].lora_int_id
    await task
    rid = next(r for r, (_, name) in lm._requests.items() if name == "vk")
    served_id = lm._sched.requests[rid].lora_request.lora_int_id
    check(
        "rebind.midbatch",
        served_id == want_id,
        f"served under id={served_id}, incoming binding is {want_id}",
    )

    # -- 7. removal evicts the name -------------------------------------------
    lm.remove_lora("vk")
    try:
        await lm.next_token_logprobs(prompt, lora_name="vk")
        removed = False
    except ValueError:
        removed = True
    check("remove.raises", removed)
    row_after = await lm.next_token_logprobs(list(prompt) + [11])
    check("remove.base_alive", bool(row_after.isfinite().any()))
    vk_left = [rid for rid, (_, name) in lm._requests.items() if name == "vk"]
    check("remove.purged", not vk_left, f"vk residents: {vk_left}")

    print("ALL PROBES PASSED", flush=True)
    lm.cleanup()


async def _guarded():
    await asyncio.wait_for(main(), timeout=900)


if __name__ == "__main__":
    asyncio.run(_guarded())
