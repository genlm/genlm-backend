"""Standalone GPU probe for the scheduler-native vLLM server (run manually).

Exercises the mechanism end to end, no genlm-control involved:
population decode via resident appends, ragged cadences (idle residents skip
frames), one-shot scores interleaved mid-run, placeholder discipline under
async scheduling, a window wider than the residency cap, re-asks of
already-served contexts, release, and row consistency between the resident
decode path and a fresh prefill at the same context.

    python tests/probe_vllm_server.py [model_name]
"""

import asyncio
import sys

import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM-135M"


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)
    if not cond:
        raise SystemExit(f"probe failed at: {name}")


def table_matches_engine(lm):
    """The residency table and the engine must hold exactly the same requests
    whenever the engine owes nothing."""
    ours = sorted(lm._requests)
    theirs = sorted(r for r in lm._sched.requests if r.startswith("resident-"))
    return ours == theirs, f"table={len(ours)} engine={len(theirs)}"


async def main():
    from genlm.backend.llm.vllm import AsyncVirtualLM

    lm = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.5,
            "max_num_seqs": 24,  # residency cap 16: the over-cap window below
        },
    )
    sched = lm._sched
    prompt = lm.tokenizer.encode("The quick brown fox")
    n_rows = 8

    # -- 1. population decode: N rows advancing concurrently ------------------
    contexts = [list(prompt) for _ in range(n_rows)]
    for step in range(10):
        rows = await asyncio.gather(*[lm.next_token_logprobs(c) for c in contexts])
        if step == 0:
            check("rows.shape", rows[0].dim() == 1 and rows[0].shape[-1] > 1000)
        for i, row in enumerate(rows):
            # diverge the rows: rank-i token among top-32
            tok = int(torch.topk(row, 32).indices[i * 3 % 32])
            contexts[i].append(tok)
    check("population.decode", True, f"{n_rows} rows x 10 steps")
    check("population.table", *table_matches_engine(lm))

    # -- 2. placeholder discipline: all resident requests at 0 -------------------
    bad = [
        r.request_id
        for r in sched.requests.values()
        if r.request_id.startswith("resident-") and r.num_output_placeholders != 0
    ]
    check("placeholders.zero", not bad, f"nonzero={bad}")

    # -- 3. ragged cadence: half advance, half idle ----------------------------
    for step in range(5):
        rows = await asyncio.gather(
            *[lm.next_token_logprobs(c) for c in contexts[: n_rows // 2]]
        )
        for i, row in enumerate(rows):
            contexts[i].append(int(torch.argmax(row)))
    check("ragged.advance", True, "half advanced 5 steps, half idle")

    # idle rows still resident and resumable
    rows = await asyncio.gather(*[lm.next_token_logprobs(c) for c in contexts])
    check("ragged.resume", all(r is not None for r in rows))

    # -- 4. one-shot scores interleaved with decode ---------------------------
    async def one_shots():
        alt = lm.tokenizer.encode("A completely different prompt about ships")
        return await asyncio.gather(
            *[lm.next_token_logprobs(alt + [i]) for i in range(4)]
        )

    async def keep_decoding():
        for _ in range(4):
            rws = await asyncio.gather(
                *[lm.next_token_logprobs(c) for c in contexts[:4]]
            )
            for i, row in enumerate(rws):
                contexts[i].append(int(torch.argmax(row)))

    scores, _ = await asyncio.gather(one_shots(), keep_decoding())
    check("oneshot.interleave", all(s.isfinite().any() for s in scores))

    # -- 4b. decode rows vs an independent backend (HF fp32) -------------------
    # The resident-decode path checked against a computation that shares NO
    # engine state — in particular no prefix-cache blocks, which a same-engine
    # comparison can recycle, blessing its own corruption.
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf.eval()
    ctx = list(prompt)
    agree = 0
    for step in range(6):
        row = await lm.next_token_logprobs(ctx)
        with torch.no_grad():
            hf_row = torch.log_softmax(
                hf(torch.tensor([ctx])).logits[0, -1].float(), dim=-1
            )
        tv_step = 0.5 * (row.float().exp().cpu() - hf_row.exp()).abs().sum().item()
        same = int(row.argmax()) == int(hf_row.argmax())
        agree += same
        check(
            f"hfcross.row.{step}",
            tv_step < 0.25,
            f"TV={tv_step:.3e} argmax_same={same}",
        )
        ctx.append(int(hf_row.argmax()))
    check("hfcross.argmax", agree >= 5, f"{agree}/6 argmax agree")

    # -- 5. one window wider than the residency cap ---------------------------
    # Every ask must still get its row: newborns owe rows, so eviction may not
    # take them, and the table trims back under the cap afterward.
    wide = [list(prompt) + [i + 1] for i in range(lm._max_residents + 4)]
    rows = await asyncio.gather(*[lm.next_token_logprobs(c) for c in wide])
    check("overcap.rows", all(r.isfinite().any() for r in rows), f"{len(wide)} asks")
    await asyncio.gather(*[lm.next_token_logprobs(c) for c in contexts[:2]])
    check(
        "overcap.trimmed",
        len(lm._requests) <= lm._max_residents,
        f"table={len(lm._requests)} cap={lm._max_residents}",
    )

    # -- 6. re-asks of already-served contexts ---------------------------------
    # An ask at an exact-resident content births a duplicate (a resident only
    # serves rows through extension), and the newcomer takes the content index;
    # displaced requests stay ordinary table rows — nothing may leak past the
    # table.
    base = list(contexts[0])
    row_a = await lm.next_token_logprobs(base)  # duplicate birth at base
    step = base + [int(torch.argmax(row_a))]
    await lm.next_token_logprobs(step)  # extends base's index holder
    await lm.next_token_logprobs(base)  # another duplicate takes the index
    await lm.next_token_logprobs(step)  # extension displaces step's holder
    check("reask.table", *table_matches_engine(lm))

    # -- 7. row consistency: resident decode row vs fresh prefill row ---------
    probe_ctx = list(contexts[0])
    row_resident = await lm.next_token_logprobs(probe_ctx)
    await lm.release_all()
    check("release.table", len(lm._requests) == 0)
    row_fresh = await lm.next_token_logprobs(probe_ctx)
    # Compare the distributions, not their log tails: a column at p=1e-4 can move
    # 0.1 in log space while carrying no mass. Total variation is the honest
    # summary of the warm-KV-vs-reprefill residual.
    tv = 0.5 * (row_resident.exp() - row_fresh.exp()).abs().sum().item()
    full_diff = (row_resident - row_fresh).abs().max().item()
    same_top = int(torch.argmax(row_resident)) == int(torch.argmax(row_fresh))
    check(
        "rows.consistent",
        same_top and tv < 1e-2,
        f"TV={tv:.2e} argmax_same={same_top} full max|logp diff|={full_diff:.2e}",
    )

    # -- 8. dead requests actually left the engine -----------------------------
    await lm.release_all()
    live = [r for r in sched.requests if r.startswith("resident-")]
    check("release.engine", not live, f"live={live}")

    print("ALL PROBES PASSED", flush=True)
    lm.cleanup()


async def _guarded():
    await asyncio.wait_for(main(), timeout=600)


if __name__ == "__main__":
    asyncio.run(_guarded())
