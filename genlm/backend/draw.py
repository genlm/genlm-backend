"""Batched reductions over next-token log-weight rows.

Concurrent asks on one event loop meet in a window and run as one stacked
reduction per kind and (device, V) group, with one host readback per group. Rows
stay on their device; the readback is the designed crossing.
"""

import asyncio
import weakref
from collections import defaultdict

import torch

from genlm.backend.batching import (
    batch_abandoned,
    batch_stats,
    fail_futures,
    join_batch,
)

# --- token-picker family ---
# Each maps a log-weight tensor -> drawn index over dim=-1 (1-D draw or batched [N, V]),
# at the row's own dtype. They draw from the global torch RNG, so `torch.manual_seed`
# is what makes a run's draws reproducible.


def gumbel_max(logps):
    """Argmax of `logps + Gumbel noise`; the default picker."""
    g = -torch.log(-torch.log(torch.rand_like(logps)))
    return (logps + g).argmax(dim=-1)


def multinomial(logps):
    """Categorical draw over the last dim (scalar for `[V]`, `[N]` for `[N, V]`)."""
    p = (logps - torch.logsumexp(logps, dim=-1, keepdim=True)).exp()
    # A row with no support is NaN here; it draws arbitrarily.
    return torch.multinomial(p.nan_to_num(nan=1.0), 1).squeeze(-1)


def inverse_cdf(logps):
    """Single-uniform inverse-CDF draw over the last dim (scalar for `[V]`, `[N]` for
    `[N, V]`); one uniform per row, on `logps`'s device."""
    p = (logps - torch.logsumexp(logps, dim=-1, keepdim=True)).exp()
    cdf = p.nan_to_num(nan=1.0).cumsum(
        dim=-1
    )  # a row with no support draws arbitrarily
    u = torch.rand((*cdf.shape[:-1], 1), dtype=cdf.dtype, device=cdf.device)
    return torch.searchsorted(cdf, u).squeeze(-1).clamp_(max=cdf.shape[-1] - 1)


DRAW_METHODS = {
    "gumbel_max": gumbel_max,
    "multinomial": multinomial,
    "inverse_cdf": inverse_cdf,
}
# Process-wide picker for the draw window; set it via `set_draw_method`.
_picker = gumbel_max


def set_draw_method(method):
    """Set the token picker used by the draw window, process-wide.

    Args:
        method (str | callable): A name in `DRAW_METHODS`, or a custom
            `(logps_tensor) -> index` callable.
    """
    global _picker
    _picker = DRAW_METHODS[method] if isinstance(method, str) else method


_WINDOW = weakref.WeakKeyDictionary()  # event loop -> _Batch


async def _ask(kind, weights, *args):
    future = asyncio.get_running_loop().create_future()
    batch = await join_batch(_WINDOW, [(kind, weights, *args, future)])
    if batch is not None:
        _flush(batch)
    return await future


async def draw_from(weights, target=None):
    """Draw an index from a `[V]` log-weight row and weigh it.

    Args:
        weights (torch.Tensor | np.ndarray): Log-weights over the row's vocabulary.
        target (torch.Tensor | np.ndarray, optional): A second row over the same
            vocabulary, making this an importance draw: `weights` is the proposal
            and the index is weighed under `target`.

    Returns:
        (tuple): `(index, logw, logp)`. `logw` is the row's log-normalizer, or
            `target[index] - logp` with a target. A row with no support draws an
            arbitrary index at `logp = logw = -inf`.
    """
    return await _ask("draw", weights, target)


async def logsumexp_from(weights):
    """The log-normalizer of a `[V]` log-weight row.

    Args:
        weights (torch.Tensor | np.ndarray): Log-weights over the row's vocabulary.

    Returns:
        (float): `logsumexp(weights)`.
    """
    return await _ask("logsumexp", weights)


async def logp_at(weights, index):
    """One entry of a `[V]` log-weight row.

    Args:
        weights (torch.Tensor | np.ndarray): Log-weights over the row's vocabulary.
        index (int): The entry to read.

    Returns:
        (float): `weights[index]`.
    """
    return await _ask("at", weights, index)


def _stack(entries):
    return torch.stack([torch.as_tensor(e[1]) for e in entries])


def _reduce_draw(entries):
    rows = _stack(entries)
    logZ = torch.logsumexp(rows, dim=-1)
    # A row with no support stays -inf throughout: the pickers draw it arbitrarily
    # and its draw comes back at logp = -inf.
    empty = logZ.isneginf()
    logps = rows.sub_(logZ.masked_fill(empty, 0.0).unsqueeze(-1))
    idx = _picker(logps)
    logp = logps.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    logp = logp.masked_fill(empty, float("-inf"))
    logw = logZ
    targeted = [k for k, e in enumerate(entries) if e[2] is not None]
    if targeted:
        t_rows = torch.stack([torch.as_tensor(entries[k][2]) for k in targeted])
        t_logp = logp[targeted]
        t_logw = t_rows.gather(-1, idx[targeted].unsqueeze(-1)).squeeze(-1) - t_logp
        # Nothing to propose from is a dead draw, not an infinite weight.
        logw[targeted] = t_logw.masked_fill(t_logp.isneginf(), float("-inf"))
    # The ids ride the values' readback; float32 and up hold any vocabulary index
    # exactly, and the row's own precision is kept.
    dt = torch.promote_types(rows.dtype, torch.float32)
    return torch.stack([idx.to(dt), logw.to(dt), logp.to(dt)])


def _reduce_logsumexp(entries):
    return torch.logsumexp(_stack(entries), dim=-1)


def _reduce_at(entries):
    return torch.stack([torch.as_tensor(e[1])[e[2]] for e in entries])


_REDUCE = {"draw": _reduce_draw, "logsumexp": _reduce_logsumexp, "at": _reduce_at}


def _flush(queue):
    """Resolve a window's asks with one reduction and one host readback per kind and
    (backend, device, V) group; scalar reads group across V. Every future is
    resolved, with its value or with a failure."""
    groups = defaultdict(list)
    for entry in queue:
        w = entry[1]
        where = ("torch", w.device) if torch.is_tensor(w) else ("np",)
        width = () if entry[0] == "at" else (w.shape[-1],)
        groups[(entry[0], *where, *width)].append(entry)
    for key, entries in groups.items():
        kind = key[0]
        batch_stats[(kind, len(entries))] += 1
        try:
            vals = _REDUCE[kind](entries).tolist()
        except Exception as exc:
            fail_futures(entries, exc)
            continue
        except BaseException as exc:
            # This flush is unwinding, so nothing else will resolve what it still
            # owes. Groups already resolved are skipped by `fail_futures`.
            for rest in groups.values():
                fail_futures(rest, batch_abandoned(exc))
            raise
        if kind == "draw":
            ids, logws, logps = vals
            vals = [(int(i), w, p) for i, w, p in zip(ids, logws, logps)]
        for val, entry in zip(vals, entries):
            if not entry[-1].done():
                entry[-1].set_result(val)
