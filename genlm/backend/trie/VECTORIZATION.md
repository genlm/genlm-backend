# Trie construction is slow (~7 s for a 128k-token vocab) — vectorization notes

`TokenCharacterTrie.__init__` (`base.py`) + `ParallelTokenCharacterTrie._build_reachability_matrix`
(`parallel.py`) take **~7 s** to build over a Llama-3.1 vocab (128,255 tokens). That is the
construction cost; once built the trie is fast. It matters because callers that build a trie
per item (e.g. genlm-control's `TrieSetSampler`/`eager_token_sampler`, constructed once per
question) pay it `N` times — an N-question run spends `~7·N` s of single-core CPU with the GPU
idle before any generation. (bomk works around the *repetition* by caching the trie across
questions — see `bomk/trie_cache.py` — but the **single build is itself far slower than it
should be**: a 128k-entry byte trie should build in well under a second. This note is about
fixing the build itself, which benefits every caller.)

## Where the time goes

Both phases are unvectorized Python loops over the whole vocabulary.

1. **Byte-trie insert** — `TokenCharacterTrie.__init__` (`base.py:24-70`): a Python loop over
   all 128k tokens, and an inner loop over each token's bytes, growing dict-of-dicts
   `self.children`. Then `_order`/`_order_full` (DFS over all nodes), `_rename`, and the
   `node2prefix` pass (another loop over every node/edge). All pure-Python, dominated by dict
   churn and per-byte Python overhead.
2. **Reachability matrix** — `ParallelTokenCharacterTrie._build_reachability_matrix`
   (`parallel.py:33-60`): builds a parent **dict** (`_build_parent_map`, a Python loop over all
   nodes × children), then for **each leaf** runs a Python `while current in parent` walk up to
   the root, appending `(row, col)` per ancestor. That is `O(#leaves × avg_depth)` ≈ 128k × ~8
   ≈ ~1M Python-level iterations and list appends, then one `torch.sparse_coo_tensor` over ~1M
   entries. This per-leaf ancestor walk is the largest single cost.

`numba` is already a dependency (`base.py` imports it and uses `numba.typed.List`), so the hot
loops are JIT-able without new deps.

## Vectorization plan

### Reachability matrix (biggest win, fully vectorizable)
The per-leaf walk-to-root is an ancestor enumeration on a tree — do it in tensors, not a Python
`while` loop:

- Replace the parent **dict** with a parent **array** `parent[node] -> node` (root = self), an
  `int64` tensor of length `n_nodes`. Build it once by scattering child→node from `self.jump`
  (already an array-of-arrays) instead of the Python double loop.
- Enumerate ancestors by **pointer-doubling / fixed-depth iteration**: start `cur = leaf_nodes`
  (a `[L]` tensor of all leaf node ids); for `d in range(max_depth)` append `(arange(L), cur)`
  to the COO and set `cur = parent[cur]`, stopping rows that have reached the root (mask
  `cur != parent[cur]`). `max_depth` = trie depth ≈ longest token in bytes (small, ~tens). That
  is `O(max_depth)` vectorized torch steps instead of `O(#leaves × depth)` Python iterations,
  and the COO `rows/cols` come out as tensors directly — no Python lists, no per-entry append.
- Equivalent alternative: process nodes in reverse topological order (`self.ordering` already
  exists) and propagate leaf membership up with a scatter-add; same complexity, also tensorized.

### Byte-trie insert (`base.py`)
Trie insertion is order-dependent, but the cost is Python overhead, not algorithm:
- JIT the insert loop with `numba` over an integer view of the bytes (tokens are byte strings;
  pass a flat `int8` buffer + offsets), building `children` as typed arrays rather than a
  dict-of-dicts. Or build the trie by **sorting** the (token, byte-position) pairs and computing
  shared-prefix boundaries — a radix/LCP construction is vectorizable and avoids per-byte dict
  lookups.
- `node2prefix`, `_order`, `_rename` are O(nodes) passes that JIT cleanly once `children` is in
  array form. (`node2prefix` materializes a full prefix list per node — only build it if a
  consumer needs it; it can be large and is pure overhead otherwise.)

### Quick wins regardless
- `_build_parent_map` dict → array (above) removes a Python loop and makes downstream
  vectorization possible.
- Avoid rebuilding per caller: a vocab-keyed cache at `load_trie`/`load_async_trie`
  (`genlm/control/util.py`) keyed on the model's `token_maps`/vocab identity would make the
  build a one-time cost for everyone (bomk does this externally today via a monkey patch).

## Validating a rewrite
The trie has a cross-check already (`base.py` keeps the reference structure); a vectorized
`_build_reachability_matrix` must produce a sparse `M` whose dense form is identical to the
current one (`M[i,j]=1` iff node `j` is leaf `i` or an ancestor of it). Compare `M.to_dense()`
against the current implementation on a small vocab, and assert `weight_sum`/`weight_max`
outputs are unchanged on a random weight vector before/after.
