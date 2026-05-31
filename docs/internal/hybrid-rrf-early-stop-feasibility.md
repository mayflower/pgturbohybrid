# Hybrid RRF BM25 early-stop: feasibility analysis (no code shipped)

Status: **analysis only — not implemented.** Date: 2026-05-31.

Goal evaluated: online fused-top-k early termination for the BM25 branch of an
RRF hybrid scan — run dense first as a prior, then stop BM25 once the RRF
top-`finalK` can no longer change. The request asked for this to be **exact**
(RRF results unchanged) while reducing BM25 work.

Conclusion up front: **an exact *and* work-reducing RRF early-stop is not
achievable in this engine for the default `rrfK` regime.** The two goals
conflict. The feature is inherently an *approximate* recall/latency knob (which
is why the cited prior art — Elasticsearch `rank_window_size`, Qdrant prefetch
limits — is approximate, not exact). No production code was changed.

## What already exists

`PgturbohybridMaybeApplyBm25HybridBound()` (`src/pgturbohybrid_am.c`) +
`turbohybrid.bm25_hybrid_bound` (`off` / `safe` / `approx`) already implement a
coarser version of this idea: from the dense prior it computes

```
fusedFloor = denseWeight / (rrfK + finalTarget)            // valid lower bound on the kth fused score when denseCount >= finalTarget
capRank    = floor(bm25Weight / fusedFloor - rrfK)         // BM25 rank beyond which a BM25-only doc cannot enter the fused top-k
```

and caps `bm25K` to `capRank`, which feeds the BM25 engine's existing WAND /
impact kth-score threshold pruning. Its `safe` mode is **deliberately a no-op**,
with this author comment:

> The RRF rank bound alone is not exact-safe unless BM25 contributions for dense
> candidates are also preserved. Truncating BM25 to the bound can otherwise
> change the final top-k when dense candidates are tied or appear after the
> truncated lexical head. Keep safe mode conservative until the BM25 branch can
> explicitly score those dense candidates.

The task was essentially "do what that TODO defers." The analysis below shows
why that TODO cannot be discharged exactly with a work *reduction*.

## Engine reality (the constraints)

- `PgturbohybridBm25TopK()` (`src/pgturbohybrid_bm25_query.c`) **accumulates all
  touched candidates, then sorts at the end**; the BM25 `rank` is assigned
  `i+1` at output (`PgturbohybridBm25SelectFinalTopK` → output loop). It does
  *not* stream candidates to the caller in rank order.
- WAND / impact early-termination is driven by a scalar **BM25-score** threshold
  (the kth score in a top-`k` min-heap) and a remaining **score** upper bound
  (`PgturbohybridBm25IteratorUpperBound`). The merge advances by **docid**
  (pivot = smallest docid among active iterators), not by score.
- RRF fusion (`PgturbohybridScoreResults`) uses **rank**, not score:
  `fused = denseWeight/(rrfK+denseRank) + bm25Weight/(rrfK+bm25Rank)`.

## Why exact + work-reducing is infeasible

1. **RRF ignores BM25 score magnitude.** Only `bm25Rank` matters. So the
   engine's only cheap early-out signal — a BM25 *score* upper bound — is the
   wrong unit. You cannot compare a residual BM25 score to an RRF `1/(rrfK+r)`
   term; an unprocessed doc with a small score can still be `bm25Rank` 1 if few
   docs outscore it.

2. **A doc's exact `bm25Rank` needs the full ranking.** `bm25Rank(d) = 1 +
   |{docs with higher BM25 score}|`. Determining it for a doc whose score is
   below the lexical head requires computing the docs between `capRank` and
   `bm25K` — exactly the work the early-stop wants to skip. Explicitly scoring a
   dense doc gives its *score*, but not its *rank*, because the skipped tail
   docs that outrank it are unknown.

3. **The only way to save work is to drop docs, and dropping dense docs' tail
   BM25 terms is not exact.** Dropping BM25-*only* docs beyond `capRank` is exact
   (their fused `= bm25Weight/(rrfK+r) < fusedFloor <= kth`; this is what the
   existing cap proves). But the same cap also strips the BM25 term from any
   *dense* doc whose `bm25Rank > capRank`, perturbing that doc's fused score by
   up to `bm25Weight/(rrfK+capRank+1)`. At the default `rrfK = 60`,
   equal weights, that perturbation dwarfs the gap between adjacent dense docs —
   gap at dense rank `i` is `denseWeight/((rrfK+i)(rrfK+i+1)) ≈ dW/3782` at
   `i=1`, while the perturbation can be ~`dW/70`. So it readily reorders the
   top-k. **Inexact.**

4. **You cannot cheaply tell whether the unsafe case applies.** Whether any
   dense doc sits in the `(capRank, bm25K]` BM25 tail is unknowable without
   scoring it. (For high-overlap queries dense docs are usually in the head, so
   capping is *often* exact there; for low-overlap dense docs have no BM25 term
   at all; the problematic middle — dense docs with weak BM25 — can't be
   detected for free.)

The one genuinely-exact win — "skip BM25 entirely because dense provably
dominates" — requires every dense rank-gap in the top region to exceed the max
BM25 term `bm25Weight/(rrfK+1)`. Solving `dW/((rrfK+i)^2) > bW/(rrfK+1)` at
`rrfK=60`, equal weights, needs `(60+i)^2 < 61` → impossible for `i >= 1`. So it
never fires at default `rrfK`.

## If pursued anyway: approximate design sketch (for a future implementer)

Treat it as an **approximate** knob, default off, with a measured quality gate —
not as exact. The pieces:

- `turbohybrid.hybrid_rrf_early_stop = auto|off|on` (default off / off-equivalent
  auto). Active only for `fusion=RRF`, vector+tsquery present, positive weights,
  `denseCount >= finalTarget`, and **not** `REQUIRE_BM25_MATCH` (which removes
  dense-only docs and invalidates the dense floor).
- A `HybridFusionController` built after dense collection: `finalTarget`,
  `rrfK`, weights, `fusedFloor`, a `denseRankByNode` lookup (a flat
  `int32[tqNodeCount]`, `0`=not dense, else 1-based rank), and dense-touched
  tracking (`minUntouchedRank` → `maxDenseTermUntouched = dW/(rrfK+minUntouched)`).
- Thread the controller into `PgturbohybridBm25TopK` and store it on the
  accumulator (`acc->hybrid`) so strategy functions reach it without signature
  churn. Mark dense docs touched in the single choke point
  `PgturbohybridBm25AccumulatorAddTermScore`.
- Engine hook: at the WAND main-loop top (`PgturbohybridBm25ScoreBaseWand`,
  ~line 4249) and the IMPACT_OR tier-loop top, terminate when the residual is
  provably (approximately) irrelevant. Because the residual is a *score* bound,
  the honest version converts it to an *approximate* rank cap (the existing
  `capRank` logic) plus the dense-touched shrink — i.e. it is the existing
  `approx` cap made adaptive, still inexact for dense tails.
- Quality gate: benchmarks must report `overlap@finalK` / nDCG vs the exact
  (early-stop off) result across rare-/broad-/high-overlap/low-overlap query
  shapes, and fail if recall drops below a chosen threshold. Without a gate the
  knob is an unbounded recall risk.

Counters worth surfacing as stats if built: `hybrid_rrf_early_stop_enabled`,
`hybrid_bm25_candidates_pruned`, `hybrid_bm25_blocks_pruned_by_fusion`,
`hybrid_bm25_early_stop_rank`, `hybrid_kth_fused_score` (the `fusedFloor` used).
Note that a true "online fused top-k heap" / `hybrid_fusion_heap_replacements`
is **not meaningful mid-scan**, because BM25 ranks are only final at output —
the heap could only ever hold the static dense-only floor.

## Recommendation

Do **not** ship this as "exact." If the latency win is wanted, ship it as an
explicitly approximate `rank_window_size`-style knob (default off) with the
quality gate above, and rename acceptance criterion #1 from "results unchanged"
to "overlap@finalK >= threshold." Otherwise leave the exact baseline and the
existing `bm25_hybrid_bound` as they are.
