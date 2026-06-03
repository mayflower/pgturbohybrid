# AMD64/x86_64 retrieval-feature validation report

Validation of the current retrieval features and named profiles on AMD64/x86_64
(Intel Xeon Platinum 8375C; AVX2 + AVX-512-VNNI, compile + runtime). All runs
use deterministic synthetic corpora at 1536 dims; results are reproducible.
**No runtime code or profile defaults were changed to produce this report.**

Method note: each section drives the feature/profile under `turbohybrid.simd`
on/off (and `dense_u8_batch_x4` on/off where relevant), compares result IDs /
recall / overlap and the relevant `turbohybrid_last_scan_stats()` fields, and
verifies SIMD/scalar equivalence. Exact KNN ground truth uses a seqscan over the
pgvector `<=>` cosine operator; lexical ground truth uses `ts_rank_cd`.

## 1. Build & test
- `make installcheck` (default AMD64 SIMD): **10/10 tests pass** (incl. `u8split`,
  `querysplit`, `codebook`, `nibble_guard`, `x4_safety`, `rescore`).
- With `turbohybrid.simd=off`: 9/10 — the only diff is `pgturbohybrid_query`
  asserting `current_setting('turbohybrid.simd')='on'` (a profile-defaults
  self-check the global override trips), not a scoring failure.

## 2. Dense scoring kernel parity (scalar / AVX2 / AVX-512-VNNI / x4)
- Per-kernel parity via `turbohybrid_scorer_distances` across dims
  100/128/130/250/384/1024/1500/1536/3072: the u8 kernel's **scalar == AVX2 ==
  AVX-512-VNNI == x4 batch, bit-exact**, every case.
- Deterministic index, top-20 result IDs identical across {simd on (avx512),
  simd on (avx2), simd off (scalar), x4 off}. `graph_scored_codes` constant;
  `scalar_scored_codes` non-zero only under `simd=off`.
- Kernel dispatch by bit width: 1→`avx2_lut_gather`, 2→`signed_split_avx512vnni`,
  4→`unsigned_split_avx512vnni` (u8 split, **4-bit-only**, dims ≥ 1024), 8→`scalar_8bit`.
- AVX-512 fallback: forcing `dense_graph_avx512vnni=off` routes to AVX2,
  parity-clean — the path a non-AVX-512 host takes; no crash or misdispatch.

## 3. entry_sidecar_strategy (hash / farthest_code / level_covering / hybrid_level_covering)
Indexes with identical graph params, varying only the sidecar setting.
- Recall **preserved** across all strategies and vs `entry_sidecar=off`
  (dense 1.000, ambiguous 0.700). Strategies engage distinctly
  (`sidecar_selected` differs: hash 8, level_covering 5, farthest_code 2, hybrid 1).
- SIMD on/off: recall & result-ID **sets** identical; differences only tie-order
  reshuffles on the ambiguous dense case.

## 4. dense_uncertainty_retry (off / auto / on)
- **off** never retries (0/16). **on** = `forced`, bounded `passes=2` (1 retry,
  `max_passes=1`), and safely skips degenerate filtered scans. **auto** fires only
  on a valid reason: `flat_top10` (top-10 gap ≤ `min_gap`=0.03) and `heap_reordered`
  (heap-band reorder) surfaced; `residual_reordered`/`sidecar_unused`/
  `payload_underfilled` correctly did **not** fire when their preconditions were
  unmet.
- **SIMD on vs off: identical** retry decision (triggered/reason/passes) and
  recall — 0 diffs, only tie-order reshuffles.
- Behavioral note: in 1536-d the top-10 gap is ≤ `min_gap` on most queries, so
  `flat_top10` (high priority) tends to mask lower-priority reasons in `auto`.
- Retry **reported no gain** here (widened ef/scored codes, recall unchanged) —
  ceilings were quantization/tie/fusion-bound, not search-reach-bound.

## 5. Calibrated hybrid fusion (rrf / weighted / fast_weighted / calibrated)
- `fusion=>'calibrated'` works; `turbohybrid_query_out` prints `calibrated`.
- Shape detection sets alpha: rare/lexical → `rare_identifier` (α 0.35),
  broad NL → `broad_natural_language` (α 0.70), mixed → `mixed` (α 0.50).
- **Improves** on rare/mixed: rare-identifier target rank 1 vs rrf 2; mixed
  overlap 1.00 vs rrf 0.50. `calibrated_fusion_enabled=false` for the other three
  modes (they are unaffected).
- **SIMD on vs off**: fusion logic fields (enabled/shape/alpha/bonus/norms/
  candidates) **bit-identical**; only dense candidate ranks reshuffle.

## 6. BM25 heap-tsvector rerank (off / topk / band / auto)
- Uses `ts_rank_cd` (proximity-aware). Bounds: topk = `final_k`, band/auto =
  `final_k × multiplier` (4). **auto is phrase-gated** (engages only when the
  tsquery has a phrase; skipped the bag query).
- **off** never reranks. Phrase/proximity overlap **preserved** (`topk_changed=false`;
  pre-rerank top-k already correct on this corpus).
- **SIMD on vs off**: `rerank_count`/`topk_changed`/overlap/`bm25_candidates`
  identical; only dense ranks reshuffle. p95 +~0.1 ms for band.

## 7. final_diversity = group_payload
- **off** preserves relevance order. **group_payload** reduces duplicate groups
  when the int4 INCLUDE slot is valid — scales with pool: pool×3 → dup 9→7,
  pool×10 → **dup 9→0**, 10 distinct documents, 9 suppressed.
- **Invalid slot falls back safely** (slot beyond payload count, or −1, or pool×1
  → no-op, behaves like off, no crash).
- lambda 1.0 = pure relevance (≡ off), lower diversifies. **SIMD on/off
  identical** (diversity is a post-ranking step over payload slots; `final_diversity_us`
  ≈ 10 µs, heap-free).

## 8. quantization_bits = 8
- 8-bit accepted by reloption validation; `dense_scorer=scalar_8bit`,
  `graph_simd_scored_codes=0`, **zero u8/signed-split kernel calls** under all
  simd/x4 combos — **never misdispatches into the 4-bit-only SIMD kernels**
  (gated by `bits==4` in `…TqUseU8Split`). 1/2/4-bit kernels unchanged.
- Recall/latency/size tradeoff (dense, exact_storage=off): 4-bit 0.95 recall /
  ~730 µs / 5.6 MB vs 8-bit **1.00** recall / **~2160 µs** / 9.2 MB. 8-bit is
  scalar-only (no 8-bit SIMD kernel) → ~3× scan latency. `exact_storage=on`
  reaches 1.00 at either width.
- Insert into an 8-bit index is immediately searchable; VACUUM/ANALYZE succeed.

## 9. Profile benchmark (latency / balanced / matched_recall / high_recall / quality / debug)
Build reloptions: latency m16/efc128/efs64/ovs4/fast → balanced 192/96/4/heuristic
→ matched_recall 192/128/8/heuristic → high_recall 256/192/12/fast → quality
256/192/8/heuristic → debug 256/192/4/heuristic (retry=auto, eff_ef 200→300). All
4-bit, exact_storage=off. New features (retry/bm25-rerank/final-diversity) are
**off by default** in every profile except debug (retry=auto). `residual_rerank_mode=calibrated`
is the GUC default but inert unless the index sets `residual_rerank=on`.

Recall@10 / latency (simd on):
| profile | easy recall | hard recall | hard p50 (ms) | hard p95 (ms) |
|---|---:|---:|---:|---:|
| latency | 1.00 | 0.60 | 0.59 | 0.62 |
| balanced | 1.00 | 0.60 | 0.60 | 0.63 |
| matched_recall | 1.00 | 0.60 | 0.60 | 0.65 |
| high_recall | 1.00 | **1.00** | 1.40 | 1.75 |
| quality | 1.00 | 0.60 | 0.63 | 0.66 |
| debug | 1.00 | 0.60 | 0.94 | 0.99 |

- **Best latency:** `latency` (p50 0.55–0.59 ms).
- **Best high-recall:** `high_recall` — the only profile to recover full recall on
  the hard out-of-distribution query (0.60 → 1.00) and payload_filter (0.90 →
  1.00), at ~2.4× latency; the gain comes from wider `efs=192`/`ovs=12`.
- **Best matched-recall:** `matched_recall` (purpose-built for HNSW-matched
  recall); on this synthetic it sits at latency-class recall — its edge should be
  confirmed on real HNSW-comparison data.
- **SIMD parity:** recall/dup-group diffs = 0 across all profiles/cases (both
  corpora); native cache used (`auto_fits_max_mb`), scorer `unsigned_split_avx512vnni`.
- **No AMD64-specific profile behavior** observed — all profiles are
  SIMD/scalar-equivalent (consistent with §2's bit-exact kernels).

## Recommendations (not code changes)
1. The effective recall lever is `efs`/oversampling, not the new features —
   `high_recall`'s wider search recovered 0.60→1.00 where retry recovered nothing.
2. Consider `dense_uncertainty_retry=auto` for `high_recall`/`quality` (debug
   already enables it) **only if** real-data tests show recall gain at acceptable
   p95 — it widens ef on uncertain top-k.
3. For hybrid-heavy workloads, consider `bm25_heap_tsvector_rerank=auto` in
   `quality` (phrase-gated, bounded, SIMD-neutral).
4. Document that `residual_rerank_mode=calibrated` is a global default but inert
   without `residual_rerank=on`, so it isn't mistaken for an active rescore.
5. 8-bit is correct but scalar-only (~3× scan latency). A dedicated 8-bit SIMD
   kernel would remove the penalty; until then prefer 4-bit + rescore
   (heap-band / exact_storage) as the speed/recall sweet spot.
