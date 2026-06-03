# AMD64 current-state check

Pre-validation audit of `main` on AMD64/x86_64. Read-only: this note adds no
runtime code, GUCs, or reloptions. It records what the **current source**
actually contains (not earlier prompt assumptions).

- HEAD audited: `0a94f1d` (feat: add quality tuning controls).
- Build: `make all` succeeds; `pgturbohybrid.so` is produced. No code changes
  were made, so existing regression tests are unaffected.

## "AMD64-specific" defined

The only x86/AMD64-specific code in the extension is the SIMD **scoring**
surface:

- `src/pgturbohybrid_quant_score_u8_x86.c` — unsigned-codebook ("u8") query-split
  kernel (`maddubs`/`VPDPBUSD`, AVX2 / AVX-512-VNNI). Per
  `pgturbohybrid_quant_score.c:471` (`PgturbohybridGraphTqUseU8Split`) this kernel
  is **4-bit-only** (`tq->bits != PGTURBOHYBRID_DEFAULT_BITS` ⇒ disabled),
  needs AVX2, and dims ≥ 1024.
- `src/pgturbohybrid_quant_score_signed_x86.c` — signed integer query-split kernel
  (codebook via shuffle).
- Dispatch / CPU-feature probes: `src/pgturbohybrid_quant_score.c`
  (`pgturbohybrid_quant_score_internal.h` documents the split).

ARM equivalents live in `pgturbohybrid_quant_score_arm.c`. A scalar/LUT reference
runs when SIMD is forced off or a kernel does not apply.

Consequence: the eight retrieval features below are **control-flow / scalar**
(which candidates to seed, retry, rerank, fuse, diversify) layered *on top of*
the scoring kernels. They do not add x86 code, so they need only the normal
cross-platform regression on AMD64 — **not** feature-specific x86 validation. The
genuine AMD64 validation target is the underlying **default 4-bit SIMD scoring
path** (u8-split + signed-split, AVX2/AVX-512-VNNI, and the AVX-512 frequency
behaviour), which every feature here exercises indirectly. `quantization_bits=8`
is the one listed feature that changes kernel selection (to a scalar 8-bit path),
so it warrants its own AMD64 pass.

## Feature inventory (confirmed present in current source)

| # | Feature | Source files | User-facing setting | AMD64-specific validation? |
|---|---|---|---|---|
| 1 | entry_sidecar_strategy | `pgturbohybrid_graph.c:51,600` (relopt enum + registration); Options field `entrySidecarStrategy` in `pgturbohybrid.h:707`, `pgturbohybrid_am.h:46`, `pgturbohybrid_quant.h:172` | **index reloption** `entry_sidecar_strategy` = `hash` \| `farthest_code` \| `level_covering` \| `hybrid_level_covering` | No — scalar entry-point selection; feeds the existing scorer, adds no x86 code |
| 2 | payload entry seeding | `pgturbohybrid_graph.c:264-266,322` (vars/enum/names); `pgturbohybrid_am.c:5121,5128` (GUCs) | GUCs `turbohybrid.payload_entry_seeding` (enum), `turbohybrid.payload_entry_seed_count` | No — scalar seed selection |
| 3 | dense uncertainty retry | `pgturbohybrid_graph.c:255-257,449` (vars/enum/names); `pgturbohybrid_am.c:5076,5083,5088,5092` (GUCs); stats in `pgturbohybrid_stats.c` | GUCs `turbohybrid.dense_uncertainty_retry` (enum), `…_max_passes`, `…_multiplier`, `turbohybrid.dense_uncertainty_min_gap` | No — scalar retry decision; re-invokes the same (already-validated) scorer |
| 4 | calibrated residual rerank | `pgturbohybrid.h:250`, `pgturbohybrid_graph.c:247` (var); `pgturbohybrid_am.c:280,4972,4979,4984` (enum + GUCs); `pgturbohybrid_stats.c:65,413,670` (stats) | GUC `turbohybrid.dense_residual_rerank_mode` = `off` \| `fixed` \| `calibrated`; `…_weight`, `…_max_adjust_ratio` | No — score adjustment is scalar (grep finds no SIMD in the rerank/calibration path) |
| 5 | calibrated fusion | `pgturbohybrid_query.h:25` (enum `PGTURBOHYBRID_FUSION_CALIBRATED`); `pgturbohybrid_query.c:113,285` (output/parser); `pgturbohybrid_am.c:3168,3231,3562,3600,4900-4914` (dispatch + tuning GUCs); `pgturbohybrid_stats.c:1609+` (stats) | `turbohybrid_query(..., fusion => 'calibrated')`; tuning GUCs `turbohybrid.calibrated_fusion_{both_match_bonus,identifier_bm25_alpha,broad_dense_alpha,default_alpha}` | No — scalar score fusion |
| 6 | BM25 heap-tsvector rerank | `pgturbohybrid_am.h:153-155` (externs); `pgturbohybrid_am.c:108-111,5198,5205,5210` (vars + GUCs); BM25 query path in `pgturbohybrid_bm25_query.c` | GUCs `turbohybrid.bm25_heap_tsvector_rerank` (enum), `…_multiplier`, `…_weight` | No — PostgreSQL `ts_rank`-style scoring on the heap tsvector; architecture-independent |
| 7 | final group-payload diversity | `pgturbohybrid.h:268-271,323-326` (externs + enum `…FINAL_DIVERSITY_GROUP_PAYLOAD`); `pgturbohybrid_am.c:298-300,5134,5141,5147,5153` (options + GUCs) | GUC `turbohybrid.final_diversity` = `off` \| `group_payload`; `…_payload_slot`, `…_lambda`, `…_pool_multiplier` | No — scalar MMR-style selection over payloads |
| 8 | quantization_bits = 8 | reloption: `pgturbohybrid_am.c:4843` (range 1..8, **effective**) and `pgturbohybrid_graph.c:587` (range 1..`PGTURBOHYBRID_DEFAULT_BITS`=4, shadowed — see note); 8-bit encode/score/storage in `pgturbohybrid_graph_utils.c:1035,1348,1412,1605,3978`; stats `pgturbohybrid_stats.c:1006,1038,1064` | **index reloption** `quantization_bits = 8` | **Yes (recommended)** — 8-bit selects the `scalar_8bit` kernel (`graph_utils.c:3978`), i.e. it does *not* use the 4-bit x86 u8-split; validate that the 8-bit path builds/scores correctly and that kernel routing is right on AMD64 |

## quantization_bits = 8 — confirmed accepted, with a registration caveat

`quantization_bits = 8` **is accepted** by reloption validation. Verified
empirically: `CREATE INDEX ... USING turbohybrid (...) WITH (quantization_bits = 8)`
on a throwaway 8-d table succeeds.

Caveat for maintainers (no action taken here): the option is registered twice on
the same `pgturbohybrid_relopt_kind` —
`PgturbohybridGraphInit()` (`graph.c:587`, max `PGTURBOHYBRID_DEFAULT_BITS` = 4)
then `PgturbohybridInit()` (`am.c:4843`, max 8). `_PG_init`
(`pgturbohybrid.c:74-75`) calls Graph init first, then the am init, and the
later (max-8) registration is the effective one — hence 8 is accepted. The
duplicate registration with conflicting bounds is latent and worth de-duplicating
later, but it is not a runtime bug today.

## Summary for the AMD64 validation run

- All eight features are present and the tree builds on AMD64.
- Features 1–7 are scalar/algorithmic and architecture-independent: cover them
  with the standard regression suite on AMD64; no x86-specific kernel work.
- Feature 8 (`quantization_bits = 8`) changes scoring-kernel selection to a
  scalar 8-bit path — give it an explicit AMD64 build+query pass.
- The real AMD64-specific effort is the **default 4-bit SIMD scoring path**
  (`pgturbohybrid_quant_score_u8_x86.c`, `…_signed_x86.c`, dispatch in
  `pgturbohybrid_quant_score.c`): AVX2/AVX-512-VNNI correctness vs the scalar
  reference, and AVX-512 frequency/throughput behaviour. Every feature above
  rides on this path but none of them modify it.
