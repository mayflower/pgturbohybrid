# Native sparse embedding (SPLADE) retrieval — design

> Status: **design / in progress**. This documents a native sparse-vector
> retrieval branch for pgturbohybrid. It is forward-looking: the canonical
> `turbohybrid_sparse_vector` type and its `tsvector`/`tsquery` bridges exist
> today, but **native sparse retrieval (a sparse opclass + a sparse index branch)
> does not yet exist**. Nothing here claims production compatibility.

## 1. Goals

- Support SPLADE-style learned sparse vectors as a first-class retrieval signal,
  alongside the existing dense, multivector, and BM25 branches.
- Preserve float sparse inner-product semantics: `score(q,d) = Σ_t q_t · d_t`.
- Add TurboQuant-style per-term impact quantization (q16, q8) with an exact f32
  baseline and exact rerank.
- Add SIMD-optimized postings scoring (AVX2/NEON), scalar reference first.
- Integrate with existing fusion (RRF first; weighted/calibrated later).

## 2. Non-goals

- Do **not** route sparse embeddings through PostgreSQL `tsvector`/`tsquery` as
  the native retrieval path. (The existing `turbohybrid_sparse_vector_terms` /
  `_query_terms` bridges remain, but only as a convenience for FTS interop.)
- Do **not** model sparse embeddings as dense TurboQuant graph vectors.
- Do **not** modify pgvector source or pgvector-owned SQL objects.

## 3. Architectural constraint: the index is dense-graph-primary

This is the single most important fact shaping the design. Today:

- `aminsert` (`pgturbohybrid_am.c:7215`) drops any row whose **first key** is NULL
  (`if (isnull[0]) return false;`), then derives `node_id` from the dense /
  multivector graph insert, then appends BM25 postings keyed by that `node_id`
  (`PgturbohybridBm25AppendDelta(index, nodeId, ...)`).
- `PgturbohybridValidateIndex` requires the first key to be `vector` /
  `turbohybrid_multivector` and limits the index to 1–2 keys.
- `node_id ↔ heap TID` identity and MVCC liveness are recovered from the dense
  graph's code tuples (`PgturbohybridReadNodeMap` / `PgturbohybridReadNodeStates`,
  `pgturbohybrid_bm25_build.c`).

So the dense/multivector graph **owns** the node-identity space; BM25 is a
*secondary* branch hanging off it. Sparse therefore splits into two cases:

| Case | Index shapes | Approach | Milestone |
| --- | --- | --- | --- |
| **(a) dense-present** | `dense+sparse`, `dense+sparse+bm25` | sparse is a secondary branch off the dense node space — mirrors BM25 | ships first (Prompts 4–11) |
| **(b) sparse-primary** | `sparse-only`, `sparse+bm25` (no dense) | needs a *new* dense-independent primary node space (node_id alloc, liveness, NULL-gating) | deferred architectural milestone |

## 4. SQL surface

```sql
-- opclass (new)
... USING turbohybrid (col sparse_ip_turbohybrid_ops)
-- query payload (new named args on the existing turbohybrid_query(...))
turbohybrid_query(
  sparse_query        => turbohybrid_sparse_vector,
  sparse_weight       => float8 DEFAULT 1.0,
  sparse_k            => int4   DEFAULT NULL,
  require_sparse_match=> bool    DEFAULT false,
  ... existing dense/multivector/bm25 args ...)
-- ORDER BY operator (new)
col <~*> turbohybrid_query(...)
```

Examples: sparse-only (case b), `dense+sparse` RRF, `dense+sparse+bm25` RRF.

## 5. Data model

Canonical `turbohybrid_sparse_vector` (already implemented,
`src/pgturbohybrid_query.c`):

```c
struct PgturbohybridSparseVector { int32 vl_len_; uint16 version; uint16 flags; uint32 count; };
struct PgturbohybridSparseVectorEntry { int32 termId; float4 weight; int16 fieldId; uint16 reserved; };
```

- entries sorted by `term_id` (canonical form, produced by
  `turbohybrid_sparse_vector_build`);
- `term_id ≥ 0`; weights finite; `field_id` optional (default
  `PGTURBOHYBRID_SPARSE_VECTOR_FIELD_NONE`);
- duplicate `term_id` handling: `sum` (default), `max`, or `error`;
- non-positive weights dropped when `drop_non_positive` (default true).

## 6. Index storage (mirrors BM25, `pgturbohybrid_bm25.h`)

- sparse meta tuple (`PGTURBOHYBRID_SPARSE_VERSION`, quant bits/mode, encoding);
- sparse lexicon keyed by `term_id` (`df`, `max_doc_weight`, per-term quant scale,
  postings location/count/bytes);
- sparse postings tuples (`node_id` + weight), chunked on-disk;
- block-max tuples (for WAND);
- optional delta tuples (incremental insert/update).

Postings key on the same `node_id` as the dense graph in case (a); case (b) gets
its own node space.

## 7. Quantization

- **exact f32** baseline (`sparse_quant_bits = 0`).
- **q16**: `scale = max_doc_weight / 65535`, `q = round(weight/scale)`. Mirrors the
  BM25 impact quantizer `QuantizeTfNorm` (`pgturbohybrid_bm25_build.c:211`).
- **q8**: `scale = max_doc_weight / 255`. **Net-new** — BM25 has no q8 path.
- score: `Σ_t (query_weight_t · scale_t) · q_weight`. Exact rerank fetches the heap
  sparse column and recomputes f32 over the candidate band (mirrors
  `PgturbohybridBm25HeapTSVectorRerank`).

## 8. SIMD

Sparse scoring is a **scatter** (`score[node] += w·q`); all existing kernels are
**gather** (code→codebook→accumulate), so the sparse kernels are a net-new family.
Pattern: SIMD load+widen quantized weights, multiply by a broadcast
`term_multiplier`, store to a small stack buffer, then apply the accumulator
updates with a scalar loop (AVX2/NEON have no cheap general scatter). Follow the
established convention: `turbohybrid.simd` master GUC, `__attribute__((target))`
guards + runtime `__builtin_cpu_supports` dispatch, **scalar reference first** (the
SIMD kernels must match it), per-kernel scan-stat naming, and a SIMD/scalar parity
regression (mirroring `test/sql/pgturbohybrid_simd_parity.sql`).

## 9. Query algorithms

1. exact OR accumulation (baseline; correctness oracle);
2. block-max / WAND / MaxScore pruning (mirrors BM25);
3. impact-ordered traversal (later).

## 10. Fusion

- RRF first: extend `PgturbohybridRrfScore` (`pgturbohybrid_am.c:4432`) with one
  additive term `sparse_weight · 1/(rrf_k + sparse_rank)`; add `hasSparse` /
  `sparseRank` to `PgturbohybridResult`.
- weighted / calibrated / dbsf may reject sparse initially with a clear error.

## 11. Diagnostics (`turbohybrid_last_scan_stats()`)

`sparse_branch_available`, `sparse_branch_used`, `sparse_terms`,
`sparse_resolved_terms`, `sparse_postings_touched`, `sparse_candidates_scored`,
`sparse_quant_bits`, `sparse_score_kernel`, `sparse_simd_blocks`,
`sparse_wand_pruned`, `sparse_exact_rerank_count`, `sparse_elapsed_us`. Fields are
zero/false/absent when sparse is not used.

## 12. Milestones (implementation order)

0. this design doc
1. canonical `turbohybrid_sparse_vector_build` + inspectors
2. `sparse_query` payload + `<~*>` + `sparse_ip_turbohybrid_ops` skeleton
3. index key map (replace hardcoded dense=key0 / lexical=key1)
4. exact f32 sparse as a **dense-present** secondary branch
5. sparse RRF fusion
6. q16/q8 quantized postings (scalar)
7. exact sparse rerank
8. AVX2/NEON sparse SIMD (scatter)
9. WAND / block-max
10. hot-postings cache + memory estimator
11. delta insert/update + compaction
12. **sparse-primary node space** (sparse-only / no-dense) — architectural
13. `llama_embed_sparse` stub backend
14. real sparse backend seam
15. benchmark harness
16. hardening

Sequencing rule: **exact f32 → quantized scalar → SIMD** at every layer, and
dense-present sparse ships before sparse-primary.
