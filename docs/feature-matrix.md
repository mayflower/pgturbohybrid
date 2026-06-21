# Feature & Maturity Matrix

This is the single source of truth for which `pgturbohybrid` capabilities exist
and how much you should trust each one. README, `docs/architecture.md`,
`docs/how-it-works.md`, `docs/compatibility.md`, and the SQL comments in
`sql/pgturbohybrid--0.1.0.sql` are kept consistent with this table; if any of
them disagree, this file wins and the other should be fixed.

The whole extension is still **alpha software** (see the README Status section).
The maturity labels below are *relative within the alpha*: they say how stable a
feature's behavior and on-disk/SQL surface are expected to be, not that any of it
is finished production software.

## Maturity labels

| Label | Meaning |
| --- | --- |
| **stable public** | Supported surface. Names and SQL semantics are not expected to change without a release note. Still alpha, so on-disk format can change between alpha tags. |
| **experimental public** | Real, tested feature you can use, but the SQL surface, defaults, or on-disk format may still change. Validate on your data; expect `REINDEX` across versions. |
| **research-only** | Built for investigation and benchmarking. Not a serving path; correctness/latency are not guaranteed and the option may be removed. |
| **diagnostic** | Read-only introspection (stats, estimators, probes). Safe to call; individual JSON keys may be stable or experimental (see `docs/diagnostics-schema.md`). |
| **internal** | Not part of the user contract. Support functions, on-disk identifiers, executor hooks. May change at any time. |

## Matrix

| Feature | Maturity | Primary SQL surface | Docs |
| --- | --- | --- | --- |
| Dense-only vector search | **stable public** | `vector_{l2,ip,cosine}_turbohybrid_ops`, `<~->`/`<~#>`/`<~>`, `turbohybrid_query(vector_query => ...)` | [how-it-works](how-it-works.md), [architecture](architecture.md) |
| Dense + BM25 hybrid search | **stable public** | dense opclass + `bm25_tsvector_turbohybrid_ops`, `turbohybrid_query(vector_query, text_query)` | [how-it-works](how-it-works.md) |
| RRF fusion | **stable public** | `turbohybrid_query(fusion => 'rrf')` (default) | [how-it-works](how-it-works.md) |
| Score-level fusion (`weighted`, `fast_weighted`, `calibrated`, `dbsf`) | **experimental public** | `turbohybrid_query(fusion => ...)` | README "What It Does" |
| Sparse-only learned-sparse (SPLADE) search | **experimental public** | `turbohybrid_sparse_vector`, `sparse_ip_turbohybrid_ops`, `<~*>`, `turbohybrid_query(sparse_query => ...)` | [sparse-embeddings](sparse-embeddings.md) |
| Sparse + BM25 | **experimental public** | sparse opclass + `bm25_tsvector_turbohybrid_ops` | [sparse-embeddings](sparse-embeddings.md) |
| Multivector dense-only (late interaction / MaxSim) | **experimental public** | `multivector` type, `multivector_cosine_turbohybrid_ops` / `multivector_maxsim_ip_turbohybrid_ops`, `turbohybrid_query(multivector_query => ...)` | [multivector-late-interaction](multivector-late-interaction.md) |
| Multivector + BM25 | **experimental public** | multivector opclass + `bm25_tsvector_turbohybrid_ops` (document-keyed fusion only) | [multivector-late-interaction](multivector-late-interaction.md) |
| ColBERT exact heap MaxSim rerank | **experimental public** | `turbohybrid.multivector_exact_rerank`, `turbohybrid_multivector_maxsim(...)` | [multivector-late-interaction](multivector-late-interaction.md) |
| Native ColBERT candidate source: `proxy_vector`, `document_nodes` | **experimental public** | `multivector_doc_storage` / candidate-source index options | [multivector-late-interaction](multivector-late-interaction.md) |
| Native ColBERT candidate source: `centroid_lite` | **experimental public** | `centroid_only` storage + candidate source | [multivector-late-interaction](multivector-late-interaction.md) |
| `quantized_inverted_experimental` candidate source | **research-only** | codeword/posting admission option | README "Native ColBERT candidate generation" |
| `exact_doc_scan` / `exact_token_scan` candidate sources | **diagnostic** | exact oracles, not serving paths | README "Native ColBERT candidate generation" |
| `pg_colbert_llama` / `llama_embed` companion | **experimental public** (optional, separate extension) | `llama_embed_*`, legacy `colbert_*` | [colbert-llama-extension](colbert-llama-extension.md) |
| Diagnostics: `turbohybrid_last_scan_stats()`, `turbohybrid_last_scan_diagnosis()`, `turbohybrid_index_stats()`, `turbohybrid_simd_capabilities()`, `turbohybrid_last_build_stats()` | **diagnostic** | read-only jsonb | [diagnostics-schema](diagnostics-schema.md) |
| Memory estimator: `turbohybrid_estimate_memory()` | **diagnostic** | read-only jsonb | [architecture](architecture.md) (concurrency sizing) |
| Native graph prewarm / shared cache: `turbohybrid_prewarm()`, `turbohybrid.native_cache_*` | **experimental public** | jsonb + GUCs | [architecture](architecture.md) |
| Maintenance: `turbohybrid_sparse_compact()`, `turbohybrid_graph_repair_dry_run()` | **experimental public** (compact) / **diagnostic** (repair dry-run, read-only) | jsonb | [sparse-embeddings](sparse-embeddings.md) |
| Scoring probes: `turbohybrid_scorer_distances()`, `turbohybrid_query_split_probe()`, `turbohybrid_scorer_x4_batch_parity()`, `turbohybrid_scorer_bench()`, `turbohybrid_experimental_compact_code_score()` | **research-only** / **diagnostic** | jsonb kernel introspection | README Diagnostics |
| Quantization: 4-bit (default), 2-bit codes; `exact_storage`, residual sketches | **stable public** (`quantization_bits`, `exact_storage`) / **experimental public** (`residual_rerank*`) | index reloptions | [architecture](architecture.md) |
| Build-time graph knobs: `graph_backbone`, `entry_sidecar*` | **experimental public** | index reloptions | [architecture](architecture.md) |

## Notes

- **Stable public is still alpha.** A stable label means we will not casually
  rename the SQL object or flip its semantics; it does *not* promise on-disk
  index compatibility across alpha tags. Plan to `REINDEX` when upgrading.
- **Research-only paths must not be used as serving paths.** They exist so the
  candidate-source and kernel experiments can be measured against the exact
  oracles. Latency, recall, and even result stability are not guaranteed.
- **Diagnostic functions are read-only** and never change scan execution. The
  set of JSON keys they return is itself an evolving surface — stable vs
  experimental keys are tracked in [`diagnostics-schema.md`](diagnostics-schema.md).
- **The companion `pg_colbert_llama` / `llama_embed` extension is optional.**
  You do not need it to use any `pgturbohybrid` index feature; it only helps you
  generate dense, token-level, sparse, or multivector embeddings locally.

## Supported versions

PostgreSQL **14 through 19** and pgvector **0.8.2+** (PostgreSQL 19 is tested
against pgvector `master`). See [`compatibility.md`](compatibility.md) for the
authoritative version range; README, `RELEASE.md`, `META.json`, and the CI
workflows are kept consistent with it.
