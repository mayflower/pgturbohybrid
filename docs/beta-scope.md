# Beta Scope

`pgturbohybrid` is heading toward a **narrow beta**, not a broad feature beta.
Beta freezes a small **stable public core** and keeps everything else visibly
experimental. The point: beta users should expect rough edges, but *not* a
moving public-product boundary.

This document is the contract for that boundary. It complements the per-feature
[feature & maturity matrix](feature-matrix.md) and the
[roadmap](../ROADMAP.md).

## In beta scope (the stable contract)

These surfaces are intended to be stable through beta: names and SQL semantics
will not change without a release note, and they are the paths we commit to
support.

- **Dense-only vector search** — `vector_{l2,ip,cosine}_turbohybrid_ops`, the
  `<~->` / `<~#>` / `<~>` operators, and `turbohybrid_query(vector_query => ...)`
  (plus the `turbohybrid_dense_query(...)` wrapper).
- **Dense + BM25 hybrid search** — a dense opclass plus
  `bm25_tsvector_turbohybrid_ops`, via
  `turbohybrid_query(vector_query, text_query)` (and `turbohybrid_hybrid_query`).
- **RRF fusion** — `turbohybrid_query(fusion => 'rrf')`, the default.
- **Support diagnostics** — the read-only functions needed to debug the above:
  `turbohybrid_last_scan_stats()` (its **stable** keys, per
  [diagnostics-schema.md](diagnostics-schema.md)),
  `turbohybrid_last_scan_diagnosis()`, `turbohybrid_index_stats()`,
  `turbohybrid_estimate_memory()`, `turbohybrid_validate_index()`, and
  `turbohybrid_simd_capabilities()`.
- **Install / build / package surfaces** — `CREATE EXTENSION pgturbohybrid`
  over an unmodified pgvector, the PGXS build, and the documented version range.

It is still alpha-grade software: even "stable" here does not promise on-disk
index compatibility across pre-1.0 tags. Expect to `REINDEX` on upgrade (see
[operations.md](operations.md)).

## Out of beta scope (experimental — use, but no beta promise)

These are real and tested, but their SQL surface, defaults, or on-disk format
may still change; they are **not** part of the beta support promise and should
not be read as production-ready:

- **Sparse / learned-sparse (SPLADE)** retrieval — `turbohybrid_sparse_vector`,
  `sparse_ip_turbohybrid_ops`, `<~*>`.
- **Multivector / ColBERT late interaction** — the
  `turbohybrid_multivector` type, its
  opclasses, and the native ColBERT candidate sources. The document-node build
  scaling limitation is a **beta blocker for the multivector feature**
  (tracked in
  [docs/dev/multivector-document-node-build-scaling.md](dev/multivector-document-node-build-scaling.md)
  and the [roadmap](../ROADMAP.md)).
- **`quantized_inverted_experimental`** candidate source — research-only.
- **`pg_colbert_llama` / `llama_embed`** companion extension — optional; you do
  not need it to use any in-scope feature.
- **Score-level fusion modes** (`weighted`, `fast_weighted`, `calibrated`,
  `dbsf`), and **developer/benchmark GUCs and scoring probes**.

All SQL objects in this section require the explicit companion install
`CREATE EXTENSION pgturbohybrid_experimental`; the core extension never installs
them implicitly.

## Why narrow

The dense+BM25 core is the strongest, best-tested path (build, runtime, and
operational story). Sparse and multivector remain in the repository and are
worth using for evaluation, but their build/runtime/upgrade stories are not yet
as strong as the dense+BM25 path, so promising beta-level stability for them
would overstate where the product is.
