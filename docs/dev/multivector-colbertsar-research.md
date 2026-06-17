# ColBERTSaR-Style Quantized Inverted Research Branch

This document tracks the research-only
`quantized_inverted_experimental` multivector candidate source. It is inspired
by ColBERTSaR-style quantized inverted indexes for late-interaction retrieval,
where token embeddings are mapped to codebook terms and searched through
postings before exact MaxSim rerank.

The branch is intentionally not a production on-disk format. Any codebook,
posting, residual, or score-payload layout introduced here is allowed to change
without compatibility guarantees. Indexes built with an experimental storage
version will need explicit REINDEX guidance before they can be used as a stable
feature.

## Guard

The branch is selected only with:

```sql
SET turbohybrid.multivector_candidate_source =
  'quantized_inverted_experimental';
```

Current executable status:

- The GUC value is registered so benchmark and SQL plumbing can name the branch.
- Document-node indexes persist experimental codeword metadata and posting
  tuples in the multivector docmap sidecar. Query tokens are assigned with the
  same deterministic or explicit external codebook metadata used at build time,
  matching persisted postings are accumulated into approximate document scores,
  and candidates are exact-reranked with heap MaxSim. The scan validates
  document vectors lazily for touched postings rather than pre-validating every
  document in the sidecar.
- Token-node indexes fail explicitly with `feature_not_supported` because the
  branch needs document sidecar vectors for exact rerank.
- Plain fallback is bypassed when the branch is selected so benchmark output
  cannot silently substitute a different candidate source.
- Fresh document-node indexes always write the current experimental posting
  sidecar. The missing-sidecar error path is for old indexes and malformed
  sidecars, and reports REINDEX guidance rather than falling back.

The default codeword assignment is deterministic and not learned: each token
maps to the signed largest-magnitude dimension, so the temporary codebook has
`2 * dim` terms. This is useful for validating the branch, stats, and benchmark
comparison surface, but it is not a production ColBERTSaR index.

An explicit external text codebook can be selected for experiments:

```sql
SET turbohybrid.multivector_quantized_inverted_codebook = 'external';
SET turbohybrid.multivector_quantized_inverted_codebook_path =
  '/path/to/codebook.txt';
SET turbohybrid.multivector_quantized_inverted_codebook_top_m = 1;
```

The first executable external format is intentionally small and administrator
provided:

```text
pgturbohybrid_quantized_inverted_codebook_v1 <dim> <codebook_size> <checksum>
<codebook_size * dim float values>
```

The checksum is stored as declared metadata and used as a mismatch guard. It is
not a compatibility promise or a cryptographic verification layer. SQL table
and model-profile codebook sources remain target-design items for later
prototypes; they are not executable in this slice.

The DBpedia benchmark can exercise the same path without changing production
defaults by selecting the experimental external-codebook profile. First build a
sampled cosine k-means codebook from already imported document token vectors:

```bash
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m_colbert \
  --reuse-data \
  --document-node-colbert-quantized-codebook-build-only \
  --quantized-inverted-codebook-output \
    .nix-dev/tmp/dbpedia-colbert-1m-quantized-codebook.txt \
  --quantized-inverted-codebook-size 256 \
  --quantized-inverted-codebook-sample-docs 10000 \
  --quantized-inverted-codebook-sample-tokens 100000 \
  --quantized-inverted-codebook-kmeans-iterations 20
```

Then run the explicit external-codebook profile:

```bash
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m_colbert \
  --reuse-data \
  --document-node-colbert-candidate-source-focus \
  --include-quantized-inverted-experimental \
  --document-node-colbert-candidate-source-profiles \
    quantized_inverted_external_centroid_only_compact_topk_032 \
  --multivector-quantized-inverted-codebook-path \
    .nix-dev/tmp/dbpedia-colbert-1m-quantized-codebook.txt \
  --candidate-budgets 800,1600 \
  --exact-rerank-k 100
```

The named profile is still experimental. It sets the candidate source to
`quantized_inverted_experimental`, uses compact score-top-k admission, records
the external codebook metadata in the physical index signature, and reranks the
retained heap documents with exact MaxSim.

For 1M-scale experiments prefer the `centroid_only` compact external profile:
it stores centroids plus quantized posting/codeword payloads, not a full f32
document multivector sidecar. The older
`quantized_inverted_external_f32_compact_topk_032` profile remains useful only
when explicitly measuring full-sidecar behavior.

The codebook builder is benchmark plumbing, not a production trainer: it samples
loaded document token vectors, runs deterministic cosine k-means in Python, and
writes the current text file format. It does not build an index, run retrieval,
or introduce a compatibility promise.

## Current Sidecar

The current sidecar stores experimental codebook metadata plus inverted
postings:

```text
codebook source, dim, codebook_size, top_m, checksum
codeword -> docId, tokenOrdinal, approximate score payload
```

Bulk build emits posting tuples during index creation. Incremental document-node
insert appends matching posting tuples when the existing index already carries
the quantized-postings flag, or when an empty document-node index is being
bootstrapped. Older indexes without the sidecar fail explicitly and require
`REINDEX` before the experimental branch can run.

The deterministic codeword assignment and score payload are deliberately
simple: the deterministic codeword is the signed largest-magnitude dimension
and the payload stores a quantized token-norm proxy. External codebooks assign
the nearest codeword by dot product. Exact rerank still uses the original
float32 document multivectors, not the payload.

External codebook metadata is part of the experimental sidecar. Query execution
fails with REINDEX guidance if the active codebook source, dimension, size,
`top_m`, or checksum does not match the index metadata. Existing deterministic
indexes without the metadata tuple are treated as deterministic-only for
backward compatibility.

## Target Design

The next prototype stages should replace the deterministic codeword assignment
with a real codebook and add residual payloads:

1. Learn or load a codebook for the model dimension.
2. Quantize each stored document token to one or more codewords.
3. Store postings keyed by codeword and document identity.
4. Preserve the original document multivector for exact MaxSim rerank.
5. Mark every persisted structure with an experimental magic/version.

Query should:

1. Quantize every unmasked query token with the same codebook.
2. Retrieve postings for matching or nearby codewords.
3. Accumulate document-level approximate scores.
4. Keep a bounded candidate set.
5. Exact-rerank candidates with float32 MaxSim over the original multivectors.

The approximate score must remain admission-only until benchmark evidence shows
that it can safely participate in fusion. Final SQL ordering for multivector
queries remains document-keyed exact MaxSim unless a later prompt explicitly
changes that contract.

## Benchmark Requirements

Compare this branch against:

- `learned_sparse`
- `centroid_lite`
- `token_nodes`
- `document_nodes`
- `proxy_vector`

At minimum, reports must include:

- index bytes for codebook, postings, residuals, and original multivectors;
- admission top-1 and top-10 recall against exact MaxSim;
- final Recall/NDCG/MRR where qrels exist;
- p50/p95 latency;
- postings touched;
- documents scored approximately;
- documents exact-reranked.

The default benchmark grids should keep this branch opt-in because the storage
format is explicitly unstable. It can be selected manually today to collect
persisted-posting admission, storage, and latency numbers.

Compact-code admission scoring is also explicit and diagnostic-only:

```sql
SET turbohybrid.multivector_quantized_inverted_compact_scoring = experimental;
```

The default is `off`. The current prototype uses the persisted posting
`scorePayload` as a compact token-norm proxy and dispatches through the
experimental compact scalar/SIMD scorer. It does not implement a stable residual
or PQ payload format, and it does not change SQL result ordering: retained
documents are still reranked with exact heap MaxSim over the original
multivectors.

Use the compact serving grid only with the experimental opt-in when comparing
against `proxy_vector`, `document_nodes`, `centroid_mean`, and `centroid_lite`:

```bash
nix --extra-experimental-features 'nix-command flakes' develop --command \
  python benchmarks/dbpedia_colbert_multivector.py \
    --database pgturbohybrid_dbpedia_colbert_research \
    --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
    --max-docs 10000 \
    --max-queries 100 \
    --reuse-data \
    --document-node-serving-grid \
    --document-node-serving-grid-include-experimental \
    --admission-budget-sweep 50,100,200,400,800 \
    --output .nix-dev/tmp/dbpedia-colbert-serving-experimental.json \
    --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-experimental.md
```

Generated JSON, Markdown, local logs, model files, and datasets are local
artifacts and must not be committed.

## Reported Fields

`turbohybrid_last_scan_stats()` reports these branch-specific fields when the
candidate source is selected:

- `quantized_inverted_lists_visited`
- `quantized_inverted_postings_touched`
- `quantized_inverted_docs_scored`
- `quantized_inverted_candidates`
- `quantized_inverted_exact_rerank_docs`
- `quantized_inverted_codebook_source`
- `quantized_inverted_codebook_size`
- `quantized_inverted_codebook_dim`
- `quantized_inverted_codebook_checksum`
- `quantized_inverted_codebook_top_m`
- `quantized_inverted_compact_kernel`
- `quantized_inverted_compact_score_us`
- `quantized_inverted_compact_docs_scored`
- `quantized_inverted_compact_payload_bytes`
- `quantized_inverted_compact_topk_changed_vs_scalar`
- `quantized_inverted_assignment_us`
- `quantized_inverted_list_offset_bytes`
- `quantized_inverted_posting_bytes`
- `quantized_inverted_sidecar_bytes`

The benchmark harness carries those fields into admission rows, serving-grid
rows, and Markdown summaries when present. Standard benchmark metrics still
come from the normal harness:

- p50 and p95 latency;
- exact top-1 admission and exact top-10 admission recall;
- Recall@10, NDCG@10, and MRR@10 when qrels exist;
- index bytes for the full PostgreSQL index relation.

Final SQL ranking remains exact heap MaxSim. The approximate posting score is
used only to admit documents into the bounded exact rerank band.

## Limitations

- The default codebook is deterministic and dimension-sign based, not learned.
- External codebooks are file-only today; SQL-table and model-profile sources
  are design targets for later prompts.
- `top_m` values greater than 1 are rejected until the posting sidecar has an
  explicitly versioned multi-assignment format.
- The score payload is a temporary token-norm proxy, not a residual or
  calibrated ColBERTSaR payload.
- The persisted posting layout is research-only and can change without
  compatibility promises.
- The branch is not enabled by normal CI, production profiles, or default
  serving grids.
- Token-node indexes are rejected; the executable path requires
  `multivector_graph = document_nodes`.

## Next Steps

1. Add SQL-table and model-profile codebook sources with explicit metadata.
2. Add a versioned multi-assignment posting format for `top_m > 1`.
3. Store residual or compressed score payloads that can approximate MaxSim
   without consulting full document vectors during admission.
4. Report separate codebook, posting, residual, and original multivector byte
   components once those payloads exist.
5. Keep exact heap MaxSim as the final SQL ordering contract until a later
   prompt explicitly changes it.
