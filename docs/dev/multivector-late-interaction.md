# Multivector Late Interaction Design

This document defines the implementation contract for ColBERT-style
multivector retrieval in `pgturbohybrid`.

## Target Semantics

A query and a document are both represented as multiple dense vectors. For a
query `Q` and document `D`, the score is:

```text
score(Q, D) = sum_i max_j sim(q_i, d_j)
```

For the MVP, `sim()` is dot product over float32 vectors. The score is a
similarity, so larger is better. PostgreSQL `ORDER BY` operators exposed by the
access method must remain distance-oriented, so the external distance is:

```text
distance = -score(Q, D)
```

## Node And Result Separation

Late interaction indexes subvectors, but returns documents. That separation is
the main correctness invariant:

- `nodeId` identifies one subvector/token node.
- `docId` identifies the source document.
- `heaptid` identifies the PostgreSQL heap tuple returned to the executor.
- `tokenOrdinal` identifies the subvector position inside the document.

One document can produce many `nodeId`s. A result must be ranked and deduplicated
by document/heap tuple, never by subvector node. Subvector hits are only evidence
used to update a document-level MaxSim accumulator.

## Complexity Variables

- `D`: number of documents.
- `L`: document vectors per document.
- `N`: total subvector nodes, `N = D * L` for fixed `L`.
- `Q`: query vectors.
- `d`: vector dimensions.
- `Ks`: raw subvector hits per query vector.
- `C`: unique candidate documents reached by subvector hits.
- `R`: documents selected for exact rerank.

Expected complexity:

- Build expansion: `O(D * L)`.
- Approximate query candidate generation: `O(Q * ANN + Q * Ks)`.
- Document accumulator memory: `O(C * Q)`, not `O(D * Q)` and not
  `O(N * Q)`.
- Exact rerank: `O(R * Q * L * d)`.

## MVP Phases

1. Packed `turbohybrid_multivector` type.
2. Exact MaxSim reference functions.
3. Query v2 with dense kind and multivector payload.
4. Multivector SQL operator and opclass validation.
5. Build-time expansion into subnodes.
6. Approximate MaxSim aggregation over subvector hits.
7. Exact float32 rerank.
8. BM25/RRF fusion at document level.

## MVP Non-Goals

- No int8/VNNI exact rerank.
- No per-subnode exact float32 storage as the default.
- Hybrid multivector fusion is document-level RRF only; score-level fusion modes
  are deferred until their document-level semantics are specified and tested.
- No on-disk format expansion without version or compatibility checks.

## Scan Diagnostics

`turbohybrid_last_scan_stats()` should make multivector cost drivers visible:

- `multivector_query_vectors` and `multivector_subvector_searches` describe
  `Q`, the number of query-token graph traversals.
- `multivector_raw_subvector_hits` tracks bounded subnode hits across query
  tokens.
- `multivector_candidate_source`,
  `multivector_exact_token_scan_enabled`, and
  `multivector_exact_token_scan_nodes_scored` identify whether the scan used
  the normal graph token path or the developer exact-token oracle.
- `multivector_plain_fallback_used`,
  `multivector_plain_fallback_reason`,
  `multivector_plain_fallback_docs_scored`, and
  `multivector_plain_fallback_pairs` identify exact heap MaxSim fallback work
  for small or near-exhaustive scans.
- `multivector_reservoirs_enabled`, `multivector_reservoir_score_docs`,
  `multivector_reservoir_coverage_docs`, `multivector_reservoir_mean_docs`,
  `multivector_reservoir_per_token_docs`,
  `multivector_reservoir_bm25_docs`,
  `multivector_reservoir_union_docs`, and
  `multivector_reservoir_duplicates` describe the bounded multi-reservoir
  candidate selector when it is enabled.
- `multivector_bm25_injection_enabled`,
  `multivector_bm25_injection_candidates`,
  `multivector_bm25_injection_retained`, and
  `multivector_bm25_injection_exact_reranked` describe BM25-backed candidate
  injection into the exact MaxSim rerank set.
- `multivector_doc_graph_nodes`,
  `multivector_doc_graph_docs_scored`,
  `multivector_doc_graph_edges_visited`,
  `multivector_doc_graph_candidates`,
  `multivector_doc_graph_quantized_scores`,
  `multivector_doc_graph_heap_fetches`,
  `multivector_doc_graph_exact_rerank_docs`, and
  `multivector_doc_graph_warning` describe document-level prototype or
  `multivector_graph = document_nodes` candidate generation.
- `multivector_unique_docs`, `multivector_duplicate_doc_hits`, and
  `multivector_maxsim_updates` describe the document-level MaxSim accumulator.
- `multivector_doc_candidates` is the retained document candidate count before
  final SQL output.
- `multivector_exact_rerank_docs`, `multivector_exact_rerank_pairs`, and
  `multivector_exact_kernel` describe bounded heap exact rerank work.
- `multivector_accumulator_kind` and `multivector_memory_bytes_estimate` expose
  whether the implementation stayed in touched-document memory, currently a
  scan-local doc-ID hash accumulator with expected `O(C * Q)` storage.

Single-vector scans report multivector disabled, zero counters, and null
multivector kernel / accumulator names.

### Exact Token Oracle

`turbohybrid.multivector_candidate_source` defaults to `graph`. In that mode,
each query token uses the normal token-node graph traversal before document
aggregation.

For diagnostics, set:

```sql
SET turbohybrid.multivector_candidate_source = 'exact_token_scan';
```

This developer path scores every live stored token node against each query
token using the existing graph scorer, keeps the same bounded top raw hits, and
feeds the unchanged document accumulator and exact heap rerank. It does not
change SQL result identity: results remain document/heap keyed, and token
`nodeId` is only evidence.

Use it to separate failure classes:

- `graph` misses but `exact_token_scan` admits the exact top document: graph or
  quantized token-candidate recall is the problem.
- both `graph` and `exact_token_scan` miss the document at ordinary budgets:
  token-top-K admission, unique-doc caps, document-candidate truncation, or
  exact-rerank prefixing is structurally too lossy.

### Exact Document And Prototype Candidate Sources

`turbohybrid.multivector_candidate_source = 'exact_doc_scan'` is a developer
oracle that bypasses token-node admission and heap-scans MVCC-visible document
multivectors with exact float32 MaxSim. Its stats use
`multivector_candidate_source = exact_doc_scan` and
`multivector_plain_fallback_reason = exact_doc_scan`.

`turbohybrid.multivector_candidate_source = 'doc_graph_prototype'` is the
Prompt 10 validation hook for the document-level graph design. It is
non-default and does not change the on-disk index format. In the current slice
there is no index-resident document graph storage, so the prototype explicitly
uses the same heap-backed exact document MaxSim path and reports:

- `multivector_doc_graph_prototype_enabled`
- `multivector_doc_graph_docs_scored`
- `multivector_doc_graph_edges_visited` (`0` for the heap-backed prototype)
- `multivector_doc_graph_candidates`
- `multivector_doc_graph_heap_fetches`

`turbohybrid.multivector_candidate_source = 'proxy_vector'` requires an index
built with `multivector_graph = document_nodes`. It uses the existing
single-vector TurboQuant graph over document representative vectors for
admission and then exact-reranks admitted heap documents with full MaxSim. Its
stats report `multivector_candidate_source = proxy_vector` and
`multivector_doc_graph_warning =
document_node_proxy_vector_graph_traversal`.

`turbohybrid.multivector_candidate_source = 'document_nodes'` is an explicit
alias for the normal document-node graph path. It requires an index built with
`multivector_graph = document_nodes` and reports
`multivector_candidate_source = document_nodes`.
- `multivector_doc_graph_warning`

The warning is
`prototype_heap_scan_no_index_resident_doc_graph` to make benchmark output
clearly distinguish this validation mode from the future production
document-node graph.

### Multivector Graph Mode

Multivector indexes have an index option:

```sql
WITH (multivector_graph = token_nodes) -- default
```

`token_nodes` is the compatible production storage mode: each document token is
a graph node and the docmap sidecar maps token nodes back to heap/doc results.
`turbohybrid_index_stats()` exposes this persisted mode as
`multivector_graph_mode`.

`document_nodes` is explicit opt-in for the first production document-level
storage slice. It stores one graph node per heap document plus a versioned
index-resident float32 multivector sidecar. Build-time edge selection uses
symmetrized document MaxSim. Non-exhaustive scans traverse document graph
adjacency and score visited document candidates by full float32 sidecar MaxSim
before exact heap rerank; near-exhaustive scans use the exact sidecar scan.
Scans report `document_node_f32_sidecar_graph_traversal` or
`document_node_f32_sidecar_exact_scan` in
`multivector_doc_graph_warning`. The compact quantized document scorer is still
future work. Missing or malformed document-node sidecar metadata must fail with
REINDEX guidance instead of silently falling back to token-node storage.

### Synthetic Recall Gate

`benchmarks/dbpedia_colbert_multivector.py --multivector-recall-gate` is the
deterministic Prompt 12 gate for this failure class. It creates a tiny
many-moderate corpus in a temporary table:

- exact MaxSim top-1 is `good`, because it matches several query tokens
  moderately;
- low-budget token-node candidate generation admits spike documents instead,
  because each spike wins one individual query token;
- exact document paths must still return and admit `good`.

Run it inside the Nix benchmark shell:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --multivector-recall-gate \
  --database pgturbohybrid_dev \
  --output .nix-dev/tmp/multivector-recall-gate.json \
  --markdown-output .nix-dev/tmp/multivector-recall-gate.md
```

The gate intentionally does not require DBpedia data, BEIR qrels, or a GGUF
model. It is a structural correctness check for candidate admission plumbing,
not a replacement for DBpedia recall quality runs. Token graph, exact token
scan, and reservoir modes may expose the lossy token-top-K behavior at the
small candidate budget; `exact_scan`, forced plain fallback, `exact_doc_scan`,
and `doc_graph_prototype` are required to return and admit the exact top-1.

### Exact Plain Fallback

`turbohybrid.multivector_plain_fallback` defaults to `auto`:

- `off` keeps the normal token-node candidate path.
- `force` heap-scans MVCC-visible document multivectors and scores exact
  float32 MaxSim.
- `auto` chooses the exact heap path when estimated live documents are at or
  below `turbohybrid.multivector_plain_fallback_max_docs` (`1000` by default),
  or when `multivector_doc_candidate_k` / `multivector_exact_rerank_k` is near
  exhaustive according to
  `turbohybrid.multivector_plain_fallback_candidate_fraction` (`0.5` by
  default).

The fallback is a document-level safety path. It bypasses token-node admission
and returns document/heap candidates ordered by exact MaxSim, while hybrid RRF
continues to use those candidates as the dense branch. Unsupported score-level
fusion semantics remain rejected by query validation before candidate
generation.

### Multi-Reservoir Candidate Retention

`turbohybrid.multivector_candidate_reservoirs` defaults to `conservative`,
which builds the final exact-rerank candidate set from a bounded document-keyed
union while preserving the same candidate-count limits. Set it to `off` to
restore legacy score-only document truncation, or to `balanced` for a wider
coverage-oriented union of:

- approximate MaxSim-sum candidates,
- query-token coverage candidates,
- mean seen-similarity candidates,
- combined per-query-token document candidates,
- BM25-injected candidates when a hybrid/text branch is available.

The per-token reservoir width is controlled by
`turbohybrid.multivector_per_token_doc_reservoir_k` (`1` by default). Coverage
and mean seen-similarity reservoirs use
`turbohybrid.multivector_coverage_reservoir_k` (`10` by default). Final
candidate count remains bounded by `multivector_doc_candidate_k`, the dense
branch target, exact rerank caps, and accumulator memory checks. Reservoirs
deduplicate by document ID; `nodeId` remains only token evidence.

This is a mitigation for token-node candidate generation, not the final
document-level MaxSim architecture. It can retain more useful documents when
pre-rerank evidence exists in the accumulator, but it cannot recover documents
that no query-token search admitted at all.

### BM25 Candidate Injection

`turbohybrid.multivector_bm25_candidate_injection` defaults to `off`. Set it to
`hybrid_only` to inject BM25 candidates for hybrid multivector/text queries, or
to `dense_with_text` to also allow text-backed dense-only MaxSim admission.

When enabled, the scan collects BM25 candidates from the lexical branch,
deduplicates them by heap tuple/document identity, heap-fetches their stored
multivectors, scores them with exact float32 MaxSim, and merges retained
candidates into the dense rerank pool before fusion. `nodeId` remains only
subvector evidence and is never the SQL result key.

For dense-only text-backed admission, BM25 is an admission source; exact MaxSim
is still the dense score. RRF remains document-level for explicit hybrid RRF
queries. Unsupported score-level fusion validation is unchanged.

Injection remains bounded by the BM25 candidate budget, the existing
`multivector_doc_candidate_k` and `multivector_exact_rerank_k` caps, and the
same exact-rerank memory accounting as ordinary multivector candidates.

### Admission Debug Diagnostics

Candidate admission diagnostics are disabled by default with:

```sql
SET turbohybrid.multivector_debug_admission = 'off';
```

Two debug modes are available for multivector scans:

- `summary` records scan-level admission counters without a per-document trace.
- `trace` records the same counters plus a bounded document-keyed trace in
  `turbohybrid_last_scan_stats()`.

The trace is capped by `turbohybrid.multivector_debug_trace_limit`, default
`1000`, with a hard maximum of `1000`. The trace is emitted only when debug
mode is `trace`; default and `summary` mode do not include the
`multivector_admission_trace` JSON payload.

Summary fields:

- `multivector_admission_debug_enabled`
- `multivector_admission_candidates_before_rerank`
- `multivector_admission_candidates_after_truncation`
- `multivector_admission_exact_rerank_docs`
- `multivector_admission_truncated_by_doc_candidate_k`
- `multivector_admission_truncated_by_accumulator_memory`
- `multivector_admission_trace_available`

Trace entries are document-keyed and include:

- `doc_id`
- `heap_tid`, `heap_block`, and `heap_offset`
- `best_node_id`
- `approximate_score_before_rerank`
- `query_token_coverage_count`
- `raw_hit_count`
- `duplicate_hit_count`
- `candidate_rank_before_truncation`
- `retained_for_exact_rerank`
- `exact_rerank_score`, or `null` if that document was not exact reranked

`multivector_admission_truncated_by_accumulator_memory` currently reports
`false` because accumulator memory exhaustion is a hard error, not a silent
truncation path. If that behavior changes, this field must distinguish actual
memory truncation from `multivector_doc_candidate_k` truncation.

## Guardrails

The multivector scan path is bounded before allocating per-query state:

- `turbohybrid.multivector_max_doc_vectors` caps document-token expansion at
  build time.
- `turbohybrid.multivector_max_query_vectors` and
  `turbohybrid.multivector_max_dim` reject pathological query payloads.
- `turbohybrid.multivector_max_raw_hits_per_token`,
  `turbohybrid.multivector_unique_docs_per_token`, and
  `turbohybrid.multivector_doc_candidate_k` bound approximate candidate
  collection and retained document candidates.
- `turbohybrid.multivector_exact_rerank_k` bounds heap fetches and exact
  `R * Q * L * d` work.
- `turbohybrid.multivector_max_accumulator_mb` rejects queries whose
  touched-document hash accumulator would exceed the configured memory budget.

The accumulator is a touched-document hash table keyed by scan-local `TqDocId`
with per-document `Q` arrays, so it avoids global `D * Q` or `N * Q` clearing.
Bulk build and incremental insert/update both expand each multivector row into
one graph subnode per document vector. Hybrid indexes append one BM25 delta per
inserted document when the lexical key is non-null.

## Known Tradeoffs

- Token-node candidate generation can miss many-moderate-match documents. A
  document can be exact MaxSim top-1 by summing moderate matches across query
  tokens while never being the best enough hit for any individual query token.
- Exact/plain fallback is safer for small or near-exhaustive cases because it
  scores MVCC-visible document multivectors with full float32 MaxSim and
  bypasses token-node admission.
- Multi-reservoir retention and BM25 candidate injection are bounded
  mitigations. They improve retention after evidence reaches the accumulator,
  but cannot recover documents that no token or injected candidate source
  admitted.
- A document-level graph aligns candidate generation with MaxSim better than a
  token-node graph. `document_nodes` is available as explicit opt-in and scores
  document candidates with full sidecar MaxSim, but the current warning marks
  that this is an exact float32 sidecar scan rather than final compact
  quantized HNSW traversal.
- Exact rerank remains the semantic final scorer. Approximate token hits,
  reservoirs, BM25 injection, and prototype document candidates only determine
  which documents are eligible for exact MaxSim scoring.
