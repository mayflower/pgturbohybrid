# Multivector Document Graph Design

Prompt 9 design for a Qdrant-style multivector graph where graph nodes are
documents/heap tuples and every candidate is scored by full MaxSim before the
final exact rerank.

## Problem Statement

The current multivector index expands each document vector into one graph node
per token/subvector. Query execution searches that token graph once per query
token, aggregates token hits into document candidates, truncates the document
candidate set, and exact-reranks only the retained documents.

That architecture is lossy for ColBERT-style late interaction. A document can
be globally top-K by sum-of-query-token MaxSim while none of its individual
document tokens is top-K for any single query token. Once that document misses
per-token candidate admission, exact rerank cannot recover it.

The structural fix is to align candidate generation with the final scorer:

- graph node identity is document/point identity;
- graph traversal scores a candidate node with approximate full MaxSim between
  the query multivector and that document's stored multivector;
- exact heap rerank remains the semantic final scorer.

Qdrant follows this shape. Its HNSW graph works on `PointOffsetType` point
nodes. For multivectors, `MultiMetricQueryScorer::score_stored()` loads the
stored multivector for a point and calls `score_multi()`; `score_internal()`
scores two stored point multivectors for graph construction. Quantized
multivector scoring keeps per-point `MultivectorOffset { start, count }`
entries into flattened inner-vector storage and scores the whole point through
`score_point()`.

## Current Token-Node Architecture

Build-time today:

- `PgturbohybridGraphAppendBuildMultiVector()` expands one heap tuple into `L`
  graph nodes, where `L` is the document token/subvector count.
- `nodeId` means token node.
- `docId` means document aggregate identity.
- `heaptid` is retained through the document map for visibility and output.
- The docmap sidecar records `nodeId -> (docId, tokenOrdinal)` and
  `docId -> heaptid`.

Query-time today:

1. Iterate query tokens.
2. Run graph ANN over token nodes for each query token, or exact-token oracle in
   developer mode.
3. Resolve each token `nodeId` to a document.
4. Update a document accumulator with per-query-token max similarities.
5. Select bounded document candidates.
6. Heap-fetch and exact MaxSim-rerank the retained prefix.

Mitigations now exist: admission diagnostics, exact-token oracle, exact/plain
fallback, multi-reservoir retention, token diagnostics, and BM25 candidate
injection. They reduce or classify admission loss, but the default graph
candidate generator still admits documents through token-level evidence.

## Proposed Document-Node Architecture

Add an explicit document-node multivector graph mode:

```sql
CREATE INDEX ... USING turbohybrid
  (embedding multivector_maxsim_ip_turbohybrid_ops)
  WITH (multivector_graph = document_nodes);
```

In this mode:

- one graph node represents one heap document/tuple version;
- graph node ID is document-node ID, not token-node ID;
- node metadata stores `docId`, `heaptid`, token count, dimension, and compact
  multivector storage location;
- graph traversal candidate scores are approximate full MaxSim over the
  document multivector;
- exact rerank heap-fetches visible retained documents and computes float32
  MaxSim exactly.

The existing `token_nodes` mode remains the default for compatibility:

```sql
WITH (multivector_graph = token_nodes)
```

## Scoring Contract

Query-vs-document scoring is directional:

```text
MaxSim(Q, D) = sum over q in Q of max over d in D dot(q, d)
```

This is the natural ColBERT query scorer and matches existing SQL semantics:
larger MaxSim is better, and SQL distance is `-MaxSim`.

Document-vs-document graph construction needs a stable edge scorer. Directional
`MaxSim(A, B)` is not symmetric, so graph edge selection should use one of
these explicit options:

1. Symmetric MaxSim average:

   ```text
   graph_score(A, B) =
     0.5 * (MaxSim(A, B) / count(A) + MaxSim(B, A) / count(B))
   ```

2. Directional insertion scorer:

   ```text
   graph_score(new, existing) = MaxSim(new, existing) / count(new)
   ```

The recommended production default is option 1. It is more expensive during
build, but it makes bidirectional HNSW links less dependent on insertion order
and avoids silently treating one document's token count as the only normalizer.
Option 2 can be kept as a developer build option if benchmarks show the build
cost is unacceptable.

Search scoring should remain directional query-to-document:

```text
search_score(Q, D) = MaxSim(Q, D)
distance = -search_score
similarity = search_score / count(Q)
```

When `turbohybrid_query` carries query-token weights or masks, document-node
sidecar scoring and final heap rerank use:

```text
search_score(Q, D) = sum_i weight_i * max_j dot(q_i, d_j)
```

Masked query tokens are skipped, omitted weights default to `1.0`, and
similarity normalization uses the sum of unmasked weights. This keeps
document-node traversal, exact-doc fallback, and final rerank on the same
scoring contract.

Approximate quantized scoring may be used for graph traversal and oversampling,
but final exact rerank must use heap float32 MaxSim unless a later prompt
explicitly changes that contract.

## Storage Layout Options

Document graph traversal must not heap-fetch. It needs index-resident or
cache-resident multivector payloads for every document node.

### Option A: Extend The Existing Multivector Docmap Sidecar

Extend the current docmap sidecar into a document-node storage sidecar:

```c
struct TqMultiVectorDocNodeEntry {
    TqDocId docId;
    ItemPointerData heaptid;
    uint32 tokenCount;
    uint32 dim;
    uint64 codeOffset;
    uint32 codeBytes;
    uint32 flags;
};
```

The code stream stores flattened compact per-token codes for each document. The
entry table maps document-node ID to the code range. This is closest to the
current docmap responsibilities and keeps heap TID visibility metadata next to
the document node.

Pros:

- reuses docmap loading, stats, and compatibility checks;
- direct document-node ID to heap TID mapping;
- natural place for token count and code offsets.

Cons:

- current docmap is token-node oriented and would need versioned format changes;
- code payload pages need their own validation and memory accounting.

### Option B: Add A Separate Multivector Code Sidecar

Keep the existing docmap for identity and add a separate code sidecar:

```text
docnodes.meta      magic/version/dim/counts
docnodes.map       docNodeId -> heaptid/docId/tokenCount/codeOffset/codeBytes
docnodes.codes     flattened quantized or float16/f32 token code blocks
```

Pros:

- clearer on-disk compatibility boundary;
- old token-node docmaps can remain untouched;
- easier to make missing code sidecar fail fast.

Cons:

- more files/pages to manage;
- more index option combinations to validate.

### Option C: Native Cache First, Index-Resident Later

Prototype a document-node graph with a scan-local or backend-local native cache
derived from heap/docmap data. This validates recall before introducing an
on-disk format.

Pros:

- no immediate on-disk change;
- useful for Prompt 10 `doc_graph_prototype`;
- direct benchmark comparison against exact scan and token-node modes.

Cons:

- not production-safe for large indexes;
- can accidentally hide storage-layout costs;
- must report heap fetch/cache build stats prominently.

Recommended path:

1. Prototype with Option C.
2. Production storage with Option B unless the docmap versioning work is already
   being done for another reason.
3. Consider merging B into an extended docmap only after the compatibility story
   is proven.

## Build Algorithm

For `multivector_graph = document_nodes`:

1. During index build, read each visible heap tuple's multivector once.
2. Validate dimension and token count with the existing multivector limits.
3. Assign one document node ID for the heap tuple version.
4. Store `docNodeId -> heaptid/docId/tokenCount/codeOffset`.
5. Encode the document multivector into the graph code storage:
   - initial slice: f32 or f16 flattened vectors for correctness;
   - production slice: scalar/binary/product quantized token codes when ready.
6. Insert the document node into HNSW.
7. Use the configured document-document graph scorer for entry search and link
   selection.
8. Store HNSW links over document node IDs.
9. Persist metapage fields and sidecar magic/version after successful build.

Important build invariants:

- never create graph nodes for individual document tokens in document-node mode;
- never use `nodeId` as SQL result identity;
- if code storage or map storage cannot be written, fail the build;
- use PostgreSQL memory contexts and overflow-checked size arithmetic for all
  temporary and persisted allocations.

## Search Algorithm

For a query multivector:

1. Validate query dimension and token count.
2. Choose candidate source:
   - `graph` with `document_nodes` uses document-node HNSW;
   - `graph` with `token_nodes` uses the existing token path;
   - developer modes can force exact token/doc scans.
3. Prepare an approximate document MaxSim scorer over index-resident codes.
4. Score the global, segment, routing, and sampled document-node entry seeds
   with the same document MaxSim scorer, then keep the best bounded entry set.
   The sampled entry phase mirrors the single-vector graph path's distributed
   entry sampling and avoids starting large document-node traversals from one
   segment-local region.
5. Traverse HNSW document nodes with `ef_search`. The traversal breadth is the
   index/default `graph_ef_search` when
   `turbohybrid.multivector_doc_graph_search_ef = 0`, or the explicit GUC value
   when it is positive. In both cases this is a traversal cap and must not be
   inflated by a larger document candidate or exact-rerank budget.
6. For each candidate document node, compute approximate full MaxSim:
   - score all query tokens against the candidate document's stored token codes;
   - accumulate query-token maxima;
   - return a document-level score.
7. Keep an oversampled candidate pool.
8. Apply MVCC visibility against the heap TID before final output.
9. Heap-fetch the retained visible prefix and exact-rerank with float32 MaxSim.
10. Return heap/document-keyed SQL rows ordered by distance `-maxsim`.

The traversal scorer should expose counters:

- document graph nodes visited;
- document graph candidates;
- approximate MaxSim pair comparisons;
- quantized code bytes read;
- exact rerank documents and pairs.

## Exact Rerank Integration

Document-node graph candidates can reuse the existing exact heap rerank shape:

- candidate carries `heaptid`, `docId`, `distance`, `similarity`, and rank;
- exact rerank fetches the heap multivector with PostgreSQL table AM;
- `TqMultiVectorMaxSim(query, doc)` computes the final score;
- final distance is `-maxsim`;
- final rank is document/heap keyed.

The exact rerank cap remains:

```text
min(candidate_count, multivector_doc_candidate_k, multivector_exact_rerank_k)
```

`turbohybrid.multivector_exact_rerank = adaptive` is an exact top-K pruning
mode for this heap rerank prefix. It processes query tokens in deterministic
importance order, tracks the current exact top-K threshold, and uses
weighted query-token norms plus each candidate document's maximum token norm as
a safe upper bound for the remaining MaxSim contribution. Masked and
zero-weight tokens have zero remaining contribution. A candidate is stopped
only after that bound proves it cannot enter the requested top-K. If the final
top-K is not smaller than the rerank prefix, or if the bounds are unsafe, the
scan falls back to full exact MaxSim scoring.

Adaptive work is visible in `turbohybrid_last_scan_stats()` through
`exact_rerank_candidates`, `exact_rerank_tokens_evaluated`,
`exact_rerank_tokens_skipped`, `exact_rerank_pairs_saved`, and
`adaptive_rerank_topk_changed_vs_full`. The top-K changed flag should remain
false for the exact adaptive mode; approximate relaxations need a separate
explicit GUC.

For hybrid fusion, the document-node candidate list is the dense branch. BM25
continues to be heap/document keyed, and final fusion must remain keyed by
heap/document identity rather than graph node ID. RRF is still the safest
default when branch score distributions are small, highly skewed, or unstable,
because it uses branch ranks instead of comparing raw MaxSim and BM25 scales.
Score-level fusion is supported only through explicit normalization modes:
query-local `weighted`, logistic/saturating `fast_weighted`, calibrated fusion,
or `dbsf`. DBSF computes dense and BM25 branch-local mean/stddev diagnostics,
normalizes through clipped sigma endpoints, and reports
`dbsf_degenerate_branches` when a branch has too few or identical returned
scores. If DBSF reports degenerate branches on a workload, prefer RRF until an
evaluation set shows DBSF is better.

A qrel-backed 10k hybrid smoke with three DBpedia queries and `126` loaded
qrels showed that BM25 hybrid admission improves over dense document-node
retrieval but does not yet close the gap to exact MaxSim. Exact scan reached
`recall@10 = 0.800000` and `ndcg@10 = 0.736563` at p50 `350.746 ms`; dense
document nodes reached `recall@10 = 0.166667`, `ndcg@10 = 0.207257`, and
top-10 admission `0.133333` at p50 `74.467 ms`; BM25 admission/RRF/DBSF reached
`recall@10 = 0.400000` with p50 around `69..75 ms` but still had admission
failures for all three queries. Treat this as Prompt 15 harness evidence and a
candidate-generation gap, not as a default-profile quality pass.

## MVCC And Dead Tuple Handling

Document-node graph storage is an index structure and can contain dead tuple
versions. Search must handle that the same logical document may have multiple
heap tuple versions after update.

Rules:

- graph traversal may visit dead document nodes because the index is approximate
  and vacuum may not have removed them yet;
- visibility is checked through heap TID before final output;
- dead candidates are skipped and should count in stats;
- if too many retained candidates are dead, the search should widen or fill the
  candidate band using the existing dense fill-candidate machinery;
- exact rerank only scores visible heap tuples;
- VACUUM cleanup may mark document nodes dead or leave them as tombstoned graph
  nodes until rebuild, following existing graph/index AM constraints.

Additional metadata:

- document node map needs enough heap TID/version information to check
  visibility;
- no persistent logical-document deduplication should happen unless the SQL API
  defines that identity. PostgreSQL heap tuples are the result identity.

## Insert, Update, Delete, And Vacuum

Insert:

1. Assign one document node for the new heap tuple.
2. Append document-node map entry and compact code block.
3. Link the node into the document HNSW graph using document-level build score.
4. Append BM25 delta when a lexical key exists, as today.

Update:

- PostgreSQL index AM receives a new tuple version. Treat it as an insert of a
  new document node.
- Old tuple visibility naturally expires and is later vacuumed.
- Do not mutate an existing node in place unless the access method has a clear
  concurrency and WAL story.

Delete:

- Marked through heap visibility first.
- Optional tombstone sidecar can speed dead-node skips, but final correctness
  must rely on MVCC visibility.

Vacuum:

- may mark document nodes as deleted/tombstoned;
- may reclaim sidecar code blocks only with a versioned compaction/rewrite path;
- otherwise leave reclaim to REINDEX.

## Compatibility Strategy

Index option:

```text
multivector_graph = token_nodes | document_nodes
```

Defaults:

- `token_nodes` for existing and newly created indexes until the document graph
  has benchmark proof and compatibility coverage.
- `document_nodes` requires explicit opt-in.

Current production slice:

- the `multivector_graph` index option is registered and persisted in the
  native graph metapage;
- token-node indexes report `multivector_graph_mode = token_nodes` through
  `turbohybrid_index_stats()`;
- `document_nodes` is explicit opt-in and stores one graph node per heap
  document;
- the versioned docmap sidecar stores document identity plus float32
  multivector chunks for each document node;
- document-node graph edge construction uses the symmetrized document MaxSim
  score described above;
- incremental document-node inserts use the same symmetrized document MaxSim
  scorer for candidate collection, neighbor selection, and reciprocal pruning.
  The fixed-dimensional proxy vector is only the dense graph key stored for
  `proxy_vector` admission; it is not a silent graph-link fallback. Missing or
  corrupt document sidecar data raises an error with REINDEX guidance;
- non-exhaustive document-node scans traverse document graph adjacency and score
  visited candidates with the selected document sidecar storage (`f32`, `f16`,
  or `sq8`) before exact heap rerank;
- near-exhaustive document-node scans use the exact float32 sidecar scan;
- `multivector_doc_graph_warning` reports
  `document_node_f32_sidecar_graph_traversal`,
  `document_node_f16_sidecar_graph_traversal`,
  `document_node_sq8_sidecar_graph_traversal`, or
  `document_node_f32_sidecar_exact_scan`;
- the document sidecar cache mode is explicit:
  `turbohybrid.multivector_doc_storage_cache = auto | resident | paged`.
  `resident` prefers the already loaded native sidecar, `paged` keeps graph
  adjacency/cache metadata resident while loading sidecar pages through shared
  buffers for the scan, and `auto` chooses resident for the latency profile when
  the sidecar fits `turbohybrid.native_cache_max_mb`, otherwise paged;
- sidecar random-access cost is visible through
  `multivector_doc_sidecar_cache_mode`,
  `multivector_doc_sidecar_pages_read`,
  `multivector_doc_sidecar_cache_hits`,
  `multivector_doc_sidecar_cache_misses`, and
  `multivector_doc_sidecar_bytes_touched`; paged scans additionally report
  `multivector_doc_sidecar_vectors_loaded` for on-demand document multivector
  reconstruction;
- long-context metadata can be stored in one document row with
  `turbohybrid_multivector_from_contexts(...)` or
  `turbohybrid_multivector_from_contexts_and_fields(...)`. Context offsets are
  zero-based token starts that must begin at `0` and be strictly increasing.
  `multivector_context_mode = flat` keeps cross-context/global MaxSim.
  `multivector_context_mode = context_level` scores each stored context window
  independently during document-node sidecar scoring and exact heap rerank, then
  uses the best context score for the one heap/document result. This keeps long
  RAG chunks as one SQL row while avoiding accidental cross-window token mixing;
- `multivector_field_mode = weighted` records that field/section metadata is
  expected and exposes the mode in index stats. Query-specific title/body/section
  weights are still supplied through the exact
  `turbohybrid_multivector_field_weighted_maxsim()` SQL scorer until
  multivector query payloads gain field-weight metadata;
- document-token pooling is an index-time document-node option:
  `multivector_token_pooling = off | kmeans | greedy_cosine`,
  `multivector_token_pooling_target_ratio = 0.5`, and
  `multivector_token_pooling_min_tokens = 16`. It pools only stored document
  tokens, never query tokens. The sidecar metadata records both the original
  and pooled token counts per document so scan stats can expose
  `multivector_tokens_original`, `multivector_tokens_pooled`, and
  `multivector_token_pooling_ratio`. Context-aware multivectors reject pooling
  for now, because pooling changes token ordinals and must preserve or rebuild
  context-window metadata before it can be correct;
- proxy-vector admission is controlled by the persisted index option
  `multivector_proxy_encoder = normalized_mean | mean | first_token |
  max_abs_mean | centroid_mean | max_pool | random_projection_fde |
  learned_projection_placeholder | learned_projection_v1`.
  `normalized_mean` is the production document proxy for cosine/IP MaxSim
  models. `centroid_mean` requires `multivector_centroids = kmeans` and uses
  the normalized mean of persisted document centroids as the single graph
  proxy. `max_pool` and `random_projection_fde` are pluggable
  fixed-dimensional encoders for admission benchmarks.
  `learned_projection_placeholder` fails explicitly; `learned_projection_v1`
  is an opt-in file-backed first slice that requires configured projection
  weights and keeps final SQL ranking exact MaxSim;
- malformed metadata must not silently fall back to token-node search.

Document-node incremental insert stats are exposed through
`turbohybrid_last_scan_stats()` as backend-local diagnostics for the most recent
insert:

- `multivector_doc_graph_insert_full_maxsim_edges`;
- `multivector_doc_graph_insert_representative_fallbacks`;
- `multivector_doc_graph_insert_pairs_scored`.

`multivector_doc_graph_insert_representative_fallbacks` should remain zero in
the production path. A non-zero value would indicate an explicitly named debug
fallback, not an automatic correctness compromise.

Metapage and sidecar metadata should include:

- multivector graph mode;
- storage format magic;
- storage format version;
- dimension;
- max document token count at build time;
- code storage kind;
- graph scorer kind;
- document node count;
- per-document original and pooled token counts;
- flags for quantized/exact storage availability.

Compatibility rules:

- old token-node indexes keep working;
- document-node scans must fail clearly if required sidecar metadata is missing
  or malformed;
- no silent fallback from malformed document-node storage to token-node search;
- REINDEX guidance must be included in every incompatibility error;
- physical format changes require a new magic/version or option value.

## GUCs And Index Options

Current index option:

- `multivector_graph = token_nodes | document_nodes`
- `multivector_proxy_encoder = normalized_mean | mean | first_token |
  max_abs_mean | centroid_mean | max_pool | random_projection_fde |
  learned_projection_placeholder | learned_projection_v1`
- `multivector_context_mode = flat | context_level`
- `multivector_field_mode = off | weighted`
- `multivector_token_pooling = off | kmeans | greedy_cosine`
- `multivector_token_pooling_target_ratio = 0.5`
- `multivector_token_pooling_min_tokens = 16`

Current compact document-graph storage GUC:

- `turbohybrid.multivector_doc_storage = f32 | f16 | sq8`
- `turbohybrid.multivector_doc_storage_cache = auto | resident | paged`

Planned compact document-graph extensions:

- binary/product quantized document storage;
- configurable document graph build score variants beyond the current
  symmetric MaxSim objective.

Current developer GUC:

- `turbohybrid.multivector_candidate_source =
  graph | document_nodes | exact_token_scan | exact_doc_scan | doc_graph_prototype |
  proxy_vector | centroid_lite | quantized_inverted_experimental`

`document_nodes` is an explicit source alias for the document-node graph path
and requires `multivector_graph = document_nodes`. `proxy_vector` is a
document-node prototype: it uses the persisted fixed-dimensional proxy encoder
as the single-vector TurboQuant graph key for admission, then reranks admitted
documents with exact MaxSim. Proxy scans report `proxy_encoder_kind`,
`proxy_candidates`, `proxy_top1_admission`, and `proxy_exact_rerank_docs`.
`centroid_lite` is the opt-in PLAID-lite compatibility branch for indexes built
with `multivector_centroids = kmeans`, and
`quantized_inverted_experimental` is the guarded ColBERTSaR-style research
branch with unstable persisted posting tuples. Both branches remain candidate
generators only; final ranking still uses exact MaxSim rerank.

Adaptive hybrid scheduling is multivector-aware. When
`turbohybrid.hybrid_budget_policy = adaptive` and BM25 budgets were defaulted,
the scheduler uses multivector admission stats after the dense branch runs:
document-level sources with non-truncated exact-reranked dense admission can
reduce the BM25 branch, while underfilled or truncated admission keeps BM25 wide.

Planned compact document-graph GUCs:

- `turbohybrid.multivector_doc_graph_heap_fetch = off | prototype_only`
- `turbohybrid.multivector_doc_graph_exact_storage = auto | require | off`

Current scan stats:

- `multivector_graph_mode`
- `multivector_doc_graph_nodes`
- `multivector_doc_graph_docs_scored`
- `multivector_doc_graph_edges_visited`
- `multivector_doc_graph_candidates`
- `multivector_doc_graph_quantized_scores`
- `multivector_doc_graph_heap_fetches`
- `multivector_doc_graph_exact_rerank_docs`
- `multivector_doc_graph_warning`
- `proxy_encoder_kind`
- `proxy_candidates`
- `proxy_top1_admission`
- `proxy_exact_rerank_docs`
- `centroid_lists_visited`
- `centroid_docs_touched`
- `centroid_pruned_docs`
- `centroid_candidates`
- `quantized_inverted_lists_visited`
- `quantized_inverted_postings_touched`
- `quantized_inverted_docs_scored`
- `quantized_inverted_candidates`
- `quantized_inverted_exact_rerank_docs`
- `quantized_inverted_codebook_size`

Planned compact document-graph scan stats:

- `multivector_doc_graph_code_bytes_read`
- `multivector_doc_graph_dead_candidates`

## Migration And REINDEX Guidance

Existing indexes:

- remain `token_nodes`;
- do not gain document-node graph behavior automatically;
- continue to use token-node docmap, reservoirs, BM25 injection, and fallback.

To opt into document-node graph:

```sql
DROP INDEX old_idx;
CREATE INDEX new_idx ON docs USING turbohybrid
  (embedding multivector_maxsim_ip_turbohybrid_ops)
  WITH (
    multivector_graph = document_nodes,
    multivector_proxy_encoder = normalized_mean
  );
```

If an index has document-node metapage mode but missing or incompatible sidecar
metadata, queries should error:

```text
document-node multivector storage is missing or incompatible; REINDEX required
```

No compatibility bridge should reinterpret token-node sidecars as document-node
storage. The node cardinality and semantics differ.

## Testing Plan

Unit/regression:

- build a `document_nodes` multivector index;
- reject invalid option combinations and unsupported opclasses;
- assert graph stats report `multivector_graph_mode = document_nodes`;
- verify SQL results are heap/document keyed, not token-node keyed;
- deterministic many-moderate-matches case: document-node graph admits the exact
  top-1 at small candidate budgets;
- exact rerank parity with `turbohybrid_multivector_maxsim_distance()`;
- insert/update/delete visibility;
- VACUUM does not return dead tuples;
- old token-node indexes still query;
- malformed/missing sidecar produces REINDEX guidance.

Benchmark:

- compare token graph, exact token scan, exact doc scan, doc graph prototype,
  and eventual document-node graph;
- report exact top-1 admission rate and exact top-10 admission recall;
- report p50/p95 latency, nodes visited, approximate score count, exact rerank
  docs, and memory/code bytes read;
- include DBpedia ColBERT 10k, 100k, and 1M when data is available.

Current 10k DBpedia document-node evidence, using
`document_nodes`/`f32`/`auto`, EF `800`, oversampling `8`, 10 queries, and
admission budgets `50..10000`, shows that production document-node admission is
not a zero-admission path once result-doc inference is counted correctly. Exact
top-1/top-10 admission is `0.30/0.43` at budget `50`, `0.70/0.63` at budget
`800`, and `1.00/1.00` by budget `1600`. The remaining problem is cost:
budget `1600` scored all 10k documents with p50 latency around 433 ms, while
near-exhaustive fallback at budgets `6400` and `10000` also scored all
documents but avoided graph edge traversal and ran around 355 ms p50. Treat this
as evidence that the candidate-admission path works semantically on this slice,
but the graph traversal/fallback boundary still needs performance work before
1M-quality profiles are meaningful.

A later 100k reuse-index probe contradicted the 10k admission result at scale:
document-node recall@10 stayed `0.000000` at budgets `100`, `1600`, and
`10000`, while the budget-10000 query exact-reranked `10000` full multivector
documents and touched about `3.57 GB` of document sidecar data. The conclusion
was that the candidate path was entering or staying in the wrong graph region,
not that final MaxSim rerank was wrong. The fix is the sampled entry-seed phase
above: document-node scans now score a bounded deterministic spread of document
nodes with document MaxSim before base-layer traversal, instead of relying only
on the global, segment, and routing entries.

A later bounded 10k smoke with the fused symmetric build scorer,
`document_nodes`/`f32`/`auto`, one query, index build knobs `graph_m = 4`,
`graph_ef_construction = 8`, and `graph_ef_search = 8`, completed with a
single-column document-node index in `204146.563 ms`. The build remained
edge-build dominated (`build_edge_us = 190107776`) but no longer reproduced the
earlier >10 minute stalled setup. Query-side evidence came from real
document-node traversal rather than the small-doc fallback: budget `50` scored
333 documents and visited 604 graph edges; budget `800` scored 1654 documents,
visited 6428 graph edges, and admitted 4/10 exact top documents for that query.
The exact top-1 still missed at both budgets, so this is a setup/performance
improvement and not yet a Prompt 2 quality pass.

After qrel-prioritized precomputed loading was added, the same 10k document-node
smoke with three DBpedia queries retained `126` qrels instead of the invalid
zero-qrel slice. With `graph_m = 4`, `graph_ef_construction = 8`,
`graph_ef_search = 8`, and budgets `50,800,1600`, it produced
`recall@10 = 0.166667`, `ndcg@10 = 0.207257`, exact top-1 admission
`0.333333`, and exact top-10 admission recall `0.400000` at budget `1600`.
Index build took `145646.847 ms` and remained edge-build dominated
(`build_edge_us = 133304010`). This is the first qrel-backed evidence from the
precomputed dataset path; it proves the benchmark gate now measures real
retrieval quality, but it still does not satisfy the full Prompt 2 comparison
matrix across token baselines, exact scans, storage modes, and proxy branches.

A follow-up narrow admission-grid smoke on the same qrel-backed 10k slice used
`f32` only, no pooling, EF `50`, oversampling `1`, and budgets `50,800`. At
budget `800`, token nodes reached `recall@10 = 0.666667`, `ndcg@10 = 0.600137`,
exact top-1 admission `0.666667`, and exact top-10 admission `0.466667`; the
tested document-node profile reached `recall@10 = 0.300000`,
`ndcg@10 = 0.375685`, exact top-1 admission `0.333333`, and exact top-10
admission `0.300000`. Exact document scan and plain fallback reached
`recall@10 = 0.800000` with full admission. This proves the Prompt 2 comparison
report can now expose document-node losses against token and exact baselines,
but the full gate still needs the storage, EF, oversampling, proxy, and larger
query matrix.

A wider 10k qrel-backed Prompt 2 matrix now covers the requested storage and
search grids for a bounded local gate: `10` DBpedia queries, `382` qrels,
storage `f32,f16,sq8`, EF `50,100,200,400,800`, oversampling `1,2,4,8`, no
pooling, `auto` cache, and budgets `50,800,1600`. At budget `1600`, exact
document scan and forced plain fallback reached `recall@10 = 0.667143`,
`ndcg@10 = 0.591325`, and full top-1/top-10 admission. Document-node rows with
oversampling `8` matched exact-scan quality and full admission for all three
storage modes, but only by scoring all `10000` documents on this slice.
Oversampling `4` reached top-10 admission `0.490000` and `recall@10 =
0.498571`; oversampling `1` reached top-10 admission `0.420000` and
`recall@10 = 0.448571`. `proxy_vector` was faster at `138.726 ms` p50 but only
reached `recall@10 = 0.398571`; the experimental quantized-inverted branch
reached `recall@10 = 0.657143` but took `35266.755 ms` p50. The current
interpretation is that the document-node candidate path is semantically
recoverable with enough oversampling, but the useful-quality point still
collapses toward exhaustive scoring on this 10k slice.

The Prompt 15 hybrid harness was also rerun on the same 10-query qrel-backed
slice. Exact scan remained the quality, balanced, and high-recall recommendation
with `recall@10 = 0.667143`, `ndcg@10 = 0.591325`, full admission, and
`358.157 ms` p50. Raw document nodes reached only `recall@10 = 0.300000` and
top-10 admission `0.250000`; BM25 admission improved recall to `0.420000` and
top-10 admission to `0.500000` around `74 ms` p50, but still failed candidate
admission for most queries. RRF was the best BM25-fused NDCG row at
`0.372541`, DBSF reached `0.343564`, and `proxy_vector_document_nodes` was the
latency recommendation at `61.388 ms` p50 with `recall@10 = 0.227143`. This
keeps exact document scan as the only defensible quality default until
document-node admission improves without near-exhaustive oversampling.

The learned-sparse Prompt 15 path now has a separate qrel-backed code-path
validation. A deterministic text-hash sparse JSONL fixture updated `1000`
loaded documents and all `10` selected queries on a 1k DBpedia slice with `382`
loaded qrels. `learned_sparse_exact_maxsim` emitted nonzero learned-sparse
candidate stats (`p50 = 68`, mean `55.4` candidates), reached `recall@10 =
0.681429`, `ndcg@10 = 0.591116`, top-10 admission `0.910000`, and `18.756 ms`
p50. Exact scan on the same slice reached `recall@10 = 0.677143`,
`ndcg@10 = 0.593917`, and full admission at `46.206 ms` p50. This proves the
external sparse-vector ingestion and admission branch are wired into the
harness; it is not evidence for a real SPLADE/SPLATE model export.

Compatibility:

- pg_dump/restore for both graph modes;
- extension upgrade with old token-node index present;
- REINDEX after version bump;
- parallel build if supported by existing build infrastructure.

## Staged Implementation Plan

### Phase 1: In-Memory Developer Prototype

- Add `doc_graph_prototype` candidate source.
- Build a scan-local or backend-local document list from heap/docmap data.
- Score document candidates with full MaxSim.
- Optionally use a simple beam search over approximate neighbor lists, or start
  with exact doc scan plus bounded candidate admission stats.
- Report warnings and heap fetch counts so prototype cost is visible.

Exit criteria:

- DBpedia admission recall improves materially over token-node graph at smaller
  candidate budgets.
- Synthetic many-moderate case is admitted without near-exhaustive token K.

### Phase 2: Index-Resident Compact Storage

- Extend the current versioned float32 document-node sidecar with compact
  quantized code blocks.
- Traverse the document graph using compact sidecar scores instead of exact
  sidecar float32 scores.
- Keep exact heap rerank as the final scorer.

Exit criteria:

- no heap fetches during graph traversal;
- old indexes remain compatible;
- missing sidecar fails with REINDEX guidance.

### Phase 3: Quantized Scorer And Oversampling

- Implement SQ/BQ/PQ document MaxSim scorer over compact code storage.
- Add oversampling before exact rerank.
- Add SIMD kernels where the code layout allows contiguous dot-product batches.
- Report quantized score counts and exact top-K admission recall.

Exit criteria:

- approximate traversal plus exact rerank matches exact scan recall targets at
  useful latency on DBpedia.

### Phase 4: Production Safeguards

- Make option validation strict.
- Add WAL/redo and vacuum behavior tests for document-node sidecars.
- Add benchmark gate summaries for admission recall.
- Document migration, known tradeoffs, and REINDEX requirements.

Exit criteria:

- document-node mode is explicit, tested, and safe to recommend for
  ColBERT-style multivectors.

## On-Disk Compatibility Risks

- token-node graph node count is `sum(document token counts)`; document-node
  graph node count is heap tuple count;
- existing node map entries mean token nodes, not document nodes;
- HNSW links over token nodes cannot be reused for document nodes;
- compact code sidecar corruption can make traversal silently wrong unless
  magic/version/count checks are strict;
- changing code storage from f32/f16 to quantized changes score approximation
  and needs explicit metadata;
- graph build scorer changes can change edge topology and requires REINDEX;
- document-token count or dimension metadata mismatch must fail before search;
- vacuum/compaction of sidecar code blocks needs a rewrite plan and cannot be
  bolted on as a silent in-place format change.

The design therefore requires explicit graph mode metadata, versioned sidecars,
and REINDEX guidance for every persisted format or scorer change.
