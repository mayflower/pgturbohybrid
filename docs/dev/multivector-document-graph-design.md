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
4. Traverse HNSW document nodes with `ef_search`.
5. For each candidate document node, compute approximate full MaxSim:
   - score all query tokens against the candidate document's stored token codes;
   - accumulate query-token maxima;
   - return a document-level score.
6. Keep an oversampled candidate pool.
7. Apply MVCC visibility against the heap TID before final output.
8. Heap-fetch the retained visible prefix and exact-rerank with float32 MaxSim.
9. Return heap/document-keyed SQL rows ordered by distance `-maxsim`.

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

For hybrid RRF, the document-node candidate list is the dense branch. BM25
continues to be heap/document keyed. Score-level fusion remains unsupported
unless tests explicitly prove the normalization contract.

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
- non-exhaustive document-node scans traverse document graph adjacency and score
  visited candidates with full float32 sidecar MaxSim before exact heap rerank;
- near-exhaustive document-node scans use the exact sidecar scan;
- `multivector_doc_graph_warning` reports
  `document_node_f32_sidecar_graph_traversal` or
  `document_node_f32_sidecar_exact_scan`;
- compact quantized document scoring remains future work;
- malformed metadata must not silently fall back to token-node search.

Metapage and sidecar metadata should include:

- multivector graph mode;
- storage format magic;
- storage format version;
- dimension;
- max document token count at build time;
- code storage kind;
- graph scorer kind;
- document node count;
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

Planned compact document-graph options:

- `multivector_doc_graph_storage = f32 | f16 | sq8 | bq | pq`
- `multivector_doc_graph_build_score = symmetric_maxsim | directional_insert`
- `multivector_doc_graph_oversampling = <float>`

Current developer GUC:

- `turbohybrid.multivector_candidate_source =
  graph | exact_token_scan | exact_doc_scan | doc_graph_prototype`

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
  WITH (multivector_graph = document_nodes);
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
