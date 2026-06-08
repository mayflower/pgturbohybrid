# Fast Multivector Document Nodes And Qdrant Parity

This note defines the target fast architecture for `multivector_graph =
document_nodes` before any code changes. It supersedes the earlier idea that a
production document-node graph should build HNSW topology with exact symmetric
document MaxSim.

## Non-Negotiable Invariant

Exact document-document MaxSim in graph construction is forbidden by default.

The only supported default separation is:

- build topology scorer: cheap proxy/code distance;
- search candidate scorer: cheap proxy/code distance;
- final rerank scorer: exact MaxSim over the query multivector and the
  candidate document multivector.

Exact document-document graph topology may exist only as an explicitly named
diagnostic mode for tiny validation tests. It must never be selected silently by
`multivector_graph = document_nodes`.

## Current Root Cause

The current document-node build path routes graph-build distance calls through
exact symmetric document MaxSim. In `PgturbohybridGraphBuildDistance()`, the
`multivector_graph = document_nodes` branch loads both document multivectors
from build-side sidecar state and returns:

```text
distance = -TqMultiVectorSymmetricMaxSimAverageUnchecked(aDoc, bDoc)
```

That puts the final late-interaction scorer inside HNSW edge construction.
This is the wrong hot path.

Exact document MaxSim has token-matrix cost:

```text
O(tokens_a * tokens_b * dim)
```

For the loaded DBpedia/ColBERT 10k sample, document multivectors are normal RAG
chunk size: about 52 tokens on average, p50 46, p90 94, and max 256 at 128
dimensions. One document-document "distance" is therefore thousands of
128-dimensional dot products, not one proxy-vector comparison.

The graph builder multiplies this by HNSW insertion, layer search,
neighbor-selection, and reciprocal-prune calls. Micro-probes showed the shape:

- 100 rows, default graph settings: 2.57 s total, 2.45 s in `build_edges`,
  159,835 build-distance calls.
- 200 rows, default graph settings: 11.31 s total, 10.84 s in `build_edges`,
  540,017 build-distance calls.
- 250 rows with small graph settings still spent 1.45 s in `build_edges`.

The same build path also disables native build workers for multivectors:

```text
if (state.multivectorBuild)
    workerRequest = 0
```

So the slowest distance function runs single-backend. That is why the 10k
benchmark appears to hang after heap scan completion: PostgreSQL has already
loaded tuples and is burning CPU in graph edge construction.

## Target Architecture

### Document Identity

In `document_nodes` mode:

- one graph node represents one heap document/tuple version;
- graph `nodeId` is a document-node ID;
- `docId` remains the document-sidecar identity;
- `heaptid` remains the PostgreSQL result and visibility identity;
- no token/subvector nodes are created for the document-node graph.

This keeps document-node candidate generation aligned with document-level SQL
results without making graph construction pay exact MaxSim cost.

### Node Vector

Each document node stores a cheap fixed-dimensional proxy vector as its graph
key. The proxy is produced by the configured proxy encoder, for example:

- `normalized_mean`;
- `first_token`;
- `centroid_mean`;
- `mean_pool`;
- `max_pool`;
- `random_projection_fde`;
- a future learned projection only after real weights and validation exist.

The proxy vector is the build/search topology vector. It is not the final
ColBERT score.

### Build Topology Scoring

Graph build must use the existing TurboQuant build machinery over the proxy
vector:

- encode proxy vectors into normal TurboQuant codes;
- use existing code-code/query-code/proxy-vector distance paths;
- allow the normal fast-edge and native parallel build eligibility checks to
  apply where the build is code-only;
- report build stats that prove no exact document-document MaxSim was used.

The default document-node build scorer is therefore conceptually:

```text
multivector_doc_build_scorer = proxy
```

An opt-in diagnostic scorer may exist:

```text
multivector_doc_build_scorer = exact_symmetric
```

but it must be guarded as tiny-test-only. It should require an explicit index
option or developer GUC, emit a clear warning or error above a small row/token
threshold, and be disallowed for benchmark profiles that target 10k/100k/1M
DBpedia scale.

### Search Candidate Scoring

Document-node search should traverse the HNSW graph using the same cheap
candidate scorer family as build:

- proxy query vector against document proxy graph codes; or
- query proxy/code against document proxy/code, depending on the existing
  TurboQuant scorer available for the selected operator and storage.

The search stage returns a bounded candidate band of document/heap identities.
It is a candidate generator, not the final ranker.

### Exact Rerank

Final ranking uses exact MaxSim:

```text
score(Q, D) = sum_i max_j dot(q_i, d_j)
distance = -score
```

The exact rerank input is a bounded candidate band from the proxy graph. The
document multivector source is:

- sidecar-stored document multivectors when the document-node sidecar is valid
  and resident/paged access is enabled;
- heap fallback when sidecar exact values are unavailable but heap rerank is
  explicitly allowed;
- off only for diagnostic candidate-source tests that intentionally measure
  proxy admission without exact rerank.

The scan stats must expose this as:

```text
multivector_exact_rerank_source = sidecar | heap | off
```

This keeps Qdrant-style candidate generation and ColBERT-style final scoring
separate.

## Performance Invariants

The implementation must make these properties observable:

1. Default `document_nodes` build has zero exact document-document MaxSim
   build-distance calls.
2. Default `document_nodes` insert has zero exact document-document MaxSim
   graph-link calls unless an explicit diagnostic scorer is selected.
3. `exact_symmetric` is opt-in, guarded, and documented as a tiny validation
   mode, not a production/default topology scorer.
4. Proxy document-node builds should be eligible for native build workers when
   the underlying build is code-only and otherwise satisfies existing parallel
   build constraints.
5. Proxy document-node edge construction should be eligible for
   `parallel_edge_build_enabled` under the same code-only constraints as normal
   dense TurboQuant builds.
6. Query-time exact MaxSim is bounded by candidate and rerank budgets, not by
   graph node count.
7. A malformed or missing document sidecar must fail clearly or use an
   explicitly configured heap fallback. It must not silently switch to token
   nodes.
8. SQL results remain heap/document keyed; no result ranking or deduplication
   by graph `nodeId`.

## Required Stats

Add or preserve stats that make the scorer split auditable.

Build stats from `turbohybrid_last_build_stats()`:

- `multivector_doc_build_scorer`
  - expected default: `proxy`;
  - diagnostic value: `exact_symmetric`.
- `multivector_doc_exact_build_distance_calls`
  - expected default: `0`.
- `multivector_doc_exact_build_distance_us`
  - expected default: `0`.
- `parallel_edge_build_enabled`
- `native_build_workers_requested`
- `native_build_workers_launched`
- existing totals such as `build_edges_us`, `build_edges_distance_calls`,
  `build_fast_edges`, `build_neighbor_select`, `m`, and `ef_construction`.

Scan stats from `turbohybrid_last_scan_stats()`:

- `multivector_exact_rerank_source = sidecar | heap | off`
- existing document-node counters:
  - `multivector_graph_mode`;
  - `multivector_doc_graph_nodes`;
  - `multivector_doc_graph_edges_visited`;
  - `multivector_doc_graph_candidates`;
  - `multivector_doc_graph_exact_rerank_docs`;
  - sidecar cache/page counters when sidecar rerank is used.

The stats contract is part of the performance fix. A fast build that cannot
prove exact document-document build calls are zero is not acceptable.

## Qdrant Parity Benchmark Plan

Use the same generated DBpedia/ColBERT artifacts for both systems:

- same document set and ordering;
- same query set;
- same qrels;
- same ColBERT model outputs;
- same 10k, 100k, and 1M subsets;
- same final `recall@10` and latency reporting shape.

Compare at minimum:

- index build wall time;
- index size on disk;
- peak build memory if available;
- build CPU parallelism / workers used;
- candidate-generation latency p50/p95;
- final query latency p50/p95 after exact rerank;
- recall@10 after exact rerank;
- nDCG@10 where qrels support it;
- exact rerank candidate count and source;
- failed/underfilled query count.

For `pgturbohybrid`, every benchmark artifact must include:

- `multivector_doc_build_scorer`;
- `multivector_doc_exact_build_distance_calls`;
- `multivector_doc_exact_build_distance_us`;
- `parallel_edge_build_enabled`;
- `native_build_workers_requested`;
- `native_build_workers_launched`;
- `multivector_exact_rerank_source`;
- `build_edges_us`;
- `build_edges_distance_calls`;
- recall@10 after exact rerank;
- p50/p95 query latency after exact rerank.

For Qdrant, record the comparable multivector configuration and HNSW build
parameters. The comparison is not valid if one side uses exact rerank and the
other side reports only approximate candidate scores.

## 10k DBpedia Acceptance Criteria

The first acceptance gate is 10k DBpedia because it is large enough to expose
the current failure and small enough to run repeatedly during development.

A passing 10k document-node build must satisfy all of the following:

- The index builds to completion without manual interruption.
- `multivector_doc_build_scorer = proxy`.
- `multivector_doc_exact_build_distance_calls = 0`.
- `multivector_doc_exact_build_distance_us = 0`.
- `native_build_workers_requested > 0` when PostgreSQL and reloptions permit
  native workers for the same dense/proxy build shape.
- `native_build_workers_launched > 0` in the parallel acceptance profile, or
  the artifact explains the PostgreSQL-level reason workers were unavailable.
- `parallel_edge_build_enabled = true` in the parallel acceptance profile when
  code-only edge construction is otherwise eligible.
- Build stats show `build_edges_us` is no longer dominated by exact
  document-document MaxSim.
- Query artifacts report `multivector_exact_rerank_source = sidecar` for the
  preferred path or `heap` only for an explicitly labeled fallback run.
- Recall@10 is measured after exact MaxSim rerank over the candidate band.
- p50 and p95 query latency are reported for the exact-reranked path.
- Index size is reported and compared against Qdrant on the same 10k subset.
- Any exact-symmetric build-scorer run is labeled diagnostic, capped to tiny
  validation data, and excluded from the fast 10k acceptance result.

The 10k benchmark must fail or prominently mark the run invalid if default
document-node graph construction performs any exact document-document MaxSim
build-distance call.

## Test Coverage Implications

Existing small SQL tests remain useful:

- `test/sql/pgturbohybrid_multivector.sql` checks explicit
  `document_nodes` indexes, sidecar-backed document-node scan stats, exact
  rerank counts, query-token weights/masks, visibility behavior, pooling, and
  guarded candidate sources.
- `test/sql/pgturbohybrid_multivector_many_moderate.sql` captures the
  token-node admission failure where the globally best MaxSim document is lost
  by per-token candidate truncation.

The fast path needs additional tests after implementation:

- default document-node build stats show
  `multivector_doc_exact_build_distance_calls = 0`;
- an explicit `exact_symmetric` diagnostic build shows non-zero calls only on a
  tiny table and is rejected or warned above the guard threshold;
- proxy document-node search still exact-reranks enough candidates to recover
  the `many_moderate` style document when budgets allow;
- `multivector_exact_rerank_source` reports `sidecar`, `heap`, or `off`
  correctly;
- malformed document-node sidecar metadata fails with REINDEX guidance and
  does not fall back to token-node behavior.

## Relationship To The Existing Document-Graph Design

`docs/dev/multivector-document-graph-design.md` correctly identified that
token-node admission is structurally lossy and that document-node candidate
generation is needed. Its current production-scorer text, however, promotes
exact symmetric document MaxSim into graph construction. That is now known to
be the performance root cause.

The corrected target is:

```text
document-node topology: proxy/code graph
document-node candidate search: proxy/code graph
document-node final ranking: exact MaxSim rerank
```

This is the path to Qdrant-competitive behavior: document-level graph nodes and
bounded exact rerank, without exact document-document MaxSim inside the HNSW
build loop.
