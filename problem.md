# Multivector Document-Node Index Build Performance Problem

> **Status update (2026-08-17):** the recommended scorer from "Fix Direction"
> item 1 is implemented — `multivector_doc_graph_build_scorer` supports
> `proxy` (default at scale) and `exact_symmetric`, with exact MaxSim kept for
> final query rerank. Items 2–3 (defaults + parallel edge build under proxy
> scoring) and the acceptance-criteria benchmark re-run remain open. See
> `CHANGELOG.md` for the shipped behavior.

## Summary

The current `multivector_graph = document_nodes` index build path is not
usable for DBpedia-scale ColBERT/ModernBERT multivectors. The build spends most
of its time after heap scan completion, inside native graph edge construction,
on a single PostgreSQL backend.

This is not primarily a data-loading problem and not primarily a hybrid/BM25
problem. A colbert-only scratch index shows the same behavior.

## Observed Failure

The 10k DBpedia ColBERT benchmark was run inside the Nix development
environment against an already-loaded table:

- Database: `pgturbohybrid_dbpedia_colbert_10k`
- Rows: `10000`
- Populated multivectors: `10000`
- Table size: about `278 MB`
- PostgreSQL: `17.10` from the repository Nix flake
- Index mode: `multivector_graph = document_nodes`

During `CREATE INDEX`, `pg_stat_progress_create_index` showed that heap blocks
and tuples had already been consumed:

- `blocks_done = blocks_total`
- `tuples_done = 10000`

The backend then stayed CPU-bound in `CREATE INDEX` at roughly one full core.
That points to the post-scan graph edge build, not heap loading or tuple
conversion.

## Bounded Micro-Probe Evidence

The following probes used scratch tables created from the already-loaded
DBpedia rows. They built colbert-only turbohybrid indexes to isolate the
multivector graph builder from BM25/hybrid behavior.

### 100 Rows, Default Graph Build Settings

Reloptions:

```sql
WITH (
  quantization_bits = 4,
  exact_storage = off,
  multivector_graph = document_nodes
)
```

Observed stats:

- Total build time: `2.57 s`
- `build_edges_us`: `2.45 s`
- `node_count`: `100`
- `m`: `16`
- `ef_construction`: `128`
- `build_fast_edges`: `false`
- `build_edges_distance_calls`: `159835`
- `native_build_workers_requested`: `0`
- `parallel_edge_build_enabled`: `false`

### 100 Rows, Small Graph Build Settings

Reloptions:

```sql
WITH (
  quantization_bits = 4,
  exact_storage = off,
  multivector_graph = document_nodes,
  graph_m = 4,
  graph_ef_construction = 8,
  graph_ef_search = 8
)
```

Observed stats:

- Total build time: `0.56 s`
- `build_edges_us`: `0.44 s`
- `node_count`: `100`
- `m`: `4`
- `ef_construction`: `8`
- `build_fast_edges`: `false`
- `build_edges_distance_calls`: `18637`
- `native_build_workers_requested`: `0`
- `parallel_edge_build_enabled`: `false`

### 250 Rows, Small Graph Build Settings

Observed stats:

- Total build time: `1.90 s`
- `build_edges_us`: `1.45 s`
- `node_count`: `250`
- `m`: `4`
- `ef_construction`: `8`
- `build_fast_edges`: `false`
- `build_edges_distance_calls`: `56290`
- `native_build_workers_requested`: `0`
- `parallel_edge_build_enabled`: `false`

### 200 Rows, Default Graph Build Settings

Observed stats:

- Total build time: `11.31 s`
- `build_edges_us`: `10.84 s`
- `node_count`: `200`
- `m`: `16`
- `ef_construction`: `128`
- `build_fast_edges`: `false`
- `build_edges_distance_calls`: `540017`
- `native_build_workers_requested`: `0`
- `parallel_edge_build_enabled`: `false`

This is already unusable at a few hundred rows with default settings. The
10k-row build is therefore expected to run for a very long time or appear hung.

## DBpedia Multivector Shape

The loaded DBpedia sample has typical RAG chunk-sized multivectors:

- Documents: `10000`
- Minimum tokens: `9`
- Median tokens: `46`
- p90 tokens: `94`
- Maximum tokens: `256`
- Average tokens: `51.96`
- Dimensions: `128`

Each document-document distance call is therefore not a single 128-dimensional
vector comparison. It is a late-interaction token matrix comparison.

For average documents, one document-document comparison is roughly:

```text
52 * 52 * 128 float operations
```

before accounting for MaxSim bookkeeping and symmetric averaging. That cost is
then multiplied by hundreds of thousands or millions of graph-build distance
calls.

## Source-Level Cause

### 1. Document-node graph build uses exact document MaxSim for graph distances

In `src/pgturbohybrid_quant_score.c`, `PgturbohybridGraphBuildDistance()` has a
special branch for document-node multivector builds:

```c
if (state->multivectorBuild &&
    state->multivectorGraphMode ==
    PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
    state->multivectorNodeMap != NULL &&
    state->multivectorDocVectors != NULL)
{
    ...
    distance = -TqMultiVectorSymmetricMaxSimAverageUnchecked(aDoc, bDoc);
    ...
}
```

That means the HNSW-like build does not use the cheap proxy vector distance for
document nodes. It wires the graph using exact document-document symmetric
MaxSim over the full multivectors.

This is the correctness-oriented behavior we wanted for document-level graph
geometry, but it is far too expensive as the default build path for real RAG
chunks.

### 2. The exact document MaxSim primitive is token-matrix cost

`TqMultiVectorSymmetricMaxSimAverageUnchecked()` computes symmetric MaxSim by
iterating across token blocks from both documents and calculating dot products
over the full embedding dimension. SIMD helps each dot product, but it does not
change the algorithmic shape:

```text
O(tokens_a * tokens_b * dimensions)
```

With p50/p90 token counts of `46`/`94`, this is expensive even before graph
search multiplies the number of comparisons.

### 3. Default graph build settings amplify the problem

Default scratch-build stats showed:

- `m = 16`
- `ef_construction = 128`
- `build_neighbor_select = heuristic`
- `build_fast_edges = false`

The heuristic neighbor-selection path performs repeated neighbor pruning. That
adds more calls to `PgturbohybridGraphBuildDistance()` through
`PgturbohybridGraphSelectNeighbors()` and `PgturbohybridGraphPruneNeighbors()`.

For only 200 rows, this produced `540017` distance calls.

### 4. Multivector builds disable native build workers

In `src/pgturbohybrid_quant.c`, `tqgraphbuild()` requests workers and then
unconditionally disables them for multivector builds:

```c
if (heap != NULL)
    workerRequest = PgturbohybridNativeBuildWorkerRequest(heap, index);
if (state.multivectorBuild)
    workerRequest = 0;
```

The build stats confirm this:

- `native_build_workers_requested = 0`
- `native_build_workers_launched = 0`
- `parallel_edge_build_enabled = false`

So the expensive exact document-node graph construction runs on one backend.

This is probably intentional for the current implementation, because the
parallel edge path is restricted to code-only graph builds and does not carry
the full multivector document sidecar into shared worker state. But the effect
is that document-node multivector builds get the most expensive distance
function and no parallelism.

## Why This Crashes or Appears to Hang

The build is CPU-bound after scan completion. There is no useful wait event
because PostgreSQL is actively executing C code in the graph builder.

At 10k documents, the builder combines:

1. One graph node per document.
2. Exact document-document symmetric MaxSim for every build distance.
3. Default `m=16` and `ef_construction=128`.
4. Heuristic neighbor selection and reverse-neighbor pruning.
5. Single-backend execution.

That creates an enormous number of token-matrix MaxSim evaluations. Even if the
distance cache hits often, the cache miss count is still large, and the prune
path itself remains expensive.

The user-visible symptom is a `CREATE INDEX` statement that has already scanned
all table rows, then burns one CPU core for minutes or longer without producing
benchmark output.

## What Is Wrong

The implementation currently treats exact document-level MaxSim as the graph
construction distance. That is too expensive for build-time topology creation
on real ColBERT/ModernBERT RAG chunks.

The core mismatch is:

```text
Correct final scoring primitive != affordable graph construction primitive
```

Exact MaxSim is appropriate for final rerank/scoring. It is not affordable as
the default distance for every HNSW edge-search, neighbor-selection, and
reverse-prune comparison during index build.

## Fix Direction

The index needs a cheaper candidate-generation/build topology path while still
keeping exact MaxSim for final document scoring.

Viable directions:

1. Add an explicit document-node build scorer.
   - Example: `multivector_doc_graph_build_scorer = proxy | exact_symmetric`.
   - `proxy` uses the existing document proxy vector for graph construction.
   - `exact_symmetric` keeps the current exact behavior for small validation
     cases.
   - Final query rerank still uses exact MaxSim.

2. Make benchmark and production defaults avoid exact document-node build
   scoring at scale.
   - Use cheap proxy topology plus exact rerank.
   - Keep exact topology as an opt-in diagnostic mode.

3. Enable faster edge construction for document-node builds when using proxy
   build scoring.
   - The current parallel edge path requires code-only build state.
   - Proxy scoring can satisfy that; exact multivector sidecar scoring cannot
     without redesigning shared worker state.

4. Revisit neighbor-selection defaults for multivector document nodes.
   - `build_fast_edges` currently stays false for 128 dimensions because auto
     classifies it as low-dimensional.
   - For document-node multivectors, the effective distance is not
     low-dimensional. It is token-matrix MaxSim.
   - Auto selection should account for multivector document-node cost, not only
     proxy vector dimension.

5. Keep small exact-build tests.
   - Exact document-node topology remains valuable for correctness tests and
     small golden examples.
   - It should not be the large-scale benchmark or production build default.

## Immediate Operational Workaround

Until the build path is fixed, do not run 10k/100k/1M DBpedia document-node
multivector builds with default graph settings.

For bounded experiments, use small graph settings and hard statement timeouts:

```sql
WITH (
  quantization_bits = 4,
  exact_storage = off,
  multivector_graph = document_nodes,
  graph_m = 4,
  graph_ef_construction = 8,
  graph_ef_search = 8
)
```

This is still not a real fix. It only reduces the number of exact MaxSim calls.
The index build remains fundamentally dominated by document-document MaxSim.

## Required Acceptance Criteria For A Real Fix

A real fix should demonstrate:

- 10k DBpedia document-node index build completes in practical time.
- Build stats show graph edge construction no longer dominated by exact
  document-document MaxSim.
- `native_build_workers_requested` and/or fast edge construction are usable for
  the scalable build path where appropriate.
- Recall@10 is measured after exact MaxSim rerank, not just proxy retrieval.
- Exact document-node topology remains available for small validation tests.
- The benchmark refuses or clearly labels the slow exact-topology mode so it is
  not accidentally used for 100k/1M runs.
