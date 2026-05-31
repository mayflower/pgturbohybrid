# Native Parallel Edge Insertion Investigation

This document evaluates parallelizing `PgturbohybridGraphBuildEdges()` inside one
native graph. It is intentionally not an implementation plan for the default
build path. Serial edge insertion remains the reference behavior.

## Preconditions

Parallel insertion into a single graph should not be attempted until these are
all true:

- Phase 4A build stats are present and machine-readable.
- Phase 4B parallel heap scan, correction fitting, vector copy/detoast, payload
  extraction, and code encoding are present.
- Phase 4C segmented graph build has been benchmarked across build time, query
  latency, recall, and index size.
- `build_edges_us` is still the dominant build-time phase after the earlier
  phases are parallelized.
- Segmented build is either insufficiently fast or has unacceptable recall/query
  tradeoffs for the target workload.

The current insertion loop is sequential by design. It picks an order, maintains
`inserted[]`, searches the graph built so far from the current entry point,
selects neighbors, mutates neighbor arrays and reverse links, and updates the
entry node/level.

## Instrumentation

`turbohybrid_last_build_stats()` exposes edge-build counters for deciding
whether single-graph parallel insertion is worth pursuing:

- `build_edges_distance_calls`: distance calls made during `build_edges`.
- `build_edges_search_layer_us`: time in build-time greedy/search-layer
  traversal.
- `build_edges_select_neighbor_us`: time in neighbor selection.
- `build_edges_add_neighbor_us`: time adding forward/reverse links from
  selected candidates.
- `build_edges_prune_neighbor_us`: time spent pruning overfull neighbor lists.
- `build_edges_entry_update_us`: time updating segment/global entry metadata.
- `build_edges_max_frontier_size`: largest build search frontier observed.
- `build_edges_average_nearest_count`: mean nearest-list size returned by
  search-layer calls.

Some timers are intentionally inclusive. For example, reverse-link insertion can
trigger pruning, so `build_edges_add_neighbor_us` and
`build_edges_prune_neighbor_us` are not expected to sum cleanly to
`build_edges_us`. The counters are diagnostic attribution signals, not a flame
graph replacement.

## Strategy A: Lock-Striped Concurrent Insertion Into One Graph

Workers take disjoint node ranges from the chosen build order and insert into one
shared graph. Mutable node adjacency state is protected by per-node locks or
striped locks covering `neighborCounts`, `neighbors[level][]`, and
`neighborDistances`. Entry metadata (`entryNodeId`, `entryLevel`) uses a separate
lock.

Correctness risks:

- HNSW insertion assumes every new node searches a graph containing exactly the
  previously inserted nodes. Concurrent workers observe a moving graph where
  other insertions may be partially visible.
- `inserted[]` no longer defines a strict prefix of the build order.
- Search can miss better candidates that are inserted concurrently after the
  search starts.
- Reverse-link pruning can remove links chosen by another worker moments earlier.

Race conditions:

- Concurrent updates to `neighborCounts` and neighbor arrays.
- Duplicate reverse links if two workers add the same edge.
- Read/write races while search traverses a neighbor list being pruned.
- Entry node races when two higher-level nodes are inserted concurrently.

Determinism:

- Output is nondeterministic unless the implementation serializes enough work to
  erase most speedup.
- Regression tests must compare validity and recall bands, not byte-identical
  graph pages.

Recall risks:

- Medium to high. The graph topology changes because searches see a different
  candidate universe than serial insertion.
- Risks are highest for small `efConstruction`, low-bit code-only builds, and
  high duplicate/clustered data.

Memory overhead:

- Lock table sized by node count or stripe count.
- Worker-local frontier/nearest buffers.
- Optional per-worker staged reverse-link buffers to reduce lock hold time.

Expected speedup:

- Low to moderate. Distance work can parallelize, but lock contention is likely
  around popular high-level entry paths and hot reverse-link targets.
- Speedup should be measured at workers `1,2,4,8`; poor scaling is expected when
  `build_edges_prune_neighbor_us` or entry/search contention dominates.

Test plan:

- Validate no neighbor count exceeds `M`/`M0`.
- Validate every neighbor ID is in range and has the required level.
- Validate no self edges and no duplicate edges per node/level.
- Run dense queries and compare recall/precision against serial build.
- Rebuild the same input repeatedly and measure result variance.
- Stress duplicate vectors and tiny indexes.

## Strategy B: Batch/Layered Insertion With Barriers

Nodes are partitioned into batches. Each batch searches a read-only snapshot of
the graph from all prior batches, workers select candidate links in parallel,
then a barrier applies adjacency mutations and reverse links for the whole batch.
Higher layers can use smaller or stricter batches than level 0.

Correctness risks:

- Nodes in the same batch cannot use each other as search candidates unless a
  secondary intra-batch repair pass is added.
- Serial insertion order is not preserved inside a batch.
- Reverse-link application order changes pruning outcomes.

Race conditions:

- Search phase can be lock-free if prior batches are immutable.
- Apply phase still races unless adjacency writes are partitioned or locked.
- Entry metadata needs deterministic batch-level reduction.

Determinism:

- More deterministic than lock-striped insertion if batch order and apply order
  are fixed.
- Still not byte-identical to serial unless batch size is one.

Recall risks:

- Moderate. Missing same-batch links can reduce local connectivity.
- A repair pass that connects within-batch nearest neighbors can reduce the
  loss, at more distance and memory cost.

Memory overhead:

- Candidate edge buffers for one batch.
- Optional staged reverse-link and repair buffers.
- Worker-local search buffers.

Expected speedup:

- Moderate when distance/search dominates and batches are large enough.
- Barrier and apply phases limit scaling. Very small batches preserve quality
  but behave closer to serial.

Test plan:

- Sweep batch sizes and worker counts.
- Compare recall and query p95 against serial and segmented builds.
- Verify graph invariants after every batch in assertion builds.
- Include clustered and duplicate-heavy datasets where missing same-batch links
  are likely to matter.

## Strategy C: Build Multiple Graphs Then Merge/Repair Cross-Links

Workers independently build subgraphs, then a merge phase adds cross-segment
links or repairs entry/routing connectivity so the result behaves more like one
graph than pure segmented search. This differs from the current segmented build
prototype because the final structure would include cross-links and possibly a
single global entry/routing layer.

Correctness risks:

- Cross-link repair is approximate and does not recreate serial HNSW insertion.
- Choosing cross-link candidates cheaply is hard without scanning many boundary
  or representative pairs.
- Reordering after merge must preserve global node IDs and payload references.

Race conditions:

- Subgraph builds can be isolated and mostly race-free.
- Merge/repair can be serial first, or use locks only while adding cross-links.
- Entry metadata can be derived after all subgraphs are immutable.

Determinism:

- Deterministic if partitioning, per-segment build order, representative
  selection, and repair order are fixed.
- Parallel subgraph execution can still change timing but not output if each
  worker owns a segment and writes deterministic buffers.

Recall risks:

- Low to moderate if enough cross-links are added.
- Query latency can rise if search still has to try multiple entries or segments.
- Too few cross-links can behave like pure segmentation and lose recall on
  nearest neighbors split across segment boundaries.

Memory overhead:

- Per-segment build buffers and metadata.
- Representative/cross-link candidate buffers.
- Optional temporary exact vectors or sketches for repair scoring.

Expected speedup:

- High for edge construction because independent subgraphs avoid the mutable
  single-graph hot path.
- Merge/repair cost should be much smaller than full serial edge insertion if
  representative sets are bounded.

Test plan:

- Compare against current `native_segments` prototype and serial single graph.
- Sweep segment counts, cross-link fanout, and repair candidate budgets.
- Measure build time, `build_edges_us`, total time, precision/recall, query p95,
  and index size.
- Verify graph invariants before and after repair.

## Recommendation

Do not start with lock-striped insertion. It attacks the most dependency-heavy
part of HNSW directly and is likely to trade serial build time for subtle
correctness, recall, and nondeterminism problems.

The preferred order is:

1. Use the new edge-build instrumentation to prove the remaining bottleneck.
2. Finish and benchmark segmented build (`native_segments=1,2,4,8`).
3. If segmentation improves build time but recall/query cost is unacceptable,
   prototype Strategy C with a bounded cross-link repair pass.
4. Consider Strategy B only if a single logical graph is required and batch
   quality can be tuned acceptably.
5. Reserve Strategy A for an explicitly experimental GUC, for example
   `turbohybrid.native_parallel_edge_insert = experimental`, after the above
   evidence exists.

Serial edge insertion remains the default and the correctness oracle.
