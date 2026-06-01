# Problem report: graph edge construction runs serially

> Status: superseded by the parallel edge build implementation. Current code
> supports code-only native graph edge construction through the
> `turbohybrid.native_parallel_edge_build` GUC (`auto` by default), using
> batch-parallel insertion into one shared graph. The rejected
> segment-and-entry-repair prototype was fast but collapsed measured recall from
> about 0.9924 to 0.32; see `pedgeplan.md` for the current implementation plan
> and Qdrant comparison notes.

## Summary

For the native quantized `turbohybrid` index, the **graph edge-construction phase
is single-threaded** and dominates build time (≈96%). The parallel-build
machinery (parallel heap scan + parallel quantization encode, and the
`amcanbuildparallel` PG integration) only parallelizes the **scan** and
**encode** phases. The **edge build** that follows runs entirely on the leader
backend, so enabling parallel build gives almost no wall-clock speedup on
recall-oriented builds.

## Evidence (dbpedia-openai-100K-1536, build_workers=8, amcanbuildparallel=true)

`turbohybrid_last_build_stats()` for one build:

| phase | time | parallel? |
| --- | --- | --- |
| scan | 0.88 s | yes (`parallel_scan_enabled: true`, 8 workers) |
| encode (quantization) | 6.12 s | yes (`parallel_encode_enabled: true`) |
| **edge construction** | **78.8 s (fast edges) / 209 s (heuristic)** | **no — leader only** |
| └ of which `search_layer` | 75.1 s (≈95% of edge time) | no |

- `native_build_workers_launched: 8`, `worker_count: 8` — workers *do* launch.
- Sampling `pg_stat_activity` every 2 s for the whole build: **all 40 samples =
  `client backend:1 (run)`** — only the leader is active during edge
  construction; **no `parallel worker` backends are present** for that phase.
- Net effect: import time 223 s with parallel enabled vs 227 s serial — no
  speedup, because the 209 s/79 s edge phase is serial either way.

## How it is structured (code)

The native quantized build has three phases. The first two are parallel; the
third is not.

1. **Parallel scan + encode** — `pgturbohybrid_quant.c` builds a parallel
   context (`CreateParallelContext` / `LaunchParallelWorkers`, ~line 3431/3475).
   Workers read the heap and produce quantized codes. (`worker_scan_us[]`,
   `worker_merge_us`, `parallel_encode_enabled`.)

2. **Serial edge construction** — `PgturbohybridGraphBuildEdges`
   (`pgturbohybrid_quant.c:2369`) runs on the **leader only**, *after* the
   workers have finished:

   ```c
   segmentCount = PgturbohybridGraphGetNativeSegments(index);   // = native_segments reloption
   baseSegmentSize = nodeCount / segmentCount;
   for (segmentIdx = 0; segmentIdx < segmentCount; segmentIdx++) {   // SERIAL loop
       CHECK_FOR_INTERRUPTS();
       PgturbohybridGraphBuildEdgesRange(state, startNodeId, endNodeId, ...);
   }
   ```

   `PgturbohybridGraphBuildEdgesRange` is **only ever called from this serial
   loop** (no worker entry point calls it). For each node it does an HNSW
   `search_layer` greedy traversal (`PgturbohybridGraphBuildSearchLayer`) plus
   neighbor selection — `search_layer` is 95% of the edge cost.

## Why it is serial

1. **The workers are bound to scan/encode, not edges.** They are joined to the
   parallel table scan and the encode pass; once those complete the workers
   exit, and `PgturbohybridGraphBuildEdges` runs on the leader alone.

2. **`native_segments = 1`.** The edge builder partitions work into
   `native_segments` segments. The benchmark/recall config sets
   `native_segments = 1` deliberately, because independent segments are searched
   independently at query time and cost recall ("under-searched segments"). One
   segment ⇒ one serial `BuildEdgesRange` over all nodes.

3. **Even `native_segments > 1` would not parallelize the build.** The segment
   loop (`for segmentIdx …`) is serial on the leader; segments are not
   dispatched to the launched parallel workers. `PgturbohybridGraphGetNativeSegments`
   maps `auto` to `max_parallel_maintenance_workers`, which suggests the segment
   count was *intended* to map to workers — but no code builds segments
   concurrently. So more segments only changes graph topology (and lowers
   recall), without using more than one CPU for the build.

4. A genuinely concurrent insert path *does* exist for the non-quantized graph
   (`pgturbohybrid_build.c`: `InsertTupleInMemory` with a shared `entryLock` +
   per-element LWLocks, pgvector-style). The **native quantized build does not
   use it** — it uses the serial segment builder in `pgturbohybrid_quant.c`.

## Consequences

- Recall-oriented builds (`high_recall` / `matched_recall`, heuristic
  neighbor-select, `ef_construction=256`) are edge-construction-bound and
  single-threaded: ≈209 s at 100K rows, scaling to tens of minutes at 1M+.
- `amcanbuildparallel = true` (currently disabled by commit `3003d0b` for alpha
  safety) is correctness-preserving (recall unchanged at 0.9924) but yields
  almost no build-time improvement, because it accelerates only scan/encode.

## Where to fix (options, not yet implemented)

1. **Parallelize edge construction within one segment.** Dispatch
   `BuildEdgesRange` node-ranges to the launched parallel workers, writing into a
   single shared-memory graph with the fine-grained locking already used by the
   non-quantized `InsertTupleInMemory` path. Preserves recall (single graph);
   highest payoff (`search_layer` is embarrassingly parallel across query nodes,
   modulo lock contention on hot nodes near the entry point).
2. **Parallel multi-segment build + cross-link/merge.** Build `N` segments
   concurrently (one per worker), then add a merge/cross-link pass so query-time
   search is not segment-local — recovering the recall that bare
   `native_segments > 1` loses.
3. **Cheaper interim:** the per-node cost is `search_layer` (95%); reducing
   `ef_construction` or using `dense_build_neighbor_select=fast` cuts build time
   but lowers recall — not acceptable under the "not worse than pgvector" bar.

## Reproduction

```sql
LOAD 'pgturbohybrid';
SET max_parallel_maintenance_workers = 8;
SET turbohybrid.profile = 'high_recall';
SET turbohybrid.native_build_workers = '8';
-- build a turbohybrid index on ~100k+ rows, then:
SELECT turbohybrid_last_build_stats();   -- compare build_edges_us to scan_us+encode_us
-- and sample pg_stat_activity during the build: only the leader is active for build_edges.
```
