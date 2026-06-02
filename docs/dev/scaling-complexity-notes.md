# Scaling Complexity Notes

Internal map of current CPU and memory scaling paths before changing runtime
behavior. The bounds below describe the current implementation shape, not a
benchmark result.

## Variables

- `N`: native graph nodes / indexed rows.
- `d`: vector dimensions. Quantized code distance is proportional to packed
  code bytes, which scales with `d` but with smaller constants than exact
  vectors.
- `M`: graph connectivity target. Layer 0 can keep about `2M` neighbors; upper
  layers use about `M`.
- `efConstruction`: build-time graph candidate beam.
- `efSearch`: query-time graph traversal beam.
- `Kd`: dense branch candidate/result target.
- `Kb`: BM25 branch candidate/result target.
- `U`: unique candidate count after dense/BM25 union, bounded by `Kd + Kb`
  before final top-k selection.
- `Q`: concurrent backends/scans that can hold native cache memory.
- `P`: build participants, normally leader plus parallel workers.
- `R`: dense rescore/fill candidate band. This can grow to
  `min(N, turbohybrid.max_scan_tuples)` or to `N` in full-fill paths.
- `S`: resident native graph cache bytes for one cache image: code arena,
  adjacency, optional exact vectors, payload/residual arenas, node metadata and
  scratch metadata.

## Big-O Summary

| Path | CPU | Memory | Main code anchors |
| --- | --- | --- | --- |
| Native dense build | `O(N * d)` for correction fitting and encoding, plus graph linking. The linking path is roughly `O(N * efConstruction * dist(d))` for candidate search and can add `O(N * efConstruction * M * dist(d))` for heuristic neighbor pruning/backlink work. Eligible code-only builds parallelize fit/encode and much of edge link/apply work across `P`, with setup, apply, publish and synchronization overhead still present. | `O(N * (code bytes + payload/residual/exact bytes + graph metadata + M adjacency))` plus worker/DSM scratch. Edge builders also allocate `O(N)` visited/inserted arrays. | `tqgraphbuild`, `PgturbohybridGraphFitCorrection`, `PgturbohybridGraphEncodeBuildNodes`, `PgturbohybridGraphBuildEdges`, `PgturbohybridGraphBuildEdgesRange`, `PgturbohybridGraphBuildSearchLayer`, `PgturbohybridGraphSelectNeighbors`, `PgturbohybridGraphAddNeighbor`, `PgturbohybridNativeRunParallelPhase`, `PgturbohybridNativeRunParallelEdgePhase`, `PgturbohybridNativeParallelEdgeWorker`, `PgturbohybridNativeParallelEdgeBuildNodeLinks`, `PgturbohybridNativeParallelEdgeApplyNodeBacklinks`, `PgturbohybridNativeParallelApplyEdges`. |
| Native dense query | Normal traversal is approximately `O(efSearch * dist(d))`, followed by `O(R * exact_dist(d))` and sort/top-k work for rerank paths. Fill/full candidate bands become `O(N * dist(d))` when `R` reaches `N` or selectivity fill under-runs. Local expansion adds neighbor-distance work around seed nodes. | `O(R)` result storage plus scan storage/cache. The native graph cache can contribute `S` resident bytes. | `PgturbohybridGraphCollectResults`, `PgturbohybridGraphTraverse`, `PgturbohybridGraphFillCandidateBand`, `PgturbohybridGraphCollectPayloadExactBand`, `PgturbohybridGraphApplyLocalExpansion`, `PgturbohybridGraphApplyResidualRerank`, `PgturbohybridGraphHeapRescore`, `PgturbohybridGraphExactRescore`, `PgturbohybridGraphCollectDenseCandidates`. |
| BM25 query | Warm cache WAND/impact paths scale with postings visited, candidates scored and final top-k selection. Fallback DAAT can reach `O(sum(df(term)))`; a common term can be `O(N)`. Cold doc/liveness cache loading adds `O(N + code pages)`. Lazy impact-head construction can scan `O(df(term))`. | Lexicon/postings metadata plus `O(N)` `docLens`, `heapTids` and `liveNodes` caches. Dense accumulator mode allocates `O(N)` node generation arrays; hash mode is closer to touched candidates. | `PgturbohybridBm25TopK`, `PgturbohybridBm25GetCache`, `PgturbohybridBm25EnsureDocStats`, `PgturbohybridBm25LoadDocStats`, `PgturbohybridBm25EnsureLiveness`, `PgturbohybridBm25LoadHeapTids`, `PgturbohybridBm25AccumulatorInit`, `PgturbohybridBm25ScoreImpactSingle`, `PgturbohybridBm25ScoreImpactOR`, `PgturbohybridBm25ScoreBaseWand`, `PgturbohybridBm25ScoreBaseAndRarestDriver`, `PgturbohybridBm25SelectFinalTopK`. |
| Hybrid fusion | Generation-array/hash paths are `O(Kd + Kb)` plus final scoring and top-k selection over `U`. Sorted merge is `O((Kd + Kb) log(Kd + Kb))`. BM25-only exact rescore can add exact dense distance work for BM25-only candidates. | `O(U)` merged results. Generation-array fusion allocates `O(N)` `PgturbohybridFusionArrayEntry` when the array is under the configured byte cap. Hash and sorted paths allocate `O(Kd + Kb)`. | `PgturbohybridCollectScanResults`, `PgturbohybridShouldUseGenerationArray`, `PgturbohybridFuseGenerationArray`, `PgturbohybridFusionHashSlotCount`, `PgturbohybridFindMergeSlot`, `PgturbohybridFinalizeFusedResults`. |
| Native graph cache | Cold load/build is `O(N * code bytes + N * adjacency + optional N * d exact-vector load)`. Shared-cache attach avoids most resident-data loading, but backend view/scratch setup is still linear in `N`. | Per-backend policy is `O(Q * S)`. Shared policy is `O(S)` shared resident data plus `O(Q * N)` backend view/scratch metadata. Uncached scans trade resident memory for repeated page reads and per-scan arenas. | `PgturbohybridGraphInitScanStorage`, `PgturbohybridGraphBuildCache`, `PgturbohybridGraphCacheComputeResidentBytes`, `PgturbohybridGraphSharedBuildView`, `PgturbohybridGraphSharedInitStorageScratch`, `PgturbohybridGraphLoadCodePage`, `PgturbohybridGraphLoadAllAdjPages`, `PgturbohybridGraphLoadExactVectors`. |
| Single-row insert | Graph search/update is roughly `O(efConstruction * dist(d))` plus reciprocal neighbor update/prune work over selected neighbors. With valid native cache adjacency metadata, reciprocal adjacency updates avoid page-chain scans and are bounded by selected neighbors and `M`; without direct block/offset metadata, each reciprocal update may scan adjacency pages. Updates are serialized through the insert/update path. | `O(efConstruction + M * levels)` scratch, plus possible cache `repalloc` growth by new node/adjacency counts. | `PgturbohybridGraphInsertValueInPlace`, `PgturbohybridGraphInitInsertStorage`, `PgturbohybridGraphTraverse`, `PgturbohybridGraphSelectInsertNeighbors`, `PgturbohybridGraphUpdateReciprocalNeighbors`, `PgturbohybridGraphUpdateReciprocalNeighbor`, `PgturbohybridGraphUpdateAdjTuple`, `PgturbohybridGraphLoadAdjTuple`, `PgturbohybridGraphAppendInsertedCode`, `PgturbohybridGraphAppendInsertedAdj`, `PgturbohybridQuantUpdateMetaPageFromUpdate`, `PgturbohybridGraphAppendInsertCacheNode`. |

## Linear-in-N Locations

- Dense candidate fill / full candidate band:
  `PgturbohybridGraphCollectResults` can widen `resultTarget` to
  `meta.tqNodeCount` or `turbohybrid.max_scan_tuples`. The fill path
  `PgturbohybridGraphFillCandidateBand` allocates a `selected` bitmap sized by
  `meta->tqNodeCount` and scans nodes or payload refs. The exact payload band
  path `PgturbohybridGraphCollectPayloadExactBand` can scan payload refs.
- Per-backend native cache:
  `PgturbohybridGraphInitScanStorage` uses `PgturbohybridGraphBuildCache` for
  per-backend cache misses. `PgturbohybridGraphCacheComputeResidentBytes`
  derives resident bytes from `tqNodeCount`, adjacency records and optional
  exact storage. Even with shared cache, `PgturbohybridGraphSharedBuildView` and
  `PgturbohybridGraphSharedInitStorageScratch` allocate backend-local metadata
  or scratch arrays proportional to `N`.
- BM25 doc/liveness cache load:
  `PgturbohybridBm25EnsureDocStats` / `PgturbohybridBm25LoadDocStats` populate
  `docLens` for `N` nodes. `PgturbohybridBm25EnsureLiveness` /
  `PgturbohybridBm25LoadHeapTids` populate `heapTids` and `liveNodes` for `N`
  nodes and walk graph code pages. `PgturbohybridBm25AccumulatorInit` can also
  allocate `O(N)` node generation arrays.
- Generation-array fusion:
  `PgturbohybridShouldUseGenerationArray` checks whether
  `sizeof(PgturbohybridFusionArrayEntry) * meta.tqNodeCount` fits the cap.
  `PgturbohybridFuseGenerationArray` then allocates entries by `nodeCount`.
- Single-row insert reciprocal adjacency update and cache growth:
  `PgturbohybridGraphUpdateReciprocalNeighbors` updates backlinks for selected
  neighbors. When `PgturbohybridGraphInitInsertStorage` attaches a valid native
  cache, `PgturbohybridGraphUpdateAdjTuple` uses cached adjacency block/offset
  metadata, so the update path is `O(number_of_selected_neighbors * M)`.
  Without direct metadata, `PgturbohybridGraphLoadAdjTuple` and the fallback in
  `PgturbohybridGraphUpdateAdjTuple` may scan adjacency page chains.
  `PgturbohybridGraphInsertValueInPlace` emits DEBUG1 counters named
  `insert_reciprocal_neighbors_considered`,
  `insert_reciprocal_adj_cached_hits`, `insert_reciprocal_adj_chain_scans`,
  `insert_reciprocal_adj_pages_scanned`, and
  `insert_reciprocal_update_us`. If an insert cache exists,
  `PgturbohybridGraphAppendInsertCacheNode` can grow node, code, payload,
  residual, exact, visited-generation and adjacency arrays with `repalloc`.

## Later Test Plan

- Regression tests:
  add small SQL cases that build native dense and hybrid indexes, then compare
  stable top-k tuple ids before/after an optimization. Existing expected output
  should not change for a documentation-only PR; later runtime PRs should update
  expected files only when behavior intentionally changes.
- Memory estimator tests:
  add targeted checks around native cache byte estimates and scan stats for
  `off`, `per_backend` and `shared` cache policies. Verify reported bytes scale
  with `N`, code bytes, adjacency bytes and exact-vector storage.
- Scan stats tests:
  use `turbohybrid_last_scan_stats()` after queries that force dense traversal,
  dense fill, BM25 cold/warm cache paths and hybrid fusion. Assert path labels
  and counters such as native cache mode/bytes, scored codes, fusion strategy,
  BM25 docstats/liveness loading and effective dense result target.
- Simple benchmark scripts or SQL snippets:
  keep them reproducible and result-free. Use temporary tables, controlled GUCs,
  `EXPLAIN (ANALYZE, BUFFERS)` and `turbohybrid_last_scan_stats()` to measure
  slope changes over `N`, `d`, `M`, `efConstruction`, `efSearch`, `Kd` and `Kb`.
  Do not commit generated benchmark output. The local smoke scripts are
  `benchmarks/dev/dense_filter_fallback_bench.sql`,
  `benchmarks/dev/native_cache_memory_bench.sql`,
  `benchmarks/dev/bm25_cold_warm_bench.sql` and
  `benchmarks/dev/insert_scaling_bench.sql`.
