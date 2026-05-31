# Native Segmented Build Prototype

This note describes the segmented native graph prototype. The goal is to reduce
the dominant `build_edges` cost without making one HNSW insertion loop
concurrently mutable. The compatibility rule is simple: `native_segments = 1`
keeps the existing single native graph behavior and remains the default.

## Storage Format

Native graph indexes still use the existing metapage plus native code,
adjacency, exact-vector, and correction page chains. The prototype adds a
segment directory to the metapage:

- `tqSegmentCount`
- `tqSegmentBytes`
- `tqSegments[]`

Each `tqSegments[]` entry stores:

- `startNodeId`
- `nodeCount`
- `entryNodeId`
- `entryLevel`
- `codeStartBlkno`
- `adjStartBlkno`
- `exactStartBlkno`
- `correctionStartBlkno`

In the first prototype, the code, adjacency, exact, and correction starts point
at the global page chains. That keeps tuple layout and page readers unchanged.
The segment directory is still useful because it records the independent graph
components and their entry nodes. A later format can split these into true
per-segment page chains or move the directory to dedicated metadata pages if
the metapage budget becomes tight.

## Node IDs

The build keeps one global node ID space. Nodes are ordered by the same merged
build-node array used by the native single-graph path. Segments are contiguous
ranges in that global array:

```text
segment-local id = global nodeId - segment.startNodeId
global nodeId    = segment.startNodeId + segment-local id
```

Adjacency records store global node IDs, not segment-local IDs. This preserves
existing adjacency readers, heap TID lookup, exact/residual rescoring, and
vacuum repair code. Segment-local IDs are only a reasoning aid for future
per-segment storage or cache slicing.

## Build

After heap scan, correction fitting, and encoding, the build partitions
`state.nodes` into `N` contiguous ranges. Each range is passed to
`PgturbohybridGraphBuildEdgesRange()`, which runs the existing sequential
insertion/search/select/reverse-link loop against only that range. Reverse links
therefore never cross segment boundaries.

The current implementation supports serial segment builds first. That already
measures the algorithmic tradeoff: smaller independent HNSW components reduce
per-component edge-construction work, while query must search more entry points.
True parallel segment edge build is the next step. It should use
`turbohybrid.native_build_workers`, but it needs a DSM/merge format for mutable
per-node adjacency arrays before worker-built segments can be safely returned to
the leader.

Build diagnostics expose:

- `native_segments`
- `native_segment_bytes`
- `segment_build_mode`
- `parallel_segment_build_enabled`
- existing phase timings including `build_edges_us`

For now `segment_build_mode` is `single` or `serial`, and
`parallel_segment_build_enabled` is `false`.

## Query

For `segment_count == 1`, the scan path is the legacy single-entry graph path.
For `segment_count > 1`, the scan path seeds one entry candidate per segment
entry node, then searches the native graph base layer and maintains the existing
global top-k heap. Because segment adjacency components are disconnected, these
entry seeds are what make all segments reachable.

The prototype searches segments serially inside one backend. It does not add
query parallelism. Exact-storage rescoring, residual rerank, heap rescore, and
payload filtering continue to use global node IDs and the existing candidate
merge path.

Scan diagnostics expose:

- `graph_segment_count`
- `graph_segments_searched`
- `graph_entry_point_count`
- existing traversal, page-read, scoring, and rescore counters

## Cache Implications

The native cache remains an immutable per-backend cache for this prototype. Its
cache key includes segment count and segment metadata, so a cache built for one
segment layout is not reused after `REINDEX`, relfilenode replacement, or a
different segment directory. Per-segment caches are acceptable for a later shared
cache implementation, but the current reader still uses global code and
adjacency arenas.

## Vacuum And Updates

Segmented indexes keep the existing global node ID and heap TID mapping. Deletes
mark nodes dead and vacuum repair scans adjacency metadata as before. Because
prototype segments have no cross-segment edges, repair must not invent
cross-segment links unless a future format explicitly stores a cross-segment
routing layer.

Insert/update behavior should continue to invalidate stale native cache state
through the metapage generation. A future mutable segmented index can choose a
policy for new tuples, such as appending to a writable tail segment or rebuilding
segments during `REINDEX`.

## WAL

The segment directory is stored on the metapage and is covered by the existing
metapage update path. Code, adjacency, exact, and correction pages use the same
page initialization and new-page WAL behavior as the single-segment native
format. A future per-segment page-chain format will need explicit WAL coverage
for segment metadata page initialization and chain linkage.

## Compatibility

Indexes without segment metadata, or with `tqSegmentCount == 0`, are interpreted
as one segment covering all native nodes. Existing single-segment native indexes
remain readable. Changing between segment counts changes index shape and should
be done with `REINDEX` or by dropping and recreating the index.

## Benchmarking

Use `benchmarks/native_segments_bench.sql` to compare `native_segments = 1,2,4,8`.
It records build time, index size, precision against exact ordering, p50/p95,
and last-scan stats. The benchmark is expected to show the build/query/recall
tradeoff; segmented build is approximate retrieval unless the segment count is
one.
