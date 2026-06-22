# Parallel Edge Construction Plan

## Goal

Make native dense graph edge construction parallel by default for code-only
builds, so `turbohybrid.native_build_workers` parallelizes the scan, encoding,
and HNSW edge-linking phases.

## Implemented Approach

Use deterministic batch-parallel insertion into one shared graph. The first
prefix is inserted serially, matching Qdrant's warm-up pattern, then workers
compute HNSW searches and forward links for bounded batches against the shared
ready graph. Workers then shard backlink application by target node before the
leader publishes each batch for the next search wave.

1. Keep scan and encode parallelism on the existing PostgreSQL parallel CREATE
   INDEX worker infrastructure.
2. Add a new native edge phase (`PGTURBOHYBRID_NATIVE_PARALLEL_EDGES`) using the
   same `ParallelContext`.
3. Copy immutable build nodes into DSM: heap TID, code bytes, quantization
   metadata, payloads, residual sketch bytes, and level.
4. Create one deterministic insertion order.
5. Insert the first 256 points serially to avoid disconnected early components.
6. Let workers claim node positions from each bounded batch and run HNSW
   greedy/search-layer plus heuristic neighbor selection against the ready
   shared graph, storing both neighbor ids and distances in DSM.
7. Synchronize at a barrier; workers shard backlink application by target
   node modulo participant count, then the leader updates the global entry
   point and marks the batch ready.
8. Repeat until all nodes are linked, then copy final adjacency into the
   leader's build state as a single native segment.
9. Record stats that distinguish scan/encode parallelism from edge parallelism.

The first supported path is code-only native graph builds. Exact-storage and
exact-build-distance modes remain serial unless a future implementation carries
exact vectors into worker DSM or adopts a shared mutable graph.

The earlier segment-parallel implementation was rejected because it built
isolated worker-local subgraphs and only stitched segment entry nodes. That
produced a fast build but collapsed measured recall from about 0.9924 to 0.32.
The current approach keeps a single ready graph throughout edge construction.

## Qdrant Reference

Qdrant's CPU HNSW builder is lock-based rather than segment-and-repair based:

- `lib/segment/src/index/hnsw_index/hnsw.rs` creates a Rayon pool, assigns
  levels, builds a small warm-up prefix serially, then inserts remaining points
  with `into_par_iter()`.
- `graph_layers_builder.rs` stores links behind per-point/per-layer
  `RwLock<LinksContainer>`, uses a `Mutex<EntryPoints>`, and publishes inserted
  nodes through an atomic ready bitset.
- `link_new_point()` searches the ready graph, links forward edges, writes
  backlinks, and then marks the point ready.
- Qdrant's config warns against excessive indexing threads; defaults cap
  practical HNSW indexing parallelism around the 8-16 thread range.

That design gives true shared-graph insertion but adds lock contention and
thread-count sensitivity. The implemented pgturbohybrid path follows the same
single-graph/warm-up principle, but uses PostgreSQL DSM barriers and bounded
batches instead of per-neighbor locks.
