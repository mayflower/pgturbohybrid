# Parallel Edge Construction Plan

## Goal

Make native dense graph edge construction parallel by default for code-only
builds, so `turbohybrid.native_build_workers` parallelizes the scan, encoding,
and HNSW edge-linking phases.

## Implemented Approach

Use deterministic segment-parallel edge construction with a bounded leader-side
repair step for the default single-graph case.

1. Keep scan and encode parallelism on the existing PostgreSQL parallel CREATE
   INDEX worker infrastructure.
2. Add a new native edge phase (`PGTURBOHYBRID_NATIVE_PARALLEL_EDGES`) using the
   same `ParallelContext`.
3. Copy immutable build nodes into DSM: heap TID, code bytes, quantization
   metadata, payloads, residual sketch bytes, and level.
4. Split the node id range into deterministic segments.
5. Let workers claim segments and run the existing HNSW edge builder for their
   segment using local neighbor storage.
6. Copy final per-level neighbor lists back to DSM.
7. Merge adjacency into the leader's build state.
8. If the reloption-level native segment count is one, repair the temporary
   worker segments into a single graph by connecting segment entry points to
   the global entry point.
9. Record stats that distinguish scan/encode parallelism from edge parallelism.

The first supported path is code-only native graph builds. Exact-storage and
exact-build-distance modes remain serial unless a future implementation carries
exact vectors into worker DSM or adopts a shared mutable graph.

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
thread-count sensitivity. The implemented pgturbohybrid path favors PostgreSQL
parallel build integration, deterministic output, and no shared neighbor locks.
