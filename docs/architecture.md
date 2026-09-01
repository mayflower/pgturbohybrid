# Architecture

`pgturbohybrid` is one PostgreSQL extension and one index access method.

The supported paths are deliberately small:

1. Dense vectors use a quantized graph to generate candidates. Exact storage,
   when enabled, supplies final float32 distances.
2. `tsvector` columns use native BM25 postings with WAND pruning.
3. Hybrid queries merge dense and BM25 candidates and apply the requested
   fusion policy.
4. Multivector columns index token/subvector nodes. Token hits are aggregated
   by heap tuple and the final document band is reranked with exact float32
   MaxSim.

The SQL result identity is always the PostgreSQL heap tuple. A graph `nodeId`
is only an internal candidate node and is never a document ranking key.

## Storage and safety

The access method uses PostgreSQL buffer, WAL, visibility, vacuum, and memory
context rules. Size calculations are checked before allocation. Index metadata
has explicit magic and version fields; incompatible persisted changes require
REINDEX guidance instead of silent interpretation.

`turbohybrid_validate_index()` checks metadata, page kinds, graph references,
and branch-specific invariants. `turbohybrid_prewarm()` and the two stats
functions are the complete operational surface.

## Deliberate omissions

There is no second experimental extension, learned-sparse type, profile
orchestration, document-node graph, proxy encoder selection, centroid branch,
evaluator framework, or backend-local scan-report API. Performance experiments
belong outside the installed extension and must earn their way into this single
execution path before becoming product code.
