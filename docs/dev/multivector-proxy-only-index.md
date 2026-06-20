# Multivector Proxy-Only And Centroid-Only Document-Node Indexes

Status: experimental implementation exists. `multivector_doc_storage =
proxy_only` and `multivector_doc_storage = centroid_only` are SQL-visible
document-node storage tiers. They are not defaults and do not change final SQL
ordering: retained candidates are still exact MaxSim-ranked.

## Problem

Document-node indexes currently persist one graph node per heap document plus
document-level sidecar data. For the fastest `proxy_vector` path, graph
admission only needs:

- document graph adjacency
- the fixed-dimensional document proxy vector used as the graph key
- `docId` to heap TID mapping for SQL result emission and heap exact rerank

The full document multivector sidecar is not needed for proxy graph admission
when final rerank fetches the original multivector from the heap. On the 10k
DBpedia ColBERT proxy baseline, the observed index size was about
281,698,304 bytes. A proxy-only document-node layout is intended to remove the
large full-token document sidecar from that serving profile while preserving
the existing exact final MaxSim contract.

## Current Document-Node Storage

The current document-node design stores:

- one graph node per heap document
- graph adjacency and native TurboQuant graph storage
- a fixed-dimensional proxy vector/code for the graph node
- docmap metadata for document identity and heap TID resolution
- full document multivectors in the multivector docmap sidecar, unless the
  index is explicitly built as `proxy_only` or `centroid_only`
- optional centroid tuples and centroid posting tuples
- optional experimental quantized inverted posting/codebook tuples
- optional entry-sidecar representatives

This supports several candidate sources with one broad document-node format,
but it makes plain proxy serving carry storage that is only needed for
sidecar-based scoring, centroid admission, or exact sidecar scans.

## Candidate Source Storage Requirements

### Requires Full Document Multivector Sidecar

These paths need full multivector sidecar storage or sidecar-derived payloads
and must not run on a proxy-only index unless they have an explicit heap-only
fallback that preserves semantics and is accepted by the user:

- `document_nodes` full sidecar scoring
  - Scores document candidates using full document multivectors from the
    sidecar.
  - Proxy-only storage cannot provide those vectors.
- `centroid_mean`
  - Requires `multivector_centroids = kmeans`.
  - Uses persisted centroid sidecar data for centroid pre-rerank before full
    exact MaxSim.
  - Proxy-only storage is insufficient unless centroid sidecar storage is also
    retained as a separate tier.
- `centroid_lite`
  - Requires persisted k-means centroid tuples, residual summaries, and
    centroid posting tuples.
  - Uses sidecar/posting data for PLAID-inspired admission before exact rerank.
- `quantized_inverted_experimental`
  - Requires experimental codeword posting sidecar data and matching codebook
    metadata.
  - Remains research-only and must fail clearly if the sidecar payload is
    absent or incompatible.
- `exact_doc_scan`
  - Exact sidecar scan needs full float32 document multivectors in the index.
  - Proxy-only indexes should reject this source with REINDEX guidance or use a
    separately named heap exact scan diagnostic if one exists.

### Can Use Proxy-Only Storage

These paths can operate without full document multivector sidecar storage:

- `proxy_vector` with heap exact rerank
  - Graph admission uses the fixed-dimensional proxy vector/code already stored
    on each document graph node.
  - Exact final MaxSim fetches the heap tuple multivector for only the bounded
    rerank set.
- `proxy_vector` with entry-sidecar
  - Entry-sidecar representatives help choose graph entry points.
  - They are proxy/vector-level graph metadata, not full document token storage.
  - This remains compatible if entry-sidecar tuples are stored independently of
    full multivectors.
- `proxy_exact_scan`
  - Diagnostic-only path that scores all documents by the same persisted proxy
    vectors and then exact-reranks retained heap documents.
  - It requires access to all fixed-dimensional proxy vectors, but not full
    document sidecar vectors.

## Implemented Storage Modes

The current implementation exposes these storage kinds:

```text
f32/f16/sq8        proxy + docmap + full document multivector sidecar
proxy_only         graph proxy + docmap, no full document multivector sidecar
centroid_only      graph proxy + docmap + kmeans centroid/posting payloads,
                  no full document multivector sidecar
```

`proxy_only` is valid only for proxy-vector admission paths that can exact-rerank
from heap multivectors. `centroid_only` is valid for centroid-lite admission
with heap exact rerank. Full-sidecar candidate sources must fail clearly with
REINDEX guidance when the required payload is absent.

## Required Metadata

A proxy-only document-node index must persist enough metadata to support graph
admission, visibility, final heap fetch, and diagnostics:

- `docId`
  - Dense contiguous document identifier used by document-node scan state.
- `heaptid`
  - Heap tuple pointer for visibility checks and exact heap rerank.
- proxy vector/code
  - The fixed-dimensional graph key generated by
    `multivector_proxy_encoder`.
  - Dimension must match the query proxy dimension.
- proxy encoder metadata
  - Encoder name and version.
  - Any projection/checksum metadata required by file-backed encoders.
- multivector model metadata
  - Token dimension.
  - Distance/similarity family needed to build matching query proxies.
  - Optional known model/context metadata already reported by index stats.
- graph metadata
  - `multivector_graph = document_nodes`.
  - graph `m`, `ef_construction`, adjacency pages, node count, and doc count.
- storage capability flags, visible in `turbohybrid_index_stats()`:
  - `multivector_doc_storage_kind = f32|f16|sq8|centroid_only|proxy_only`.
  - `proxy_only_index = true|false`.
  - `centroid_only_index = true|false`.
  - `full_multivector_sidecar_available = true|false`.
  - `centroid_sidecar_available = true|false`.
  - `quantized_inverted_sidecar_available = true|false`.
  - `exact_rerank_source_supported = heap|sidecar|none`.

The proxy-only metadata must be included in `turbohybrid_index_stats()` so
benchmark artifacts can prove which physical index tier was used.

## MVCC Behavior

Proxy-only storage must not bypass PostgreSQL visibility:

- The index stores heap TIDs, not authoritative tuple visibility.
- Scans still return heap TIDs through the access method.
- The executor or existing AM visibility path rejects dead or invisible heap
  tuples.
- Exact heap rerank must fetch and decode the visible heap multivector for the
  retained candidate band.
- UPDATE remains insert of a new heap tuple/version; old graph/docmap entries
  are ignored through the existing dead-node and visibility machinery.
- VACUUM/dead tuple handling remains graph-node based. Proxy-only docmap
  entries may remain append-only until a future versioned compaction design.

Because proxy-only exact rerank fetches heap tuples, it may pay more heap I/O
than sidecar rerank for cold or scattered candidate bands. That is an explicit
latency/storage tradeoff and must be measured with heap fetch counters.

## Heap Exact Rerank Cost

Proxy-only serving keeps final SQL ordering exact:

1. Build query proxy vector from the query multivector.
2. Traverse the document proxy graph with existing TurboQuant proxy/code
   scoring.
3. Keep a bounded candidate band.
4. Fetch heap multivectors for the exact rerank band.
5. Compute exact float32 MaxSim.
6. Emit heap TIDs ordered by `distance = -maxsim`.

The cost moves from index-side sidecar reads to bounded heap fetches. This is
acceptable only when:

- `multivector_exact_rerank_k` remains bounded.
- heap fetch count is close to the effective exact rerank docs.
- no proxy-only scan falls back to exact document sidecar scan.
- stats expose `exact_rerank_source = heap`, heap fetch count, and MaxSim time.

If heap rerank dominates latency, the next design choice is not to put full
sidecar storage back by default. It is to compare:

- proxy-only heap rerank
- full-sidecar sidecar rerank
- `f16` or `sq8` sidecar storage
- token pooling
- smaller exact rerank bands

## Expected Index Byte Savings

The expected saving is approximately the bytes currently consumed by full
document multivector sidecar payloads and any resident native-cache mirror of
those payloads.

For a corpus with:

```text
D documents
L average tokens per document
d vector dimension
s bytes per scalar in sidecar storage
```

the raw full-vector payload is roughly:

```text
D * L * d * s
```

plus tuple headers, sidecar page overhead, chunk references, alignment, and
optional original-plus-pooled storage. For ColBERT-style `d = 128`, even
moderate token counts dominate graph adjacency and docmap bytes.

Proxy-only keeps one `d`-dimensional proxy vector/code per document rather than
`L` token vectors per document:

```text
D * d * proxy_storage_bytes
```

The 10k DBpedia proxy baseline index size of about 281,698,304 bytes provides a
concrete measurement target: a proxy-only variant should reduce index bytes
substantially while keeping latency competitive if heap exact rerank stays
bounded. The design should report:

- total index bytes
- graph adjacency bytes
- proxy vector/code bytes
- docmap bytes
- full multivector sidecar bytes omitted
- native cache bytes

## Versioning And REINDEX

Proxy-only and centroid-only storage change physical index capabilities. They
must be explicit and versioned.

Required compatibility rules:

- Record capability flags and sidecar storage kind in graph/docmap metadata.
- Preserve current full-sidecar indexes as readable.
- Do not reinterpret old full-sidecar indexes as proxy-only.
- Do not silently run full-sidecar candidate sources on proxy-only indexes.
- If a candidate source requires missing full-sidecar payloads, raise an error
  with REINDEX guidance.
- If a sidecar page exists but has unsupported magic/version/capabilities, fail
  clearly with REINDEX guidance.
- Benchmark scripts must record the physical storage mode in metadata.

Example error shape:

```text
ERROR:  document-node full multivector sidecar is not available for this index
HINT:   Rebuild the index with full document multivector sidecar storage, or use candidate_source = proxy_vector with heap exact rerank.
```

## Failure Rules

Proxy-only indexes fail, not silently fall back, for these requests:

- `candidate_source = document_nodes` when that source means full sidecar
  scoring.
- `candidate_source = centroid_lite` because proxy-only storage has no centroid
  sidecar payloads.
- `candidate_source = quantized_inverted_experimental` without quantized
  posting/codebook sidecar payloads.
- `candidate_source = exact_doc_scan` when it expects exact float32 sidecar
  scan.
- `multivector_proxy_encoder = centroid_mean` without k-means centroid sidecar
  support.
- `multivector_doc_storage_cache = resident` if there is no full sidecar to
  make resident. This should either become a no-op with a clear stat in
  proxy-only mode or fail if the user requested full-sidecar cache behavior.

Centroid-only indexes fail, not silently fall back, for requests that require a
full document multivector sidecar, including `candidate_source = exact_doc_scan`
and full `document_nodes` sidecar scoring. `centroid_lite` may run on
centroid-only storage and exact-rerank from heap multivectors.

Allowed behavior:

- `candidate_source = proxy_vector` uses proxy graph admission and heap exact
  rerank.
- `candidate_source = proxy_exact_scan` remains diagnostic-only if enabled and
  scans persisted proxy vectors, then heap exact-reranks retained candidates.
- Entry-sidecar variants run only if their own representative metadata is
  present.

## Stats And Diagnostics

Stats needed to make proxy-only and centroid-only benchmark evidence
trustworthy:

- `multivector_doc_storage_kind = f32|f16|sq8|centroid_only|proxy_only`
- `proxy_only_index = true|false`
- `centroid_only_index = true|false`
- `full_multivector_sidecar_available = true|false`
- `centroid_sidecar_available = true|false`
- `quantized_inverted_sidecar_available = true|false`
- `exact_rerank_source_supported = heap|sidecar|none`
- `multivector_proxy_vector_bytes`
- `multivector_docmap_bytes`
- `multivector_full_sidecar_bytes`
- `multivector_exact_rerank_source = heap | sidecar | off`
- `multivector_exact_rerank_heap_fetches`
- `multivector_exact_rerank_docs`
- `multivector_exact_rerank_pairs`
- `multivector_exact_maxsim_rerank_time_us`
- `proxy_candidates`
- `proxy_graph_nodes_visited`
- `proxy_graph_edges_visited`
- `proxy_vector_sidecar_touch_reason = not_applicable`

Slow-path warnings should include:

- `proxy_only_full_sidecar_unexpected`
- `centroid_only_full_sidecar_unexpected`
- `exact_sidecar_requested_but_missing`
- `heap_rerank_unbounded`

## Benchmark Plan

Compare at least these physical layouts on the same generated DBpedia/ColBERT
artifacts:

- current `proxy_normalized_mean_f16` full-sidecar index
- proxy-only `proxy_normalized_mean`
- proxy-only entry-sidecar variants, if entry-sidecar is in scope
- full-sidecar `centroid_lite` and `quantized_inverted_experimental` as
  non-proxy-only controls

Report:

- build time
- index bytes
- native cache bytes
- p50/p95/p99 latency
- heap fetch count
- exact rerank docs and pairs
- top1 admission
- top10 admission recall
- recall@10, nDCG@10, MRR@10 when qrels exist

The acceptance threshold for proxy-only is not recall improvement. It should
match the same proxy candidate source quality while reducing index size and
keeping exact heap rerank latency within an agreed budget.

## Remaining Review Questions

- Whether entry-sidecar representatives should be a common companion to
  `proxy_only` for graph-entry experiments.
- Whether centroid-only should grow a compact exact-rerank payload later, or
  remain heap-rerank only.
- What heap rerank latency target justifies dropping the full sidecar for x00k
  and 1M DBpedia serving.

## Non-Goals

- No change to final SQL ordering.
- No approximate final ranking.
- No production promise for `quantized_inverted_experimental`.
- No new centroid or quantized payload format here.
- No promotion of proxy-only or centroid-only to default serving profiles.
