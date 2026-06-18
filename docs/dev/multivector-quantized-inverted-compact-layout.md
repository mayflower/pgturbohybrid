# Quantized Inverted Compact Layout Design

Status: design only. This document describes a possible future persisted
layout for the experimental `quantized_inverted_experimental` candidate source.
It does not change the current on-disk format and does not make compatibility
promises.

## Goal

The current quantized-inverted prototype can generate pure-ColBERT candidates
without BM25, learned sparse, or hybrid fusion, and final SQL ordering remains
exact heap MaxSim over retained documents. The remaining bottleneck is the
amount of scan-local work after broad postings have admitted too many
documents into compact full-document scoring.

The next storage-oriented step, if scan-local precompact and score-bound
pruning are still insufficient, is a versioned compact document layout similar
to Qdrant-style multivector point storage:

```text
docId -> start,count
flat compact token/codeword payloads
docId-ordered compact scoring
optional residual/PQ payloads
exact heap MaxSim rerank
```

## Current Layout And Bottleneck

The current research sidecar stores experimental codebook metadata and posting
tuples keyed by codeword:

```text
codebook metadata
codeword -> docId, tokenOrdinal, approximate score payload
```

The query path retrieves postings for selected codewords, accumulates touched
documents, optionally precompacts the touched set, scores compact document
payloads, and then exact-reranks retained heap documents. This has been useful
for proving the branch is executable, but the compact score stage still works
over too many documents in the measured 50k runs. The latest acceptance report
reduced p95 from `117.839 ms` to `69.828 ms`, but still touched `8192`
compact-scored documents and failed the top-1/top-10 exact admission gates.

The current layout is also not optimized for docId-ordered scoring. Postings
are useful for broad admission, but once a document set is selected the scorer
needs a direct `docId -> compact token range` mapping so it can score each
whole document with predictable memory access.

## Proposed Versioned Layout

Any persisted compact layout must be explicitly versioned and experimental:

```text
magic: pgturbohybrid_quantized_inverted_compact_v1
format_version
flags
doc_count
dim
codebook_size
top_m
codebook_source
codebook_checksum
doc_offset_table_offset
doc_offset_table_bytes
compact_payload_offset
compact_payload_bytes
residual_payload_offset
residual_payload_bytes
heap_tid_table_offset
heap_tid_table_bytes
```

The layout separates broad postings from document-local compact scoring:

- Codebook metadata
  - source, dimension, size, `top_m`, checksum, and any external file
    provenance needed to reject mismatched query settings.
- Document offset table
  - one fixed-size entry per `docId`;
  - `start`, `count`, and byte offsets into flat compact payloads;
  - optional residual/PQ offsets;
  - flags for missing, deleted, or unsupported compact payloads.
- Flat compact token/codeword payload
  - document tokens packed contiguously by `docId`;
  - codeword IDs, quantized norms, and any compact score payload required by
    the admission scorer;
  - stable alignment chosen for SIMD-friendly scans.
- Optional residual/PQ payload
  - residual vectors, PQ codes, or block-level upper-bound terms;
  - absent in the first compatible reader unless the version/flags explicitly
    say it exists.
- MVCC and heap TID table
  - heap TID remains the SQL result identity and visibility anchor;
  - compact payloads are admission-only and do not replace heap visibility.

This is not a proposal to persist exact MaxSim final scores or to return
approximate order. The only SQL-visible final ordering remains exact heap
MaxSim.

## Query Path

A future query would keep the current branch structure while reducing random
and repeated compact-scoring work:

1. Quantize query tokens with the active codebook.
2. Retrieve broad postings for selected codewords and probes.
3. Accumulate a touched-document set plus cheap per-document upper-bound
   evidence.
4. Apply a precompact document gate:
   - score-top-k;
   - query-token coverage;
   - per-token reservoir;
   - block-max or upper-bound pruning when available.
5. Sort retained document IDs and score compact payloads in docId order using
   `docId -> start,count`.
6. Retain the candidate band for exact rerank.
7. Heap-fetch retained documents and compute exact float32 MaxSim.
8. Return heap tuples ordered by `distance = -exact_maxsim`.

The broad postings stage should be allowed to over-recall. The compact layout
exists to make the next stage cheap enough that the benchmark can choose a
quality-safe candidate band before heap rerank.

## SIMD And Payload Design

The first persisted version should remain scalar-readable and easy to validate.
SIMD implementations can be runtime-dispatched only after scalar reference
coverage exists.

Useful payload shapes:

- one byte or two byte codeword IDs when the codebook size allows it;
- quantized token norm or score payload for cheap query-codeword interaction;
- aligned document-token ranges to reduce misaligned loads;
- optional block summaries for upper-bound pruning;
- optional residual/PQ codes for improved compact MaxSim approximation.

No SIMD or residual payload should be required for correctness. Unsupported
payload flags must fail clearly or fall back to a slower exact admission path
only when explicitly requested by a diagnostic mode.

## Required Stats

`turbohybrid_last_scan_stats()` and benchmark rows should expose enough detail
to decide whether the persisted layout is actually helping:

- `quantized_inverted_compact_layout_version`
- `quantized_inverted_compact_doc_offset_bytes`
- `quantized_inverted_compact_payload_bytes`
- `quantized_inverted_compact_residual_bytes`
- `quantized_inverted_compact_payload_bytes_touched`
- `quantized_inverted_compact_docs_scored`
- `quantized_inverted_compact_us_per_doc`
- `quantized_inverted_compact_doc_order`
- `quantized_inverted_precompact_union_docs`
- `quantized_inverted_precompact_pruned_docs`
- `quantized_inverted_exact_rerank_docs`
- `multivector_exact_maxsim_rerank_time_us`

Index stats should also report layout magic/version, codebook metadata, payload
byte counts, and whether exact rerank is heap-backed or sidecar-backed.

## Compatibility And REINDEX Rules

The current quantized-inverted layout is research-only. A future persisted
compact layout still needs explicit safety checks:

- Old experimental layouts may remain readable only when the reader can prove
  the metadata and payload version are supported.
- Unsupported magic, version, flags, dimension, codebook size, `top_m`, or
  checksum must fail clearly with REINDEX guidance.
- `top_m > 1` requires a layout version that explicitly stores
  multi-assignment postings and query assignment metadata.
- Missing optional residual/PQ payloads must not be silently treated as exact
  residuals. They are absent and must be reported as absent.
- No index built with this experimental layout should be described as having
  production compatibility until a stable versioning policy exists.

## MVCC And Heap Identity

The compact payload is an index-resident candidate-generation aid. It is not
the authoritative document:

- Heap TID remains the result identity.
- PostgreSQL visibility rules still decide whether a retained tuple can be
  returned.
- UPDATE creates a new heap tuple/version and therefore a new index entry.
- Dead or invisible entries may remain in compact payloads until the existing
  graph/docmap dead-node machinery excludes them.
- Exact rerank fetches the visible heap multivector and computes exact MaxSim.

This preserves the existing access method contract and keeps approximate
payloads out of final SQL ordering.

## Benchmark Gates

A persisted compact layout should not become the experimental default-quality
candidate unless benchmark artifacts pass all gates used by the acceptance
report:

- candidate source is `quantized_inverted_experimental`;
- BM25, learned sparse, and hybrid fusion are inactive;
- final ranking is exact heap MaxSim;
- top-10 exact admission is at least `0.80`;
- top-1 exact admission is at least `0.94`;
- Recall@10 and NDCG@10 stay within `0.01` of the baseline row;
- p95 is at most `0.75x` of the baseline p95;
- compact-scored documents are normally at or below `6000`, unless the report
  proves compact scoring is no longer a top-two bottleneck;
- index bytes and payload bytes are reported;
- generated benchmark artifacts stay under `.nix-dev/tmp/`.

If those gates fail, the layout remains a research prototype even when it is
faster than the previous scan-local implementation.
