# Multivector Centroid Bitset Prefilter

This note describes a guarded EMVB-style direction for `centroid_lite`
admission. It is not a production contract. The first implementation slice is
scan-local instrumentation only and keeps final SQL ordering as exact MaxSim
over retained document multivectors.

## Current Centroid-Lite Layout

`centroid_lite` is available for document-node multivector indexes built with
`multivector_centroids = kmeans`. The index persists:

- one document graph node per heap document,
- document-local kmeans centroids,
- centroid posting-list offsets,
- centroid posting entries containing document id and centroid ordinal,
- the original document multivectors in the document sidecar for exact final
  rerank.

At query time, each query token maps to a deterministic codeword. The scan reads
the matching centroid posting list, accumulates a lightweight centroid score per
document, keeps a bounded candidate heap, then exact-reranks retained candidates
with the original multivectors.

## Candidate Docs Touched

The current posting-list union can still touch many documents when:

- common codewords have long posting lists,
- many query tokens map to high-frequency lists,
- `turbohybrid.multivector_centroid_lite_max_postings_per_token = 0`,
- candidate budgets are large enough that the prefilter admits most touched
  documents.

Existing counters such as `centroid_lists_visited`,
`centroid_postings_touched`, `centroid_docs_touched`, and
`centroid_candidates` should be inspected before changing pruning logic.

## Target Bitset Layout

A future persisted format could add a versioned centroid/codeword bitset
sidecar:

- `codeword -> compressed doc bitset` or `codeword -> block bitset`,
- optional per-block `popcount` for fast density estimates,
- optional document-frequency or IDF metadata for list selection,
- optional block-local payload offsets for compact centroid/code scoring.

Any persisted version must have an explicit magic/version/checksum and clear
REINDEX guidance. The current prototype intentionally avoids any on-disk format
change.

## Query-Time Sketch

A production bitset prefilter would:

1. choose the top centroid/codeword lists per query token,
2. OR/AND/score bitsets to form a bounded document candidate set,
3. use block counts or IDF to avoid scanning obviously weak blocks,
4. pass only retained document ids to the existing exact MaxSim rerank,
5. emit heap TIDs ordered by exact MaxSim distance.

The approximate bitset stage is admission-only. It must never become final SQL
ordering unless a later design explicitly changes that contract.

## MVCC Visibility

Bitsets can only describe index-side document ids. Heap visibility still belongs
to the access method visibility/fetch path. A bitset candidate must remain a
candidate until the normal heap tuple visibility and exact-rerank flow decides
whether it can be returned.

## Current Prototype

The first guarded slice adds:

```sql
SET turbohybrid.multivector_centroid_lite_bitset_prefilter = experimental;
```

Default is `off`.

In `experimental` mode the centroid-lite scan builds a scan-local bitset while
walking the existing posting lists. The bitset records the posting-union document
set but does not prune candidates. Therefore:

- no persisted format changes,
- no default behavior change,
- no candidate-order change,
- final ranking remains exact MaxSim over retained documents.

The purpose is to measure bitset memory and bookkeeping cost before deciding
whether a persisted compressed bitset sidecar is worth building.

## Stats

`turbohybrid_last_scan_stats()` reports:

- `centroid_bitset_prefilter_enabled`
- `centroid_bitset_lists_used`
- `centroid_bitset_docs_set`
- `centroid_bitset_docs_after_threshold`
- `centroid_bitset_prefilter_time_us`
- `centroid_bitset_memory_bytes`

Benchmarks should compare these with:

- `centroid_lists_visited`
- `centroid_postings_touched`
- `centroid_docs_touched`
- `centroid_candidates`
- `multivector_exact_rerank_docs`
- `multivector_exact_maxsim_rerank_time_us`

## Graduation Gates

Do not graduate this feature without evidence that:

- the bitset stage reduces documents/postings reaching exact rerank or a later
  safe bound,
- exhaustive-budget final top-k still matches `exact_doc_scan`,
- memory stays bounded for x00k-scale corpora,
- a persisted format, if added, is explicitly versioned and guarded,
- failures produce REINDEX guidance rather than silent fallback.
