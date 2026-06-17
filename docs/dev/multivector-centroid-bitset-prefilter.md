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
SET turbohybrid.multivector_centroid_lite_posting_selection = score_topk;
SET turbohybrid.multivector_centroid_lite_probe_centroids_per_token = 4;
SET turbohybrid.multivector_centroid_lite_score_threshold = 0.0;
SET turbohybrid.multivector_centroid_lite_bitset_prefilter = experimental;
SET turbohybrid.multivector_centroid_lite_bitset_min_token_matches = 2;
```

Default is `off`.

In `score_topk` mode the centroid-lite scan probes centroid lists and, on
freshly built indexes, reads their payload-sorted posting prefixes instead of
sampling arbitrary postings. It scores retained candidate documents with
compact deterministic query-centroid MaxSim before exact heap MaxSim rerank.
The experimental
`turbohybrid.multivector_centroid_lite_candidate_scoring = codeword_maxsim`
variant reports that scoring explicitly as a PLAID-style codeword MaxSim over
selected centroid/codeword matches and does not load the document-centroid
sidecar. The stronger but slower
`turbohybrid.multivector_centroid_lite_candidate_scoring = doc_centroid_maxsim`
variant uses the persisted document-centroid sidecar to pre-rank touched
documents with approximate full-MaxSim over document centroids. This remains a
scan-time admission scorer: it does not change the persisted bitset proposal
and it does not replace exact heap MaxSim final ordering.
The threshold is applied before broad document touching; the retained
approximate pool is bounded by the normal document candidate budget. The
relative `multivector_centroid_lite_score_drop_from_best` filter serves the
same bounded-probe purpose by keeping only centroid lists near the best
per-token codeword score, and benchmark focus rows apply both filters to the
compact `codeword_maxsim` scorer as well as the stronger `doc_centroid_maxsim`
rescoring path. In
`experimental` bitset mode the scan builds a scan-local bitset while walking
the selected posting lists. The bitset records the posting document set,
popcounts it, and can require a configurable minimum number of matched
query-token lists before a document reaches the candidate heap. Therefore:

- no persisted format changes,
- no default behavior change,
- no candidate-order change,
- final ranking remains exact MaxSim over retained documents.

The purpose is to measure whether bounded centroid probing, compact posting
scoring, and scan-local bitsets reduce broad posting/doc touches enough to
justify a future persisted compressed bitset or posting-block sidecar. Such a
future format would need explicit versioning and REINDEX guidance.

## Stats

`turbohybrid_last_scan_stats()` reports:

- `centroid_bitset_prefilter_enabled`
- `centroid_bitset_min_token_matches`
- `centroid_bitset_lists_used`
- `centroid_bitset_docs_set`
- `centroid_bitset_docs_after_threshold`
- `centroid_bitset_candidates`
- `centroid_bitset_time_us`
- `centroid_bitset_prefilter_time_us` (compatibility alias)
- `centroid_bitset_memory_bytes`
- `centroid_probe_centroids_per_token`
- `centroid_postings_selected`
- `centroid_postings_skipped`
- `centroid_score_threshold`
- `centroid_lists_skipped_by_threshold`

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
- compact posting scoring improves admission at the same posting budget,
- exhaustive-budget final top-k still matches `exact_doc_scan`,
- memory stays bounded for x00k-scale corpora,
- a persisted format, if added, is explicitly versioned and guarded,
- failures produce REINDEX guidance rather than silent fallback.
