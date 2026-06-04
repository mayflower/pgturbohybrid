# Multivector Late Interaction Design

This document defines the implementation contract for ColBERT-style
multivector retrieval in `pgturbohybrid`.

## Target Semantics

A query and a document are both represented as multiple dense vectors. For a
query `Q` and document `D`, the score is:

```text
score(Q, D) = sum_i max_j sim(q_i, d_j)
```

For the MVP, `sim()` is dot product over float32 vectors. The score is a
similarity, so larger is better. PostgreSQL `ORDER BY` operators exposed by the
access method must remain distance-oriented, so the external distance is:

```text
distance = -score(Q, D)
```

## Node And Result Separation

Late interaction indexes subvectors, but returns documents. That separation is
the main correctness invariant:

- `nodeId` identifies one subvector/token node.
- `docId` identifies the source document.
- `heaptid` identifies the PostgreSQL heap tuple returned to the executor.
- `tokenOrdinal` identifies the subvector position inside the document.

One document can produce many `nodeId`s. A result must be ranked and deduplicated
by document/heap tuple, never by subvector node. Subvector hits are only evidence
used to update a document-level MaxSim accumulator.

## Complexity Variables

- `D`: number of documents.
- `L`: document vectors per document.
- `N`: total subvector nodes, `N = D * L` for fixed `L`.
- `Q`: query vectors.
- `d`: vector dimensions.
- `Ks`: raw subvector hits per query vector.
- `C`: unique candidate documents reached by subvector hits.
- `R`: documents selected for exact rerank.

Expected complexity:

- Build expansion: `O(D * L)`.
- Approximate query candidate generation: `O(Q * ANN + Q * Ks)`.
- Document accumulator memory: `O(C * Q)`, not `O(D * Q)` and not
  `O(N * Q)`.
- Exact rerank: `O(R * Q * L * d)`.

## MVP Phases

1. Packed `turbohybrid_multivector` type.
2. Exact MaxSim reference functions.
3. Query v2 with dense kind and multivector payload.
4. Multivector SQL operator and opclass validation.
5. Build-time expansion into subnodes.
6. Approximate MaxSim aggregation over subvector hits.
7. Exact float32 rerank.
8. BM25/RRF fusion at document level.

## MVP Non-Goals

- No int8/VNNI exact rerank.
- No per-subnode exact float32 storage as the default.
- No hybrid fusion before dense-only multivector correctness is proven.
- No on-disk format expansion without version or compatibility checks.
