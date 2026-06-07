# ColBERTSaR-Style Quantized Inverted Research Branch

This document tracks the research-only
`quantized_inverted_experimental` multivector candidate source. It is inspired
by ColBERTSaR-style quantized inverted indexes for late-interaction retrieval,
where token embeddings are mapped to codebook terms and searched through
postings before exact MaxSim rerank.

The branch is intentionally not a production on-disk format. Any codebook,
posting, residual, or score-payload layout introduced here is allowed to change
without compatibility guarantees. Indexes built with an experimental storage
version will need explicit REINDEX guidance before they can be used as a stable
feature.

## Guard

The branch is selected only with:

```sql
SET turbohybrid.multivector_candidate_source =
  'quantized_inverted_experimental';
```

Current executable status:

- The GUC value is registered so benchmark and SQL plumbing can name the branch.
- Document-node indexes persist experimental codeword posting tuples in the
  multivector docmap sidecar. Query tokens are assigned deterministic
  codewords, matching persisted postings are accumulated into approximate
  document scores, and candidates are exact-reranked with heap MaxSim.
- Token-node indexes fail explicitly with `feature_not_supported` because the
  branch needs document sidecar vectors for exact rerank.
- Plain fallback is bypassed when the branch is selected so benchmark output
  cannot silently substitute a different candidate source.

The current codeword assignment is deterministic and not learned yet: each
token maps to the signed largest-magnitude dimension, so the temporary codebook
has `2 * dim` terms. This is useful for validating the branch, stats, and
benchmark comparison surface, but it is not a production ColBERTSaR index.

## Current Sidecar

The current sidecar stores experimental inverted postings:

```text
codeword -> docId, tokenOrdinal, approximate score payload
```

Bulk build emits posting tuples during index creation. Incremental document-node
insert appends matching posting tuples when the existing index already carries
the quantized-postings flag, or when an empty document-node index is being
bootstrapped. Older indexes without the sidecar fail explicitly and require
`REINDEX` before the experimental branch can run.

The current codeword assignment and score payload are deliberately simple:
the codeword is the signed largest-magnitude dimension and the payload stores a
quantized token-norm proxy. Exact rerank still uses the original float32
document multivectors, not the payload.

## Target Design

The next prototype stages should replace the deterministic codeword assignment
with a real codebook and add residual payloads:

1. Learn or load a codebook for the model dimension.
2. Quantize each stored document token to one or more codewords.
3. Store postings keyed by codeword and document identity.
4. Preserve the original document multivector for exact MaxSim rerank.
5. Mark every persisted structure with an experimental magic/version.

Query should:

1. Quantize every unmasked query token with the same codebook.
2. Retrieve postings for matching or nearby codewords.
3. Accumulate document-level approximate scores.
4. Keep a bounded candidate set.
5. Exact-rerank candidates with float32 MaxSim over the original multivectors.

The approximate score must remain admission-only until benchmark evidence shows
that it can safely participate in fusion. Final SQL ordering for multivector
queries remains document-keyed exact MaxSim unless a later prompt explicitly
changes that contract.

## Benchmark Requirements

Compare this branch against:

- `learned_sparse`
- `centroid_lite`
- `token_nodes`
- `document_nodes`
- `proxy_vector`

At minimum, reports must include:

- index bytes for codebook, postings, residuals, and original multivectors;
- admission top-1 and top-10 recall against exact MaxSim;
- final Recall/NDCG/MRR where qrels exist;
- p50/p95 latency;
- postings touched;
- documents scored approximately;
- documents exact-reranked.

The default benchmark grids should keep this branch opt-in because the storage
format is explicitly unstable. It can be selected manually today to collect
persisted-posting admission, storage, and latency numbers.
