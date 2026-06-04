# Multivector Fix Plan

This audit tracks the multivector late-interaction work in `prompts.md`.
`turbohybrid_multivector` indexes expand each document into token/subvector graph
nodes, but SQL-visible results must remain heap tuple/document keyed.

## Prioritized Checklist

### P0 correctness and SQL usability

- Add shared `turbohybrid_query` dense helpers so callers can distinguish
  vector, multivector, dense, and text payloads using `denseKind` plus flags
  instead of raw `HAS_VECTOR` checks.
- Reject scalar multivector hybrid fallback. Exact scalar MaxSim can evaluate
  only the multivector payload; a query with `text_query` must require a
  TurboHybrid index scan instead of silently ignoring the text branch.
- Implement parseable text input for `turbohybrid_multivector` so the existing
  output format can round-trip through casts, pg_dump, and restore.
- Add binary send/receive functions for binary client and `COPY BINARY`
  support without changing the internal varlena layout.

### P1 retrieval correctness

- Make `turbohybrid.multivector_unique_docs_per_token` token-local. The current
  stopping criterion must not let documents collected for earlier query tokens
  suppress collection for later tokens.
- Keep all multivector result aggregation and deduplication document/heap-tuple
  keyed. Never rank or deduplicate SQL results by subvector `nodeId`.
- Centralize unsupported multivector fusion checks. Dense-only multivector
  queries are valid, and hybrid multivector plus text currently supports RRF
  only unless a later prompt implements score-level fusion.
- Pin small-corpus indexed results against brute-force exact MaxSim and add
  MVCC lifecycle coverage for delete, update, vacuum, and insert-after-vacuum.

### P2 scan and allocation performance

- Replace repeated candidate-array `qsort` during top-k maintenance with a
  bounded heap while preserving final output ordering.
- Reuse token-to-`Vector` buffers in build and scan paths to reduce per-token
  allocation churn.
- Use slab-style allocation for document accumulators so `maxsim[Q]` and
  `seen[Q]` arrays do not require separate pallocs for every candidate
  document.
- Add blocked scalar MaxSim first as a parity reference, then add gated blocked
  AVX2/NEON kernels while preserving scalar as the semantic fallback.

### P3 storage and future fusion

- Design a persistent multivector doc map sidecar before implementing it. The
  design must include page kind, version or magic, fallback behavior, cache
  integration, vacuum/dead tuple handling, and REINDEX guidance.
- Implement the doc map sidecar only with explicit compatibility handling for
  old indexes or indexes without the sidecar.
- Add BM25-only exact multivector rescoring as a prerequisite for future
  weighted score fusion, but do not enable unsupported fusion modes until their
  full semantics and tests exist.
- Optimize incremental multivector insert batching by eliminating repeated
  detoast and validation, reusing one subvector buffer, appending the docmap once,
  and keeping BM25 delta append document-level. The current batch entry point
  still performs one graph-neighbor insertion per token; deeper correction/cache
  and metapage coalescing remains a follow-up because each graph node can change
  entrypoint and reciprocal-neighbor state.

### P4 developer validation

- Add an optional benchmark harness that measures scaling slopes across
  document count, document token count, query token count, dimensions, exact
  rerank size, and raw subvector budgets.
- Finish with a cleanup pass over direct flag checks, node/document keying,
  candidate top-k maintenance, scalar text fallbacks, fusion messages,
  compatibility docs, and multivector user documentation.

## Compatibility Care

No on-disk compatibility risk:

- Query helper functions.
- Scalar multivector `text_query` rejection.
- Text input for `turbohybrid_multivector`.
- SQL inspection helpers.
- Regression, planner, lifecycle, recall, and benchmark tests.
- Buffer reuse, accumulator slab allocation, bounded heap candidate selection,
  and exact MaxSim kernel changes when they preserve existing storage and SQL
  semantics.

SQL-visible but storage-compatible changes:

- Binary send/receive functions add a binary external representation but do not
  alter the internal varlena format.
- Adding `multivector_maxsim_ip_turbohybrid_ops` is a compatible SQL alias when
  it uses the existing operator and support functions.
- Weighted multivector fusion is a new supported SQL behavior and must remain
  document-keyed with deterministic score normalization.

On-disk compatibility risk:

- A persistent multivector doc map sidecar changes index storage. It requires an
  explicit page kind and version or magic, safe discovery, fallback for indexes
  without the sidecar or clear REINDEX guidance, native cache memory accounting,
  and tests for build, insert, scan, and vacuum/dead tuple behavior.

## Current Gaps

- `turbohybrid_multivector` text input was not parseable before this plan, even
  though text output emitted a structured literal.
- Scalar multivector distance ignored `text_query` outside an index scan before
  the fallback rejection was added.
- Several access-method paths still need cleanup from raw `HAS_VECTOR` or
  `HAS_MULTIVECTOR` checks toward dense helper functions.
- Per-token unique document accounting, bounded top-k maintenance, sidecar
  storage, blocked MaxSim kernels, and the first conservative insert-batching
  pass are implemented. Deeper multivector insert coalescing remains future work
  because graph-neighbor insertion is still intentionally per token.
