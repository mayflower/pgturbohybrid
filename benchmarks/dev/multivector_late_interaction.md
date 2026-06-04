# Multivector Late-Interaction Slopes

`multivector_late_interaction.sql` is a deterministic developer harness for the
ColBERT/Late-Interaction path. It is not intended to prove absolute latency; it
shows whether the implementation scales in the expected direction as the main
complexity variables change.

Run it from a development database:

```sh
psql -d "$PGDATABASE" -f benchmarks/dev/multivector_late_interaction.sql
```

Useful psql variables:

```sh
psql -d "$PGDATABASE" \
  -v DOCS=512 \
  -v DIMS=32 \
  -v FINAL_K=10 \
  -v ITERS=5 \
  -f benchmarks/dev/multivector_late_interaction.sql
```

The default is intentionally small (`DOCS=256`, `DIMS=32`) so it can run on a
laptop. Increase `DOCS` and `DIMS` when you want stronger slope evidence.

## Variables

- `D`: number of documents.
- `L`: document token vectors per document.
- `N`: graph subnodes, `N = D * L`.
- `Q`: query token vectors.
- `d`: vector dimension.
- `Ks`: subvector hits per query token.
- `C`: unique document candidates touched by subvector hits.
- `R`: exact heap rerank documents.

## Grid Axes

The tracked harness keeps `D` configurable through `DOCS` and sweeps these
deterministic cells:

- `L = 8, 32, 128`
- `Q = 4, 16, 64`
- `d = 32, 96, 128`
- `R = off, 25, 100`
- `Ks = 16, 64, 256`

## Expected Complexity

- Build should grow roughly with `O(D * L)` because each document token vector is
  expanded into a TurboQuant subnode.
- Approximate query collection should grow with `O(Q * ANN + Q * Ks)` because
  each query token runs a graph search and contributes bounded raw subvector
  hits.
- The MaxSim accumulator should stay `O(C * Q)`, not `O(D * Q)` and not
  `O(N * Q)`.
- Exact heap rerank should grow with `O(R * Q * L * d)`.

The SQL report prints build time, index size, graph node count, p50/p95 local
latency, exact-reference recall@K on the synthetic corpus, raw subvector hits,
unique documents, document candidate count, exact rerank pairs, accumulator
memory estimate, and `fusion_strategy` from `turbohybrid_last_scan_stats()`.
Exact rerank pair counts are bounded by available candidate documents, so very
small `DOCS` smoke runs may cap `R=25` and `R=100` at the same observed work.

## Comparing Against Single-Vector

Use the same database and run an existing dense-vector developer grid such as:

```sh
psql -d "$PGDATABASE" -f benchmarks/dev/profile_recall_latency_grid.sql
```

Compare its dense-vector index size and scan stats against the multivector
`N = D * L` rows in this harness. The comparison should be interpreted by
subnode count, not document count: a multivector document with `L = 32` stores
and searches 32 dense subnodes but returns one heap tuple.

## Output Policy

Do not commit generated tables, logs, JSON, CSV, or pasted benchmark output.
Only the harness and this explanation are tracked.
