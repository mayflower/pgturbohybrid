# pgturbohybrid v0.1.0-alpha.2

This is an alpha release.

TurboHybrid is now fast by default.

Think: hot hatch, not hypercar — practical, compact, and surprisingly quick,
but still something you should test before daily-driving in production.

## 1. What is pgturbohybrid?

`pgturbohybrid` is a PostgreSQL extension for hybrid dense-vector and lexical
retrieval on top of pgvector.

TurboHybrid, the feature provided by this package, combines pgvector dense
retrieval with PostgreSQL text search using BM25-style ranking. It then fuses
the candidate lists with reciprocal-rank fusion, or RRF, inside one
`turbohybrid` index access method.

The goal is practical PostgreSQL retrieval for RAG, docs search, product search,
support search, and other cases where dense-only search can miss exact terms and
text-only search can miss paraphrases.

## 2. What changed?

This alpha focuses on making the useful fast path the normal first experience.

- Fresh sessions default to the `latency` profile.
- New default indexes use 4-bit quantization with `exact_storage = off`.
- Default candidate budgets are 100 dense candidates and 100 BM25 candidates.
- SQL `ORDER BY ... LIMIT n` is used as the final top-k target when possible.
- Diagnostics report the effective profile, candidate budgets, index use,
  quantization, exact-storage state, timings, and final-k source.
- Public docs, release hygiene, benchmark docs, and community files have been
  cleaned up for an open-source alpha release.

## 3. Fast defaults

The intended default query shape is now simple:

1. Create a normal `turbohybrid` index.
2. Query with `ORDER BY embedding <~> turbohybrid_query(...)`.
3. Add `LIMIT n`.
4. Check `turbohybrid_last_scan_stats()` when you want to confirm the path.

You no longer need to pass `dense_k`, `bm25_k`, or `final_k` just to get the
default fast path.

## 4. Benchmark snapshot

On the FIQA/OpenAI benchmark setup documented in the repository:

- dataset: FIQA/OpenAI
- corpus rows: 57,638
- queries: 648
- embedding dimensions: 1,536
- profile: `latency`
- index: default TurboHybrid index
- query budgets: omitted, using the default fast path

In that exact setup, `pgturbohybrid` default reported:

- p95 latency: 0.910 ms
- nDCG@10: 0.421540

The SQL RRF hybrid baseline reported:

- p95 latency: 3.254 ms
- nDCG@10: 0.423341

This is a scoped benchmark result, not a global pgvector comparison and not a
production promise. Results vary by dataset, hardware, PostgreSQL settings,
embedding model, and relevance labels. Please reproduce it on your own data
before drawing conclusions.

Full benchmark context and reproduction notes are in the
[FIQA/OpenAI benchmark details](https://github.com/agentxagi/pgturbohybrid/blob/v0.1.0-alpha.2/docs/benchmarks/fiqa-openai.md).

## 5. Quality mode

Fast defaults are not always the right settings.

If relevance matters more than lowest latency, start with:

```sql
SET turbohybrid.profile = 'quality';
```

For relevance-sensitive evaluation, also benchmark an exact-storage index:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

Treat `latency` and `quality` as profiles to measure, not universal answers.

## 6. Installation

Install pgvector first, then build and install `pgturbohybrid`:

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git ../pgvector
make -C ../pgvector
make -C ../pgvector install

git clone https://github.com/agentxagi/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

Then create both extensions in your database.

## 7. Quick query example

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;

CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1,
    text_query => websearch_to_tsquery('english', $2)
)
LIMIT 10;
```

For a copy-paste local walkthrough, see the
[fast setup guide](https://github.com/agentxagi/pgturbohybrid/blob/v0.1.0-alpha.2/docs/fast_setup.md).

## 8. Known limitations

- This is alpha software.
- APIs and on-disk index formats may change before a stable release.
- Production use needs your own benchmark and relevance validation.
- `pgturbohybrid` is a standalone companion extension, not an official pgvector
  project.
- pgvector must be installed first.
- Incompatible storage changes may require `REINDEX`.
- Parallel index build is disabled for this alpha while dense graph and BM25
  build-path correctness evidence is hardened. `CREATE INDEX CONCURRENTLY` is
  covered by the release tests.
- Generated benchmark artifacts are not committed to the repository; attach
  full artifacts to releases, CI runs, or issues when publishing benchmark
  claims.

## 9. Thank you / attribution

`pgturbohybrid` depends on pgvector and contains code derived from pgvector's
HNSW implementation. Thank you to the pgvector project for the excellent
PostgreSQL vector search foundation.

This project is a separate experimental companion extension from Mayflower.
Thanks to everyone testing early builds, reporting compatibility results, and
bringing real retrieval workloads. That is what will make the little hot hatch
useful on real roads.

Useful links:

- [README](https://github.com/agentxagi/pgturbohybrid/blob/v0.1.0-alpha.2/README.md)
- [Fast setup guide](https://github.com/agentxagi/pgturbohybrid/blob/v0.1.0-alpha.2/docs/fast_setup.md)
- [FIQA/OpenAI benchmark details](https://github.com/agentxagi/pgturbohybrid/blob/v0.1.0-alpha.2/docs/benchmarks/fiqa-openai.md)
- [Changelog](https://github.com/agentxagi/pgturbohybrid/blob/v0.1.0-alpha.2/CHANGELOG.md)
