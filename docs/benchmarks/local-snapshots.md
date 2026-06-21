# Local Benchmark Snapshots

These are local benchmark snapshots moved out of the README. They are not
global claims; results vary by dataset, hardware, PostgreSQL settings, cache
state, and query workload.

## Benchmark Snapshot

These are local benchmark snapshots, not global claims. Results vary by
dataset, hardware, PostgreSQL settings, cache state, and query workload. nDCG
means normalized discounted cumulative gain, a relevance metric for ranked
results. HNSW means Hierarchical Navigable Small World, pgvector's graph index
type for approximate nearest-neighbor vector search.

On this FIQA/OpenAI setup, the run used 57,638 corpus rows, 648 qrels-backed
queries, 1,536-dimensional OpenAI `text-embedding-3-small` embeddings, the
`latency` profile, 100 dense candidates, 100 BM25 candidates, `final_k = 10`,
three warmup passes, and one measured pass. The TurboHybrid index used 4-bit
quantization with `exact_storage = off`.

| Method | Settings | p95 | nDCG@10 |
| --- | --- | ---: | ---: |
| pgturbohybrid default | default 4-bit exact-free index, adaptive widening off, LIMIT-inferred `final_k` | 1.628 ms | 0.415535 |
| pgturbohybrid adaptive auto 2.0 | explicit diagnostic setting, adaptive dense widening opt-in | 1.552 ms | 0.421465 |
| SQL RRF baseline | pgvector HNSW plus PostgreSQL GIN full-text search, 100/100 candidates | 2.009 ms | 0.423430 |
| pgvector dense-only reference | pgvector HNSW, no lexical branch | 1.412 ms | 0.442786 |

On a larger DBPedia/OpenAI3-large 1M qdrant-self setup, the run used 1,000,000
rows, 1,000 sampled self-queries, 3,072-dimensional
`text-embedding-3-large` embeddings, the `latency` profile, 100 dense
candidates, 100 BM25 candidates, `final_k = 10`, one warmup pass, and three
measured passes.

| Method | p95 | p99 | nDCG@10 | recall@10 | Index size |
| --- | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid | 3.978 ms | 7.386 ms | 0.940439 | 0.980 | 2.38 GB |
| pgvector halfvec HNSW + GIN FTS SQL RRF | 79.799 ms | 219.576 ms | 0.971086 | 0.992 | 8.41 GB |

That DBPedia result is a tradeoff, not a victory lap: TurboHybrid was much
faster and smaller on this machine, while the pgvector + FTS SQL RRF baseline
kept higher nDCG@10 and recall@10. The package default latency profile keeps
adaptive dense widening off; adaptive widening variants remain available in the
benchmark harness for controlled experiments. Benchmark details, baselines, and
reproduction notes are in
[docs/benchmarks/fiqa-openai.md](fiqa-openai.md),
[benchmarks/dbpedia_openai3_large.md](../../benchmarks/dbpedia_openai3_large.md), and
[benchmarks/README.md](../../benchmarks/README.md).

The same DBPedia/OpenAI3-large corpus can also be used as a dense-only systems
comparison. This is not a hybrid-search benchmark: it uses the dataset's
existing 3,072-dimensional embeddings, no BM25 branch, no full-text search, and
no SQL RRF fusion. Turbovec is an in-process dense vector library that runs a
flat (brute-force) 4-bit scan over all rows, so treat this as a useful
reference point rather than a PostgreSQL access-method comparison. The run below
is a fresh local snapshot (Ice Lake Xeon, current build, 200 qdrant-self
queries, one warmup pass plus measured passes); the pgturbohybrid percentiles
are end-to-end SQL latency while the Turbovec percentiles are in-process,
single-threaded `index.search()` timings.

| Dense-only method | p50 | p95 | p99 | nDCG@10 | recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid dense-only | 1.848 ms | 2.287 ms | 2.443 ms | 0.965 | 0.965 |
| Turbovec TurboQuant 4-bit | 174.176 ms | 181.312 ms | 190.260 ms | 1.000 | 1.000 |

In this qdrant-self setup each query has a single positive qrel, so recall@10 is
"is the source document in the top 10." Turbovec's flat scan is exact over the
4-bit codes and recovers that document for every query (recall@10 = 1.000), but
pays O(n) latency (~174 ms p50). `pgturbohybrid` uses a sub-linear graph index:
~94x lower latency (1.848 ms p50) while still recovering the source document for
96.5% of queries. The top-10 overlap between the two runs was 0.922 — i.e. the
graph reproduces ~92% of Turbovec's exact-over-codes top-10. As with the hybrid
numbers above, repeat this on your own hardware and query mix before drawing
conclusions.

If you already have a PostgreSQL RAG database, the bring-your-own benchmark
compares TurboHybrid with your existing retrieval SQL on your own rows and
query embeddings. See
[docs/benchmarks/bring-your-own-rag.md](bring-your-own-rag.md).

