# ValorBrain fork roadmap — 2026

Evolution plan for the **ValorBrain production fork** of `pgturbohybrid`.
Upstream alpha features (multivector ColBERT, public benchmarks) continue in
parallel; this roadmap optimizes the **ValorBrain Postgres 5433** workload:
~6k active docs/tenant, bilingual PT-BR + EN, high ingest churn, sub-second hybrid
`/search` and `/retrieve`.

Status: ✅ done · 🔄 in progress · ▫ planned · ⏸ defer

---

## Now (Q2 2026) — stability

| # | Item | Why | Effort |
|---|------|-----|--------|
| ✅ 1 | BM25 delta compaction + page pointer WAL fixes | Prod corruption under concurrent embed/FTS | M |
| ✅ 2 | Merge fixes to `main`, tag `0.1.0-valorbrain.1` | Single deploy branch | B |
| 🔄 3 | Concurrency regression in CI | Reproduce insert + FTS + hybrid query parallel | M |
| ▫ 4 | `REINDEX CONCURRENTLY` runbook automation | Post-upgrade hygiene | B |

## Next (Q3 2026) — ValorBrain retrieval quality

| # | Item | Why | Effort |
|---|------|-----|--------|
| ▫ 5 | **Scalar fallback guard** | `EXPLAIN` must show `Index Scan` on `turbohybrid`; alert when seq scan | M |
| ▫ 6 | **Bilingual BM25 probe** | `websearch_to_tsquery` config matched to query language (PT/EN) | M |
| ▫ 7 | **BM25 delta chunking** | Large tuples on write path → vacuum bloat, latency spikes | M |
| ▫ 8 | **Hybrid candidate budget tuning** | Per-tenant reloptions from `valorbrain_hybrid_embedding_coverage` | B-M |
| ▫ 9 | **Autovacuum tuning validation** | `documents_fts` churn under BullMQ embed load (migration 0041) | B |

## Later (Q4 2026+) — scale & ColBERT

| # | Item | Why | Effort |
|---|------|-----|--------|
| ▫ 10 | **Multivector document-node build perf** | `problem.md` — serial edge build blocks ColBERT scale | A |
| ▫ 11 | **Parallel edge construction hardening** | `pedgeplan.md` implemented; validate on 10k+ ColBERT | A |
| ▫ 12 | **pg_colbert_llama + 15m GGUF path** | Late interaction without leaving Postgres | A |
| ▫ 13 | **Partial index by tenant** | Optional `WHERE tenant_id = …` on hybrid index for multi-tenant PG | M |

## Explicitly deferred (not fork-critical)

- Public dbpedia-1M marketing benchmarks (upstream concern)
- `high_recall` profile as ValorBrain default (latency profile wins for hooks)
- Full multivector SQL until build time &lt; 10 min for 10k docs

## Success metrics (ValorBrain prod)

| Metric | Target |
|--------|--------|
| Hybrid index scan rate | &gt; 95% of `/search` hybrid queries |
| p95 turbohybrid latency | &lt; 500 ms (warm) |
| `valorbrain_hybrid_embedding_coverage_ratio` | &gt; 99% |
| BM25 corruption incidents | 0 (post compaction fix) |
| REINDEX downtime | 0 (always `CONCURRENTLY`) |

## Release cadence

1. ValorBrain-facing change lands on `main`
2. `th-installcheck` + ValorBrain `ptbr-benchmark` smoke
3. Tag `0.1.0-valorbrain.N` + note in ValorBrain `docs/runbooks/postgres-valorbrain.md`
4. `make install` on PG 5433 + conditional `REINDEX CONCURRENTLY`