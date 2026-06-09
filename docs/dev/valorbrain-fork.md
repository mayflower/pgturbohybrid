# pgturbohybrid — linha de manutenção ValorBrain

Fork mantido à frente do origin (`agentxagi/pgturbohybrid`) para operação
do ValorBrain em PostgreSQL 17/18 com hybrid search em produção.

## Branch ativa

`main` (fixes BM25 merged 2026-06-08)

| Commit | Fix |
|--------|-----|
| `e0125f2` | docs: ValorBrain maintenance fork section and 2026 roadmap |
| `a6e4168` | Serializa compaction BM25 delta com inserts concorrentes |
| `c7cd01c` | Refresh de page pointers após chunk linking WAL |

### 2026-06-09 — Page compaction + production stability

| Change | Detail |
|--------|--------|
| **AM page compaction** | `src/pgturbohybrid_quant_compact.c` — automatic dead-node removal during `amvacuumcleanup`. Triggers on dead-node ratio ≥ threshold% OR page bloat ≥ 3× expected. Atomic metapage swap, crash-safe (old chains untouched until swap). |
| **Reloption `page_compaction_threshold`** | Per-index, default 25, range 0–100. `WITH (page_compaction_threshold = 10)` on CREATE INDEX. |
| **Backfill script rewrite** | `scripts/valorbrain/hybrid-embed-backfill-batched.sh` — bulk UPDATE → REINDEX (compact rebuild via ambuild). Global advisory lock prevents concurrent embed worker writes. `--dry-run` flag. |
| **DB lock circuit breaker** | ValorBrain embed worker: 3-state breaker (closed/open/half_open). Skips hybrid index finalize when contention detected. Env: `VALORBRAIN_DB_LOCK_BREAKER`, `VALORBRAIN_DB_CB_THRESHOLD`, `VALORBRAIN_DB_CB_RESET_SECS`. |

**Root cause**: TurboHybrid AM page chains (code, adjacency, exact, BM25) are append-only. Incremental inserts allocate new pages without reusing space from deleted nodes. In production with continuous embed worker writes, this caused 20×+ index bloat (5.7 GB for 30 MB of data).

**Fix stack**:
1. REINDEX produces compact index via `ambuild` (confirmed: 77 MB → 14 MB)
2. AM compaction auto-triggers during vacuum/autovacuum when dead nodes accumulate
3. Backfill script uses REINDEX instead of drop+rebuild (atomic, index never missing)
4. Circuit breaker + global advisory lock prevent concurrent writes during backfill

Roadmap futuro: [valorbrain-roadmap-2026.md](./valorbrain-roadmap-2026.md)

## Deploy no PG ValorBrain (5433)

```bash
cd /opt/pgturbohybrid
nix --extra-experimental-features 'nix-command flakes' develop --command make install
sudo systemctl restart postgresql   # ou instância PG 5433

# Prewarm + prune cache + scalar fallback guard (Tier 1 ops)
PGHOST=127.0.0.1 PGPORT=5433 PGDATABASE=valorbrain \
  bash scripts/valorbrain/post-deploy.sh
```

Após corrupção ou upgrade de extensão:

```sql
REINDEX INDEX CONCURRENTLY idx_documents_fts_hybrid;
REINDEX INDEX CONCURRENTLY idx_content_vectors_turbo;
```

Depois do REINDEX, rode `post-deploy.sh` de novo (prewarm + prune).

## Scripts operacionais (Tier 1)

| Script | Função |
|--------|--------|
| `scripts/valorbrain/post-deploy.sh` | Prewarm índices, prune `.tqcache`, guard opcional |
| `scripts/valorbrain/hybrid-embed-backfill-batched.sh` | Bulk backfill embeddings + REINDEX compact rebuild. Global advisory lock. `--dry-run` |
| `scripts/valorbrain/scalar-fallback-guard.sh` | Falha se hybrid não usar `Index Scan` turbohybrid |
| `scripts/valorbrain/hybrid-tuning-report.sql` | Snapshot GUCs + `turbohybrid_last_scan_stats()` |
| `scripts/valorbrain/concurrency-check.sh` | Parallel hybrid queries no ValorBrain (PG 5433) |
| `scripts/valorbrain/stability-check.sh` | Health check: index bloat ratio, dead tuples, embed coverage |
| `scripts/ci-hybrid-concurrency.sh` | Smoke CI: insert delta + hybrid query em paralelo |

GUC de cap de disco: `turbohybrid.native_cache_disk_max_mb` (default 8192).
Prune manual: `SELECT turbohybrid_prune_shared_cache();`

## Integração ValorBrain

| Componente | Índice / path |
|------------|----------------|
| Vector 4-bit | `idx_content_vectors_turbo` |
| Hybrid RRF | `idx_documents_fts_hybrid` |
| FTS bounded | `valorbrain_fts_index_text()` (app) |

IVFFlat (`idx_content_vectors_embedding`) **não recriar** — removido em prod.

## Roadmap fork (prioridade)

1. ~~**BM25 delta chunking**~~ — done
2. **Regression concorrente** — insert/update FTS paralelo em CI
3. ~~**Scalar fallback guard**~~ — done
4. **Multivector Fase 3** — ColBERT; build performance (`problem.md`)
5. **Insert-path compaction** — page reuse on incremental inserts (currently append-only; REINDEX is the compact path)

## Releases internos

Tag sugerido: `0.1.0-valorbrain.N` com changelog separado do upstream.
Antes de tag: `th-installcheck` no flake + smoke hybrid no ValorBrain.