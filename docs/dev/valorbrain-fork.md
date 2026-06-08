# pgturbohybrid — linha de manutenção ValorBrain

Fork mantido à frente do origin (`agentxagi/pgturbohybrid`) para operação
do ValorBrain em PostgreSQL 17/18 com hybrid search em produção.

## Branch ativa

`main` (fixes BM25 merged 2026-06-08)

| Commit | Fix |
|--------|-----|
| `a6e4168` | Serializa compaction BM25 delta com inserts concorrentes |
| `c7cd01c` | Refresh de page pointers após chunk linking WAL |

Roadmap futuro: [valorbrain-roadmap-2026.md](./valorbrain-roadmap-2026.md)

## Deploy no PG ValorBrain (5433)

```bash
cd /opt/pgturbohybrid
nix --extra-experimental-features 'nix-command flakes' develop --command make install
sudo systemctl restart postgresql   # ou instância PG 5433
```

Após corrupção ou upgrade de extensão:

```sql
REINDEX INDEX CONCURRENTLY idx_documents_fts_hybrid;
```

## Integração ValorBrain

| Componente | Índice / path |
|------------|----------------|
| Vector 4-bit | `idx_content_vectors_turbo` |
| Hybrid RRF | `idx_documents_fts_hybrid` |
| FTS bounded | `valorbrain_fts_index_text()` (app) |

IVFFlat (`idx_content_vectors_embedding`) **não recriar** — removido em prod.

## Roadmap fork (prioridade)

1. **BM25 delta chunking** — tuples grandes no write path nativo
2. **Regression concorrente** — insert/update FTS paralelo em CI
3. **Scalar fallback guard** — evitar queries fora do índice turbohybrid
4. **Multivector Fase 3** — ColBERT; build performance (`problem.md`)

## Releases internos

Tag sugerido: `0.1.0-valorbrain.N` com changelog separado do upstream.
Antes de tag: `th-installcheck` no flake + smoke hybrid no ValorBrain.