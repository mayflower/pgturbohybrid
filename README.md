# pgturbohybrid

`pgturbohybrid` is a PostgreSQL index access method for fast dense vector,
BM25, hybrid, and ColBERT-style late-interaction search.

The extension has one installable package and one query type. Dense, lexical,
and token-vector branches share the same index and are fused inside PostgreSQL.

## Requirements

- PostgreSQL 14 or newer (the Nix development environment uses PostgreSQL 17)
- pgvector 0.8.2 or newer
- Nix with flakes for development and validation

## Install and develop

```sh
nix --extra-experimental-features 'nix-command flakes' develop
echo "$IN_NIX_SHELL"
pg_config --version
th-pg-init
th-installcheck
th-prove-installcheck
```

Do not run `make install` against the flake-provided `pg_config`; the target is
in the immutable Nix store.

Enable the extension after pgvector:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

## Dense and hybrid search

```sql
CREATE TABLE documents (
  id bigint PRIMARY KEY,
  embedding vector(768) NOT NULL,
  body_tsv tsvector NOT NULL
);

CREATE INDEX documents_search_idx ON documents USING turbohybrid (
  embedding vector_cosine_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);
```

Dense query:

```sql
SELECT id
FROM documents
ORDER BY embedding <~> turbohybrid_dense_query($1::vector, 20, 100)
LIMIT 20;
```

Hybrid query:

```sql
SELECT id
FROM documents
ORDER BY embedding <~> turbohybrid_hybrid_query(
  $1::vector,
  websearch_to_tsquery('english', $2),
  20,
  100,
  100
)
LIMIT 20;
```

## Late interaction

Each `turbohybrid_multivector` stores the token vectors for one document. The
graph indexes token/subvector nodes, gathers document candidates, and reranks
the final documents with exact float32 MaxSim. Similarity is larger-is-better;
SQL ordering uses `distance = -maxsim` so smaller remains better.

```sql
CREATE TABLE passages (
  id bigint PRIMARY KEY,
  tokens turbohybrid_multivector NOT NULL
);

CREATE INDEX passages_search_idx ON passages USING turbohybrid (
  tokens multivector_maxsim_ip_turbohybrid_ops
);

SELECT id
FROM passages
ORDER BY tokens <~> turbohybrid_multivector_query($1, 20, 100)
LIMIT 20;
```

See [late interaction](docs/multivector-late-interaction.md) for the exact
scoring contract and [architecture](docs/architecture.md) for the retained
execution path. The optional [pg_colbert_llama companion](docs/colbert-llama-extension.md)
can generate dense and token embeddings inside PostgreSQL.

## Operational API

```sql
SELECT turbohybrid_index_stats('documents_search_idx');
SELECT turbohybrid_estimate_memory('documents_search_idx');
SELECT turbohybrid_prewarm('documents_search_idx');
SELECT turbohybrid_validate_index('documents_search_idx', true);
```

The small public tuning surface covers candidate budgets, BM25 WAND, SIMD,
the postings cache, and exact multivector reranking. Query-specific budgets
belong in `turbohybrid_query(...)`; there are no profile presets or experimental
extension layers.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
