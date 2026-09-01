# pg_colbert_llama companion

`extensions/pg_colbert_llama` optionally generates dense embeddings and
ColBERT token vectors through llama.cpp. It requires `pgturbohybrid` and has no
independent sparse-model backend.

The primary SQL functions are:

```sql
llama_embed_vector(model text, input text, options jsonb DEFAULT '{}')
llama_embed_vector_batch(model text, inputs text[], options jsonb DEFAULT '{}')
llama_embed_tokens(model text, input text, options jsonb DEFAULT '{}')
llama_embed_mv(model text, input text, options jsonb DEFAULT '{}')
llama_embed_mv_batch(model text, inputs text[], options jsonb DEFAULT '{}')
```

Use `mode: dense` for a pooled vector and `mode: tokens` for late interaction.
Model profiles describe tokenization and projection behavior required for
parity with the model that produced the GGUF.

Development uses the repository Nix environment. The dependency-light stub is
the default test backend; set `PG_COLBERT_LLAMA_ENGINE=llama` and provide the
llama.cpp include and library paths for a live build.
