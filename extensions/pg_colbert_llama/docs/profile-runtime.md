# pg_colbert_llama Profile Runtime

`pg_colbert_llama` consumes `pg_colbert_profile_v1` metadata produced by
[`agentxagi/colbert-gguf-converter`](https://github.com/agentxagi/colbert-gguf-converter). The preferred runtime input is a
llama.cpp-loadable GGUF with an embedded `pg_colbert.profile_json` profile or a
matching sidecar named `<model>.gguf.colbert_profile.json`.

The broad compatibility claim is intentionally limited:

> supports canonicalized BERT/ModernBERT-style ColBERT GGUFs matching our
> Sauerkraut/PyLate conventions, with correct tokenization shape and usable
> ranking smoke behavior.

Do not claim universal ColBERT support. Profile-backed models should be
described by the compatibility level reported by `colbert_model_info()`.

## Compatibility Levels

- `load`: llama.cpp can load the GGUF and create an embedding context.
- `shape`: output dimension, query length, document length, and vector counts
  match the loaded profile.
- `ranking_smoke`: pgturbohybrid dense or hybrid search returns plausible
  ranked results for small controlled queries.
- `token_plan_parity`: `colbert_debug()` token IDs, padding, output masks, and
  retention decisions match the profile token-plan golden.
- `vector_parity`: token-aligned vectors match the PyLate reference within the
  configured tolerance.

Current runtime profile support targets `shape`, `ranking_smoke`, and
`token_plan_parity` diagnostics. Strict PyLate query vector parity is limited by
llama.cpp attention-mask capabilities for query expansion profiles where
expansion tokens are emitted but should not be attended by the original query
tokens.

## Why Profiles Are Required

ColBERT GGUFs need model-specific runtime metadata that cannot safely be guessed
from global GUCs:

- query and document prefixes
- query expansion length and pad token
- document max length
- punctuation and document skiplist token IDs
- tokenizer source and tokenizer capability requirements
- projection module shape
- profile compatibility and known limitations

The converter profile spec is the source of truth. `pg_colbert_llama` first
tries the GGUF metadata key `pg_colbert.profile_json`, then the sidecar
`<model>.gguf.colbert_profile.json`, then falls back to legacy GUC-derived
behavior and reports `profile_source = "guc_fallback"`.

## Runtime Boundaries

The llama engine supports canonical llama.cpp BERT/ModernBERT GGUFs that use the
Sauerkraut/PyLate ColBERT projection profile shape. Dense/linear projection
modules can read named F32/F16 weight tensors and optional named F32/F16 bias
tensors; normalize modules support `p = 2`; truncate modules can reduce the
final output dimension. The presence of `pg_colbert.profile_json` is not enough
by itself; the GGUF must also include llama.cpp's canonical BERT metadata such
as `bert.feed_forward_length` and `bert.attention.head_count`. Profile module
chains are parsed and validated, and unsupported activation functions,
unsupported normalization powers, or unknown module types fail explicitly
instead of being ignored.

Tokenizer support is also profile-aware.  A GGUF that llama.cpp can load and
tokenize is accepted even if the profile states that the original tokenizer
source was Hugging Face JSON.  A GGUF that only embeds Hugging Face tokenizer
JSON without canonical llama.cpp tokenizer metadata reports
`unsupported_tokenizer` with remediation guidance to prepare the model with
`colbert-gguf-converter --target-runtime llama_cpp`.
