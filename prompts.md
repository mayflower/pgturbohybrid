Below is a **Codex prompt set** split across the two repos:

```text
mayflower/colbert-gguf-converter
mayflower/pgturbohybrid
```

The key design is: **the converter should become the source of truth for ColBERT profile metadata**, and `pg_colbert_llama` should consume that profile instead of relying on global GUC guesses.

The converter already produces a custom `pg_colbert_v1` GGUF format and says it preserves ColBERT projection layers, tokenizer configuration, and query/document metadata.  The current spec already contains many profile ingredients, including `colbert.query_prefix`, `colbert.document_prefix`, query/document lengths, skiplist words, token IDs, and prefix token IDs.  The runtime side already exposes the right SQL functions, including `colbert`, `colbert_vectors`, `colbert_float4`, `colbert_dim`, `colbert_model_info`, and `colbert_mv`.

The missing bridge is a formal **profile contract** that the converter writes and the Postgres extension consumes.

---

# Prompt set overview

Use these in order:

```text
A. Converter profile generation
B. Runtime profile consumption
C. Parity tooling
D. Cross-repo integration
E. Documentation and claim control
```

I would do them as separate PRs so each PR is reviewable.

---

# A. `mayflower/colbert-gguf-converter` prompts

## Prompt A1 — Add ColBERT profile v1 spec

```text
Repository: mayflower/colbert-gguf-converter
Branch from: master

Goal:
Introduce a formal ColBERT runtime profile schema that can be embedded in GGUF metadata and also written as a sidecar JSON file. This profile will be consumed by pg_colbert_llama in mayflower/pgturbohybrid.

Context:
The current docs/COLBERT_GGUF_SPEC.md defines pg_colbert_v1 metadata, including projection, query/document prefixes, query/document lengths, skiplist words, token IDs, and tokenizer JSON. Extend that into a single explicit profile contract.

Tasks:
1. Add docs/COLBERT_PROFILE_SPEC.md.
2. Define schema name:
   pg_colbert_profile_v1
3. Define these top-level fields:
   - schema
   - source_model_id
   - source_revision
   - converter_version
   - backbone_family
   - colbert_family
   - similarity
   - output_dim
   - normalize
   - tokenizer
   - query
   - document
   - projection
   - compatibility
4. Define tokenizer object:
   - source: "llama" | "hf_json" | "canonical_ggml"
   - tokenizer_model
   - tokenizer_json_sha256
   - special_tokens:
       cls_token_id
       sep_token_id
       pad_token_id
       mask_token_id
       q_token_id
       d_token_id
   - prefix_token_ids:
       query
       document
5. Define query object:
   - prefix
   - max_length
   - pad_to
   - pad_token_id
   - pad_token
   - attend_to_expansion_tokens
   - retain_policy
   - output_policy
   - token_type_id
6. Define document object:
   - prefix
   - max_length
   - pad_to
   - retain_policy
   - skiplist_words
   - skiplist_token_ids
   - token_type_id
7. Define projection object:
   - kind: "identity" | "dense" | "module_chain"
   - input_dim
   - output_dim
   - modules
   - normalize_after
8. Define compatibility object:
   - llama_cpp_loadable: bool
   - requires_profile: bool
   - strict_pylate_profile: bool
   - known_limitations: string[]
9. Update docs/COLBERT_GGUF_SPEC.md to reference the new profile spec.
10. Add the GGUF metadata key:
    pg_colbert.profile_json = JSON string containing the profile.
11. Add sidecar filename convention:
    <model>.gguf.colbert_profile.json
12. Include a complete example profile for:
    VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m
13. Do not change converter code yet.

Tests:
- Documentation-only change.
- Run pytest if cheap, but no code behavior should change.
```

---

## Prompt A2 — Add profile dataclasses and JSON writer

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A1 branch

Goal:
Add Python data structures and writers for pg_colbert_profile_v1.

Tasks:
1. Create tools/colbert_profile.py.
2. Add dataclasses:
   - ColbertProfile
   - TokenizerProfile
   - SpecialTokensProfile
   - QueryProfile
   - DocumentProfile
   - ProjectionProfile
   - ProjectionModule
   - CompatibilityProfile
3. Add:
   - to_dict()
   - to_json()
   - validate_profile(profile)
   - write_profile_sidecar(profile, gguf_path)
4. Validation must check:
   - schema == "pg_colbert_profile_v1"
   - output_dim > 0
   - query.max_length > 0
   - document.max_length > 0
   - projection.output_dim == output_dim unless projection.kind == "identity"
   - special token IDs are integers or null
   - skiplist_token_ids contains only non-negative integers
5. Add unit tests in tests/test_colbert_profile.py:
   - minimal valid profile serializes
   - invalid output_dim fails
   - projection dim mismatch fails
   - skiplist token IDs validate
6. No GGUF writing yet.
```

---

## Prompt A3 — Extract profile from Hugging Face / PyLate metadata

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A2 branch

Goal:
Build a ColBERT profile from the source Hugging Face / SentenceTransformers / PyLate repository.

Files to inspect:
- tools/convert_colbert_hf_to_gguf.py
- tools/inspect_colbert_hf.py
- tools/create_pylate_golden.py
- config_sentence_transformers.json
- modules.json
- 1_Dense/config.json
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json

Tasks:
1. In tools/convert_colbert_hf_to_gguf.py, factor metadata extraction into reusable functions:
   - load_sentence_transformers_config()
   - load_tokenizer_profile()
   - load_query_document_profile()
   - load_projection_profile()
   - build_colbert_profile()
2. Extract from config_sentence_transformers.json where available:
   - query_prefix
   - document_prefix
   - query_length
   - document_length
   - similarity_fn_name
   - attend_to_expansion_tokens
   - skiplist_words
3. Extract tokenizer special IDs using AutoTokenizer:
   - cls_token_id
   - sep_token_id
   - pad_token_id
   - mask_token_id
   - q_token_id
   - d_token_id
4. Compute:
   - query_prefix_token_ids
   - document_prefix_token_ids
   - skiplist_token_ids
5. For skiplist_token_ids:
   - tokenize each skiplist word with add_special_tokens=False
   - include single-token IDs directly
   - for multi-token skiplist entries, include a structured warning in compatibility.known_limitations
6. Extract projection info:
   - Dense module config
   - in_features
   - out_features
   - bias
   - activation_function
7. Add CLI option:
   --write-profile-sidecar
   default: true
8. Add CLI option:
   --no-profile-sidecar
9. In dry-run mode, print the profile summary.
10. Add tests using existing mock fixtures:
    - profile query/document prefixes are read
    - query/document lengths are read
    - projection dims are read
    - skiplist token IDs are produced
    - profile validation passes
```

---

## Prompt A4 — Embed `pg_colbert.profile_json` into GGUF

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A3 branch

Goal:
Embed the ColBERT profile JSON into GGUF metadata and write a sidecar JSON file.

Tasks:
1. Update tools/convert_colbert_hf_to_gguf.py so every non-dry-run conversion:
   - builds a ColBERT profile
   - validates it
   - writes GGUF metadata key:
     pg_colbert.profile_json
   - writes sidecar:
     <outfile>.colbert_profile.json
     unless --no-profile-sidecar is set
2. Update docs/COLBERT_GGUF_SPEC.md:
   - document pg_colbert.profile_json
   - document sidecar JSON
3. Update tools/inspect_colbert_gguf.py:
   - detect pg_colbert.profile_json
   - parse it
   - validate it
   - print concise profile summary:
       schema
       output_dim
       query prefix/length/pad_to
       document prefix/length
       skiplist token count
       projection kind/modules
       compatibility flags
4. Add tests:
   - converted mock GGUF includes pg_colbert.profile_json
   - sidecar is written
   - inspector validates profile
```

---

## Prompt A5 — Add llama.cpp canonical export mode

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A4 branch

Goal:
Support two GGUF output targets:
1. pg_colbert_v1 custom GGUF
2. llama.cpp-loadable GGUF + ColBERT profile

The pgturbohybrid pg_colbert_llama extension should prefer llama.cpp-loadable GGUFs with profile metadata.

Tasks:
1. Add CLI option:
   --target-runtime pg_colbert|llama_cpp|both
   default: pg_colbert for backwards compatibility.
2. For target-runtime=llama_cpp:
   - write llama.cpp canonical tensor names for BERT/ModernBERT where supported
   - write tokenizer.ggml.* metadata where possible
   - embed pg_colbert.profile_json
   - store colbert.proj.weight and optional colbert.proj.bias in GGUF if llama.cpp can ignore unknown tensors safely, or write them to sidecar if necessary
3. Preserve existing pg_colbert_v1 behavior for target-runtime=pg_colbert.
4. If target-runtime=both:
   - write <outfile>.pg_colbert.gguf
   - write <outfile>.llama.gguf
   - write matching profile sidecars
5. Refactor existing tensor mapping so BERT/ModernBERT canonicalization is not hardcoded only in pgturbohybrid's canonicalize_pg_colbert_gguf.py.
6. Add tests:
   - llama_cpp target contains tokenizer.ggml.model
   - llama_cpp target contains tokenizer.ggml.tokens or fails with clear explanation
   - llama_cpp target embeds pg_colbert.profile_json
   - pg_colbert target remains unchanged
7. Update README usage examples.

Important:
Do not promise universal llama.cpp support in docs. Say "llama.cpp-loadable for supported BERT/ModernBERT ColBERT backbones."
```

---

## Prompt A6 — Generate token-plan goldens

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A5 branch

Goal:
Add a PyLate/HF-based token-plan golden generator. This will let pg_colbert_llama compare tokenization, query expansion, retention, and skiplist behavior before comparing vectors.

Tasks:
1. Create tools/create_colbert_profile_golden.py.
2. Inputs:
   --model-name-or-path
   --texts-file
   --role query|doc
   --outfile
3. For each text, output JSON containing:
   - input text
   - role
   - token_ids before padding
   - token_ids after padding/truncation
   - token pieces
   - attention_mask
   - token_type_ids if available
   - retain_mask
   - retain_reasons
   - skiplist_token_ids
   - final_vector_count
4. Use PyLate where available.
5. If PyLate does not expose token plan internals, use Hugging Face tokenizer plus profile rules and clearly mark:
   token_plan_source = "hf_tokenizer_profile_rules"
6. Add tests with a tiny fixture:
   - query "red planet" includes mask padding to query length
   - document "red planet." marks punctuation as skipped when skiplist contains "."
7. Document the tool in README.
```

---

## Prompt A7 — Add strict parity report generation

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A6 branch

Goal:
Make the converter optionally produce a parity report that downstream users can trust.

Tasks:
1. Extend tools/create_pylate_golden.py or create tools/verify_pylate_parity.py.
2. Inputs:
   --model-name-or-path
   --gguf
   --profile
   --texts-file
   --role query|doc
   --outfile
3. For now, since this repo may not run llama.cpp inference directly, produce:
   - token-plan parity report
   - projection/profile consistency report
   - PyLate vector goldens
4. Output JSON:
   - profile_valid
   - token_plan_valid
   - vector_golden_available
   - known_limitations
   - texts[]
5. Add README section:
   - "This verifies converter/profile correctness, not Postgres runtime correctness."
6. Tests:
   - report file is created
   - invalid profile fails
   - missing PyLate is a clean skip or clear error
```

---

## Prompt A8 — Publishing updates

```text
Repository: mayflower/colbert-gguf-converter
Branch from: A7 branch

Goal:
When publishing GGUF to Hugging Face, publish profile and parity artifacts too.

Tasks:
1. Update tools/publish_colbert_gguf.py.
2. Include upload of:
   - .gguf
   - .gguf.colbert_profile.json
   - optional token-plan golden JSON
   - optional parity report JSON
3. Update generated model card:
   - state profile schema
   - state target runtime
   - state whether strict PyLate token-plan parity passed
   - state whether vector parity was checked
   - include exact CLI command
4. Do not claim strict vector parity unless the parity report says it passed.
5. Add tests for model card text generation.
```

---

# B. `mayflower/pgturbohybrid` prompts

## Prompt B1 — Add runtime compatibility document

```text
Repository: mayflower/pgturbohybrid
Branch from: main

Goal:
Document the current compatibility level and the new profile-driven direction for pg_colbert_llama.

Context:
Current pg_colbert_llama works for GGUF load, projection detection, shape, normalization, and pgturbohybrid ranking smoke. It is not yet strict PyLate vector parity.

Tasks:
1. Add extensions/pg_colbert_llama/docs/profile-runtime.md.
2. Explain compatibility levels:
   - load
   - shape
   - ranking_smoke
   - token_plan_parity
   - vector_parity
3. State that current broad claim must remain:
   "supports canonicalized BERT/ModernBERT-style ColBERT GGUFs matching our Sauerkraut/PyLate conventions, with correct tokenization shape and usable ranking smoke behavior."
4. Explain why profile support is needed:
   - query expansion attention mask
   - punctuation/document skiplist
   - tokenizer metadata
   - projection module chain
   - model-specific query/document lengths
5. Link to converter profile spec once available.
6. No code changes.
```

---

## Prompt B2 — Add profile parser to `pg_colbert_llama`

```text
Repository: mayflower/pgturbohybrid
Branch from: B1 branch

Goal:
Teach pg_colbert_llama to load pg_colbert_profile_v1 from either GGUF metadata or sidecar JSON.

Files:
- extensions/pg_colbert_llama/src/colbert_engine.h
- extensions/pg_colbert_llama/src/colbert_engine_llama.cpp
- extensions/pg_colbert_llama/src/pg_colbert_llama.c

Tasks:
1. Add profile structs in C/C++:
   - PgColbertRuntimeProfile
   - PgColbertTokenizerProfile
   - PgColbertQueryProfile
   - PgColbertDocumentProfile
   - PgColbertProjectionProfile
   - PgColbertCompatibilityProfile
2. Add profile load order:
   - GGUF metadata key pg_colbert.profile_json
   - sidecar <model>.gguf.colbert_profile.json
   - fallback to legacy GUC-derived profile
3. Add profile validation:
   - schema must be pg_colbert_profile_v1
   - output_dim must match expected_dim unless expected_dim is explicitly overridden
   - query/document max lengths must be positive
   - projection output dim must match output_dim
4. Avoid adding a heavy JSON dependency if possible:
   - simple parser for known fields is acceptable
   - or use PostgreSQL JSON routines on C side if cleaner
5. colbert_model_info() must report:
   - profile_loaded
   - profile_source: gguf | sidecar | guc_fallback
   - profile_schema
   - compatibility_level
   - query_length source
   - document_length source
   - skiplist_token_count
6. Add tests in stub mode:
   - profile sidecar is loaded from model_dir for stub fake model
   - colbert_model_info reports profile_loaded=true
   - invalid schema errors cleanly
   - missing profile falls back to GUCs and reports guc_fallback
```

---

## Prompt B3 — Add `colbert_debug()` with token-plan output

```text
Repository: mayflower/pgturbohybrid
Branch from: B2 branch

Goal:
Add a debug function that exposes the ColBERT token plan, not just final vectors.

Current colbert() returns token_ids and vectors. We need more detail to diagnose PyLate parity failures.

Tasks:
1. Add SQL function:
   colbert_debug(model text, input text)
   RETURNS jsonb
2. Keep colbert() unchanged for compatibility.
3. colbert_debug() must include:
   - engine
   - alias
   - role
   - profile_source
   - dim
   - vector_count
   - normalized
   - input
   - prefix
   - token_plan:
       tokens[]
4. Each token object should include:
   - index
   - id
   - piece if available through llama_token_to_piece
   - position_id
   - token_type_id if known
   - attention_mask value if known
   - output_enabled
   - retained
   - retain_reason
5. For now, if token_type_id or attention_mask is unavailable, output null and add known_limitations.
6. Add regression tests in stub mode for JSON shape.
7. Add live test guarded by PG_COLBERT_LLAMA_TEST_MODEL:
   - colbert_debug(alias:query, 'red planet') includes token_plan.tokens
   - token count is >= vector count
```

---

## Prompt B4 — Implement profile-driven retention and skiplist

```text
Repository: mayflower/pgturbohybrid
Branch from: B3 branch

Goal:
Replace the current hardcoded retention policy.

Current behavior:
- query retains every token
- document drops only PAD
This causes document punctuation mismatches against PyLate.

Tasks:
1. Replace PgColbertShouldRetainToken() with a profile-driven function.
2. The function must support:
   - retain all query tokens when query.retain_policy says so
   - document PAD drop
   - document punctuation skiplist drop
   - explicit skiplist_token_ids from profile
   - optional skiplist_words fallback using token pieces
3. Use llama_vocab_pad(), llama_vocab_cls(), llama_vocab_sep(), llama_vocab_mask() where available.
4. Use llama_token_to_piece() to expose token piece strings for fallback punctuation detection.
5. Add Unicode-basic punctuation fallback:
   - ASCII punctuation at minimum
   - document clearly that full Unicode category support is limited unless profile provides token IDs
6. colbert_debug() must show retain_reason:
   - retained_query
   - retained_document
   - dropped_pad
   - dropped_skiplist_token
   - dropped_punctuation
7. Add tests:
   - with fake profile skiplist_token_ids containing token X, doc drops token X
   - query still retains mask tokens
   - document "red planet." drops punctuation when profile says "."
8. Add live parity helper test:
   - if PG_COLBERT_LLAMA_TEST_MODEL and a profile sidecar exist, document count for "red planet." should match profile golden when provided.
```

The current implementation only drops PAD for documents, so this prompt directly targets the documented mismatch.

---

## Prompt B5 — Build explicit query/document encode plans

```text
Repository: mayflower/pgturbohybrid
Branch from: B4 branch

Goal:
Separate tokenization, query expansion, attention policy, output mask, and retention into an explicit encode plan.

Current behavior directly tokenizes prefix+input, then pads queries with mask tokens up to query_length. That gets shape right but does not expose enough to match PyLate attention behavior. :contentReference[oaicite:4]{index=4}

Tasks:
1. Add internal struct:
   PgColbertEncodePlan
   containing:
   - token_ids
   - token_pieces
   - position_ids
   - token_type_ids
   - attention_mask
   - output_mask
   - retain_mask
   - retain_reasons
   - n_tokens
2. Add:
   PgColbertBuildEncodePlan(profile, role, input)
3. Rules:
   - use profile query/document prefix
   - use profile max_length
   - use profile query pad_to
   - use profile pad_token_id / mask_token_id
   - support truncation before padding
   - retain [MASK] tokens for query when profile says so
4. llama_batch construction must consume the encode plan instead of raw tokens.
5. colbert_debug() must output the encode plan.
6. Do not attempt custom llama.cpp attention masking yet.
7. Add model_info known limitation:
   attention_mask_policy="llama_default_noncausal" unless exact policy is implemented.
8. Tests:
   - query "red planet" with pad_to=32 yields 32 tokens
   - doc "red planet" is not mask padded
   - debug token count equals encode plan count
```

---

## Prompt B6 — Add strict attention-mask capability flag

```text
Repository: mayflower/pgturbohybrid
Branch from: B5 branch

Goal:
Make attention-mask parity explicit instead of silently pretending to match PyLate.

Tasks:
1. Add profile field handling:
   query.attention_mask_policy
   document.attention_mask_policy
2. Supported values initially:
   - llama_default_noncausal
   - pylate_query_expansion_requested
3. If profile requests exact PyLate query expansion attention and pg_colbert_llama cannot express it in llama.cpp:
   - default behavior: warn in colbert_model_info known_limitations
   - if GUC pg_colbert_llama.strict_profile = on, raise ERROR
4. Add GUC:
   pg_colbert_llama.strict_profile bool default off
5. colbert_model_info must report:
   - strict_profile
   - attention_mask_status: ok | approximated | unsupported
6. colbert_debug must show attention_mask if known, and attention_mask_status.
7. Tests:
   - strict_profile=off allows approximated mask
   - strict_profile=on rejects profile requesting unsupported exact mask
8. Add TODO comment pointing to future llama.cpp encoder attention-mask support.
```

---

## Prompt B7 — Support projection module chains

```text
Repository: mayflower/pgturbohybrid
Branch from: B6 branch

Goal:
Replace one hardcoded colbert.proj.weight path with a profile-driven projection chain.

Current engine loads colbert.proj.weight or .colbert_proj sidecar and supports manual F32/F16 dense projection. Keep that as module_chain length 1, but generalize it.

Tasks:
1. Parse projection.modules from profile.
2. Support module:
   - type: dense
   - weight tensor name
   - optional bias tensor name
   - input_dim
   - output_dim
   - activation: identity initially
3. Support module:
   - type: normalize
   - p: 2
4. Support module:
   - type: truncate
   - output_dim
5. Dense module:
   - load weight from GGUF or projection sidecar
   - support F32/F16 initially
   - support optional bias F32/F16
6. If activation is not identity:
   - if strict_profile=on, error
   - otherwise error with precise message; do not silently ignore activation
7. colbert_model_info reports:
   - projection_kind
   - projection_modules
   - projection_status
8. Tests:
   - one dense no bias still works
   - dense with bias fixture works
   - unsupported activation errors cleanly
   - truncate module changes output_dim
```

---

## Prompt B8 — Stop rejecting HF tokenizer JSON too early

```text
Repository: mayflower/pgturbohybrid
Branch from: B7 branch

Goal:
Make tokenizer support profile-aware.

Current code rejects GGUFs with embedded HF tokenizer JSON unless canonical tokenizer.ggml metadata is present. Keep a clear failure mode, but move the decision to profile/model load capability rather than a blanket rejection.

Tasks:
1. Replace PgColbertUnsupportedGgufMetadata() with:
   PgColbertCheckTokenizerCapability(profile, model, path)
2. If llama.cpp can load and tokenize the model, do not reject just because HF tokenizer JSON exists.
3. If llama.cpp cannot tokenize and profile tokenizer.source == hf_json:
   - error with remediation:
     "prepare this GGUF with tokenizer.ggml metadata using colbert-gguf-converter --target-runtime llama_cpp"
4. colbert_model_info reports:
   - tokenizer_source
   - tokenizer_status
   - tokenizer_known_limitations
5. Tests:
   - GGUF metadata fixture with HF tokenizer JSON + canonical tokenizer metadata is accepted
   - HF tokenizer JSON only reports unsupported_tokenizer unless stub profile says otherwise
6. Keep existing safe behavior for truly unsupported tokenizers.
```

The current rejection is visible in the llama engine: if `tokenizer.huggingface.json` exists without canonical `tokenizer.ggml.*` metadata, the engine rejects the model.

---

## Prompt B9 — Upgrade PyLate comparison tool to token-plan parity

```text
Repository: mayflower/pgturbohybrid
Branch from: B8 branch

Goal:
Make compare_pylate.py diagnose where parity fails.

Current tool compares vector rows and ranking. It needs token-plan and retention comparison first.

Tasks:
1. Update extensions/pg_colbert_llama/tools/compare_pylate.py.
2. Add optional input:
   --golden-token-plan path.json
3. Add ability to call:
   SELECT colbert_debug(%s, %s)::text
4. Compare:
   - token_ids_after_padding
   - token pieces when available
   - retain_mask
   - final retained token_ids
   - vector count
5. Output JSON fields:
   - token_plan_parity_passed
   - retain_parity_passed
   - vector_parity_passed
   - ranking_passed
   - first_token_mismatch
   - first_retain_mismatch
6. Change vector comparison:
   - align vectors by retained token id/order when token plan is available
   - otherwise retain existing row-wise comparison
7. Add tests using stub debug JSON.
8. Keep ranking smoke behavior.
```

The current comparison tool already computes vector parity and ranking smoke, so this is an incremental upgrade rather than a replacement.

---

## Prompt B10 — Add profile-aware live TAP tests

```text
Repository: mayflower/pgturbohybrid
Branch from: B9 branch

Goal:
Add live tests that prove profile consumption and pgturbohybrid search integration without requiring CI model downloads.

Tasks:
1. Extend extensions/pg_colbert_llama/test/t/002_live_pgturbohybrid_multivector.pl.
2. Keep existing PG_COLBERT_LLAMA_TEST_MODEL gating.
3. Add optional env:
   PG_COLBERT_LLAMA_TEST_PROFILE
   PG_COLBERT_LLAMA_TOKEN_PLAN_GOLDEN
4. If profile env is present:
   - copy or symlink it into model_dir as <alias>.gguf.colbert_profile.json
   - assert colbert_model_info profile_loaded=true
   - assert profile_source=sidecar or gguf
5. If token-plan golden env is present:
   - call colbert_debug()
   - compare token ids after padding
   - compare retain mask
   - compare count
6. Keep existing tests:
   - model loads
   - projection_status ok
   - vectors finite and normalized
   - query/doc prefix embeddings differ
   - pgturbohybrid dense indexed search works
   - pgturbohybrid hybrid indexed search works
7. Do not make these live tests run by default in normal CI.
```

The existing live TAP test already verifies model load, projection status, finite normalized vectors, and indexed pgturbohybrid dense/hybrid search.

---

# C. Cross-repo prompts

## Prompt C1 — Converter/runtime contract test fixture

```text
Repositories:
- mayflower/colbert-gguf-converter
- mayflower/pgturbohybrid

Goal:
Create a shared fixture contract so converter output can be consumed by pg_colbert_llama.

In colbert-gguf-converter:
1. Add tests/fixtures/profile_sauerkraut_15m_minimal.json.
2. Add tests/fixtures/token_plan_red_planet_query.json.
3. Add tests/fixtures/token_plan_red_planet_doc.json.
4. Add a small README describing fixture meaning.

In pgturbohybrid:
1. Copy or vendor those fixtures under:
   extensions/pg_colbert_llama/test/fixtures/
2. Add a test that loads profile fixture in stub mode.
3. Add a test that colbert_debug() shape can be compared against token plan fixture.

Important:
The fixture should not contain large model weights.
```

---

## Prompt C2 — End-to-end local smoke script

```text
Repositories:
- mayflower/colbert-gguf-converter
- mayflower/pgturbohybrid

Goal:
Add a documented local smoke procedure that converts a model, writes a profile, installs it into PostgreSQL model_dir, and runs pg_colbert_llama live tests.

In colbert-gguf-converter:
1. Add examples/convert_for_pg_colbert_llama.sh.
2. It should run:
   python tools/convert_colbert_hf_to_gguf.py \
     --model-id VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m \
     --target-runtime llama_cpp \
     --outfile ./build/sauerkraut-modern.gguf \
     --outtype f16 \
     --write-profile-sidecar
3. It should print:
   - output GGUF path
   - profile sidecar path
   - recommended PG_COLBERT_LLAMA_TEST_MODEL
   - recommended PG_COLBERT_LLAMA_TEST_PROFILE

In pgturbohybrid:
1. Add extensions/pg_colbert_llama/README section:
   "Using a profile generated by colbert-gguf-converter"
2. Include exact env vars:
   PG_COLBERT_LLAMA_TEST_MODEL
   PG_COLBERT_LLAMA_TEST_PROFILE
3. Do not download models in default CI.
```

---

## Prompt C3 — Compatibility matrix

```text
Repositories:
- mayflower/colbert-gguf-converter
- mayflower/pgturbohybrid

Goal:
Create a compatibility matrix for supported ColBERT model families.

In colbert-gguf-converter:
1. Add docs/COMPATIBILITY.md with rows:
   - BERT WordPiece PyLate ColBERT
   - ModernBERT PyLate ColBERT
   - SentenceTransformers Dense ColBERT
   - HF tokenizer JSON only
   - llama.cpp canonical tokenizer metadata
2. Columns:
   - converter support
   - llama.cpp load support
   - pg_colbert_llama load support
   - token-plan parity support
   - vector parity support
   - known limitations

In pgturbohybrid:
1. Add a link from extensions/pg_colbert_llama/README.md.
2. Add a short local compatibility table specific to runtime.
3. Make sure claims are conservative:
   - "shape/ranking smoke" when strict parity is not proved
   - "strict PyLate parity" only when token-plan and vector parity tests pass
```

---

# D. Follow-up prompts for strict parity

## Prompt D1 — Fix document punctuation parity first

```text
Repository: mayflower/pgturbohybrid
Depends on:
- profile loader
- colbert_debug
- converter skiplist_token_ids

Goal:
Eliminate document count mismatch caused by punctuation retention.

Tasks:
1. Use profile.document.skiplist_token_ids for document filtering.
2. Use profile.document.skiplist_words only as fallback.
3. Add test:
   document text "red planet."
   profile skiplist contains "."
   final retained tokens exclude "."
4. Add PyLate comparison:
   compare_pylate.py reports retain_parity_passed=true for this case.
5. Do not chase vector cosine delta until token count and retain mask match exactly.
```

---

## Prompt D2 — Investigate query cosine delta

```text
Repository: mayflower/pgturbohybrid
Depends on:
- token-plan parity

Goal:
Determine whether query vector delta comes from token plan, attention mask, projection, normalization, or llama.cpp numerics.

Tasks:
1. Add debug mode:
   pg_colbert_llama.debug_dump_stage = off|token|hidden|projected|normalized
   or expose stage data only in test builds if too large.
2. For a single query "red planet", compare against PyLate:
   - token ids
   - attention mask
   - hidden state before projection if PyLate can expose it
   - projection output before normalization
   - final normalized vectors
3. Add compare_pylate.py options:
   --compare-stage hidden|projected|normalized
4. Print first layer/stage where cosine delta exceeds threshold.
5. If the first mismatch is before projection and token plan matches:
   document as likely llama.cpp encoder-mask/numeric difference.
6. If mismatch starts at projection:
   fix projection orientation/bias/activation.
7. If mismatch starts after normalization:
   fix normalization precision/order.
```

---

## Prompt D3 — Upstream or local llama.cpp attention-mask support

```text
Repository: mayflower/pgturbohybrid
Potential external dependency: llama.cpp

Goal:
Prepare the work needed for exact PyLate query expansion attention mask support.

Tasks:
1. Add docs/llama-encoder-attention-mask-gap.md.
2. Document current llama.cpp batch fields used by pg_colbert_llama.
3. Document PyLate/HF attention mask required for query expansion.
4. Add a minimal C++ reproduction outside PostgreSQL:
   extensions/pg_colbert_llama/tools/llama_colbert_attention_mask_probe.cpp
5. The probe should:
   - load the same GGUF
   - run current default llama encode
   - optionally run a patched/custom attention mask path if available
   - dump final token embeddings
6. If llama.cpp exposes a public API for encoder attention mask by the current version, use it.
7. If not, prepare an upstream issue/PR summary.
8. In pg_colbert_llama, keep strict_profile=on rejecting exact mask profiles until support exists.
```

---

# E. Final documentation prompt

## Prompt E1 — Update claims and release notes

```text
Repositories:
- mayflower/colbert-gguf-converter
- mayflower/pgturbohybrid

Goal:
Update docs so users know exactly what is supported.

In colbert-gguf-converter:
1. README should describe:
   - pg_colbert_v1 GGUF
   - llama_cpp target runtime
   - profile sidecar
   - parity report artifacts
2. docs/COLBERT_PROFILE_SPEC.md must be linked.
3. Publishing docs must explain profile upload.

In pgturbohybrid:
1. extensions/pg_colbert_llama/README.md should state:
   - profile-aware models are preferred
   - shape/ranking smoke is not strict vector parity
   - strict parity requires token-plan and vector parity reports
2. colbert_model_info docs should explain:
   - profile_source
   - compatibility_level
   - attention_mask_status
   - tokenizer_status
   - projection_status
3. Add examples:
   - profile-based model install
   - dense pgturbohybrid multivector search
   - hybrid pgturbohybrid multivector + BM25 search
4. Add release note:
   "Do not claim universal ColBERT support. Claim profile-backed BERT/ModernBERT ColBERT support at the compatibility level reported by colbert_model_info()."
```

---

# Suggested PR order

I would merge in this order:

```text
1. converter: profile spec + dataclasses
2. converter: profile extraction + sidecar + GGUF metadata
3. pgturbohybrid: profile parser + colbert_model_info reporting
4. pgturbohybrid: colbert_debug token plan
5. pgturbohybrid: profile-driven retention / skiplist
6. converter: token-plan golden generator
7. pgturbohybrid: compare_pylate token-plan parity
8. pgturbohybrid: projection module chain
9. converter: llama_cpp target runtime export
10. cross-repo live profile integration
11. attention-mask investigation / upstream llama.cpp work
```

The first concrete parity win should be **document skiplist parity**, because the current code only drops PAD for documents. After that, token-plan parity will tell you whether the remaining query cosine delta is an attention-mask gap, projection issue, or llama.cpp numerical/model-path issue.
