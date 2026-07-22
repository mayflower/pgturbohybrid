# Learned Projection Proxy for Multivector Admission

## Goal

`learned_projection_v1` is the first guarded slice of an FDE-style document
proxy. It reduces a document or query multivector to one fixed-dimensional
proxy vector for graph admission, then keeps the existing exact float32 MaxSim
rerank for final SQL ordering.

This is not a training path. Projection weights are installed by an
administrator outside PostgreSQL and loaded by the extension only when the
encoder is explicitly selected.

## Scope of the First Slice

The first implementation keeps the projected proxy dimension equal to the
multivector token dimension. That avoids changing the graph/index storage
contract or adding a persisted format version solely for projection output.

The encoder is opt-in:

```sql
SET turbohybrid.multivector_learned_projection_path =
  '/path/to/projection.weights';

CREATE INDEX ... WITH (
  multivector_graph = document_nodes,
  multivector_proxy_encoder = learned_projection_v1
);
```

If no projection file is configured, `learned_projection_v1` fails explicitly.

## Projection File Format

The current text format is intentionally simple and unstable:

```text
pgturbohybrid_learned_projection_v1 <model> <input_dim> <output_dim> <checksum>
<w_0_0> <w_0_1> ... <w_0_input_dim-1>
...
<w_output_dim-1_0> ... <w_output_dim-1_input_dim-1>
```

The implementation requires:

- the magic string to match exactly;
- `input_dim == output_dim == multivector_dim`;
- `turbohybrid.multivector_learned_projection_model`, when set, to match the
  file model;
- `turbohybrid.multivector_learned_projection_checksum`, when set, to match the
  file checksum.

The checksum is a caller-provided model/version string in this slice. It is not
a cryptographic validation of the file contents.

## Encoding

Document and query encoding use the same conservative flow:

1. build a normalized mean vector from the multivector tokens;
2. multiply by the configured projection matrix;
3. normalize the projected proxy vector;
4. use the proxy graph for candidate admission;
5. exact-rerank retained candidates with the original multivectors.

This keeps final ranking exact and makes projection quality a candidate
admission concern only.

## Future Versioned Format

A production projection format needs explicit versioning before the projection
dimension can differ from the token dimension or before projection metadata is
embedded in index storage. A future format should include:

- magic/version;
- model profile name;
- input and output dimensions;
- projection orientation;
- checksum over the complete payload;
- query/document asymmetric projection flags;
- clear `REINDEX` guidance for metadata mismatch.

## Stats and Benchmark Requirements

The existing stats already expose the active proxy encoder through:

- `turbohybrid_index_stats(...)->>'multivector_proxy_encoder'`;
- `turbohybrid_last_scan_stats()->>'proxy_encoder_kind'`.

The first implementation also exposes dedicated learned-projection stats:

- `learned_projection_loaded`;
- `learned_projection_dim`;
- `learned_projection_weight_bytes`;
- `learned_projection_model`;
- `learned_projection_checksum`;
- `learned_projection_query_encode_us`;
- `learned_projection_doc_encode_build_us`.

Benchmark evidence must report admission quality, build cost, query latency, and
exact-rerank cost against `normalized_mean`, `max_pool`, and `centroid_mean`.

## Safety Invariants

- No final approximate SQL ordering.
- No default selection.
- No silent on-disk format change.
- Missing or mismatched weights fail before a misleading index or scan can run.
- Exact MaxSim over original multivectors remains authoritative for retained
  candidates.
