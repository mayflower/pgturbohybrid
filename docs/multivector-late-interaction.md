# Multivector late interaction

`turbohybrid_multivector` stores a non-empty sequence of float32 token vectors
with a shared dimension.

```sql
SELECT turbohybrid_multivector(ARRAY[
  '[1,0,0]'::vector,
  '[0,1,0]'::vector
]);
```

For query tokens `q_i` and document tokens `d_j`, exact MaxSim is:

```text
maxsim(q, d) = sum_i max_j dot(q_i, d_j)
```

MaxSim is a similarity, so larger is better. PostgreSQL index ordering expects
smaller distances first; `turbohybrid_multivector_maxsim_distance` and the
`<~>` operator therefore return `-maxsim`.

## Index execution

```sql
CREATE INDEX passages_tokens_idx ON passages USING turbohybrid (
  tokens multivector_maxsim_ip_turbohybrid_ops
);

SELECT id
FROM passages
ORDER BY tokens <~> turbohybrid_multivector_query($1, 20, 100)
LIMIT 20;
```

The graph node is a token/subvector, not a SQL result. Approximate token scores
generate a bounded candidate set. Candidates are aggregated by document heap
tuple, then final ordering is computed from the original float32 token vectors.
Token ordinals preserve the position of each subvector inside its document.

The direct functions are useful for exact checks:

```sql
SELECT turbohybrid_multivector_dims(tokens),
       turbohybrid_multivector_count(tokens),
       turbohybrid_multivector_subvector(tokens, 0),
       turbohybrid_multivector_maxsim(tokens, $1),
       turbohybrid_multivector_maxsim_distance(tokens, $1)
FROM passages
LIMIT 1;
```

Documents and queries must obey the configured dimension and token limits.
Mixed dimensions, empty values, non-finite values, and incompatible operands
are rejected.
