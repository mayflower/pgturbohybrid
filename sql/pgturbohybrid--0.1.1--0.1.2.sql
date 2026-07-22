-- pgturbohybrid 0.1.1 --> 0.1.2
-- Public mechanical index validation diagnostics.

CREATE FUNCTION turbohybrid_validate_index(index regclass, deep boolean DEFAULT false)
RETURNS pg_catalog.jsonb
AS 'MODULE_PATHNAME', 'pgturbohybrid_validate_index'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE;

COMMENT ON FUNCTION turbohybrid_validate_index(regclass, boolean) IS
'Mechanically validate pgturbohybrid page chains, tuples, branches, and optional deep graph reachability without modifying the index. [diagnostic]';
