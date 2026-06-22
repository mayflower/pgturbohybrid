-- pgturbohybrid_api_ledger: a snapshot of the public SQL API surface (each
-- object plus the maturity label parsed from its COMMENT ON). A diff here means
-- the public API changed -- intentional or not. If intentional, regenerate this
-- expected file AND update docs/api-ledger.json to match
-- (scripts/check-api-ledger.py enforces that the ledger and this snapshot agree,
-- so accidental additions/removals/relabels are caught). See
-- docs/feature-matrix.md and docs/beta-scope.md for what the labels mean.
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

SELECT DISTINCT kind || '|' || name || '|' || COALESCE(maturity, '(none)') AS api_object
FROM (
  SELECT 'type'::text AS kind, t.typname AS name,
         substring(obj_description(t.oid, 'pg_type') from '\[([^]]+)\]') AS maturity,
         1 AS ord
    FROM pg_type t
    WHERE t.typname IN ('turbohybrid_query', 'turbohybrid_sparse_vector',
                        'turbohybrid_multivector', 'multivector')
  UNION ALL
  SELECT 'opclass', oc.opcname,
         substring(obj_description(oc.oid, 'pg_opclass') from '\[([^]]+)\]'), 2
    FROM pg_opclass oc JOIN pg_am a ON a.oid = oc.opcmethod
    WHERE a.amname = 'turbohybrid'
  UNION ALL
  SELECT 'operator', o.oprname || '(' || format_type(o.oprleft, NULL) || ')',
         substring(obj_description(o.oid, 'pg_operator') from '\[([^]]+)\]'), 3
    FROM pg_operator o
    WHERE o.oprname IN ('<~>', '<~->', '<~#>', '<~*>')
      AND o.oprright = 'turbohybrid_query'::regtype
  UNION ALL
  SELECT 'function', p.proname,
         substring(obj_description(p.oid, 'pg_proc') from '\[([^]]+)\]'), 4
    FROM pg_proc p
    WHERE p.proname LIKE 'turbohybrid\_%'
) s
ORDER BY 1;
