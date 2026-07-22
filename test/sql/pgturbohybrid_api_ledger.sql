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
CREATE EXTENSION IF NOT EXISTS pgturbohybrid_experimental;
RESET client_min_messages;

WITH objects AS (
  SELECT 'type'::text AS kind, t.typname AS name,
         substring(obj_description(t.oid, 'pg_type') from '\[([^]]+)\]') AS maturity,
         t.oid AS object_oid, 'pg_type'::regclass AS class_oid
    FROM pg_type t
    WHERE t.typname IN ('turbohybrid_query', 'turbohybrid_sparse_vector',
                        'turbohybrid_multivector')
  UNION ALL
  SELECT 'opclass', oc.opcname,
         substring(obj_description(oc.oid, 'pg_opclass') from '\[([^]]+)\]'),
         oc.oid, 'pg_opclass'::regclass
    FROM pg_opclass oc JOIN pg_am a ON a.oid = oc.opcmethod
    WHERE a.amname = 'turbohybrid'
  UNION ALL
  SELECT 'operator', o.oprname || '(' || format_type(o.oprleft, NULL) || ')',
         substring(obj_description(o.oid, 'pg_operator') from '\[([^]]+)\]'),
         o.oid, 'pg_operator'::regclass
    FROM pg_operator o
    WHERE o.oprname IN ('<~>', '<~->', '<~#>', '<~*>')
      AND o.oprright = 'turbohybrid_query'::regtype
  UNION ALL
  SELECT 'function', p.proname,
         substring(obj_description(p.oid, 'pg_proc') from '\[([^]]+)\]'),
         p.oid, 'pg_proc'::regclass
    FROM pg_proc p
    WHERE p.proname LIKE 'turbohybrid\_%'
)
SELECT DISTINCT e.extname || '|' || kind || '|' || name || '|' ||
       COALESCE(maturity, '(none)') AS api_object
FROM objects o
JOIN pg_depend d ON d.classid = o.class_oid
                AND d.objid = o.object_oid
                AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid
WHERE e.extname IN ('pgturbohybrid', 'pgturbohybrid_experimental')
ORDER BY 1;
