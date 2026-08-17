/* pgturbohybrid 0.1.0 -> 0.1.1
 *
 * Ships the per-tenant BM25 statistics reporting function.  The per-tenant
 * statistics themselves are computed by the shared library (bm25Version 2
 * metadata, written since the library update); indexes built before it keep
 * global statistics until REINDEX.  See turbohybrid.bm25_tenant_stats and
 * the bm25_tenant_payload_slot reloption.
 */

CREATE OR REPLACE FUNCTION turbohybrid_bm25_tenant_stats(index pg_catalog.regclass)
	RETURNS TABLE(tenant pg_catalog.int4, doc_count pg_catalog.int8,
				  total_doc_len pg_catalog.int8, avg_doc_len pg_catalog.float8)
	AS 'MODULE_PATHNAME', 'pgturbohybrid_bm25_tenant_stats_fn'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

-- CREATE OR REPLACE because the 0.1.0 base script was amended in-place with
-- this same function: installs that run the base and then this update path
-- (CREATE EXTENSION at default version 0.1.1) would otherwise collide with
-- themselves. Pre-amend 0.1.0 installs (production) simply create it here.
