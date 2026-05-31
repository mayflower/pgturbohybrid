\set ON_ERROR_STOP on

-- Native graph build phase benchmark.
--
-- Override from psql with, for example:
--   psql -v ROWS=100000 -v DIMS=100 -v EF_CONSTRUCTION=192 -f benchmarks/native_build_phase_bench.sql
\if :{?ROWS}
\else
\set ROWS 10000
\endif
\if :{?DIMS}
\else
\set DIMS 100
\endif
\if :{?GRAPH_M}
\else
\set GRAPH_M 16
\endif
\if :{?EF_CONSTRUCTION}
\else
\set EF_CONSTRUCTION 128
\endif
\if :{?QUANTIZATION_BITS}
\else
\set QUANTIZATION_BITS 4
\endif
\if :{?EXACT_STORAGE}
\else
\set EXACT_STORAGE off
\endif

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

DROP TABLE IF EXISTS th_native_build_phase_docs;

CREATE TABLE th_native_build_phase_docs (
	id bigint PRIMARY KEY,
	embedding vector(:DIMS)
);

INSERT INTO th_native_build_phase_docs (id, embedding)
SELECT row_id,
	('[' || string_agg(
		(sin((row_id * dim_id)::double precision * 0.017) +
		 cos((row_id + dim_id)::double precision * 0.031))::text,
		',' ORDER BY dim_id
	) || ']')::vector(:DIMS)
FROM generate_series(1, :ROWS) AS row_id
CROSS JOIN generate_series(1, :DIMS) AS dim_id
GROUP BY row_id
ORDER BY row_id;

SET client_min_messages = warning;

ANALYZE th_native_build_phase_docs;

CREATE TEMP TABLE th_native_build_phase_results (
	worker_setting text PRIMARY KEY,
	stats jsonb NOT NULL,
	index_bytes bigint NOT NULL
);

\echo 'native_build_workers=0'
DROP INDEX IF EXISTS th_native_build_phase_docs_idx;
SET turbohybrid.native_build_workers = '0';
\timing on
CREATE INDEX th_native_build_phase_docs_idx
ON th_native_build_phase_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (
	graph_m = :GRAPH_M,
	graph_ef_construction = :EF_CONSTRUCTION,
	quantization_bits = :QUANTIZATION_BITS,
	exact_storage = :EXACT_STORAGE
);
\timing off
INSERT INTO th_native_build_phase_results
SELECT '0', turbohybrid_last_build_stats(),
	pg_relation_size('th_native_build_phase_docs_idx'::regclass);

\echo 'native_build_workers=2'
DROP INDEX IF EXISTS th_native_build_phase_docs_idx;
SET turbohybrid.native_build_workers = '2';
\timing on
CREATE INDEX th_native_build_phase_docs_idx
ON th_native_build_phase_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (
	graph_m = :GRAPH_M,
	graph_ef_construction = :EF_CONSTRUCTION,
	quantization_bits = :QUANTIZATION_BITS,
	exact_storage = :EXACT_STORAGE
);
\timing off
INSERT INTO th_native_build_phase_results
SELECT '2', turbohybrid_last_build_stats(),
	pg_relation_size('th_native_build_phase_docs_idx'::regclass);

\echo 'native_build_workers=4'
DROP INDEX IF EXISTS th_native_build_phase_docs_idx;
SET turbohybrid.native_build_workers = '4';
\timing on
CREATE INDEX th_native_build_phase_docs_idx
ON th_native_build_phase_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (
	graph_m = :GRAPH_M,
	graph_ef_construction = :EF_CONSTRUCTION,
	quantization_bits = :QUANTIZATION_BITS,
	exact_storage = :EXACT_STORAGE
);
\timing off
INSERT INTO th_native_build_phase_results
SELECT '4', turbohybrid_last_build_stats(),
	pg_relation_size('th_native_build_phase_docs_idx'::regclass);

\echo 'native_build_workers=8'
DROP INDEX IF EXISTS th_native_build_phase_docs_idx;
SET turbohybrid.native_build_workers = '8';
\timing on
CREATE INDEX th_native_build_phase_docs_idx
ON th_native_build_phase_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (
	graph_m = :GRAPH_M,
	graph_ef_construction = :EF_CONSTRUCTION,
	quantization_bits = :QUANTIZATION_BITS,
	exact_storage = :EXACT_STORAGE
);
\timing off
INSERT INTO th_native_build_phase_results
SELECT '8', turbohybrid_last_build_stats(),
	pg_relation_size('th_native_build_phase_docs_idx'::regclass);

SELECT
	worker_setting,
	jsonb_pretty(stats) AS native_build_stats
FROM th_native_build_phase_results
ORDER BY worker_setting::int;

SELECT
	worker_setting,
	pg_size_pretty(index_bytes) AS index_size,
	index_bytes,
	(stats ->> 'native_build_workers_requested')::int AS workers_requested,
	(stats ->> 'native_build_workers_launched')::int AS workers_launched,
	(stats ->> 'parallel_fit_enabled')::boolean AS parallel_fit,
	(stats ->> 'parallel_scan_enabled')::boolean AS parallel_scan,
	(stats ->> 'parallel_encode_enabled')::boolean AS parallel_encode,
	(stats ->> 'worker_merge_us')::bigint AS worker_merge_us,
	(stats -> 'worker_scan_us') AS worker_scan_us
FROM th_native_build_phase_results
ORDER BY worker_setting::int;

WITH phases(phase, ord) AS (
	VALUES
		('fit_correction_scan_us', 1),
		('scan_us', 2),
		('fit_correction_us', 3),
		('encode_us', 4),
		('build_edges_us', 5),
		('free_exact_vectors_us', 6),
		('reorder_nodes_us', 7),
		('connect_backbone_us', 8),
		('entry_sidecar_us', 9),
		('write_pages_us', 10),
		('wal_us', 11),
		('total_us', 12)
)
SELECT
	r.worker_setting,
	phase,
	(r.stats ->> phase)::bigint AS us,
	round(((r.stats ->> phase)::numeric / 1000.0), 3) AS ms
FROM th_native_build_phase_results r
JOIN phases ON true
ORDER BY r.worker_setting::int, ord;
