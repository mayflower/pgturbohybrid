SET enable_seqscan = off;

CREATE TABLE tq (id int, bucket int, val vector(3));
INSERT INTO tq VALUES
	(1, 10, '[0,0,0]'),
	(2, 20, '[1,2,3]'),
	(3, 30, '[1,1,1]'),
	(4, 40, '[3,2,1]');

CREATE INDEX tq_val_l2_idx ON tq
	USING turboquant (val vector_l2_ops)
	INCLUDE (bucket)
	WITH (routing = graph, tq_exact_storage = on);

DO $$
DECLARE
	ids int[];
	top_bucket int;
BEGIN
	SELECT array_agg(id), min(bucket) FILTER (WHERE ord = 1)
	INTO ids, top_bucket
	FROM (
		SELECT id, bucket, row_number() OVER () AS ord
		FROM (
			SELECT id, bucket
			FROM tq
			ORDER BY val <-> '[1,2,3]'
			LIMIT 3
		) s
	) ranked;

	IF ids <> ARRAY[2,3,4] OR top_bucket <> 20 THEN
		RAISE EXCEPTION 'unexpected l2 results %, bucket %', ids, top_bucket;
	END IF;
END
$$;

INSERT INTO tq VALUES (5, 50, '[1,2,4]');

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tq
	ORDER BY val <-> '[1,2,4]'
	LIMIT 1;

	IF top_id <> 5 THEN
		RAISE EXCEPTION 'unexpected inserted top id: %', top_id;
	END IF;
END
$$;

DELETE FROM tq WHERE id = 5;
VACUUM tq;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tq
		ORDER BY val <-> '[1,2,4]'
		LIMIT 3
	) s;

	IF 5 = ANY(ids) THEN
		RAISE EXCEPTION 'deleted tuple returned by turboquant scan: %', ids;
	END IF;
END
$$;

DROP INDEX tq_val_l2_idx;

CREATE INDEX tq_val_cos_idx ON tq
	USING turboquant (val vector_cosine_ops)
	WITH (routing = graph, tq_bits = 4, tq_exact_storage = on);

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tq
		ORDER BY val <=> '[1,2,3]'
		LIMIT 3
	) s;

	IF ids <> ARRAY[2,3,4] THEN
		RAISE EXCEPTION 'unexpected cosine results: %', ids;
	END IF;
END
$$;

DROP INDEX tq_val_cos_idx;

CREATE INDEX tq_val_ip_idx ON tq
	USING turboquant (val vector_ip_ops)
	WITH (routing = graph, tq_exact_storage = on);

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tq
		ORDER BY val <#> '[1,2,3]'
		LIMIT 3
	) s;

	IF ids <> ARRAY[2,4,3] THEN
		RAISE EXCEPTION 'unexpected inner product results: %', ids;
	END IF;
END
$$;

DROP INDEX tq_val_ip_idx;

CREATE INDEX tq_val_flat_idx ON tq
	USING turboquant (val vector_l2_ops)
	WITH (routing = flat);

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tq
		ORDER BY val <-> '[1,2,3]'
		LIMIT 3
	) s;

	IF ids <> ARRAY[2,3,4] THEN
		RAISE EXCEPTION 'unexpected flat l2 results: %', ids;
	END IF;
END
$$;

DROP INDEX tq_val_flat_idx;

CREATE INDEX tq_auto_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = auto);
CREATE INDEX tq_graph_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = graph, graph_m = 16, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4);
CREATE INDEX tq_flat_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = flat);

CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (routing = bad);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_m = 1);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_ef_construction = 3);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_ef_search = 0);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_oversampling = 0);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (tq_bits = 3);

DROP TABLE tq;
