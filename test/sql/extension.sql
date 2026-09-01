DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgturbohybrid') THEN
		EXECUTE 'DROP EXTENSION pgturbohybrid CASCADE';
	END IF;
	IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
		EXECUTE 'DROP EXTENSION vector CASCADE';
	END IF;
	IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'vecschema') THEN
		EXECUTE 'DROP SCHEMA vecschema CASCADE';
	END IF;
	IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'thschema') THEN
		EXECUTE 'DROP SCHEMA thschema CASCADE';
	END IF;
END
$$;

CREATE EXTENSION pgturbohybrid;

CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;

SELECT extname
FROM pg_extension
WHERE extname IN ('vector', 'pgturbohybrid')
ORDER BY extname;

SELECT extversion
FROM pg_extension
WHERE extname = 'pgturbohybrid';

DROP EXTENSION pgturbohybrid;

SELECT extname
FROM pg_extension
WHERE extname IN ('vector', 'pgturbohybrid')
ORDER BY extname;

DROP EXTENSION vector;

CREATE SCHEMA vecschema;
CREATE EXTENSION vector WITH SCHEMA vecschema;

CREATE SCHEMA thschema;
CREATE EXTENSION pgturbohybrid WITH SCHEMA thschema;

CREATE TABLE schema_reloc_docs (
	id int,
	embedding vecschema.vector(3),
	body_tsv pg_catalog.tsvector
);

INSERT INTO schema_reloc_docs VALUES
	(1, '[1,0,0]', pg_catalog.to_tsvector('english', 'postgres vector search')),
	(2, '[0,1,0]', pg_catalog.to_tsvector('english', 'lexical search'));

CREATE INDEX schema_reloc_docs_idx ON schema_reloc_docs
USING turbohybrid (
	embedding thschema.vector_l2_turbohybrid_ops,
	body_tsv thschema.bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

SELECT id
FROM schema_reloc_docs
ORDER BY embedding OPERATOR(thschema.<~->) thschema.turbohybrid_query(
	vector_query => '[1,0,0]'::vecschema.vector,
	dense_k => 2,
	final_k => 1
)
LIMIT 1;

SET client_min_messages = warning;
DROP EXTENSION pgturbohybrid CASCADE;
RESET client_min_messages;

SELECT extname
FROM pg_extension
WHERE extname IN ('vector', 'pgturbohybrid')
ORDER BY extname;

SELECT '[1,2,3]'::vecschema.vector;

DROP TABLE schema_reloc_docs;
DROP EXTENSION vector;
DROP SCHEMA thschema;
DROP SCHEMA vecschema;
