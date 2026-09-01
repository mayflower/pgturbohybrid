SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE SCHEMA hostile;
CREATE FUNCTION hostile.to_tsvector(regconfig, text) RETURNS tsvector
LANGUAGE SQL IMMUTABLE AS $$ SELECT 'fake:1'::tsvector $$;

SET search_path = hostile, public, pg_catalog;

CREATE TABLE secure_docs (
	id integer,
	embedding public.vector(2),
	body pg_catalog.tsvector
);
INSERT INTO secure_docs VALUES
	(1, '[1,0]', pg_catalog.to_tsvector('simple', 'real content'));
CREATE INDEX secure_docs_idx ON secure_docs USING turbohybrid (
	embedding public.vector_cosine_turbohybrid_ops,
	body public.bm25_tsvector_turbohybrid_ops
);

SELECT id FROM secure_docs
ORDER BY embedding OPERATOR(public.<~>) public.turbohybrid_dense_query('[1,0]'::public.vector)
LIMIT 1;

RESET search_path;
DROP TABLE hostile.secure_docs;
DROP SCHEMA hostile CASCADE;
