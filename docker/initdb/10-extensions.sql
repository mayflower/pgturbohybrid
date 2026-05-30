-- Runs once, on first cluster initialization (docker-entrypoint-initdb.d).
-- pgturbohybrid requires vector, so create it first.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
