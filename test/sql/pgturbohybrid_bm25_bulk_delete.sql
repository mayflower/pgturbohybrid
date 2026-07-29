-- Exclusão em volume num índice híbrido: a estatística do BM25 tem que acompanhar,
-- e a varredura híbrida tem que continuar de pé.
--
-- Dois defeitos que este teste fixa, ambos reproduzidos em 2026-07-28:
--
--   1. `PgturbohybridGraphMaybeCompactPageChains` renumerava os identificadores de
--      nó, e as estruturas do BM25 (postagens, estatística de documento, teto por
--      bloco, camadas de impacto) são todas indexadas por esse identificador e não
--      eram reescritas. Depois de `DELETE` de metade das linhas mais `VACUUM`, o
--      grafo dizia 1.000 nós, o metadado do BM25 seguia dizendo 2.000 documentos, e
--      a verificação de invariante derrubava **toda** varredura híbrida com
--      "BM25 metadata document count is invalid". A busca vetorial seguia
--      funcionando; a híbrida morria até o REINDEX.
--
--   2. `PgturbohybridBm25MaybeCompact` só disparava por acúmulo de delta, então
--      carga que só apaga nunca refazia a base: `N` inflado no idf e `avgdl`
--      inflado na normalização por tamanho. Em produção, depois de um expurgo, o
--      índice contava 12.480 documentos com a tabela em 9.489 (avgdl 1.359 contra
--      1.040) e a ablação de outro tenant caiu dois cenários de oito — o índice
--      lexical é um só para todos.

SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE TABLE bulk_del (
    id int PRIMARY KEY,
    tsv_bi tsvector,
    embedding vector(8)
);

INSERT INTO bulk_del
SELECT i,
       to_tsvector('simple', repeat('palavra' || (i % 50) || ' ', 20 + (i % 30))),
       (SELECT ('[' || string_agg((i % 7 + s)::text, ',') || ']')::vector
        FROM generate_series(1, 8) s)
FROM generate_series(1, 2000) i;

CREATE INDEX bulk_del_ix ON bulk_del
    USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                       tsv_bi bm25_tsvector_turbohybrid_ops);

-- Recém-construído: o metadado do BM25 conta o que a tabela tem.
SELECT (turbohybrid_index_stats('bulk_del_ix'::regclass) ->> 'bm25_document_count')::int
       = (SELECT count(*)::int FROM bulk_del) AS bm25_igual_a_tabela_no_build;

DELETE FROM bulk_del WHERE id > 1000;
VACUUM bulk_del;

-- Depois do expurgo: a contagem e o comprimento médio refletem o que sobrou.
SELECT (turbohybrid_index_stats('bulk_del_ix'::regclass) ->> 'bm25_document_count')::int
       = (SELECT count(*)::int FROM bulk_del) AS bm25_igual_a_tabela_apos_expurgo;

-- A varredura híbrida continua de pé — era aqui que estourava — e só devolve
-- linha viva.
SET enable_seqscan = off;
SELECT count(*) AS devolvidos, bool_and(id <= 1000) AS todos_vivos
FROM (
    SELECT id FROM bulk_del
    ORDER BY embedding <~> turbohybrid_query(
        vector_query => (SELECT embedding FROM bulk_del WHERE id = 5)::vector,
        text_query => to_tsquery('simple', 'palavra5'))
    LIMIT 20
) t;

-- Segundo ciclo, com escrita depois do expurgo: a base se refaz de novo e o
-- identificador de nó dos vivos não se move.
DELETE FROM bulk_del WHERE id > 100;
VACUUM bulk_del;
INSERT INTO bulk_del
SELECT 3000 + i,
       to_tsvector('simple', repeat('palavra' || (i % 50) || ' ', 25)),
       (SELECT ('[' || string_agg((i % 7 + s)::text, ',') || ']')::vector
        FROM generate_series(1, 8) s)
FROM generate_series(1, 50) i;
VACUUM bulk_del;

SELECT (turbohybrid_index_stats('bulk_del_ix'::regclass) ->> 'bm25_document_count')::int
       = (SELECT count(*)::int FROM bulk_del) AS bm25_igual_a_tabela_no_segundo_ciclo;

SELECT count(*) AS devolvidos, bool_and(id <= 100 OR id > 3000) AS todos_vivos
FROM (
    SELECT id FROM bulk_del
    ORDER BY embedding <~> turbohybrid_query(
        vector_query => (SELECT embedding FROM bulk_del WHERE id = 5)::vector,
        text_query => to_tsquery('simple', 'palavra5'))
    LIMIT 20
) t;

RESET enable_seqscan;
DROP TABLE bulk_del;
