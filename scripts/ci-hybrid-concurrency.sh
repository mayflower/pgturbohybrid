#!/usr/bin/env bash
# CI smoke: parallel BM25 delta inserts + hybrid index scans on a throwaway DB.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${PGDATABASE:-pgturbohybrid_concurrency_smoke}"
PSQL=(psql -v ON_ERROR_STOP=1)
PGPORT="${PGPORT:-}"

if [[ -n "$PGPORT" ]]; then
  PSQL+=(-p "$PGPORT")
fi

createdb --if-not-exists "$DB" 2>/dev/null || createdb "$DB"
trap 'dropdb --if-exists "$DB" 2>/dev/null || true' EXIT

"${PSQL[@]}" -d "$DB" <<'SQL'
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;

DROP TABLE IF EXISTS cc_hybrid_docs;
CREATE TABLE cc_hybrid_docs (
  id int PRIMARY KEY,
  embedding vector(3) NOT NULL,
  body_tsv tsvector NOT NULL
);

INSERT INTO cc_hybrid_docs (id, embedding, body_tsv)
SELECT g,
       format('[%s,%s,%s]', 1.0 - (g % 17)::float8 / 100,
              (g % 31)::float8 / 100, (g % 43)::float8 / 100)::vector(3),
       to_tsvector('simple', 'alpha beta hybrid doc ' || g::text)
FROM generate_series(1, 32) g;

CREATE INDEX cc_hybrid_docs_idx ON cc_hybrid_docs
USING turbohybrid (
  embedding vector_cosine_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);
ANALYZE cc_hybrid_docs;
SQL

insert_worker() {
  local start="$1"
  local end="$2"
  for id in $(seq "$start" "$end"); do
    "${PSQL[@]}" -d "$DB" -q -c "
      INSERT INTO cc_hybrid_docs (id, embedding, body_tsv)
      VALUES (
        ${id},
        format('[%s,%s,%s]', 0.5, 0.3, 0.2)::vector(3),
        to_tsvector('simple', 'fresh delta insert term${id} hybrid')
      );
    "
  done
}

query_worker() {
  local n="$1"
  for _ in $(seq 1 12); do
    "${PSQL[@]}" -d "$DB" -q -c "
      SET enable_seqscan = off;
      SELECT count(*) FROM (
        SELECT id FROM cc_hybrid_docs
        ORDER BY embedding <~> turbohybrid_query(
          vector_query => '[1,0,0]'::vector,
          text_query => websearch_to_tsquery('simple', 'hybrid delta')
        )
        LIMIT 5
      ) q;
    " >/dev/null
  done
  echo "query worker ${n} ok"
}

pids=()
insert_worker 100 111 &
pids+=("$!")
insert_worker 200 211 &
pids+=("$!")
for q in 1 2 3 4; do
  query_worker "$q" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"${PSQL[@]}" -d "$DB" -c "
SET enable_seqscan = off;
SELECT id FROM cc_hybrid_docs
ORDER BY embedding <~> turbohybrid_query(
  text_query => websearch_to_tsquery('simple', 'term105'),
  dense_k => 0,
  bm25_k => 10,
  final_k => 3
)
LIMIT 1;
"

echo "ci-hybrid-concurrency: pass"