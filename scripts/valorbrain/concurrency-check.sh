#!/usr/bin/env bash
# Lightweight ValorBrain-style concurrency smoke: parallel hybrid queries while
# BM25 delta inserts run. Fails on SQL errors or scalar-fallback guard regression.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5433}"
PGDATABASE="${PGDATABASE:-valorbrain}"
PGUSER="${PGUSER:-postgres}"
WORKERS="${WORKERS:-4}"
ROUNDS="${ROUNDS:-8}"

PSQL=(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1)

echo "concurrency-check: ${WORKERS} workers x ${ROUNDS} rounds on ${PGDATABASE}"

hybrid_worker() {
  local wid="$1"
  local round
  for round in $(seq 1 "$ROUNDS"); do
    "${PSQL[@]}" -q -c "
      SELECT count(*) FROM (
        SELECT d.id
        FROM documents_fts fts
        JOIN documents d ON d.id = fts.doc_id
        WHERE fts.embedding IS NOT NULL AND d.active = 1
        ORDER BY fts.embedding <~> turbohybrid_query(
          vector_query => (SELECT embedding FROM documents_fts WHERE embedding IS NOT NULL LIMIT 1),
          text_query => websearch_to_tsquery('english', 'postgres hybrid search')
        )
        LIMIT 5
      ) q;
    " >/dev/null
  done
  echo "worker ${wid} ok"
}

pids=()
for w in $(seq 1 "$WORKERS"); do
  hybrid_worker "$w" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "concurrency-check: hybrid workers failed" >&2
  exit 1
fi

echo "== scalar fallback guard =="
bash "$ROOT_DIR/scripts/valorbrain/scalar-fallback-guard.sh"

echo "concurrency-check: pass"