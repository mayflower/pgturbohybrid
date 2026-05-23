#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONFIG="${PG_CONFIG:-pg_config}"
DBNAME="${DBNAME:-pgturbohybrid_dev}"

"$ROOT_DIR/scripts/install-pgvector.sh"

make -C "$ROOT_DIR" PG_CONFIG="$PG_CONFIG" clean
make -C "$ROOT_DIR" PG_CONFIG="$PG_CONFIG"
make -C "$ROOT_DIR" PG_CONFIG="$PG_CONFIG" install

dropdb --if-exists "$DBNAME" >/dev/null 2>&1 || true
createdb "$DBNAME"
psql -d "$DBNAME" -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
SELECT '[1,0,0]'::vector <~-> turbohybrid_query(vector_query => '[1,0,0]'::vector);
SQL

printf 'ready: database %s has vector and pgturbohybrid installed\n' "$DBNAME"
