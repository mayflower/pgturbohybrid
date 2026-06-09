#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SQL_FILE="$ROOT_DIR/scripts/valorbrain/scalar-fallback-guard.sql"

PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5433}"
PGDATABASE="${PGDATABASE:-valorbrain}"
PGUSER="${PGUSER:-postgres}"

PSQL="${PSQL:-psql}"
PSQL_OPTS=(
  -h "$PGHOST"
  -p "$PGPORT"
  -U "$PGUSER"
  -d "$PGDATABASE"
  -v ON_ERROR_STOP=1
)

echo "scalar fallback guard: ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
exec "$PSQL" "${PSQL_OPTS[@]}" -f "$SQL_FILE"