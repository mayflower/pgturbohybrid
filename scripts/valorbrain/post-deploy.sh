#!/usr/bin/env bash
# Post-install / post-REINDEX hygiene for ValorBrain pgturbohybrid indexes.
# - Ensures turbohybrid_prune_shared_cache() exists
# - Prewarms production indexes (moves cold cache build off first user query)
# - Prunes stale .tqcache orphans
# - Optionally runs scalar fallback guard
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5433}"
PGDATABASE="${PGDATABASE:-valorbrain}"
PGUSER="${PGUSER:-postgres}"
RUN_GUARD="${RUN_GUARD:-1}"

PSQL="${PSQL:-psql}"
PSQL_OPTS=(
  -h "$PGHOST"
  -p "$PGPORT"
  -U "$PGUSER"
  -d "$PGDATABASE"
  -v ON_ERROR_STOP=1
)

run_sql() {
  "$PSQL" "${PSQL_OPTS[@]}" -c "$1"
}

echo "pgturbohybrid post-deploy: ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"

run_sql "
CREATE OR REPLACE FUNCTION turbohybrid_prune_shared_cache(index regclass DEFAULT NULL)
RETURNS jsonb LANGUAGE C PARALLEL UNSAFE
AS '\$libdir/pgturbohybrid', 'pgturbohybrid_prune_shared_cache';
"

echo "== prewarm hybrid index =="
run_sql "SELECT turbohybrid_prewarm('idx_documents_fts_hybrid'::regclass);"

if run_sql "SELECT to_regclass('public.idx_content_vectors_turbo') IS NOT NULL AS ok;" | grep -q t; then
  echo "== prewarm dense turbo index =="
  run_sql "SELECT turbohybrid_prewarm('idx_content_vectors_turbo'::regclass);"
fi

echo "== prune shared disk cache =="
run_sql "SELECT turbohybrid_prune_shared_cache();"

if [[ -n "${PGDATA:-}" && -d "${PGDATA}/pg_turbohybrid_cache" ]]; then
  echo "== cache dir (${PGDATA}/pg_turbohybrid_cache) =="
  du -sh "${PGDATA}/pg_turbohybrid_cache" || true
elif command -v sudo >/dev/null 2>&1; then
  for cache_dir in /var/lib/postgresql/*/main/pg_turbohybrid_cache; do
    if [[ -d "$cache_dir" ]]; then
      echo "== cache dir (${cache_dir}) =="
      du -sh "$cache_dir" || true
    fi
  done
fi

if [[ "$RUN_GUARD" == "1" ]]; then
  echo "== stability check =="
  STABILITY_WORKERS="${STABILITY_WORKERS:-2}" \
  STABILITY_ROUNDS="${STABILITY_ROUNDS:-2}" \
  bash "$ROOT_DIR/scripts/valorbrain/stability-check.sh"
fi

echo "post-deploy complete"