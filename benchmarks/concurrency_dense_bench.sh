#!/usr/bin/env bash
#
# Runner for benchmarks/concurrency_dense_bench.sql -- the concurrent-client
# dense scaling diagnostics that explain the 1->8 client collapse on
# glove-100-angular before any algorithm is changed.
#
# The SQL file already sweeps the full matrix internally
# (clients {1,2,4,8,16} x native_cache_scope {per_backend, shared, off} x
# prewarm {A,B}, with default/HIGH cache caps for cached scopes) and prints the diagnosis
# tables.  This wrapper just:
#   - points it at a database (existing `items` index, or a synthetic dataset it
#     builds on first run),
#   - forwards the tunables as psql -v vars,
#   - tees the full output to a timestamped log under benchmarks/output/.
#
# USAGE:
#   benchmarks/concurrency_dense_bench.sh [DB]
#
# Against the loaded glove-100-angular dataset (the real collapse):
#   benchmarks/concurrency_dense_bench.sh pgturbohybrid_benchmark
#
# Build + run a synthetic glove-sized dataset in a throwaway DB (no real data
# needed -- this reproduces the per-backend duplication at scale):
#   CCB_NROWS=1183514 CCB_DIMS=100 \
#     benchmarks/concurrency_dense_bench.sh pgturbohybrid_ccbench
#
# Tunables (env vars, all optional):
#   PGDATABASE / first arg  target database (default: $PGTURBOHYBRID_DB or
#                           pgturbohybrid_benchmark)
#   CCB_TBL                 base table to target if present (default items)
#   CCB_VCOL                vector column (default embedding)
#   CCB_NROWS               synthetic rows if the table is built here (default 200000)
#   CCB_DIMS                synthetic vector dims (default 100)
#   CCB_QSET                fixed query-vector count (default 64)
#   CCB_WARM                prewarm queries per client, mode B (default 40)
#   CCB_TIMED               timed queries per client (default 200)
#   CCB_MAX_CLIENTS         cap the client sweep (default 16)
#   CCB_HIGH_CACHE_MB       high native cache cap, MB (default 4096)
#   PSQL                    psql binary (default psql)
#
# NOTE: this measures SERVER-SIDE latency via dblink (no client/network
# round-trip), so absolute RPS differs from the Python vector-db-benchmark
# numbers; the scaling SHAPE and per-backend cause signals are the point.  For
# authoritative end-to-end throughput numbers, drive the same dense query with
#   pgbench -n -c <N> -j <N> -T 30 -f <(printf '%s\n' "SELECT ...") <DB>
# but pgbench cannot capture the per-backend native_cache_* signals this does.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="$SCRIPT_DIR/concurrency_dense_bench.sql"
PSQL="${PSQL:-psql}"

DB="${1:-${PGDATABASE:-${PGTURBOHYBRID_DB:-pgturbohybrid_benchmark}}}"

OUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUT_DIR"
# Portable timestamp (avoids relying on GNU date).
TS="$(date '+%Y%m%d-%H%M%S')"
LOG="$OUT_DIR/concurrency_dense_bench-${DB}-${TS}.txt"

echo "concurrency_dense_bench: db=$DB" >&2
echo "  tbl=${CCB_TBL:-items} vcol=${CCB_VCOL:-embedding} nrows=${CCB_NROWS:-200000} dims=${CCB_DIMS:-100}" >&2
echo "  qset=${CCB_QSET:-64} warm=${CCB_WARM:-40} timed=${CCB_TIMED:-200} max_clients=${CCB_MAX_CLIENTS:-16} high_cache_mb=${CCB_HIGH_CACHE_MB:-4096}" >&2
echo "  log -> $LOG" >&2

# Each -v is only forwarded when the matching env var is set; otherwise the SQL
# file's own defaults apply.
declare -a VARS=()
[ -n "${CCB_TBL:-}" ]            && VARS+=(-v "TBL=${CCB_TBL}")
[ -n "${CCB_VCOL:-}" ]           && VARS+=(-v "VCOL=${CCB_VCOL}")
[ -n "${CCB_NROWS:-}" ]          && VARS+=(-v "NROWS=${CCB_NROWS}")
[ -n "${CCB_DIMS:-}" ]           && VARS+=(-v "DIMS=${CCB_DIMS}")
[ -n "${CCB_QSET:-}" ]           && VARS+=(-v "QSET=${CCB_QSET}")
[ -n "${CCB_WARM:-}" ]           && VARS+=(-v "WARM=${CCB_WARM}")
[ -n "${CCB_TIMED:-}" ]          && VARS+=(-v "TIMED=${CCB_TIMED}")
[ -n "${CCB_MAX_CLIENTS:-}" ]    && VARS+=(-v "MAX_CLIENTS=${CCB_MAX_CLIENTS}")
[ -n "${CCB_HIGH_CACHE_MB:-}" ]  && VARS+=(-v "HIGH_CACHE_MB=${CCB_HIGH_CACHE_MB}")

"$PSQL" -d "$DB" -v ON_ERROR_STOP=1 "${VARS[@]}" -f "$SQL_FILE" 2>&1 | tee "$LOG"

echo "" >&2
echo "concurrency_dense_bench: full output saved to $LOG" >&2
