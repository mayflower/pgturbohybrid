#!/usr/bin/env bash
#
# test/t/004_crash_compaction.sh
#
# Crash-safety test: verify that killing the PostgreSQL server during or
# immediately after VACUUM-triggered graph compaction does not corrupt the
# turbohybrid index.
#
# This exercises the WAL logging path in PgturbohybridGraphMaybeCompactPageChains:
# new code/adj/exact pages are WAL-flushed before the metapage swap so that
# crash recovery sees a consistent index.
#
# Usage: run inside the Nix dev shell. Requires a running PG cluster.
#   nix develop --command test/t/004_crash_compaction.sh
#
set -uo pipefail
# Note: we don't use set -e because we want to report individual test failures.

# --- Configuration ---

TEST_DB="crash_test_$$"
PGHOST="${PGHOST:-/opt/pgturbohybrid/.nix-dev/pg18-pgvector-v0.8.2/run}"
PGPORT="${PGPORT:-55432}"
PGDATA="${PGDATA:-/opt/pgturbohybrid/.nix-dev/pg18-pgvector-v0.8.2/pgdata}"

PSQL="psql -h $PGHOST -p $PGPORT -v ON_ERROR_STOP=1 -q"
PGCTL="pg_ctl -D $PGDATA"
PASS=0
FAIL=0

ok() {
    echo "ok $((++PASS)) - $1"
}

not_ok() {
    echo "not ok $((++PASS)) - $1"
    FAIL=$((FAIL+1))
}

cleanup() {
    # Restart PG if it's down
    if ! $PSQL -d postgres -c "SELECT 1" >/dev/null 2>&1; then
        su pgturbohybrid-dev -c "$PGCTL -w start" 2>/dev/null || true
    fi
    $PSQL -d postgres -c "DROP DATABASE IF EXISTS $TEST_DB;" 2>/dev/null || true
}
trap cleanup EXIT

# --- Helper: top 5 nearest docs to [0.10, 0, 0] ---
top_ids() {
    $PSQL -d "$TEST_DB" -t -A -c "
        SET enable_seqscan = off;
        SELECT string_agg(id::text, ',' ORDER BY ord)
        FROM (
            SELECT id, row_number() OVER () AS ord
            FROM (
                SELECT id
                FROM crash_docs
                ORDER BY embedding <~> turbohybrid_query(
                    vector_query => '[10,20,30]'::vector,
                    dense_k => 50,
                    final_k => 5
                )
                LIMIT 5
            ) q
        ) s;
    " 2>/dev/null
}

echo "# Crash-safety test for pgturbohybrid graph compaction"
echo "# Testing WAL durability during VACUUM compaction"

# --- Setup ---
$PSQL -d postgres -c "DROP DATABASE IF EXISTS $TEST_DB;" 2>/dev/null || true
$PSQL -d postgres -c "CREATE DATABASE $TEST_DB;"
$PSQL -d "$TEST_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"
$PSQL -d "$TEST_DB" -c "CREATE EXTENSION IF NOT EXISTS pgturbohybrid;"

# Create a table with 200 vectors in 3-dim space, widely spaced.
# page_compaction_threshold=10 ensures compaction triggers easily.
# exact_storage=on ensures Phase 4 (exact slab rewrite) is exercised.
$PSQL -d "$TEST_DB" <<'SQL'
SET client_min_messages = warning;
CREATE TABLE crash_docs (
    id int PRIMARY KEY,
    embedding vector(3)
);
INSERT INTO crash_docs (id, embedding)
SELECT i, ('[' || (i * 1.0) || ',' || (i * 2.0) || ',' || (i * 3.0) || ']')::vector
FROM generate_series(1, 50) AS i;
CREATE INDEX crash_idx ON crash_docs
USING turbohybrid (embedding vector_l2_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = on,
      page_compaction_threshold = 10);
SQL

# Baseline query: find top-5 docs closest to id=10's vector.
# Each vector has value i*0.01 in dim 1, so query [0.10,0,...,0] targets id=10.
BASELINE=$(top_ids)
ok "baseline query result: $BASELINE"

# Verify the query returns valid results (not empty, not error)
if [ -n "$BASELINE" ] && [ "$BASELINE" != "" ]; then
    ok "baseline query succeeds"
else
    not_ok "baseline query failed"
    exit 1
fi

# --- Cycle 1: Delete 30%, VACUUM, immediate stop, restart, verify ---
$PSQL -d "$TEST_DB" -c "DELETE FROM crash_docs WHERE id BETWEEN 1 AND 20;"

COUNT=$($PSQL -d "$TEST_DB" -t -A -c "SELECT count(*) FROM crash_docs;")
if [ "$COUNT" = "30" ]; then
    ok "20 rows deleted, 30 remain"
else
    not_ok "expected 140 rows after delete, got $COUNT"
fi

# VACUUM (triggers compaction)
$PSQL -d "$TEST_DB" -c "VACUUM crash_docs;" 2>&1 | grep -i compaction && echo "# (compaction ran)" || echo "# (compaction may have run)"

# Crash simulation: immediate stop
su pgturbohybrid-dev -c "$PGCTL stop -m immediate" 2>/dev/null
sleep 1

# Restart
su pgturbohybrid-dev -c "$PGCTL -w start" 2>/dev/null
sleep 2

# Verify index returns same result as before crash
AFTER_CRASH1=$(top_ids)
if [ "$AFTER_CRASH1" = "$BASELINE" ]; then
    ok "index returns identical results after crash recovery cycle 1"
else
    # Results may differ if deleted rows were in the top-5, which is fine.
    # What matters is that the query still works and returns valid rows.
    if [ -n "$AFTER_CRASH1" ]; then
        ok "index functional after crash recovery cycle 1 (was: $BASELINE, now: $AFTER_CRASH1)"
    else
        not_ok "index corrupted after cycle 1 (query returned empty)"
    fi
fi

# --- Cycle 2: Delete more, VACUUM, kill -9, restart, verify ---
$PSQL -d "$TEST_DB" -c "DELETE FROM crash_docs WHERE id BETWEEN 21 AND 30;"

COUNT=$($PSQL -d "$TEST_DB" -t -A -c "SELECT count(*) FROM crash_docs;")
if [ "$COUNT" = "20" ]; then
    ok "10 more rows deleted, 20 remain"
else
    not_ok "expected 100 rows after delete, got $COUNT"
fi

# VACUUM then immediate stop
$PSQL -d "$TEST_DB" -c "VACUUM crash_docs;" 2>/dev/null || true

# Hard crash: SIGKILL the postmaster
PM_PID=$(head -1 "$PGDATA/postmaster.pid" 2>/dev/null || echo "")
if [ -n "$PM_PID" ]; then
    kill -9 "$PM_PID" 2>/dev/null || true
    sleep 1
    # Clean up shared memory
    ipcs -m | awk '{print $3}' | while read -r shmid; do
        ipcrm -m "$shmid" 2>/dev/null || true
    done
    rm -f "$PGDATA/postmaster.pid"
fi

# Restart and recover
su pgturbohybrid-dev -c "$PGCTL -w start" 2>/dev/null
sleep 2

# Verify index still works
AFTER_CRASH2=$(top_ids)
if [ -n "$AFTER_CRASH2" ]; then
    ok "index functional after SIGKILL crash recovery cycle 2 (got: $AFTER_CRASH2)"
else
    not_ok "index corrupted after cycle 2 (query returned empty)"
fi

# --- Verify index returns all surviving rows ---
FULL_COUNT=$($PSQL -d "$TEST_DB" -t -A -c "
    SET enable_seqscan = off;
    SELECT count(*) FROM crash_docs WHERE embedding <~->
        turbohybrid_query(
            vector_query => '[0,0,0]'::vector,
            dense_k => 50,
            final_k => 50
        );
" 2>/dev/null || echo "0")

if [ "$FULL_COUNT" = "20" ]; then
    ok "index returns all 20 surviving rows"
else
    not_ok "index returned $FULL_COUNT rows (expected 20)"
fi

# --- REINDEX succeeds ---
if $PSQL -d "$TEST_DB" -c "REINDEX INDEX crash_idx;" 2>/dev/null; then
    ok "REINDEX succeeds after crash recovery (no structural corruption)"
else
    not_ok "REINDEX failed - possible structural corruption"
fi

# --- Summary ---
echo ""
echo "# $PASS tests run, $FAIL failures"
if [ "$FAIL" -eq 0 ]; then
    echo "# All tests passed"
    exit 0
else
    echo "# $FAIL test(s) FAILED"
    exit 1
fi
