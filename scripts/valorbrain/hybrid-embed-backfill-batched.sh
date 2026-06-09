#!/usr/bin/env bash
# hybrid-embed-backfill-batched.sh
#
# Bulk backfill documents_fts.embedding from content_vectors (seq=0).
# Uses drop-index → bulk UPDATE → rebuild pattern to avoid TurboHybrid
# page-lock contention that causes index bloat.
#
# IMPORTANT: This script acquires a global advisory lock so the embed worker
# detects it via copyHybridEmbeddingForHash() and defers its own writes.
#
# Usage:
#   PGHOST=127.0.0.1 PGPORT=5433 PGDATABASE=valorbrain PGUSER=postgres \
#     bash hybrid-embed-backfill-batched.sh
#
# Options:
#   --dry-run   Count pending rows and report index stats without modifying anything.
#
# Environment:
#   PGHOST, PGPORT, PGDATABASE, PGUSER — PostgreSQL connection (defaults below)
#   LOCK_TIMEOUT                        — advisory lock acquire timeout (default: 5s)
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5433}"
PGDATABASE="${PGDATABASE:-valorbrain}"
PGUSER="${PGUSER:-postgres}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-5s}"
GLOBAL_LOCK_KEY="valorbrain:hybrid_backfill_global"
INDEX_NAME="idx_documents_fts_hybrid"

PSQL=(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# ── Helpers ─────────────────────────────────────────────────────────
function sql() { "${PSQL[@]}" -tAc "$@"; }
function sql_val() { sql "$1" | tr -d '[:space:]'; }

# ── Pre-flight ──────────────────────────────────────────────────────
echo "[$(date -Iseconds)] hybrid-embed-backfill starting (dry_run=${DRY_RUN})"

pending=$(sql_val "
  SELECT COUNT(*)
  FROM documents d
  JOIN documents_fts fts ON fts.doc_id = d.id
  JOIN content_vectors cv ON cv.hash = d.hash AND cv.seq = 0
  WHERE d.active = 1
    AND fts.embedding IS NULL
    AND cv.embedding IS NOT NULL;
")
echo "[$(date -Iseconds)] Pending embeddings: ${pending}"

index_size=$(sql_val "
  SELECT COALESCE(pg_size_pretty(pg_relation_size('${INDEX_NAME}'::regclass)), 'none');
")
echo "[$(date -Iseconds)] Current index size: ${index_size}"

if [[ "$pending" -eq 0 ]]; then
  echo "[$(date -Iseconds)] Nothing to backfill. Exiting."
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[$(date -Iseconds)] --dry-run: would backfill ${pending} rows. Exiting."
  exit 0
fi

# ── Acquire global advisory lock ────────────────────────────────────
echo "[$(date -Iseconds)] Acquiring global advisory lock (${LOCK_TIMEOUT})..."
if ! sql "SET lock_timeout = '${LOCK_TIMEOUT}'; SELECT pg_advisory_lock(hashtext('${GLOBAL_LOCK_KEY}'));" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] FATAL: Could not acquire global backfill lock." >&2
  echo "[$(date -Iseconds)] The embed worker or another backfill may be running." >&2
  exit 1
fi
echo "[$(date -Iseconds)] Global lock acquired."
# Ensure lock is released on exit (even on error)
trap "sql \"SELECT pg_advisory_unlock(hashtext('${GLOBAL_LOCK_KEY}'));\" 2>/dev/null || true" EXIT

start_ms=$(date +%s%3N)

	# ── Bulk UPDATE ─────────────────────────────────────────────────────
	echo "[$(date -Iseconds)] Running bulk UPDATE..."
	updated=$(sql_val "
	  WITH upd AS (
	    UPDATE documents_fts fts
	    SET embedding = cv.embedding
	    FROM content_vectors cv, documents d
	    WHERE d.hash = cv.hash
	      AND d.id = fts.doc_id
	      AND cv.seq = 0
	      AND fts.embedding IS NULL
	      AND d.active = 1
	    RETURNING 1
	  )
	  SELECT COUNT(*) FROM upd;
	")
	echo "[$(date -Iseconds)] Updated ${updated} rows."

	# ── REINDEX (compact rebuild via ambuild) ───────────────────────────
	echo "[$(date -Iseconds)] Reindexing ${INDEX_NAME}..."
	sql "REINDEX INDEX ${INDEX_NAME};"
	echo "[$(date -Iseconds)] Index compacted."

	# ── Analyze ────────────────────────────────────────────────────────
	echo "[$(date -Iseconds)] ANALYZE documents_fts..."
	sql "ANALYZE documents_fts;"

# ── Prewarm ─────────────────────────────────────────────────────────
echo "[$(date -Iseconds)] Prewarming index..."
sql "SELECT turbohybrid_prewarm('${INDEX_NAME}'::regclass);" 2>/dev/null || true

# ── Stats ───────────────────────────────────────────────────────────
end_ms=$(date +%s%3N)
duration_ms=$(( end_ms - start_ms ))
duration_s=$(( duration_ms / 1000 ))

new_index_size=$(sql_val "
  SELECT pg_size_pretty(pg_relation_size('${INDEX_NAME}'::regclass));
")
total=$(sql_val "SELECT COUNT(*) FROM documents_fts;")
with_emb=$(sql_val "SELECT COUNT(*) FROM documents_fts WHERE embedding IS NOT NULL;")
coverage=$(sql_val "
  SELECT round(100.0 * COUNT(*) FILTER (WHERE embedding IS NOT NULL)
    / GREATEST(COUNT(*), 1), 2) FROM documents_fts;
")

echo "[$(date -Iseconds)] ───────────────────────────────────────"
echo "[$(date -Iseconds)] Backfill complete"
echo "[$(date -Iseconds)]   Rows updated:   ${updated}"
echo "[$(date -Iseconds)]   Index size:     ${new_index_size} (was ${index_size})"
echo "[$(date -Iseconds)]   Embed coverage: ${with_emb}/${total} (${coverage}%)"
echo "[$(date -Iseconds)]   Duration:       ${duration_s}s"
echo "[$(date -Iseconds)] ───────────────────────────────────────"
