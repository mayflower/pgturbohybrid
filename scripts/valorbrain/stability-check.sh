#!/usr/bin/env bash
# ValorBrain + pgturbohybrid stability gate after deploy or code changes.
# Exits non-zero on any hard failure. Suitable for cron / CI smoke / post-deploy.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5433}"
PGDATABASE="${PGDATABASE:-valorbrain}"
PGUSER="${PGUSER:-postgres}"
ENGINE_URL="${ENGINE_URL:-http://127.0.0.1:7438}"
WORKERS="${STABILITY_WORKERS:-2}"
ROUNDS="${STABILITY_ROUNDS:-3}"
MIN_EMBED_RATIO="${STABILITY_MIN_EMBED_RATIO:-0.95}"
MIN_HYBRID_EMBED_RATIO="${STABILITY_MIN_HYBRID_EMBED_RATIO:-0.95}"

PSQL=(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1)
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

failures=0
warnings=0
note() { echo "[$TS] $*"; }
fail() { note "FAIL: $*"; failures=$((failures + 1)); }
warn() { note "WARN: $*"; warnings=$((warnings + 1)); }
pass() { note "OK: $*"; }

note "stability-check start (${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE})"

# 1) PostgreSQL reachable
if ! "${PSQL[@]}" -q -c "SELECT 1" >/dev/null 2>&1; then
  fail "PostgreSQL unreachable"
  exit 1
fi
pass "PostgreSQL reachable"

# 2) Extensions present
ext_ok="$("${PSQL[@]}" -tAc "
  SELECT count(*) = 2
  FROM pg_extension
  WHERE extname IN ('vector', 'pgturbohybrid');
" | tr -d '[:space:]')"
if [[ "$ext_ok" != "t" ]]; then
  fail "missing vector and/or pgturbohybrid extension"
else
  pass "extensions vector + pgturbohybrid"
fi

# 2b) pg_ripple KG shadow path
ripple_ext="$("${PSQL[@]}" -tAc "
  SELECT count(*) FROM pg_extension WHERE extname = 'pg_ripple';
" | tr -d '[:space:]')"
if [[ "$ripple_ext" != "1" ]]; then
  warn "pg_ripple extension not installed (KG shadow/inference disabled)"
else
  pass "extension pg_ripple present"
  preload_ok="$("${PSQL[@]}" -tAc "
    SELECT current_setting('shared_preload_libraries') LIKE '%pg_ripple%';
  " | tr -d '[:space:]')"
  if [[ "$preload_ok" != "t" ]]; then
    warn "pg_ripple not in shared_preload_libraries (HTAP/CWB disabled)"
  else
    pass "pg_ripple preloaded (HTAP/CWB enabled)"
  fi
  ripple_n="$("${PSQL[@]}" -tAc "SELECT pg_ripple.triple_count();" 2>/dev/null | tr -d '[:space:]' || echo 0)"
  if [[ "${ripple_n:-0}" -lt 1 ]]; then
    warn "pg_ripple graph empty — run /opt/valorbrain/scripts/kg/load-ripple.sh"
  else
    pass "pg_ripple triple_count global ${ripple_n}"
  fi
  main_tenant="${VALORBRAIN_DEFAULT_TENANT_ID:-3ceeb048-754c-4910-a798-112ae867e9a4}"
  graph_iri="https://valorbrain.valor.digital/kg/tenant/${main_tenant}"
  tenant_ripple_n="$("${PSQL[@]}" -tAc "
    SELECT COALESCE((s::jsonb->>'c')::int, 0)
    FROM (SELECT pg_ripple.sparql(
      'SELECT (COUNT(?o) AS ?c) WHERE { GRAPH <${graph_iri}> { ?s ?p ?o } }'
    )::text AS s) q;
  " 2>/dev/null | tr -d '[:space:]' || echo 0)"
  sql_triple_n="$("${PSQL[@]}" -tAc "
    SELECT count(*)::int FROM entity_triples
    WHERE tenant_id = '${main_tenant}' AND valid_to IS NULL;
  " 2>/dev/null | tr -d '[:space:]' || echo 0)"
  if [[ "${tenant_ripple_n:-0}" -lt 1 && "${sql_triple_n:-0}" -gt 0 ]]; then
    warn "tenant ripple graph empty (sql=${sql_triple_n}) — run backfill-ripple-sync"
  else
    pass "tenant ripple graph triples ${tenant_ripple_n} (sql current ${sql_triple_n})"
  fi
  if command -v bun >/dev/null 2>&1 && [[ -f /opt/valorbrain/scripts/kg/kg-sync-drift.ts ]]; then
    if POSTGRES_USER="$PGUSER" POSTGRES_PASSWORD="${PGPASSWORD:-}" \
      bun run /opt/valorbrain/scripts/kg/kg-sync-drift.ts >/dev/null 2>&1; then
      pass "kg SQL/ripple drift sample OK"
    else
      warn "kg SQL/ripple drift — run scripts/kg/backfill-ripple-sync.ts"
    fi
  fi
  mirror_enforce="$("${PSQL[@]}" -tAc "
    SELECT current_setting('valorbrain.kg_sql_mirror_enforce', true);
  " 2>/dev/null | tr -d '[:space:]' || echo off)"
  if [[ "$mirror_enforce" == "on" ]]; then
    pass "ADR-002 entity_triples mirror enforce on"
  else
    warn "ADR-002 mirror enforce off — set valorbrain.kg_sql_mirror_enforce=on"
  fi
  if command -v bun >/dev/null 2>&1 && [[ -f /opt/valorbrain/scripts/kg/test-kg-visibility-ff15.ts ]]; then
    if POSTGRES_SUPERUSER_PASSWORD="${POSTGRES_SUPERUSER_PASSWORD:-${PGPASSWORD:-}}" \
      bun run /opt/valorbrain/scripts/kg/test-kg-visibility-ff15.ts >/dev/null 2>&1; then
      pass "FF.15 KG visibility probe OK"
    else
      warn "FF.15 KG visibility probe failed"
    fi
  fi
fi

# 3) Critical indexes exist
for idx in idx_documents_fts_hybrid idx_content_vectors_turbo; do
  exists="$("${PSQL[@]}" -tAc "SELECT to_regclass('public.${idx}') IS NOT NULL;" | tr -d '[:space:]')"
  if [[ "$exists" != "t" ]]; then
    fail "index ${idx} missing"
  else
    pass "index ${idx} present"
  fi
done

# 4) Embed / hybrid coverage
read -r embed_ratio hybrid_ratio <<<"$("${PSQL[@]}" -tAc "
  SELECT
    round((COUNT(*) FILTER (WHERE last_embedded_at IS NOT NULL AND embed_state != 'skipped')::numeric
      / GREATEST(COUNT(*) FILTER (WHERE embed_state != 'skipped'), 1)), 4),
    round((SELECT COUNT(*) FILTER (WHERE embedding IS NOT NULL)::numeric
      / GREATEST(COUNT(*),1) FROM documents_fts), 4)
  FROM documents WHERE active = 1;
" | tr '|' ' ')"

if awk -v r="$embed_ratio" -v m="$MIN_EMBED_RATIO" 'BEGIN { exit !(r+0 >= m+0) }'; then
  pass "embed coverage ${embed_ratio} (>= ${MIN_EMBED_RATIO})"
else
  warn "embed coverage ${embed_ratio} below ${MIN_EMBED_RATIO} (ingest backlog?)"
fi

if awk -v r="$hybrid_ratio" -v m="$MIN_HYBRID_EMBED_RATIO" 'BEGIN { exit !(r+0 >= m+0) }'; then
  pass "hybrid embedding coverage ${hybrid_ratio} (>= ${MIN_HYBRID_EMBED_RATIO})"
else
  warn "hybrid embedding coverage ${hybrid_ratio} below ${MIN_HYBRID_EMBED_RATIO} (run hybrid backfill?)"
fi

# 5) FTS table health
dead_pct="$("${PSQL[@]}" -tAc "
  SELECT round(100.0 * n_dead_tup / GREATEST(n_live_tup + n_dead_tup, 1), 2)
  FROM pg_stat_user_tables WHERE relname = 'documents_fts';
" | tr -d '[:space:]')"
if [[ -n "$dead_pct" ]] && awk -v d="$dead_pct" 'BEGIN { exit !(d+0 > 20) }'; then
  warn "documents_fts dead tuple ratio ${dead_pct}% (>20%) — schedule VACUUM (ANALYZE)"
else
  pass "documents_fts dead tuples ${dead_pct:-0}%"
fi

# 6) Native cache disk (if visible)
for cache_dir in /var/lib/postgresql/*/main/pg_turbohybrid_cache; do
  if [[ -d "$cache_dir" ]]; then
    cache_mb="$(du -sm "$cache_dir" | awk '{print $1}')"
    if [[ "$cache_mb" -gt 16384 ]]; then
      fail "pg_turbohybrid_cache ${cache_mb}MB (>16GB)"
    else
      pass "pg_turbohybrid_cache ${cache_mb}MB"
    fi
  fi
done

# 7) Scalar fallback guard (hybrid must use Index Scan)
if ! bash "$ROOT_DIR/scripts/valorbrain/scalar-fallback-guard.sh" >/dev/null 2>&1; then
  fail "scalar fallback guard"
else
  pass "scalar fallback guard"
fi

# 8) Light concurrent hybrid reads
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
  echo "hybrid worker ${wid} ok"
}

pids=()
for w in $(seq 1 "$WORKERS"); do
  hybrid_worker "$w" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail "concurrent hybrid worker"
  fi
done
pass "concurrent hybrid queries (${WORKERS}x${ROUNDS})"

# 9) Engine health (optional — skip if engine down)
if curl -sf "${ENGINE_URL}/health" >/tmp/vb-health.json 2>/dev/null; then
  if grep -q '"status":"ok"' /tmp/vb-health.json; then
    pass "valorbrain engine /health ok"
  else
    fail "valorbrain engine unhealthy: $(tr -d '\n' </tmp/vb-health.json)"
  fi
else
  note "WARN: valorbrain engine not reachable at ${ENGINE_URL}/health (skipped)"
fi

# 10) Recent PG errors (pgturbohybrid / BM25)
if command -v journalctl >/dev/null 2>&1; then
  err_count="$(journalctl -u postgresql --since "15 min ago" --no-pager 2>/dev/null \
    | grep -ciE 'pgturbohybrid|BM25 page kind|unexpected' || true)"
  if [[ "${err_count:-0}" -gt 0 ]]; then
    fail "postgresql logs: ${err_count} pgturbohybrid/BM25 errors (15m)"
  else
    pass "no pgturbohybrid errors in PG logs (15m)"
  fi
fi

note "stability-check done: ${failures} failure(s), ${warnings} warning(s)"
exit "$failures"