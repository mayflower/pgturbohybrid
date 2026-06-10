#!/usr/bin/env bash
# Emit JSON stability metrics for trend monitoring (cron / post-deploy / CI artifacts).
# Exits 0 when hard checks pass (same gates as stability-check.sh, without concurrent load).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PGHOST="${PGHOST:-127.0.0.1}"
PGPORT="${PGPORT:-5433}"
PGDATABASE="${PGDATABASE:-valorbrain}"
PGUSER="${PGUSER:-postgres}"
ENGINE_URL="${ENGINE_URL:-http://127.0.0.1:7438}"
OUT="${STABILITY_SNAPSHOT_OUT:-}"

PSQL=(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1)
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

failures=0
warnings=()

pg_ok=false
ext_ok=false
hybrid_idx=false
dense_idx=false
embed_ratio="0"
hybrid_ratio="0"
dead_pct="0"
cache_mb="null"
guard_ok=false
engine_ok="null"
log_errors=0
ripple_ext=0
ripple_global=0
kg_sql_triples=0
kg_ripple_triples=0
kg_drift_ok="null"

if "${PSQL[@]}" -q -c "SELECT 1" >/dev/null 2>&1; then
  pg_ok=true
  ext_ok="$("${PSQL[@]}" -tAc "
    SELECT count(*) = 2 FROM pg_extension
    WHERE extname IN ('vector', 'pgturbohybrid');
  " | tr -d '[:space:]')"
  hybrid_idx="$("${PSQL[@]}" -tAc "
    SELECT to_regclass('public.idx_documents_fts_hybrid') IS NOT NULL;
  " | tr -d '[:space:]')"
  dense_idx="$("${PSQL[@]}" -tAc "
    SELECT to_regclass('public.idx_content_vectors_turbo') IS NOT NULL;
  " | tr -d '[:space:]')"
  read -r embed_ratio hybrid_ratio <<<"$("${PSQL[@]}" -tAc "
    SELECT
      round((COUNT(*) FILTER (WHERE last_embedded_at IS NOT NULL AND embed_state != 'skipped')::numeric
        / GREATEST(COUNT(*) FILTER (WHERE embed_state != 'skipped'), 1)), 4),
      round((SELECT COUNT(*) FILTER (WHERE embedding IS NOT NULL)::numeric
        / GREATEST(COUNT(*),1) FROM documents_fts), 4)
    FROM documents WHERE active = 1;
  " | tr '|' ' ')"
  dead_pct="$("${PSQL[@]}" -tAc "
    SELECT coalesce(round(100.0 * n_dead_tup / GREATEST(n_live_tup + n_dead_tup, 1), 2), 0)
    FROM pg_stat_user_tables WHERE relname = 'documents_fts';
  " | tr -d '[:space:]')"
  ripple_ext="$("${PSQL[@]}" -tAc "
    SELECT count(*) FROM pg_extension WHERE extname = 'pg_ripple';
  " 2>/dev/null | tr -d '[:space:]' || echo 0)"
  if [[ "${ripple_ext:-0}" == "1" ]]; then
    ripple_global="$("${PSQL[@]}" -tAc "SELECT pg_ripple.triple_count();" 2>/dev/null | tr -d '[:space:]' || echo 0)"
    main_tenant="${VALORBRAIN_DEFAULT_TENANT_ID:-3ceeb048-754c-4910-a798-112ae867e9a4}"
    graph_iri="https://valorbrain.valor.digital/kg/tenant/${main_tenant}"
    kg_ripple_triples="$("${PSQL[@]}" -tAc "
      SELECT COALESCE((s::jsonb->>'c')::int, 0)
      FROM (SELECT pg_ripple.sparql(
        'SELECT (COUNT(?o) AS ?c) WHERE { GRAPH <${graph_iri}> { ?s ?p ?o } }'
      )::text AS s) q;
    " 2>/dev/null | tr -d '[:space:]' || echo 0)"
    kg_sql_triples="$("${PSQL[@]}" -tAc "
      SELECT count(*)::int FROM entity_triples
      WHERE tenant_id = '${main_tenant}' AND valid_to IS NULL;
    " 2>/dev/null | tr -d '[:space:]' || echo 0)"
    if command -v bun >/dev/null 2>&1 && [[ -f /opt/valorbrain/scripts/kg/kg-sync-drift.ts ]]; then
      if POSTGRES_USER="$PGUSER" POSTGRES_PASSWORD="${PGPASSWORD:-}" \
        bun run /opt/valorbrain/scripts/kg/kg-sync-drift.ts >/dev/null 2>&1; then
        kg_drift_ok=true
      else
        kg_drift_ok=false
        warnings+=("kg_sql_ripple_drift")
      fi
    fi
  fi
else
  failures=$((failures + 1))
fi

for cache_dir in /var/lib/postgresql/*/main/pg_turbohybrid_cache; do
  if [[ -d "$cache_dir" ]]; then
    cache_mb="$(du -sm "$cache_dir" | awk '{print $1}')"
    break
  fi
done

if bash "$ROOT_DIR/scripts/valorbrain/scalar-fallback-guard.sh" >/dev/null 2>&1; then
  guard_ok=true
else
  failures=$((failures + 1))
fi

if curl -sf "${ENGINE_URL}/health" | grep -q '"status":"ok"'; then
  engine_ok=true
elif curl -sf "${ENGINE_URL}/health" >/dev/null 2>&1; then
  engine_ok=false
  failures=$((failures + 1))
fi

if command -v journalctl >/dev/null 2>&1; then
  log_errors="$(journalctl -u postgresql --since "15 min ago" --no-pager 2>/dev/null \
    | grep -ciE 'pgturbohybrid|BM25 page kind|unexpected' || true)"
  if [[ "${log_errors:-0}" -gt 0 ]]; then
    failures=$((failures + 1))
  fi
fi

[[ "$pg_ok" != true ]] && failures=$((failures + 1))
[[ "$ext_ok" != "t" ]] && failures=$((failures + 1))
[[ "$hybrid_idx" != "t" ]] && failures=$((failures + 1))
[[ "$dense_idx" != "t" ]] && failures=$((failures + 1))
[[ -n "$cache_mb" && "$cache_mb" != "null" && "$cache_mb" -gt 16384 ]] && failures=$((failures + 1))

awk -v r="$embed_ratio" 'BEGIN { exit !(r+0 < 0.95) }' && warnings+=("embed_coverage_below_95")
awk -v r="$hybrid_ratio" 'BEGIN { exit !(r+0 < 0.95) }' && warnings+=("hybrid_embedding_below_95")
[[ -n "$dead_pct" ]] && awk -v d="$dead_pct" 'BEGIN { exit !(d+0 > 20) }' && warnings+=("documents_fts_dead_pct_high")

warn_json="[]"
if [[ ${#warnings[@]} -gt 0 ]]; then
  warn_json="$(printf '%s\n' "${warnings[@]}" | jq -R . | jq -s .)"
fi

json="$(jq -n \
  --arg ts "$TS" \
  --arg host "$PGHOST" \
  --argjson port "$PGPORT" \
  --arg db "$PGDATABASE" \
  --argjson pg_ok "$pg_ok" \
  --argjson ext_ok "$( [[ "$ext_ok" == "t" ]] && echo true || echo false )" \
  --argjson hybrid_idx "$( [[ "$hybrid_idx" == "t" ]] && echo true || echo false )" \
  --argjson dense_idx "$( [[ "$dense_idx" == "t" ]] && echo true || echo false )" \
  --argjson embed_ratio "$embed_ratio" \
  --argjson hybrid_ratio "$hybrid_ratio" \
  --argjson dead_pct "$dead_pct" \
  --argjson cache_mb "$( [[ "$cache_mb" == "null" ]] && echo null || echo "$cache_mb" )" \
  --argjson guard_ok "$guard_ok" \
  --argjson engine_ok "$( [[ "$engine_ok" == "null" ]] && echo null || ( [[ "$engine_ok" == true ]] && echo true || echo false ) )" \
  --argjson log_errors "$log_errors" \
  --argjson ripple_ext "$ripple_ext" \
  --argjson ripple_global "$ripple_global" \
  --argjson kg_sql_triples "$kg_sql_triples" \
  --argjson kg_ripple_triples "$kg_ripple_triples" \
  --argjson kg_drift_ok "$( [[ "$kg_drift_ok" == "null" ]] && echo null || ( [[ "$kg_drift_ok" == true ]] && echo true || echo false ) )" \
  --argjson failures "$failures" \
  --argjson warnings "$warn_json" \
  '{
    ts: $ts,
    target: {host: $host, port: $port, database: $db},
    checks: {
      postgres: $pg_ok,
      extensions: $ext_ok,
      idx_documents_fts_hybrid: $hybrid_idx,
      idx_content_vectors_turbo: $dense_idx,
      scalar_guard: $guard_ok,
      engine_health: $engine_ok,
      pg_log_errors_15m: $log_errors,
      pg_ripple: ($ripple_ext == 1),
      kg_drift: $kg_drift_ok
    },
    metrics: {
      embed_coverage: $embed_ratio,
      hybrid_embedding_coverage: $hybrid_ratio,
      documents_fts_dead_pct: $dead_pct,
      cache_mb: $cache_mb,
      ripple_triples_global: $ripple_global,
      kg_sql_triples: $kg_sql_triples,
      kg_ripple_triples: $kg_ripple_triples
    },
    failures: $failures,
    warnings: $warnings,
    ok: ($failures == 0)
  }')"

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  echo "$json" >>"$OUT"
fi

echo "$json"
exit "$failures"