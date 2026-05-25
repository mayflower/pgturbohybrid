#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="${FIQA_DATASET:?set FIQA_DATASET to a real FIQA/OpenAI-compatible dataset directory}"
DATABASE="${PGDATABASE:-pgturbohybrid_fiqa_quick}"
PG_CONFIG="${PG_CONFIG:-pg_config}"
PGVECTOR_REF="${PGVECTOR_REF:-v0.8.2}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results}"
MAX_DOCS="${MAX_DOCS:-5000}"
MAX_QUERIES="${MAX_QUERIES:-50}"
PROFILE="${PROFILE:-latency}"
DENSE_K="${DENSE_K:-100}"
BM25_K="${BM25_K:-100}"
FINAL_K="${FINAL_K:-10}"
WARMUP="${WARMUP:-1}"
METHODS="${METHODS:-pgvector_hnsw_dense_only,postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit}"
BM25_CACHE_PROBE="${BM25_CACHE_PROBE:-0}"
DEV_DIAGNOSTICS_SQL="${DEV_DIAGNOSTICS_SQL:-sql/pgturbohybrid_dev_diagnostics.sql}"
FORCE_TURBOHYBRID_INDEX="${FORCE_TURBOHYBRID_INDEX:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${OUTPUT:-$RESULT_DIR/fiqa_openai_quick_${STAMP}.json}"
CREATE_DATABASE="${CREATE_DATABASE:-1}"
INSTALL_PGVECTOR="${INSTALL_PGVECTOR:-0}"
BUILD_PGTURBOHYBRID="${BUILD_PGTURBOHYBRID:-0}"

mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$OUTPUT")"

if [[ "$INSTALL_PGVECTOR" == "1" ]]; then
	PG_CONFIG="$PG_CONFIG" PGVECTOR_REF="$PGVECTOR_REF" scripts/install-pgvector.sh
fi

if [[ "$BUILD_PGTURBOHYBRID" == "1" ]]; then
	make PG_CONFIG="$PG_CONFIG" clean
	make PG_CONFIG="$PG_CONFIG"
	make PG_CONFIG="$PG_CONFIG" install
fi

if [[ "$CREATE_DATABASE" == "1" ]]; then
	dropdb --if-exists "$DATABASE"
	createdb "$DATABASE"
fi

ARGS=(
	--database "$DATABASE" \
	--dataset "$DATASET" \
	--max-docs "$MAX_DOCS" \
	--max-queries "$MAX_QUERIES" \
	--profile "$PROFILE" \
	--dense-k "$DENSE_K" \
	--bm25-k "$BM25_K" \
	--final-k "$FINAL_K" \
	--warmup "$WARMUP" \
	--methods "$METHODS" \
	--budget-matrix \
	--explain \
	--output "$OUTPUT"
)

if [[ "$FORCE_TURBOHYBRID_INDEX" == "1" ]]; then
	ARGS+=(--force-turbohybrid-index)
fi

if [[ "$BM25_CACHE_PROBE" == "1" ]]; then
	ARGS+=(--bm25-cache-probe --dev-diagnostics-sql "$DEV_DIAGNOSTICS_SQL")
fi

python3 benchmarks/fiqa_openai.py "${ARGS[@]}"
