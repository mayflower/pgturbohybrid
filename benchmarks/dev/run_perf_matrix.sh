#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="${FIQA_DATASET:-}"
DATABASE="${PGDATABASE:-pgturbohybrid_perf}"
PG_CONFIG="${PG_CONFIG:-pg_config}"
PGVECTOR_REF="${PGVECTOR_REF:-v0.8.2}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results}"
DENSE_K="${DENSE_K:-100}"
BM25_K="${BM25_K:-100}"
RRF_K="${RRF_K:-60}"
FINAL_K="${FINAL_K:-10}"
WARMUP="${WARMUP:-1}"
MEASURE="${MEASURE:-5}"
PLANNER_DEBUG="${PLANNER_DEBUG:-1}"
PROFILE="${PROFILE:-latency}"
DEV_DIAGNOSTICS="${DEV_DIAGNOSTICS:-0}"
DEV_DIAGNOSTICS_SQL="${DEV_DIAGNOSTICS_SQL:-sql/pgturbohybrid_dev_diagnostics.sql}"
INSTALL_PGVECTOR="${INSTALL_PGVECTOR:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_JSON="${OUTPUT:-$RESULT_DIR/fiqa_perf_matrix_${STAMP}.json}"
if [[ -n "${OUTPUT:-}" ]]; then
	CATEGORY_TXT="${OUTPUT%.*}_categories.txt"
else
	CATEGORY_TXT="$RESULT_DIR/fiqa_perf_categories_${STAMP}.txt"
fi

if [[ -z "$DATASET" ]]; then
	echo "set FIQA_DATASET to a FIQA/OpenAI dataset directory" >&2
	exit 1
fi

for file in corpus.jsonl corpus_embeddings.jsonl queries.jsonl query_embeddings.jsonl qrels/test.tsv; do
	if [[ ! -f "$DATASET/$file" ]]; then
		echo "missing real FIQA/OpenAI dataset file: $DATASET/$file" >&2
		exit 1
	fi
done

mkdir -p "$RESULT_DIR"
mkdir -p "$(dirname "$SUMMARY_JSON")" "$(dirname "$CATEGORY_TXT")"

if [[ "$INSTALL_PGVECTOR" == "1" ]]; then
	PG_CONFIG="$PG_CONFIG" PGVECTOR_REF="$PGVECTOR_REF" scripts/install-pgvector.sh
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
	make PG_CONFIG="$PG_CONFIG" install
fi

dropdb --if-exists "$DATABASE"
createdb "$DATABASE"

PYTHON_ARGS=(
	--database "$DATABASE"
	--dataset "$DATASET"
	--dense-k "$DENSE_K"
	--bm25-k "$BM25_K"
	--final-k "$FINAL_K"
	--profile "$PROFILE"
	--warmup "$WARMUP"
	--methods "pgvector_hnsw_dense_only,postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit"
	--explain
	--output "$SUMMARY_JSON"
)

if [[ "$DEV_DIAGNOSTICS" == "1" ]]; then
	if [[ ! -f "$DEV_DIAGNOSTICS_SQL" ]]; then
		echo "DEV_DIAGNOSTICS=1 requires DEV_DIAGNOSTICS_SQL to point to an existing SQL file" >&2
		exit 1
	fi
	PYTHON_ARGS+=(--bm25-cache-probe --dev-diagnostics-sql "$DEV_DIAGNOSTICS_SQL")
fi

python3 benchmarks/fiqa_openai.py "${PYTHON_ARGS[@]}"

psql -X -v ON_ERROR_STOP=1 -d "$DATABASE" -f benchmarks/dev/setup_perf_db.sql

run_category_matrix() {
	local label="$1"
	shift
	psql -X -v ON_ERROR_STOP=1 \
		-v label="${label}_${STAMP}" \
		-v dataset="fiqa-openai" \
		-v dense_k="$DENSE_K" \
		-v bm25_k="$BM25_K" \
		-v rrf_k="$RRF_K" \
		-v final_k="$FINAL_K" \
		-v warmup="$WARMUP" \
		-v measure="$MEASURE" \
		-v planner_debug="$PLANNER_DEBUG" \
		"$@" \
		-d "$DATABASE" \
		-f benchmarks/dev/run_queries.sql
}

{
	run_category_matrix exact_storage_off_balanced -v profile=balanced
	run_category_matrix exact_storage_off_latency -v profile=latency
	run_category_matrix exact_storage_off_latency_cache_off \
		-v profile=latency \
		-v bm25_hot_postings_cache_mb=0
	run_category_matrix exact_storage_off_latency_impact_or_off \
		-v profile=latency \
		-v bm25_impact_or_mode=off
	run_category_matrix exact_storage_off_latency_wand_off \
		-v profile=latency \
		-v enable_wand=off
} | tee "$CATEGORY_TXT"

if [[ -n "${OLD_PGVECTOR_REF:-}" ]]; then
	echo "OLD_PGVECTOR_REF=$OLD_PGVECTOR_REF was provided." | tee -a "$CATEGORY_TXT"
	echo "Old patched-branch comparison is handled by compare_old_branch.sh and is not run by this local matrix yet." | tee -a "$CATEGORY_TXT"
fi

echo "$SUMMARY_JSON"
echo "$CATEGORY_TXT"
