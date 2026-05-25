#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="${FIQA_DATASET:?set FIQA_DATASET to a real FIQA/OpenAI-compatible dataset directory}"
OLD_PGVECTOR_REPO="${OLD_PGVECTOR_REPO:-https://github.com/mayflower/pgvector.git}"
OLD_PGVECTOR_REF="${OLD_PGVECTOR_REF:-}"
PGVECTOR_BASELINE_REF="${PGVECTOR_BASELINE_REF:-v0.8.2}"
PGTURBOHYBRID_REF="${PGTURBOHYBRID_REF:-current-worktree}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results}"
WORK_DIR="${WORK_DIR:-$(mktemp -d /tmp/pgturbohybrid-old-compare.XXXXXX)}"
MAX_DOCS="${MAX_DOCS:-0}"
MAX_QUERIES="${MAX_QUERIES:-0}"
DENSE_K="${DENSE_K:-100}"
BM25_K="${BM25_K:-100}"
FINAL_K="${FINAL_K:-10}"
WARMUP="${WARMUP:-1}"
PROFILE="${PROFILE:-latency}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$RESULT_DIR" "$WORK_DIR"

for file in corpus.jsonl corpus_embeddings.jsonl queries.jsonl query_embeddings.jsonl qrels/test.tsv; do
	if [[ ! -f "$DATASET/$file" ]]; then
		echo "missing real FIQA/OpenAI dataset file: $DATASET/$file" >&2
		exit 2
	fi
done

if [[ "$PGTURBOHYBRID_REF" != "current-worktree" ]]; then
	echo "PGTURBOHYBRID_REF=$PGTURBOHYBRID_REF is recorded as metadata only; run this script from the checked-out pgturbohybrid worktree to benchmark it." >&2
fi

run_fiqa() {
	local database="$1"
	local output="$2"
	local methods="$3"
	local extension_mode="$4"

	dropdb --if-exists "$database"
	createdb "$database"

	python3 benchmarks/fiqa_openai.py \
		--database "$database" \
		--dataset "$DATASET" \
		--methods "$methods" \
		--dense-k "$DENSE_K" \
		--bm25-k "$BM25_K" \
		--final-k "$FINAL_K" \
		--warmup "$WARMUP" \
		--profile "$PROFILE" \
		--max-docs "$MAX_DOCS" \
		--max-queries "$MAX_QUERIES" \
		--budget-matrix \
		--explain \
		--force-turbohybrid-index \
		--extension-mode "$extension_mode" \
		--output "$output"
}

clone_pgvector() {
	local ref="$1"
	local target="$2"

	git clone --quiet "$OLD_PGVECTOR_REPO" "$target"
	git -C "$target" checkout --quiet "$ref"
}

echo "Building pgvector baseline $PGVECTOR_BASELINE_REF"
BASELINE_DIR="$WORK_DIR/pgvector-baseline"
clone_pgvector "$PGVECTOR_BASELINE_REF" "$BASELINE_DIR"
make -C "$BASELINE_DIR" clean
make -C "$BASELINE_DIR"
make -C "$BASELINE_DIR" install

echo "Building current pgturbohybrid worktree"
make clean
make
make install

CURRENT_JSON="$RESULT_DIR/current_pgturbohybrid_${STAMP}.json"
run_fiqa "pgturbohybrid_current_${STAMP}" "$CURRENT_JSON" \
	"pgvector_hnsw_dense_only,postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit" \
	"pgturbohybrid"

OLD_JSON=""
if [[ -n "$OLD_PGVECTOR_REF" ]]; then
	echo "Building old patched pgvector $OLD_PGVECTOR_REF"
	OLD_DIR="$WORK_DIR/pgvector-old"
	clone_pgvector "$OLD_PGVECTOR_REF" "$OLD_DIR"
	make -C "$OLD_DIR" clean
	make -C "$OLD_DIR"
	make -C "$OLD_DIR" install

	OLD_JSON="$RESULT_DIR/old_patched_pgvector_${STAMP}.json"
	run_fiqa "pgturbohybrid_old_${STAMP}" "$OLD_JSON" \
		"pgturbohybrid,pgturbohybrid_recovered_explicit" \
		"patched_pgvector"
else
	echo "OLD_PGVECTOR_REF is not set; skipping old patched pgvector run" >&2
fi

SUMMARY_MD="$RESULT_DIR/old_vs_new_${STAMP}.md"
python3 - "$SUMMARY_MD" "$CURRENT_JSON" "$OLD_JSON" "$OLD_PGVECTOR_REF" "$PGVECTOR_BASELINE_REF" "$PGTURBOHYBRID_REF" <<'PY'
import json
import sys
from pathlib import Path

summary, current_path, old_path, old_ref, baseline_ref, pgturbohybrid_ref = sys.argv[1:]

def rows(path):
    if not path:
        return []
    payload = json.loads(Path(path).read_text())
    out = []
    for item in payload.get("results", []):
        latency = item.get("latency", {})
        metrics = item.get("metrics", {})
        out.append([
            Path(path).name,
            item.get("method", ""),
            latency.get("p50_ms", ""),
            latency.get("p95_ms", ""),
            latency.get("p99_ms", ""),
            metrics.get("ndcg@10", ""),
            metrics.get("recall@10", ""),
            item.get("index_size_bytes", ""),
            item.get("build_ms", ""),
        ])
    return out

lines = [
    "# Old vs Current FIQA Comparison",
    "",
    f"- pgvector baseline ref: `{baseline_ref}`",
    f"- current pgturbohybrid ref: `{pgturbohybrid_ref}`",
    f"- old patched pgvector ref: `{old_ref or 'not run'}`",
    "",
    "| source | method | p50 ms | p95 ms | p99 ms | nDCG@10 | recall@10 | index bytes | build ms |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows(current_path) + rows(old_path):
    lines.append("| " + " | ".join(str(value) for value in row) + " |")

Path(summary).write_text("\n".join(lines) + "\n")
print(summary)
PY

echo "wrote $SUMMARY_MD"
