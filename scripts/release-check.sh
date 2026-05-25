#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONFIG="${PG_CONFIG:-pg_config}"
INSTALL_COMMAND="${INSTALL_COMMAND:-make install}"

cd "$ROOT_DIR"

fail() {
	echo "release-check: $*" >&2
	exit 1
}

run() {
	printf '\n==> %s\n' "$*"
	"$@"
}

run_sh() {
	printf '\n==> %s\n' "$*"
	sh -c "$*"
}

require_clean_tree() {
	local status
	status="$(git status --short --untracked-files=all)"
	if [[ -n "$status" ]]; then
		printf '%s\n' "$status" >&2
		fail "working tree must be clean"
	fi
}

check_no_local_paths() {
	local host_matches tmp_matches disallowed_tmp

	host_matches="$(git grep -n -E '(/Volumes|/Users|/home/|\.cache/beir)' -- . ':!scripts/release-check.sh' || true)"
	if [[ -n "$host_matches" ]]; then
		printf '%s\n' "$host_matches" >&2
		fail "host-specific absolute paths found"
	fi

	tmp_matches="$(git grep -n '/tmp/' -- . ':!scripts/release-check.sh' || true)"
	disallowed_tmp="$(printf '%s\n' "$tmp_matches" | grep -v 'OUTPUT=/tmp/pgturbohybrid-fiqa\.json' || true)"
	if [[ -n "$disallowed_tmp" ]]; then
		printf '%s\n' "$disallowed_tmp" >&2
		fail "unexpected /tmp path found; use a documented environment variable"
	fi
}

check_no_generated_benchmark_artifacts() {
	local tracked

	tracked="$(git ls-files | grep -E '(^|/)regression\.(diffs|out)$|(^|/)\.DS_Store$|(^|/)(benchmarks/(results|output)|results)/|(^|/)perf-smoke-results\.json$|^benchmarks/.*\.(csv|md|json)$' | grep -v '^benchmarks/README\.md$' | grep -v '^benchmarks/config/.*\.json$' || true)"
	if [[ -n "$tracked" ]]; then
		printf '%s\n' "$tracked" >&2
		fail "generated benchmark or regression artifacts are tracked"
	fi
}

check_no_root_scratch_files() {
	local root_files tracked_files

	root_files="$(find . -maxdepth 1 -type f \( \
		-iname 'fixes*.md' -o \
		-iname 'prompts*.md' -o \
		-iname 'scratch*.md' -o \
		-name 'FAST_DEFAULTS_PLAN.md' -o \
		-name 'FAST_DEFAULTS_RELEASE_SUMMARY.md' -o \
		-name 'PERF_RECOVERY_SUMMARY.md' -o \
		-name 'PERF_REGRESSION_REPORT.md' \
	\) -print)"
	if [[ -n "$root_files" ]]; then
		printf '%s\n' "$root_files" >&2
		fail "root scratch or planning files found"
	fi

	tracked_files="$(git ls-files | grep -E '^(fixes|prompts|scratch).*\.md$|^(FAST_DEFAULTS_PLAN|FAST_DEFAULTS_RELEASE_SUMMARY|PERF_RECOVERY_SUMMARY|PERF_REGRESSION_REPORT)\.md$' || true)"
	if [[ -n "$tracked_files" ]]; then
		printf '%s\n' "$tracked_files" >&2
		fail "root scratch or planning files are tracked"
	fi
}

check_readme_links() {
	python3 - <<'PY'
from pathlib import Path
import re
import sys

readme = Path("README.md")
text = readme.read_text(encoding="utf-8")
targets = []

targets.extend(match.group(1) for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text))
targets.extend(match.group(1) for match in re.finditer(r'<img[^>]+src="([^"]+)"', text))

missing = []
for target in targets:
    if target.startswith(("http://", "https://", "mailto:")):
        continue
    path = target.split("#", 1)[0]
    if not path:
        continue
    if not Path(path).exists():
        missing.append(target)

if missing:
    print("README links point at missing local files:", file=sys.stderr)
    for target in missing:
        print(f"  {target}", file=sys.stderr)
    sys.exit(1)
PY
}

printf '==> checking clean tree\n'
require_clean_tree

printf '\n==> checking release hygiene\n'
check_no_local_paths
check_no_generated_benchmark_artifacts
check_no_root_scratch_files
check_readme_links

run make PG_CONFIG="$PG_CONFIG" clean
run make PG_CONFIG="$PG_CONFIG"

if [[ "${RELEASE_CHECK_SKIP_INSTALL:-0}" != "1" ]]; then
	run_sh "PG_CONFIG='$PG_CONFIG' $INSTALL_COMMAND"
fi

run make PG_CONFIG="$PG_CONFIG" installcheck
run make PG_CONFIG="$PG_CONFIG" prove_installcheck
run make PG_CONFIG="$PG_CONFIG" dist

printf '\n==> checking final clean tree\n'
require_clean_tree

printf '\nrelease-check: ok\n'
