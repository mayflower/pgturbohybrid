#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

extversion="$(sed -n 's/^EXTVERSION[[:space:]]*=[[:space:]]*//p' Makefile | head -n1)"
control_version="$(sed -n "s/^default_version[[:space:]]*=[[:space:]]*'\\([^']*\\)'.*/\\1/p" pgturbohybrid.control | head -n1)"

if [[ -z "$extversion" || "$extversion" != "$control_version" ]]; then
	echo "version mismatch: Makefile=$extversion control=$control_version" >&2
	exit 1
fi

if [[ ! -f "sql/pgturbohybrid--${extversion}.sql" ]]; then
	echo "missing sql/pgturbohybrid--${extversion}.sql" >&2
	exit 1
fi

python3 - "$extversion" <<'PY'
import json
import sys

version = sys.argv[1]
with open("META.json", encoding="utf-8") as source:
    metadata = json.load(source)

if metadata.get("version") != version:
    raise SystemExit("META.json version does not match Makefile")

for name, provided in (metadata.get("provides") or {}).items():
    if provided.get("version") != version:
        raise SystemExit(f"META.json provides.{name}.version does not match Makefile")
PY

echo "version consistency: ok ($extversion)"
