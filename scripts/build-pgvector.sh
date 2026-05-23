#!/usr/bin/env bash
set -euo pipefail

PG_CONFIG="${PG_CONFIG:-pg_config}"
PGVECTOR_REF="${PGVECTOR_REF:-v0.8.2}"
PGVECTOR_REPO="${PGVECTOR_REPO:-https://github.com/pgvector/pgvector.git}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${DEPS_DIR:-$ROOT_DIR/.deps}"
PGVECTOR_DIR="${PGVECTOR_DIR:-$DEPS_DIR/pgvector-$PGVECTOR_REF}"

mkdir -p "$DEPS_DIR"

if [ ! -d "$PGVECTOR_DIR/.git" ]; then
	rm -rf "$PGVECTOR_DIR"
	git clone --depth 1 --branch "$PGVECTOR_REF" "$PGVECTOR_REPO" "$PGVECTOR_DIR"
fi

git -C "$PGVECTOR_DIR" fetch --depth 1 origin "$PGVECTOR_REF" >/dev/null 2>&1 || true
git -C "$PGVECTOR_DIR" reset --hard FETCH_HEAD >/dev/null 2>&1 || true
git -C "$PGVECTOR_DIR" clean -xfd

make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" clean
make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG"

printf '%s\n' "$PGVECTOR_DIR"
