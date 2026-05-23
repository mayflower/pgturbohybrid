#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONFIG="${PG_CONFIG:-pg_config}"
PGVECTOR_REF="${PGVECTOR_REF:-v0.8.2}"
DEPS_DIR="${DEPS_DIR:-$ROOT_DIR/.deps}"
PGVECTOR_DIR="${PGVECTOR_DIR:-$DEPS_DIR/pgvector-$PGVECTOR_REF}"

"$ROOT_DIR/scripts/build-pgvector.sh"

make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" install
