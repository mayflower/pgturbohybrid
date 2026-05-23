#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONFIG="${PG_CONFIG:-pg_config}"

make -C "$ROOT_DIR" PG_CONFIG="$PG_CONFIG" installcheck
