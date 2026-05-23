#!/usr/bin/env bash
set -euo pipefail

DBNAME="${DBNAME:-pgturbohybrid_dev}"

dropdb --if-exists "$DBNAME"
