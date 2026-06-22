#!/usr/bin/env bash
#
# check-version-consistency.sh
#
# Verify that version strings and supported-PostgreSQL wording agree across the
# files that humans and packagers read:
#
#   - Makefile EXTVERSION  ==  pgturbohybrid.control default_version
#   - META.json version (and provides.*.version / .file)  ==  EXTVERSION
#   - README.md, RELEASE.md, docs/compatibility.md  agree on the supported
#     PostgreSQL major-version range (docs/compatibility.md is authoritative)
#   - When GITHUB_REF_NAME is set (a tag build), a GitHub release-notes file
#     exists for that tag.
#
# The SQL extension version may stay 0.1.0 even when the Git tag is
# v0.1.0-alpha.N; this script checks the SQL/package version, not the Git tag.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

status=0
fail() {
	echo "check-version-consistency: $*" >&2
	status=1
}

# --- extension / package version --------------------------------------------
extversion="$(sed -n 's/^EXTVERSION[[:space:]]*=[[:space:]]*//p' Makefile | head -n1)"
if [[ -z "$extversion" ]]; then
	fail "could not read EXTVERSION from Makefile"
fi

control_version="$(sed -n "s/^default_version[[:space:]]*=[[:space:]]*'\\([^']*\\)'.*/\\1/p" pgturbohybrid.control | head -n1)"
if [[ -z "$control_version" ]]; then
	fail "could not read default_version from pgturbohybrid.control"
fi

if [[ -n "$extversion" && -n "$control_version" && "$extversion" != "$control_version" ]]; then
	fail "Makefile EXTVERSION ($extversion) != pgturbohybrid.control default_version ($control_version)"
fi

# META.json version, provides.*.version, and provides.*.file must match EXTVERSION.
if ! python3 - "$extversion" <<'PY'
import json, sys
extversion = sys.argv[1]
with open("META.json", encoding="utf-8") as fh:
    meta = json.load(fh)
ok = True
if meta.get("version") != extversion:
    print(f"META.json version {meta.get('version')!r} != EXTVERSION {extversion!r}", file=sys.stderr)
    ok = False
for name, prov in (meta.get("provides") or {}).items():
    if prov.get("version") != extversion:
        print(f"META.json provides.{name}.version {prov.get('version')!r} != EXTVERSION {extversion!r}", file=sys.stderr)
        ok = False
    expected_file = f"sql/pgturbohybrid--{extversion}.sql"
    if prov.get("file") and prov["file"] != expected_file:
        print(f"META.json provides.{name}.file {prov['file']!r} != {expected_file!r}", file=sys.stderr)
        ok = False
sys.exit(0 if ok else 1)
PY
then
	fail "META.json version/provides do not match EXTVERSION ($extversion)"
fi

# The install script named by EXTVERSION must exist.
if [[ ! -f "sql/pgturbohybrid--${extversion}.sql" ]]; then
	fail "install script sql/pgturbohybrid--${extversion}.sql does not exist"
fi

# --- supported PostgreSQL wording -------------------------------------------
# docs/compatibility.md is the authoritative range; README and RELEASE must
# phrase it consistently and nobody may carry the stale "14 through 18".
if ! grep -qF '14, 15, 16, 17, 18, 19' docs/compatibility.md; then
	fail "docs/compatibility.md is missing the canonical range '14, 15, 16, 17, 18, 19'"
fi

for f in README.md RELEASE.md SUPPORT.md; do
	if ! grep -qF '14 through 19' "$f"; then
		fail "$f is missing the supported-PostgreSQL wording '14 through 19'"
	fi
done

if stale="$(grep -rnF '14 through 18' README.md RELEASE.md SUPPORT.md docs/compatibility.md 2>/dev/null)"; then
	printf '%s\n' "$stale" >&2
	fail "stale 'PostgreSQL 14 through 18' wording found (supported range is 14 through 19)"
fi

# META.json runtime prerequisite must require PostgreSQL 14.x.
if ! grep -qE '"PostgreSQL"[[:space:]]*:[[:space:]]*"14\.' META.json; then
	fail "META.json runtime prereq PostgreSQL is not 14.x"
fi

# --- release notes for the current tag (CI only) ----------------------------
if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
	notes="docs/release-notes/github-${GITHUB_REF_NAME}.md"
	if [[ ! -f "$notes" ]]; then
		fail "release notes file missing for tag ${GITHUB_REF_NAME}: create $notes"
	fi
fi

if [[ "$status" -ne 0 ]]; then
	echo "check-version-consistency: FAILED" >&2
	exit 1
fi

echo "check-version-consistency: ok (version ${extversion}, PostgreSQL 14 through 19)"
