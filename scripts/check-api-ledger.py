#!/usr/bin/env python3
"""check-api-ledger.py

Keep the machine-readable public-API ledger (docs/api-ledger.json) in lockstep
with the live catalog snapshot captured by the pgturbohybrid_api_ledger
regression test (test/expected/pgturbohybrid_api_ledger.out).

The regression test snapshots the installed catalog (types, operators,
opclasses, turbohybrid_* functions) with the maturity label parsed from each
object's COMMENT ON, so a public-API addition/removal/relabel changes that
expected file. This script then fails if docs/api-ledger.json does not match the
snapshot exactly -- so the ledger cannot silently drift from the real catalog,
and a new turbohybrid_* public object cannot exist without a ledger entry.

It also checks that diagnostics_stable_keys in the ledger matches the
PGTURBOHYBRID_DIAG_KEY_* constants in src/pgturbohybrid_diagnostics.h.

File-vs-file only; no database required.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "test/expected/pgturbohybrid_api_ledger.out"
LEDGER = ROOT / "docs/api-ledger.json"
DIAG_H = ROOT / "src/pgturbohybrid_diagnostics.h"

# snapshot "kind" -> ledger top-level key
KINDS = {"type": "types", "operator": "operators", "opclass": "opclasses", "function": "functions"}


def parse_snapshot():
    out = {ext: {v: {} for v in KINDS.values()}
           for ext in ("pgturbohybrid", "pgturbohybrid_experimental")}
    for line in SNAPSHOT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(pgturbohybrid(?:_experimental)?)\|(type|operator|opclass|function)\|(.+)\|([a-z][a-z -]*)\s*$", line)
        if not m:
            continue
        ext, kind, name, mat = m.groups()
        key = KINDS[kind]
        out[ext][key].setdefault(name, set()).add(mat)
    # collapse single-element sets to a str, multi to a sorted list (matches ledger)
    return {ext: {k: {n: (sorted(v) if len(v) > 1 else next(iter(v)))
                      for n, v in d.items()} for k, d in kinds.items()}
            for ext, kinds in out.items()}


def parse_ledger():
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    extensions = led.get("extensions", {})
    return {ext: {key: extensions.get(ext, {}).get(key, {}) for key in KINDS.values()}
            for ext in ("pgturbohybrid", "pgturbohybrid_experimental")}, led


def main():
    snap = parse_snapshot()
    led, led_full = parse_ledger()
    errors = []

    for ext in snap:
      for key in KINDS.values():
        s, l = snap[ext][key], led[ext][key]
        for name in sorted(set(s) - set(l)):
            errors.append(f"{ext}.{key}: '{name}' is in the catalog snapshot but missing from docs/api-ledger.json")
        for name in sorted(set(l) - set(s)):
            errors.append(f"{ext}.{key}: '{name}' is in docs/api-ledger.json but not in the catalog snapshot")
        for name in sorted(set(s) & set(l)):
            sv = s[name] if isinstance(s[name], str) else sorted(s[name])
            lv = l[name] if isinstance(l[name], str) else sorted(l[name])
            if sv != lv:
                errors.append(f"{ext}.{key}: '{name}' maturity differs (snapshot={sv!r}, ledger={lv!r})")

    # diagnostics stable keys must match the C header constants.
    header_keys = sorted(re.findall(r'#define PGTURBOHYBRID_DIAG_KEY_\w+\s+"([a-z0-9_]+)"',
                                    DIAG_H.read_text(encoding="utf-8")))
    ledger_keys = sorted(led_full.get("diagnostics_stable_keys", []))
    if header_keys != ledger_keys:
        errors.append("diagnostics_stable_keys in docs/api-ledger.json do not match the "
                      "PGTURBOHYBRID_DIAG_KEY_* constants in pgturbohybrid_diagnostics.h")

    if not SNAPSHOT.exists():
        errors.append(f"missing snapshot {SNAPSHOT} (run the pgturbohybrid_api_ledger test)")

    if errors:
        print("check-api-ledger: FAILED", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        print("  Fix: regenerate test/expected/pgturbohybrid_api_ledger.out and rerun "
              "scripts/gen-api-ledger (or update docs/api-ledger.json) so they agree.", file=sys.stderr)
        sys.exit(1)

    total = sum(len(snap[ext][k]) for ext in snap for k in KINDS.values())
    print(f"check-api-ledger: ok ({total} public SQL objects + {len(ledger_keys)} stable "
          f"diagnostic keys match docs/api-ledger.json)")


if __name__ == "__main__":
    main()
