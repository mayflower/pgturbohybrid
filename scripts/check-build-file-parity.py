#!/usr/bin/env python3
"""check-build-file-parity.py

Guard against drift between the Unix `Makefile` and the Windows `Makefile.win`
object lists. Both must compile the same `pgturbohybrid_*.c` sources, EXCEPT for
a small, documented set of architecture-specific SIMD kernels that MSVC cannot
build (the Windows build uses the portable scalar fallbacks instead).

Fails if:
  - a source object is in one Makefile but not the other AND is not in the
    documented WINDOWS_OMITTED allowlist,
  - an OBJS entry has no matching src/<name>.c source file, or
  - Makefile.win runs a regression test the Unix Makefile does not.

The Unix > Windows regression *reduction* is expected and is not flagged:
Windows is a documented reduced-coverage build profile (see docs/compatibility.md).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Architecture-specific SIMD kernels intentionally absent from the MSVC build.
# These use GCC/clang __attribute__((target(...))) + __builtin_cpu_supports,
# which MSVC does not support; their callers fall back to the scalar paths in
# pgturbohybrid_quant_score / pgturbohybrid_sparse_score.
WINDOWS_OMITTED = {
    "pgturbohybrid_quant_score_u8_x86",
    "pgturbohybrid_quant_score_signed_x86",
    "pgturbohybrid_quant_score_arm",
    "pgturbohybrid_sparse_simd_x86",
    "pgturbohybrid_sparse_simd_arm",
}


def objs(path, ext):
    text = (ROOT / path).read_text(encoding="utf-8")
    return set(re.findall(r"src[\\/](pgturbohybrid[A-Za-z0-9_]+)\." + ext + r"\b", text))


def regress(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(r"^REGRESS\s*=\s*(.+)$", text, re.M)
    return set(m.group(1).split()) if m else set()


def main():
    unix = objs("Makefile", "o")
    win = objs("Makefile.win", "obj")
    errors = []

    unix_only = unix - win
    win_only = win - unix

    unexpected = unix_only - WINDOWS_OMITTED
    if unexpected:
        errors.append("objects in Makefile but missing from Makefile.win, and not "
                      "in the documented Windows-omitted allowlist: "
                      + ", ".join(sorted(unexpected)))
    if win_only:
        errors.append("objects in Makefile.win but missing from Makefile: "
                      + ", ".join(sorted(win_only)))
    stale = WINDOWS_OMITTED - unix_only
    if stale:
        errors.append("WINDOWS_OMITTED allowlist entries no longer Unix-only "
                      "(update this script): " + ", ".join(sorted(stale)))

    for obj in sorted(unix | win):
        if not (ROOT / "src" / (obj + ".c")).exists():
            errors.append(f"OBJS references a missing source file: src/{obj}.c")

    win_extra = regress("Makefile.win") - regress("Makefile")
    if win_extra:
        errors.append("Makefile.win REGRESS runs tests absent from the Unix REGRESS: "
                      + ", ".join(sorted(win_extra)))

    if errors:
        print("check-build-file-parity: FAILED", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        sys.exit(1)

    print(f"check-build-file-parity: ok ({len(unix)} Unix objects, {len(win)} Windows "
          f"objects; {len(WINDOWS_OMITTED)} intentionally Windows-omitted SIMD kernels)")


if __name__ == "__main__":
    main()
