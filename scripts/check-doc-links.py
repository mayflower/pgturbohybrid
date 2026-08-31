#!/usr/bin/env python3
"""check-doc-links.py

Verify that every *local* Markdown link and image in the tracked docs resolves.
Covers all tracked .md files (root + docs/**), not just README.md, so doc moves
or renames cannot silently leave dangling links. Relative links are resolved
against the linking file's directory; in-page #anchors and http(s)/mailto links
are not followed. No database or network required.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"')


def tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, text=True,
                         capture_output=True, check=True).stdout.split()
    return [ROOT / p for p in out if (ROOT / p).is_file()]


def main():
    missing = []
    checked = 0
    for md in tracked_markdown():
        text = md.read_text(encoding="utf-8")
        base = md.parent
        targets = LINK_RE.findall(text) + IMG_RE.findall(text)
        for target in targets:
            t = target.strip()
            if t.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = t.split("#", 1)[0]
            if not path:
                continue
            checked += 1
            if not (base / path).exists():
                missing.append(f"{md.relative_to(ROOT)} -> {target}")

    if missing:
        print("check-doc-links: FAILED -- local links pointing at missing files:", file=sys.stderr)
        for m in missing:
            print("  - " + m, file=sys.stderr)
        sys.exit(1)

    print(f"check-doc-links: ok ({checked} local links across "
          f"{len(tracked_markdown())} markdown files resolve)")


if __name__ == "__main__":
    main()
