"""Verify that files referenced from the docs are present *and tracked by git*.

A plain existence check is not enough. `.gitignore` once carried an unanchored
``results/`` pattern, which also matched ``docs/results/`` -- so committed run
evidence was silently dropped while every local path still resolved. The docs
looked fine to anyone who had run the training and were broken in a fresh clone.

Usage:
    python scripts/check_docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Markdown links to repo files (not URLs, not bare anchors).
_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)\s]+)\)")
_SKIP = ("http://", "https://", "mailto:", "#")


def tracked_files() -> set[Path]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {Path(p) for p in out}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tracked = tracked_files()
    missing: list[str] = []
    untracked: list[str] = []

    for md in sorted(root.glob("docs/**/*.md")) + [root / "README.md"]:
        if not md.exists():
            continue
        for m in _LINK.finditer(md.read_text(encoding="utf-8")):
            target = m.group("target").split("#")[0]
            if not target or target.startswith(_SKIP):
                continue
            resolved = (md.parent / target).resolve()
            rel_md = md.relative_to(root).as_posix()
            if not resolved.exists():
                missing.append(f"{rel_md} -> {target}")
                continue
            if resolved.is_dir():
                continue
            rel = resolved.relative_to(root)
            if rel not in tracked:
                untracked.append(f"{rel_md} -> {target}")

    for label, items in (("MISSING", missing), ("NOT TRACKED BY GIT", untracked)):
        if items:
            print(f"\n{label}:")
            for i in items:
                print(f"  {i}")

    if missing or untracked:
        print(f"\n{len(missing)} missing, {len(untracked)} untracked")
        return 1
    print("all doc references exist and are tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
