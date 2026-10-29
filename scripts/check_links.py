"""Check that every relative markdown link in the repository's docs resolves to an existing file.

    python scripts/check_links.py            # exit code 1 if any link is broken
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache"}


def markdown_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.md") if not SKIP_DIRS & set(p.relative_to(ROOT).parts))


def broken_links() -> list[tuple[Path, str]]:
    bad = []
    for md in markdown_files():
        text = md.read_text()
        text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)  # ignore code blocks
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+://|^mailto:|^#", target):
                continue
            path = target.split("#", 1)[0]
            if path and not (md.parent / path).exists():
                bad.append((md.relative_to(ROOT), target))
    return bad


def main() -> int:
    bad = broken_links()
    for md, target in bad:
        print(f"BROKEN {md}: {target}")
    print(f"checked {len(markdown_files())} markdown files, {len(bad)} broken links")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
