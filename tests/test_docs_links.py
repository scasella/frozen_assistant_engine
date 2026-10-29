import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_relative_markdown_links_resolve():
    spec = importlib.util.spec_from_file_location("check_links", ROOT / "scripts" / "check_links.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.broken_links() == []
