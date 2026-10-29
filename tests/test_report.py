"""RESULTS.md regenerates from the logged generations and matches the recorded summaries."""

import json
from pathlib import Path

import pytest

from eval.conditions import summarize
from eval.report import _load, write_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["Qwen__Qwen2.5-3B", "Qwen__Qwen2.5-1.5B"])
def test_rescored_summaries_match_recorded(model):
    out_dir = ROOT / "results" / model / "test40"
    recorded = json.loads((out_dir / "summary.json").read_text())["summary"]
    _, recs = _load(out_dir)
    for cond, rs in recs.items():
        assert summarize(rs)["composite"] == pytest.approx(recorded[cond]["composite"]), cond


def test_report_regenerates(tmp_path):
    out = tmp_path / "RESULTS.md"
    write_report(ROOT / "results" / "Qwen__Qwen2.5-3B" / "test40", out,
                 [ROOT / "results" / "Qwen__Qwen2.5-1.5B" / "test40"])
    text = out.read_text()
    assert "| `10_engine_v2_1` | 100.0 | 87.5 | 92.5 | 97.5 | **94.4** |" in text
    assert "| `1_raw_base` | 100.0 | 75.0 | 52.5 | 95.0 | **80.6** |" in text
    assert "## Appendix: other runs" in text
