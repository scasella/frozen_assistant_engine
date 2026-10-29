"""RESULTS.md regenerates from the logged generations and matches the recorded summaries."""

import json
from pathlib import Path

import pytest

from eval.conditions import summarize
from eval.report import _load, recovery_ci, write_report

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
    assert "| `11_engine_v2_2_lean` |" in text and "## 7. Economics versus post-training" in text
    # fp32-score sensitivity rows: paired change vs. the protocol stack, and recovery on a matched 512-token stack
    assert "| `10_engine_v2_1` | 512 tokens | 87.5 | 90.0 | +2.5 [-3.1, +8.1] | 4/8 → 7/8 | 6/40 |" in text
    assert ("| fp32-score attention, 512 tokens | `10_engine_v2_1` | `12_official_instruct_chat` | +18.1 [+11.2, +25.0] "
            "| -6.9 [-12.5, -1.9] | **62%** [28%, 90%] |") in text
    assert "fp32-score attention, 160 tokens; Instruct from `test40_sdpa`" in text


def _recs(name, scores):
    return [{"prompt_id": f"p{i}", "metrics": {"composite": v}} for i, v in enumerate(scores)]


def test_recovery_ratio():
    recs = {"raw": _recs("raw", [0.5] * 10), "eng": _recs("eng", [0.75] * 10), "inst": _recs("inst", [1.0] * 10)}
    rc = recovery_ci(recs, "raw", "eng", "inst", n_boot=200)
    assert rc["point"] == pytest.approx(0.5) and rc["lo"] == pytest.approx(0.5) and rc["undefined_frac"] == 0
    recs["inst"] = _recs("inst", [0.5] * 10)
    assert recovery_ci(recs, "raw", "eng", "inst", n_boot=50)["point"] is None
