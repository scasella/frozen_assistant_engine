"""Condition definitions and the cached-log consistency guard."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from eval.conditions import ALL_CONDITIONS, engine_config, load_hparams, relevant_config

ROOT = Path(__file__).resolve().parents[1]
HP = load_hparams(ROOT / "configs" / "hparams.json")
MODELS = ["Qwen__Qwen2.5-3B", "Qwen__Qwen2.5-1.5B"]


def test_nineteen_conditions():
    assert len(ALL_CONDITIONS) == 19


def test_v2_base_resolution():
    cfg = engine_config(ALL_CONDITIONS["9_engine_v2_full"][2], HP)
    assert cfg.erase_mode == "hinge" and cfg.signed_attention and cfg.unembed_project
    cfg10 = engine_config(ALL_CONDITIONS["10_engine_v2_1"][2], HP)
    assert not cfg10.signed_attention and cfg10.sibling_orth and cfg10.unembed_center and cfg10.stop_min_step == 6


def test_relevant_config_drops_unused_parameters():
    rel = relevant_config(asdict(engine_config(ALL_CONDITIONS["8_engine_v2_signed_attn"][2], HP)))
    assert "gamma_unembed" not in rel and "beta_neg_attn" in rel


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("cond", [c for c, (_, _, t) in ALL_CONDITIONS.items() if t is not None])
def test_cached_logs_match_current_config(model, cond):
    f = ROOT / "results" / model / "test40" / f"{cond}.jsonl"
    assert f.exists(), f"missing cached log {f.relative_to(ROOT)}"
    records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    assert len(records) == 40
    current = relevant_config(asdict(engine_config(ALL_CONDITIONS[cond][2], HP)))
    for rec in records:
        stored = relevant_config(rec["engine_config"])
        assert {k: v for k, v in stored.items() if current.get(k) != v} == {}
