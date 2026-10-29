"""Condition definitions and the cached-log consistency guard."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from eval.conditions import ALL_CONDITIONS, INSTRUCT_CONDITIONS, engine_config, load_hparams, relevant_config

ROOT = Path(__file__).resolve().parents[1]
HP = load_hparams(ROOT / "configs" / "hparams.json")
MODELS = ["Qwen__Qwen2.5-3B", "Qwen__Qwen2.5-1.5B"]


def test_twenty_conditions():
    assert len(ALL_CONDITIONS) == 20
    assert len(INSTRUCT_CONDITIONS) == 4 and not set(INSTRUCT_CONDITIONS) & set(ALL_CONDITIONS)


def test_logged_conditions_keep_the_exact_ban():
    """The v2.3 demo's stem morphology is opt-in; every benchmark condition still bans surface forms only."""
    for conds in (ALL_CONDITIONS, INSTRUCT_CONDITIONS):
        for name, (_, _, toggles) in conds.items():
            if toggles is not None:
                assert engine_config(toggles, HP).ban_morphology == "exact", name


def test_lean_engine_differs_from_x9_only_in_lean_settings():
    x9 = asdict(engine_config(ALL_CONDITIONS["x9_v2_1_minus_anti_echo"][2], HP))
    lean = asdict(engine_config(ALL_CONDITIONS["11_engine_v2_2_lean"][2], HP))
    assert {k for k in x9 if x9[k] != lean[k]} == {"drift_stop", "boundary_trigger", "sibling_fast"}
    assert lean["sibling_k"] == HP["sibling_k"] == 4 and not lean["anti_echo"]


def test_instruct_ban_uses_condition_11_lexical_settings():
    ban = asdict(engine_config(INSTRUCT_CONDITIONS["13_instruct_plus_orth_ban_chat"][2], HP))
    lean = asdict(engine_config(ALL_CONDITIONS["11_engine_v2_2_lean"][2], HP))
    lexical = ["unembed_project", "gamma_unembed", "unembed_center", "sibling_orth", "sibling_k", "sibling_min_overlap",
               "anti_echo", "sibling_fast"]
    assert {k: ban[k] for k in lexical} == {k: lean[k] for k in lexical}
    assert not any(ban[k] for k in ("lifecycle", "dynamic_stop", "erase", "predictor", "pin_attention", "boundary_hinge"))


def test_v2_base_resolution():
    cfg = engine_config(ALL_CONDITIONS["9_engine_v2_full"][2], HP)
    assert cfg.erase_mode == "hinge" and cfg.signed_attention and cfg.unembed_project
    cfg10 = engine_config(ALL_CONDITIONS["10_engine_v2_1"][2], HP)
    assert not cfg10.signed_attention and cfg10.sibling_orth and cfg10.unembed_center and cfg10.stop_min_step == 6


def test_relevant_config_drops_unused_parameters():
    rel = relevant_config(asdict(engine_config(ALL_CONDITIONS["8_engine_v2_signed_attn"][2], HP)))
    assert "gamma_unembed" not in rel and "beta_neg_attn" in rel


_LOGGED = [(m, c, ALL_CONDITIONS[c][2]) for m in MODELS for c, (_, _, t) in ALL_CONDITIONS.items() if t is not None]
_LOGGED += [(m + "-Instruct", c, t) for m in MODELS for c, (_, _, t) in INSTRUCT_CONDITIONS.items() if t is not None]


@pytest.mark.parametrize("model,cond,toggles", _LOGGED)
def test_cached_logs_match_current_config(model, cond, toggles):
    f = ROOT / "results" / model / "test40" / f"{cond}.jsonl"
    assert f.exists(), f"missing cached log {f.relative_to(ROOT)}"
    records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    assert len(records) == 40
    current = relevant_config(asdict(engine_config(toggles, HP)))
    for rec in records:
        stored = relevant_config(rec["engine_config"])
        assert {k: v for k, v in stored.items() if current.get(k) != v} == {}
