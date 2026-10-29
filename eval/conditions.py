"""Ablation conditions and the shared prompt-runner used by the benchmark and the dev-set tuner."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
from tqdm import tqdm

from engine import AssistantEngine, EngineConfig, SteeringArtifact, encode_prompt, generate
from engine.prompts import EVAL_STOP_STRINGS

from .verifiers import score_response

_OFF = dict(lifecycle=False, dynamic_stop=False, erase=False, predictor=False, pin_attention=False, static_caa=False)

# name -> (use URIAL prefix, use stop strings, module toggles or None for "no engine")
CONDITIONS: dict[str, tuple[bool, bool, dict | None]] = {
    "1_raw_base": (False, False, None),
    "2_raw_base_plus_stops": (False, True, None),
    "3_urial_3shot_prefix": (True, True, None),
    "4_static_caa": (False, True, {**_OFF, "static_caa": True}),
    "5_lifecycle_plus_nullspace": (False, True, {**_OFF, "lifecycle": True, "dynamic_stop": True, "erase": True, "predictor": True}),
    "6_full_assistant_engine": (False, True, {**_OFF, "lifecycle": True, "dynamic_stop": True, "erase": True, "predictor": True, "pin_attention": True}),
}
# Extra single-module ablations (not part of the required 6-way matrix) for attributing gains.
EXTRA_CONDITIONS: dict[str, tuple[bool, bool, dict | None]] = {
    "x1_lifecycle_only": (False, True, {**_OFF, "lifecycle": True, "dynamic_stop": True}),
    "x2_lifecycle_plus_erase": (False, True, {**_OFF, "lifecycle": True, "dynamic_stop": True, "erase": True}),
    "x3_full_minus_dynamic_stop": (False, True, {**_OFF, "lifecycle": True, "erase": True, "predictor": True, "pin_attention": True}),
    "x4_pin_attention_only": (False, True, {**_OFF, "pin_attention": True}),
    "x5_full_minus_erase": (False, True, {**_OFF, "lifecycle": True, "dynamic_stop": True, "predictor": True, "pin_attention": True}),
}
# Phase 2 ("engine v2"). Conditions 8 and 9 build on whichever of x5 / 7 scored higher on the 1.5B dev set,
# recorded as hparams["v2_base"] by `eval.tune --phase 2`.
_V1_FULL = {**_OFF, "lifecycle": True, "dynamic_stop": True, "erase": True, "predictor": True, "pin_attention": True}
V2_CONDITIONS: dict[str, tuple[bool, bool, dict | None]] = {
    "7_engine_v2_hinge_erase": (False, True, {**_V1_FULL, "erase_mode": "hinge"}),
    "8_engine_v2_signed_attn": (False, True, {"_base": "v2_base", "signed_attention": True}),
    "9_engine_v2_full": (False, True, {"_base": "v2_base", "signed_attention": True, "unembed_project": True}),
    # v2.1: condition 7 + boundary-coupled hinge + drift-assisted stop (t_min 6) + sibling-orthogonal unembedding
    # projection + anti-echo guard; no signed attention (pinning uses the Phase-1 beta_attn)
    "10_engine_v2_1": (False, True, {**_V1_FULL, "erase_mode": "hinge", "unembed_project": True, "unembed_center": True,
                                     "boundary_hinge": True, "drift_stop": True, "stop_min_step": 6,
                                     "sibling_orth": True, "anti_echo": True}),
}
# Leave-one-out ablations of condition 10, for attributing its gains (not used for any selection).
_V21 = V2_CONDITIONS["10_engine_v2_1"][2]
V21_ABLATIONS: dict[str, tuple[bool, bool, dict | None]] = {
    "x6_v2_1_minus_boundary_hinge": (False, True, {**_V21, "boundary_hinge": False}),
    "x7_v2_1_minus_drift_stop": (False, True, {**_V21, "drift_stop": False}),
    "x8_v2_1_minus_sibling_orth": (False, True, {**_V21, "sibling_orth": False}),
    "x9_v2_1_minus_anti_echo": (False, True, {**_V21, "anti_echo": False}),
}
ALL_CONDITIONS = {**CONDITIONS, **EXTRA_CONDITIONS, **V2_CONDITIONS, **V21_ABLATIONS}


def load_hparams(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("hparams", data)


def resolve_toggles(toggles: dict, hparams: dict) -> dict:
    """Expand a ``_base`` reference (conditions 8/9 inherit the dev-selected v2 base) into concrete toggles."""
    if "_base" not in toggles:
        return toggles
    base = hparams.get(toggles["_base"], "7_engine_v2_hinge_erase")
    return {**resolve_toggles(ALL_CONDITIONS[base][2], hparams), **{k: v for k, v in toggles.items() if k != "_base"}}


def engine_config(toggles: dict, hparams: dict) -> EngineConfig:
    names = {f.name for f in fields(EngineConfig)}
    toggles = resolve_toggles(toggles, hparams)
    return replace(EngineConfig(), **{k: v for k, v in hparams.items() if k in names and k not in toggles}, **toggles)


# Parameters that cannot affect generation when the module that reads them is off (or in the other mode).
_PARAMS_OF = {
    "lifecycle": ["alpha_init", "gamma", "init_steps", "alpha_sustain"],
    "dynamic_stop": ["alpha_stop", "stop_min_step", "stop_logit_boost", "newturn_logprob"],
    "erase": ["lambda_erase", "erase_kappa", "erase_mode", "lambda_hinge", "hinge_kappa_floor", "boundary_hinge",
              "kappa_boundary", "boundary_after_newline", "boundary_score_margin"],
    "predictor": ["predictor_weight", "predictor_clip"],
    "pin_attention": ["beta_attn", "signed_attention", "beta_neg_attn", "neg_span_mode"],
    "signed_attention": ["beta_neg_attn", "neg_span_mode"],
    "static_caa": ["alpha_caa"],
    "unembed_project": ["gamma_unembed", "unembed_center", "sibling_orth", "sibling_k", "sibling_min_overlap",
                        "anti_echo", "anti_echo_window"],
    "boundary_hinge": ["kappa_boundary", "boundary_after_newline", "boundary_score_margin"],
    "drift_stop": ["tau_drift_stop", "drift_stop_score_margin"],
    "sibling_orth": ["sibling_k", "sibling_min_overlap"],
    "anti_echo": ["anti_echo_window"],
}


def relevant_config(cfg: dict) -> dict:
    """Engine config restricted to parameters that can influence generation under its module toggles."""
    drop: set[str] = set()
    for flag, params in _PARAMS_OF.items():
        if not cfg.get(flag, False):
            drop |= set(params)
    if cfg.get("erase_mode", "v1") == "v1":
        drop |= {"lambda_hinge", "hinge_kappa_floor", "boundary_hinge", "kappa_boundary", "boundary_after_newline",
                 "boundary_score_margin"}
    else:
        drop |= {"lambda_erase", "erase_kappa"}
    return {k: v for k, v in cfg.items() if k not in drop}


def load_prompts(path: Path, n: int | None = None, per_category: bool = True) -> list[dict]:
    prompts = json.loads(path.read_text())
    if n is None or n >= len(prompts):
        return prompts
    if not per_category:
        return prompts[:n]
    by_cat: dict[str, list[dict]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    picked, i = [], 0
    while len(picked) < n:  # round-robin across categories so small runs stay balanced
        for cat in by_cat:
            if i < len(by_cat[cat]) and len(picked) < n:
                picked.append(by_cat[cat][i])
        i += 1
    return picked


def run_condition(lm, art: SteeringArtifact | None, name: str, prompts: list[dict], hparams: dict,
                  max_new_tokens: int = 160, out_file: Path | None = None, desc: str | None = None,
                  conditions: dict | None = None) -> list[dict]:
    urial, stops, toggles = (conditions or ALL_CONDITIONS)[name]
    cfg = engine_config(toggles, hparams) if toggles is not None else None
    done: dict[str, dict] = {}
    if out_file is not None and out_file.exists():
        for line in out_file.read_text().splitlines():
            rec = json.loads(line)
            done[rec["prompt_id"]] = rec
    engine = AssistantEngine(lm, art, cfg).attach() if cfg is not None else None
    records = []
    try:
        for p in tqdm(prompts, desc=desc or name, leave=False):
            if p["id"] in done:
                rec = done[p["id"]]
                stored = relevant_config(rec.get("engine_config") or {})
                current = relevant_config(asdict(cfg)) if cfg is not None else {}
                if cfg is not None and any(current.get(k) != v for k, v in stored.items()):
                    diff = {k: (v, current.get(k)) for k, v in stored.items() if current.get(k) != v}
                    raise RuntimeError(f"cached {name}/{p['id']} was generated with a different engine config {diff}; "
                                       "use a new --tag or --fresh")
                rec["metrics"] = score_response(rec["response"], rec["stop_reason"], p)  # verifiers may have changed
                records.append(rec)
                continue
            torch.manual_seed(0)
            enc = encode_prompt(lm.tokenizer, p["instruction"], urial=urial)
            res = generate(lm, enc, max_new_tokens, EVAL_STOP_STRINGS if stops else None, engine)
            rec = {
                "condition": name, "prompt_id": p["id"], "category": p["category"], "instruction": p["instruction"],
                "response": res.text, **{k: v for k, v in res.to_dict().items() if k != "text"},
                "extra_prompt_tokens": enc.extra_tokens,
                "metrics": score_response(res.text, res.stop_reason, p),
                "engine_config": asdict(cfg) if cfg is not None else None,
            }
            records.append(rec)
            if out_file is not None:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with out_file.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
    finally:
        if engine is not None:
            engine.detach()
    return records


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    keys = ("turn_cross", "constraint", "clean_stop", "fluency", "composite", "natural_eos")
    out = {k: 100.0 * sum(r["metrics"][k] for r in records) / n for k in keys}
    tot_tok = sum(r["n_new_tokens"] for r in records)
    tot_s = sum(r["latency_s"] for r in records)
    out.update(
        n=n,
        ms_per_token=1000.0 * tot_s / max(tot_tok, 1),
        tok_per_s=tot_tok / max(tot_s, 1e-9),
        avg_new_tokens=tot_tok / n,
        extra_prompt_tokens=sum(r["extra_prompt_tokens"] for r in records) / n,
        stop_reasons={s: sum(r["stop_reason"] == s for r in records) for s in ("eos", "stop_string", "max_tokens")},
    )
    return out
