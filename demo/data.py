"""Benchmark data for the studio, read from the cached logs with the report's own loaders (so every number matches
RESULTS.md): logs are always re-scored by the frozen verifiers, CIs use ``eval.report.paired_bootstrap_ci``."""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

from engine.negation import parse_negations, stem_of
from eval.conditions import ALL_CONDITIONS, INSTRUCT_CONDITIONS, INSTRUCT_SUFFIX, summarize
from eval.report import (CATEGORIES, _load, _load_instruct, _negation_ids, _read_json, _variant_recs,
                         paired_bootstrap_ci, recovery_ci)
from eval.verifiers import contamination

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"3B": "Qwen/Qwen2.5-3B", "1.5B": "Qwen/Qwen2.5-1.5B"}
TAGS = {"test40": "protocol (eager attention, 160 tokens)", "test40_512tok": "512-token budget",
        "test40_sdpa": "SDPA attention, 160 tokens"}

LABELS = {
    "1_raw_base": ("Raw base", "Baselines"),
    "2_raw_base_plus_stops": ("Raw base + stop strings", "Baselines"),
    "3_urial_3shot_prefix": ("URIAL 3-shot prefix (768 tokens)", "Baselines"),
    "4_static_caa": ("Static CAA (single vector)", "Baselines"),
    "5_lifecycle_plus_nullspace": ("v1: lifecycle + nullspace + predictor", "Phase 1 · v1"),
    "6_full_assistant_engine": ("v1 full engine (M1–M4)", "Phase 1 · v1"),
    "x1_lifecycle_only": ("x1 lifecycle only", "Phase 1 · v1"),
    "x2_lifecycle_plus_erase": ("x2 lifecycle + erase", "Phase 1 · v1"),
    "x3_full_minus_dynamic_stop": ("x3 full − dynamic stop", "Phase 1 · v1"),
    "x4_pin_attention_only": ("x4 pin attention only", "Phase 1 · v1"),
    "x5_full_minus_erase": ("x5 full − erase", "Phase 1 · v1"),
    "7_engine_v2_hinge_erase": ("v2: hinge erase", "Phase 2 · v2"),
    "8_engine_v2_signed_attn": ("v2: + signed attention", "Phase 2 · v2"),
    "9_engine_v2_full": ("v2 full (+ unembedding projection)", "Phase 2 · v2"),
    "10_engine_v2_1": ("v2.1 engine", "Phase 3 · v2.1"),
    "x6_v2_1_minus_boundary_hinge": ("x6 v2.1 − boundary hinge", "Phase 3 · v2.1"),
    "x7_v2_1_minus_drift_stop": ("x7 v2.1 − drift stop", "Phase 3 · v2.1"),
    "x8_v2_1_minus_sibling_orth": ("x8 v2.1 − sibling orth", "Phase 3 · v2.1"),
    "x9_v2_1_minus_anti_echo": ("x9 v2.1 − anti-echo", "Phase 3 · v2.1"),
    "11_engine_v2_2_lean": ("v2.2 lean engine", "Phase 4 · lean"),
    "12_official_instruct_chat": ("Official Instruct (chat template)", "Phase 4 · post-trained"),
    "12_official_instruct_bare": ("Official Instruct (bare frame)", "Phase 4 · post-trained"),
    "13_instruct_plus_orth_ban_chat": ("Instruct + sibling-orth ban (chat)", "Phase 4 · post-trained"),
    "13_instruct_plus_orth_ban_bare": ("Instruct + sibling-orth ban (bare)", "Phase 4 · post-trained"),
}
INSTRUCT = set(INSTRUCT_CONDITIONS)

# One-click scenarios for the arena. Every claim here was read off the cached 3B logs; the sidebar also shows the logged
# pass/fail chips per column, computed below, so the copy cannot drift from the data silently.
PRESETS = [
    {"id": "flan_leak", "title": "The FLAN pretraining leak", "prompts": ["t_qa_04"],
     "blurb": "Raw base answers in one sentence, then continues with a FLAN-style “single-select problem” template. "
              "Static CAA runs into a “is this text about computer science?” classifier prompt and hits the cap. "
              "The engine explains both immune branches and ends with EOS; Instruct is still mid-list at 160 tokens."},
    {"id": "instruct_fails", "title": "Where official Instruct fails", "prompts": ["t_ng_01"],
     "blurb": "Raw base, static CAA and Qwen2.5-3B-Instruct all use “force” (Instruct: “Gravity is a fundamental "
              "force…”). The engine and Instruct + the sibling-orthogonal ban answer without it."},
    {"id": "morphology", "title": "Morphology & sub-word trap", "prompts": ["t_ng_04"],
     "blurb": "Every logged condition breaks the ban while restating the rule: raw base writes “balance”; v2.1 and "
              "Instruct + ban write “balancing” as bal + ancing; the v2.2 lean engine writes “balanc” (bal + anc), "
              "which the frozen verifier misses. Stem mode bans every token that starts a word with “balanc” and masks "
              "completions of an unfinished “bal…”. Live, it writes the misspelling “balace” instead."},
    {"id": "loop", "title": "Repetition loop vs. sibling protection", "prompts": ["t_ng_02"],
     "blurb": "The v1 engine (condition 6) loops “Swift, Kotlin, Go, Rust…” to the cap; Instruct restates the rule "
              "(“…without mentioning Python:”) and fails. The engine answers “C, C++, Java” in 5 tokens — as does the "
              "raw base on this prompt."},
    {"id": "format", "title": "Strict format & length", "prompts": ["t_cd_01", "t_fm_06"],
     "blurb": "SQL: static CAA appends a tutorial after the code block (104 tokens) and fails “only code”. Blockchain "
              "in < 40 words: raw base and CAA run past 100 tokens. The engine returns the bare 28-token query and a "
              "36-word answer. Instruct passes both too (fenced block; 21 tokens)."},
    {"id": "math", "title": "Math worksheet run-on", "prompts": ["t_rm_07", "t_rm_05"],
     "blurb": "Raw base and static CAA solve it, then start a new “[Question] …” worksheet item. The engine stops after "
              "the answer. Second prompt is an honest failure: on t_rm_05 the lean engine appends a “[Reasoning: …]” "
              "tag and runs to the cap; only v2.1's drift-assisted stop (removed in v2.2 lean) caught it."},
    {"id": "fix_instruct", "title": "Fix Instruct with the ban", "prompts": ["t_ng_06"],
     "blurb": "Instruct writes “interconnected computers”. Flip “+ ban” on the Instruct card: the same zero-training "
              "lexical operator (no artifact, no re-tuning) makes it answer “devices and servers”."},
]

# Arena column -> the benchmark condition it reproduces with default settings (for logged-outcome chips and replay).
COLUMN_CONDITION = {"raw": "1_raw_base", "caa": "4_static_caa", "engine": "11_engine_v2_2_lean",
                    "instruct": "12_official_instruct_chat", "instruct_ban": "13_instruct_plus_orth_ban_chat"}


def slug(model: str) -> str:
    return model.replace("/", "__")


@lru_cache(maxsize=1)
def test_prompts() -> dict[str, dict]:
    return {p["id"]: p for p in json.loads((ROOT / "data" / "test_prompts.json").read_text())}


def negation_info(instruction: str) -> dict:
    targets = [t for n in parse_negations(instruction) for *_, t in n.targets]
    return {"targets": targets, "stems": sorted({stem_of(t) for t in targets})}


def stem_strict(text: str, stems: list[str]) -> bool | None:
    """No word in ``text`` starts with a banned stem (stricter than the frozen verifier's surface-form regex)."""
    if not stems:
        return None
    return not any(re.search(rf"\b{re.escape(s)}", text, flags=re.IGNORECASE) for s in stems)


_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def infer_checks(instruction: str) -> list[dict]:
    """Machine-checkable rules for a custom prompt (benchmark prompts keep their frozen checks)."""
    low, checks = instruction.lower(), []
    for s in negation_info(instruction)["stems"]:
        checks.append({"type": "forbidden", "patterns": [rf"\b{re.escape(s)}"]})
    num = r"(\d+|" + "|".join(_WORDNUM) + r")"

    def n_of(g: str) -> int:
        return int(g) if g.isdigit() else _WORDNUM[g]
    if m := re.search(rf"(?:fewer|less) than {num} words", low):
        checks.append({"type": "max_words", "n": n_of(m.group(1)) - 1})
    elif m := re.search(rf"(?:in|under|at most|no more than|maximum of|max) {num} words", low):
        checks.append({"type": "max_words", "n": n_of(m.group(1))})
    if re.search(r"\bone paragraph\b", low):
        checks.append({"type": "one_paragraph"})
    if m := re.search(rf"exactly {num} bullet", low):
        checks.append({"type": "bullet_count", "n": n_of(m.group(1))})
    if m := re.search(rf"(?:in|exactly) {num} sentences?\b", low):
        checks.append({"type": "sentence_count", "n": n_of(m.group(1))})
    if re.search(r"\breturn only\b|\bonly the (?:sql|python|code)\b|\bcode block only\b", low):
        checks.append({"type": "only_code"})
    if re.search(r"\ball caps\b|\buppercase\b|\bcapital letters\b", low):
        checks.append({"type": "all_upper"})
    return checks


def _lite(r: dict) -> dict:
    m = r["metrics"]
    return {"response": r["response"], "stop_reason": r["stop_reason"], "stop_string": r.get("stop_string"),
            "n_new_tokens": r["n_new_tokens"], "extra_prompt_tokens": r.get("extra_prompt_tokens", 0),
            "latency_s": r.get("latency_s"), "metrics": {k: m[k] for k in ("turn_cross", "constraint", "clean_stop",
            "fluency", "composite", "natural_eos")} | {"constraint_detail": m["constraint_detail"],
            "contamination": m["contamination"]}}


@lru_cache(maxsize=None)
def logs(model_key: str, tag: str = "test40") -> dict[str, dict[str, dict]]:
    """{condition: {prompt_id: record}} for the base model and its Instruct reference, re-scored."""
    out_dir = ROOT / "results" / slug(MODELS[model_key]) / "test40"
    recs = _variant_recs(out_dir, "" if tag == "test40" else tag.removeprefix("test40"))
    return {c: {r["prompt_id"]: r for r in rs} for c, rs in recs.items()}


def logs_lite(model_key: str, tag: str) -> dict:
    return {c: {pid: _lite(r) for pid, r in rs.items()} for c, rs in logs(model_key, tag).items()}


def logged_outcome(model_key: str, pid: str) -> dict[str, dict]:
    """Per arena column: the logged record of the condition it reproduces (protocol tag)."""
    out = {}
    for col, cond in COLUMN_CONDITION.items():
        r = logs(model_key).get(cond, {}).get(pid)
        if r is not None:
            out[col] = _lite(r) | {"condition": cond}
    return out


def prompts_payload() -> dict:
    ps = []
    for p in test_prompts().values():
        ps.append({"id": p["id"], "category": p["category"], "instruction": p["instruction"], "checks": p["checks"],
                   **negation_info(p["instruction"])})
    presets = []
    for pr in PRESETS:
        presets.append(pr | {"logged": {mk: {pid: logged_outcome(mk, pid) for pid in pr["prompts"]} for mk in MODELS}})
    return {"prompts": ps, "presets": presets, "categories": CATEGORIES}


# ---------------------------------------------------------------------------------------------
# Ablations, economics (heavy: computed once, in the background)
# ---------------------------------------------------------------------------------------------

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def _cat_scores(rs: list[dict]) -> dict[str, float]:
    out = {}
    for cat in CATEGORIES:
        sub = [r for r in rs if r["category"] == cat]
        if sub:
            out[cat] = 100.0 * sum(r["metrics"]["composite"] for r in sub) / len(sub)
    return out


def ablation_data(model_key: str) -> dict:
    key = f"ablation:{model_key}"
    with _lock:
        if key in _cache:
            return _cache[key]
    out_dir = ROOT / "results" / slug(MODELS[model_key]) / "test40"
    _, recs = _load(out_dir)
    recs = {**recs, **_load_instruct(out_dir)}
    long = {c: summarize(rs) for c, rs in _variant_recs(out_dir, "_512tok").items()}
    sdpa = {c: summarize(rs) for c, rs in _variant_recs(out_dir, "_sdpa").items()}
    rows = []
    for c in [*ALL_CONDITIONS, *INSTRUCT_CONDITIONS]:
        if c not in recs:
            continue
        s = summarize(recs[c])
        lo, hi = paired_bootstrap_ci(recs, "1_raw_base", c) if c != "1_raw_base" else (0.0, 0.0)
        label, group = LABELS.get(c, (c, ""))
        rows.append({"condition": c, "label": label, "group": group, "instruct": c in INSTRUCT,
                     **{k: s[k] for k in ("turn_cross", "constraint", "clean_stop", "fluency", "composite",
                                          "natural_eos", "avg_new_tokens", "tok_per_s", "extra_prompt_tokens")},
                     "stop_reasons": s["stop_reasons"], "delta_vs_raw": s["composite"] - summarize(recs["1_raw_base"])["composite"],
                     "ci_vs_raw": [lo, hi], "categories": _cat_scores(recs[c]),
                     "composite_512": long.get(c, {}).get("composite"), "composite_sdpa": sdpa.get(c, {}).get("composite")})
    data = {"model": MODELS[model_key], "rows": rows, "categories": CATEGORIES,
            "note": "Composite = mean of TurnCross, Constraint, CleanStop and Fluency over the 40 test prompts, re-scored "
                    "from the logged text. CI: paired bootstrap of the difference to the raw base (10,000 resamples, "
                    "seed 0), as in RESULTS.md."}
    with _lock:
        _cache[key] = data
    return data


def _pooled(rows: list[dict], keep) -> dict | None:
    rs = [r for r in rows if keep(r["id"])]
    n = sum(r["n"] for r in rs)
    if not n:
        return None
    return {"mean_kl": sum(r["kl_sum"] for r in rs) / n, "top1_agree": sum(r["top1_agree"] for r in rs) / n,
            "n_tokens": n, "n_texts": len(rs)}


def economics(model_key: str) -> dict:
    key = f"economics:{model_key}"
    with _lock:
        if key in _cache:
            return _cache[key]
    s = slug(MODELS[model_key])
    out_dir = ROOT / "results" / s / "test40"
    _, recs = _load(out_dir)
    irecs = _load_instruct(out_dir)
    all_recs = {**recs, **irecs}
    long_recs = _variant_recs(out_dir, "_512tok")
    tax = _read_json(ROOT / "results" / s / "alignment_tax.json")
    delta = _read_json(ROOT / "results" / (s + INSTRUCT_SUFFIX) / "weight_delta.json")
    bench = _read_json(ROOT / "results" / s / "speed_bench.json").get("conditions", {})
    art = ROOT / "artifacts" / s / "calib80" / "calibration.pt"
    secs = (_read_json(out_dir / "summary.json").get("run", {}).get("calibration_meta") or {}).get("calibration_seconds")
    neg = _negation_ids()

    def kl(c: str) -> dict:
        if c == "1_raw_base":
            return {"reference": True}
        if not tax or c not in tax["summary"].get("heldout", {}):
            return {}
        per = tax["per_text"]
        return {"heldout_all": tax["summary"]["heldout"][c], "calibration_all": tax["summary"]["calibration"][c],
                "heldout_no_negation": _pooled(per["heldout"][c], lambda i: i not in neg),
                "calibration_no_negation": _pooled(per["calibration"][c], lambda i: i not in neg)}

    size = art.stat().st_size if art.exists() else None
    approaches = [
        ("Raw base", "1_raw_base", 0.0, 0, "none", "base"),
        ("URIAL 3-shot prefix", "3_urial_3shot_prefix", 0.0, 0, "none", "prompt"),
        ("Static CAA", "4_static_caa", 0.0, size, "calibration (same artifact)", "steer"),
        ("Engine v2.1", "10_engine_v2_1", 0.0, size, "calibration", "engine"),
        ("Engine v2.2 lean", "11_engine_v2_2_lean", 0.0, size, "calibration", "engine"),
        ("Official Instruct (chat)", "12_official_instruct_chat", delta.get("params_changed_frac"),
         delta.get("instruct_weights_bytes"), "SFT on > 1M examples + multi-stage RL", "instruct"),
        ("Official Instruct (bare)", "12_official_instruct_bare", delta.get("params_changed_frac"),
         delta.get("instruct_weights_bytes"), "SFT on > 1M examples + multi-stage RL", "instruct"),
        ("Instruct + sibling-orth ban", "13_instruct_plus_orth_ban_chat", delta.get("params_changed_frac"),
         delta.get("instruct_weights_bytes"), "post-training + 0 (no artifact)", "instruct"),
    ]
    rows = []
    for label, c, wfrac, nbytes, cost, kind in approaches:
        if c not in all_recs:
            continue
        sm = summarize(all_recs[c])
        lg = summarize(long_recs[c]) if c in long_recs else None
        rows.append({"label": label, "condition": c, "kind": kind, "weights_changed_frac": wfrac, "bytes": nbytes,
                     "one_off_cost": cost, "extra_prompt_tokens": sm["extra_prompt_tokens"], "kl": kl(c),
                     "composite_160": sm["composite"], "composite_512": lg["composite"] if lg else None,
                     "avg_tokens_160": sm["avg_new_tokens"], "avg_tokens_512": lg["avg_new_tokens"] if lg else None,
                     "tok_per_s": bench[c]["tok_per_s"] if c in bench else sm["tok_per_s"],
                     "tok_per_s_source": "same-session benchmark" if c in bench else "log"})
    rec = {}
    for tag, rr in (("160", all_recs), ("512", long_recs)):
        for eng in ("10_engine_v2_1", "11_engine_v2_2_lean"):
            if all(k in rr for k in ("1_raw_base", eng, "12_official_instruct_chat")):
                rec[f"{eng}@{tag}"] = recovery_ci(rr, "1_raw_base", eng, "12_official_instruct_chat")
    data = {"model": MODELS[model_key], "rows": rows, "recovery": rec, "weight_delta": delta,
            "artifact_bytes": size, "calibration_seconds": secs,
            "kl_note": "Mean D_KL(P_raw ‖ P_condition) in nats per token, teacher-forced at answer steps t ≥ 5 on clean "
                       "answers (eval/alignment_tax.py). held-out = URIAL's clean answers to the test prompts; "
                       "no-negation drops prompts with a parsed negation, where the lexical ban suppresses words the "
                       "teacher answer uses by design. A distance from the base model, not a quality score.",
            "kl_floor": (tax.get("summary", {}).get("calibration", {}).get("raw_base_single_pass", {}) or {}).get("mean_kl")}
    with _lock:
        _cache[key] = data
    return data


def warm() -> None:
    """Fill the caches (bootstrap CIs take a few seconds per model)."""
    for mk in MODELS:
        try:
            ablation_data(mk)
            economics(mk)
            for tag in TAGS:
                logs(mk, tag)
        except FileNotFoundError:
            continue


def contamination_spans(text: str) -> list[list[int]]:
    """Character spans of run-on markers (the verifier's patterns), for highlighting."""
    from eval.verifiers import _TURN_MARKERS, _WEB_MARKERS
    spans = [[m.start(), m.end()] for pat in (_TURN_MARKERS, _WEB_MARKERS) for m in pat.finditer(text)]
    return sorted(spans)


__all__ = ["MODELS", "TAGS", "PRESETS", "COLUMN_CONDITION", "ablation_data", "contamination", "contamination_spans",
           "economics", "infer_checks", "logged_outcome", "logs", "logs_lite", "negation_info", "prompts_payload",
           "slug", "stem_strict", "test_prompts", "warm"]
