"""Render RESULTS.md from logged generations (re-runnable without regenerating anything).

    python -m eval.report results/Qwen__Qwen2.5-3B/test40 --out RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

from .conditions import ALL_CONDITIONS, INSTRUCT_CONDITIONS, INSTRUCT_SUFFIX, summarize
from .verifiers import score_response

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["direct_qa", "format_constraints", "negative_constraints", "reasoning_math", "code_and_sql"]
SHOWCASE = ["1_raw_base", "4_static_caa", "6_full_assistant_engine"]
METRICS = [("turn_cross", "TurnCross"), ("constraint", "Constraint"), ("clean_stop", "CleanStop"),
           ("fluency", "Fluency"), ("composite", "**Composite**"), ("natural_eos", "NaturalEOS")]


def _load(out_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    summary = json.loads((out_dir / "summary.json").read_text())
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / summary["run"]["prompts_file"]).read_text())}
    recs: dict[str, list[dict]] = {}
    for name in ALL_CONDITIONS:
        f = out_dir / f"{name}.jsonl"
        if f.exists():
            recs[name] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for r in recs[name]:  # always re-score from the logged text so the report tracks the current verifiers
                r["metrics"] = score_response(r["response"], r["stop_reason"], prompts[r["prompt_id"]])
    return summary, recs


def _instruct_dir(out_dir: Path) -> Path:
    """results/<base slug>/<tag> -> results/<base slug>-Instruct/<tag> (the post-trained reference checkpoint)."""
    return out_dir.parent.parent / (out_dir.parent.name + INSTRUCT_SUFFIX) / out_dir.name


def _load_logs(d: Path, names) -> dict[str, list[dict]]:
    if not d.exists():
        return {}
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "test_prompts.json").read_text())}
    recs: dict[str, list[dict]] = {}
    for name in names:
        f = d / f"{name}.jsonl"
        if f.exists():
            recs[name] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for r in recs[name]:
                r["metrics"] = score_response(r["response"], r["stop_reason"], prompts[r["prompt_id"]])
    return recs


def _load_instruct(out_dir: Path) -> dict[str, list[dict]]:
    return _load_logs(_instruct_dir(out_dir), INSTRUCT_CONDITIONS)


# Sensitivity runs of the economics comparison (separate tags; never mixed with the protocol logs).
VARIANTS = [("", "protocol: eager attention, 160 tokens"), ("_sdpa", "SDPA attention, 160 tokens"),
            ("_512tok", "eager attention, 512 tokens")]
SENS_CONDITIONS = ["1_raw_base", "2_raw_base_plus_stops", "10_engine_v2_1", "11_engine_v2_2_lean",
                   "12_official_instruct_chat", "12_official_instruct_bare", "13_instruct_plus_orth_ban_chat"]


def _variant_recs(out_dir: Path, suffix: str) -> dict[str, list[dict]]:
    d = out_dir.parent / (out_dir.name + suffix)
    return {**_load_logs(d, ALL_CONDITIONS), **_load_logs(_instruct_dir(d), INSTRUCT_CONDITIONS)}


def _sensitivity(out_dir: Path) -> list[str]:
    var = [(sfx, lab, _variant_recs(out_dir, sfx)) for sfx, lab in VARIANTS]
    var = [v for v in var if v[2]]
    if len(var) < 2:
        return []
    out = ["### Sensitivity of the post-training comparison", "",
           "| Condition | " + " | ".join(f"Composite, {lab}" for _, lab, _ in var) + " | "
           + " | ".join(f"hit token cap, {lab.split(': ')[-1]}" for _, lab, _ in var) + " |",
           "|" + "---|" * (1 + 2 * len(var))]
    for c in SENS_CONDITIONS:
        cells = []
        caps = []
        for _, _, rs in var:
            if c in rs:
                s = summarize(rs[c])
                cells.append(f"{s['composite']:.1f}")
                caps.append(f"{s['stop_reasons']['max_tokens']}/{s['n']}")
            else:
                cells.append("–")
                caps.append("–")
        if any(x != "–" for x in cells):
            out.append(f"| `{c}` | " + " | ".join(cells) + " | " + " | ".join(caps) + " |")
    out.append("")
    rows = []
    for sfx, lab, rs in var:
        eng_src = rs if "10_engine_v2_1" in rs else var[0][2]   # the engine needs steer_eager: SDPA rows reuse the protocol engine
        for eng in ("10_engine_v2_1", "11_engine_v2_2_lean"):
            for inst in ("12_official_instruct_chat", "12_official_instruct_bare"):
                if eng in eng_src:
                    note = "" if eng_src is rs else " (engine on the protocol stack)"
                    rows.append((lab + note, {**rs, eng: eng_src[eng]}, "1_raw_base", eng, inst))
    out += _recovery_table(rows)
    return out + [""]


def _fmt(x: float, nd: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _summary_table(summ: dict[str, dict]) -> list[str]:
    head = "| Condition | " + " | ".join(m[1] for m in METRICS) + " | Extra ctx tokens | ms/token | tok/s | avg new tokens |"
    lines = [head, "|" + "---|" * (len(METRICS) + 5)]
    for name, s in summ.items():
        cells = [f"{s[k]:.1f}" for k, _ in METRICS]
        cells[4] = f"**{cells[4]}**"
        lines.append(f"| `{name}` | " + " | ".join(cells)
                     + f" | {s['extra_prompt_tokens']:.0f} | {s['ms_per_token']:.1f} | {s['tok_per_s']:.1f} | {s['avg_new_tokens']:.0f} |")
    return lines


def _category_table(recs: dict[str, list[dict]], metric: str) -> list[str]:
    cats = [c for c in CATEGORIES if any(r["category"] == c for rs in recs.values() for r in rs)]
    lines = ["| Condition | " + " | ".join(cats) + " |", "|" + "---|" * (len(cats) + 1)]
    for name, rs in recs.items():
        cells = []
        for c in cats:
            sub = [r for r in rs if r["category"] == c]
            cells.append(_fmt(100.0 * sum(r["metrics"][metric] for r in sub) / len(sub)) if sub else "–")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return lines


def _delta(summ: dict, a: str, b: str, k: str = "composite") -> float | None:
    if a in summ and b in summ:
        return summ[b][k] - summ[a][k]
    return None


def paired_bootstrap_ci(recs: dict, a: str, b: str, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the Composite difference (b - a), resampling prompts with replacement (paired by prompt)."""
    import random
    ra = {r["prompt_id"]: r["metrics"]["composite"] for r in recs[a]}
    rb = {r["prompt_id"]: r["metrics"]["composite"] for r in recs[b]}
    ids = [i for i in ra if i in rb]
    diffs = [100.0 * (rb[i] - ra[i]) for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _cmp_line(summ: dict, recs: dict, label: str, a: str, b: str) -> str | None:
    d = _delta(summ, a, b)
    if d is None:
        return None
    lo, hi = paired_bootstrap_ci(recs, a, b)
    parts = ", ".join(f"{lab} {summ[b][k] - summ[a][k]:+.1f}" for k, lab in
                      [("turn_cross", "TurnCross"), ("constraint", "Constraint"), ("clean_stop", "CleanStop"),
                       ("fluency", "Fluency"), ("natural_eos", "NaturalEOS")])
    return f"- **{label}** (`{a}` → `{b}`): Composite **{d:+.1f}** [{lo:+.1f}, {hi:+.1f}] ({parts})."


def _category_metric(recs: dict, cond: str, cat: str, metric: str) -> float | None:
    sub = [r for r in recs.get(cond, []) if r["category"] == cat]
    return 100.0 * sum(r["metrics"][metric] for r in sub) / len(sub) if sub else None


def _phase2_analysis(summ: dict, recs: dict, run: dict) -> list[str]:
    if not any(c in summ for c in ("x5_full_minus_erase", "7_engine_v2_hinge_erase", "8_engine_v2_signed_attn", "9_engine_v2_full")):
        return []
    p2 = ((run.get("hparams_meta") or {}).get("phase2") or {})
    hp = run.get("hparams") or {}
    base = hp.get("v2_base", "7_engine_v2_hinge_erase")
    best = p2.get("dev_selected_best_v2", "9_engine_v2_full")
    out = ["### Phase 2 (engine v2) comparisons", "",
           f"Conditions 8 and 9 build on `{base}`, the better of x5 / 7 on the 1.5B dev set. The \"best v2\" condition "
           f"is `{best}`, selected on **dev** composite ({', '.join(f'{k} {v:.1f}' for k, v in p2.get('dev_composite', {}).items())}), "
           "not on these test numbers. Conditions 8 and 9 only differ from their base on prompts that contain a negation "
           "(8 of 40 test prompts), so their deltas are driven by the `negative_constraints` category.", ""]
    for label, a, b in [
        ("Dropping v1 Module 2 from the full engine", "6_full_assistant_engine", "x5_full_minus_erase"),
        ("p99 hinge erasure vs. no erasure", "x5_full_minus_erase", "7_engine_v2_hinge_erase"),
        ("p99 hinge vs. v1 erasure (both with Modules 1+3+4)", "6_full_assistant_engine", "7_engine_v2_hinge_erase"),
        ("Signed attention on its dev-selected base", base, "8_engine_v2_signed_attn"),
        ("+ Unembedding projection", "8_engine_v2_signed_attn", "9_engine_v2_full"),
        ("v1 full engine → v2 full engine", "6_full_assistant_engine", "9_engine_v2_full"),
        *[(f"Best v2 (`{best}`) vs. {ref}", ref, best) for ref in
          ("1_raw_base", "2_raw_base_plus_stops", "3_urial_3shot_prefix", "4_static_caa", "6_full_assistant_engine")],
    ]:
        if a != b and (ln := _cmp_line(summ, recs, label, a, b)):
            out.append(ln)
        if label.startswith("Signed attention") and base != "7_engine_v2_hinge_erase":
            if ln := _cmp_line(summ, recs, "Condition 7 → 8 (as specified; confounded because the base is x5)",
                               "7_engine_v2_hinge_erase", "8_engine_v2_signed_attn"):
                out.append(ln)
    neg = [(c, _category_metric(recs, c, "negative_constraints", "constraint"))
           for c in ("1_raw_base", "6_full_assistant_engine", base, "8_engine_v2_signed_attn", "9_engine_v2_full")]
    neg = [(c, v) for c, v in neg if v is not None]
    if neg:
        out.append("- **Negative-constraint ConstraintScore** (8 prompts): " + " → ".join(f"`{c}` {v:.1f}%" for c, v in neg) + ".")
    return out + [""]


def _phase3_analysis(summ: dict, recs: dict, run: dict) -> list[str]:
    if "10_engine_v2_1" not in summ:
        return []
    out = ["### Phase 3 (engine v2.1, condition 10) comparisons", ""]
    for ref in ("1_raw_base", "2_raw_base_plus_stops", "3_urial_3shot_prefix", "6_full_assistant_engine",
                "7_engine_v2_hinge_erase", "9_engine_v2_full"):
        if ln := _cmp_line(summ, recs, f"v2.1 vs. `{ref}`", ref, "10_engine_v2_1"):
            out.append(ln)
    for abl, lab in (("x6_v2_1_minus_boundary_hinge", "boundary-coupled hinge"), ("x7_v2_1_minus_drift_stop", "drift-assisted stop"),
                     ("x8_v2_1_minus_sibling_orth", "sibling orthogonalisation"), ("x9_v2_1_minus_anti_echo", "anti-echo guard")):
        if ln := _cmp_line(summ, recs, f"Leave-one-out: removing the {lab}", "10_engine_v2_1", abl):
            out.append(ln)
    cats = [(c, _category_metric(recs, c, "reasoning_math", "composite")) for c in
            ("6_full_assistant_engine", "7_engine_v2_hinge_erase", "9_engine_v2_full", "10_engine_v2_1")]
    neg = [(c, _category_metric(recs, c, "negative_constraints", "constraint")) for c in
           ("6_full_assistant_engine", "7_engine_v2_hinge_erase", "9_engine_v2_full", "10_engine_v2_1")]
    out.append("- **reasoning_math Composite**: " + " → ".join(f"`{c}` {v:.1f}%" for c, v in cats if v is not None) + ".")
    out.append("- **negative_constraints ConstraintScore**: " + " → ".join(f"`{c}` {v:.1f}%" for c, v in neg if v is not None) + ".")
    ev = [e for r in recs["10_engine_v2_1"] for e in r.get("stop_events", []) if "reason" in e]
    tr = [r.get("engine_trace") or {} for r in recs["10_engine_v2_1"]]
    out.append(f"- **Mechanism activity on these prompts (condition 10):** EOS boosts by reason "
               + ", ".join(f"{k} {sum(e['reason'] == k for e in ev)}" for k in ("centroid", "drift", "newturn"))
               + f"; boundary-window steps {sum(t.get('boundary_steps', 0) for t in tr)}; unembedding projections "
               f"{sum(t.get('unembed_projections', 0) for t in tr)} (sibling fallbacks {sum(t.get('sibling_fallbacks', 0) for t in tr)}); "
               f"prompts with an anti-echo set {sum('anti_echo_tokens' in t for t in tr)}.")
    return out + [""]


def _analysis(summ: dict, recs: dict, diag: dict) -> list[str]:
    out: list[str] = [
        "Deltas are in Composite points on the same prompts; brackets give a 95% paired-bootstrap CI over prompts "
        "(10k resamples). With 40 prompts one prompt flipping one metric moves Composite by 0.625 pts, so intervals "
        "that straddle 0 should be read as \"no detectable difference\".", ""]

    def line(label: str, a: str, b: str) -> None:
        if ln := _cmp_line(summ, recs, label, a, b):
            out.append(ln)

    line("Full engine vs. raw base (headline)", "1_raw_base", "6_full_assistant_engine")
    line("Full engine vs. unsteered (+stops)", "2_raw_base_plus_stops", "6_full_assistant_engine")
    line("Full engine vs. static CAA", "4_static_caa", "6_full_assistant_engine")
    line("Hard stop strings alone", "1_raw_base", "2_raw_base_plus_stops")
    line("Static single-vector CAA", "2_raw_base_plus_stops", "4_static_caa")
    line("Module 1 alone (lifecycle schedule + dynamic stop)", "2_raw_base_plus_stops", "x1_lifecycle_only")
    line("+ Module 2 (drift nullspace erasure)", "x1_lifecycle_only", "x2_lifecycle_plus_erase")
    line("+ Module 3 (self-contrastive predictor)", "x2_lifecycle_plus_erase", "5_lifecycle_plus_nullspace")
    line("Modules 1+2+3 vs. unsteered (+stops)", "2_raw_base_plus_stops", "5_lifecycle_plus_nullspace")
    line("Modules 1+2+3 vs. static CAA", "4_static_caa", "5_lifecycle_plus_nullspace")
    line("+ Module 4 (instruction-pinning attention)", "5_lifecycle_plus_nullspace", "6_full_assistant_engine")
    line("Module 4 alone vs. unsteered (+stops)", "2_raw_base_plus_stops", "x4_pin_attention_only")
    line("Removing dynamic stop from the full engine", "6_full_assistant_engine", "x3_full_minus_dynamic_stop")
    line("Full engine vs. URIAL 3-shot reference", "3_urial_3shot_prefix", "6_full_assistant_engine")

    steps = [(lab, _delta(summ, a, b)) for lab, a, b in [
        ("Module 1 (lifecycle + stop)", "2_raw_base_plus_stops", "x1_lifecycle_only"),
        ("Module 2 (nullspace erasure)", "x1_lifecycle_only", "x2_lifecycle_plus_erase"),
        ("Module 3 (predictor)", "x2_lifecycle_plus_erase", "5_lifecycle_plus_nullspace"),
        ("Module 4 (attention pinning)", "5_lifecycle_plus_nullspace", "6_full_assistant_engine"),
    ]]
    steps = [(lab, d) for lab, d in steps if d is not None]
    if steps:
        best = max(steps, key=lambda x: x[1])
        out.append(f"- In the cumulative ablation chain, the largest single-module gain comes from **{best[0]}** "
                   f"({best[1]:+.1f} Composite pts). Chain: " + " → ".join(f"{lab} {d:+.1f}" for lab, d in steps) + ".")
    if "6_full_assistant_engine" in summ and "3_urial_3shot_prefix" in summ:
        s6, s3 = summ["6_full_assistant_engine"], summ["3_urial_3shot_prefix"]
        out.append(f"- Context cost: URIAL spends {s3['extra_prompt_tokens']:.0f} extra prompt tokens per query; the engine spends 0 "
                   f"and decodes at {s6['tok_per_s']:.1f} tok/s vs. {s3['tok_per_s']:.1f} tok/s for URIAL and "
                   f"{summ.get('2_raw_base_plus_stops', s6)['tok_per_s']:.1f} tok/s unsteered.")

    layers = diag.get("layers", [])
    if layers:
        auc = [d["stop_auc_final_vs_midpunct"] for d in layers]
        r2 = [d["predictor_cv"][f"ridge_{d.get('predictor_lambda', 10.0):g}"]["cv_r2_vs_train_mean"] for d in layers]
        lam = "CV-selected λ per layer: " + ", ".join(f"{d.get('predictor_lambda', 10.0):g}" for d in layers)
        rel = [d["v_init_rel"] for d in layers]
        act = [d.get("drift_active_directions", 0) for d in layers]
        z1 = [d.get("drift_trap_tokens_mean_z", [0.0])[0] for d in layers]
        out.append(f"- Calibration diagnostics: completion-centroid AUC (final token vs. mid-answer punctuation) ranges "
                   f"{min(auc):.2f}–{max(auc):.2f} across target layers (stop check at layer {diag.get('stop_layer')}); "
                   f"{min(act)}–{max(act)} drift direction(s) per layer pass the separation test, with trap tokens "
                   f"{min(z1):.1f}–{max(z1):.1f} clean-σ along the leading one; ‖v_init‖/‖h‖ = {min(rel):.2f}–{max(rel):.2f}.")
        if r2:
            out.append(f"- Module 3 cross-validated R² of the predicted t=0 URIAL shift (beyond the mean shift v_init; {lam}): "
                       f"{min(r2):+.2f} to {max(r2):+.2f} across layers — "
                       + ("the input-dependent component carries real signal." if max(r2) > 0.05 else
                          "with N≈80 prompts the input-dependent part adds little beyond v_init."))
    return out


def _pick_examples(recs: dict[str, list[dict]], k_min: int = 8) -> list[str]:
    have = [c for c in SHOWCASE if c in recs]
    if len(have) < 2:
        return []
    by_id = {c: {r["prompt_id"]: r for r in recs[c]} for c in have}
    ids = [pid for pid in by_id[have[0]] if all(pid in by_id[c] for c in have)]
    first, last = have[0], have[-1]

    def gain(pid: str) -> float:
        return by_id[last][pid]["metrics"]["composite"] - by_id[first][pid]["metrics"]["composite"]

    chosen: list[str] = []
    for cat in CATEGORIES:
        cand = sorted([p for p in ids if by_id[first][p]["category"] == cat], key=gain, reverse=True)
        chosen += cand[:2]
    regress = sorted([p for p in ids if p not in chosen and gain(p) < 0], key=gain)
    chosen += regress[:2]
    rest = sorted([p for p in ids if p not in chosen], key=gain, reverse=True)
    while len(chosen) < k_min and rest:
        chosen.append(rest.pop(0))
    return chosen


def _examples(recs: dict[str, list[dict]]) -> list[str]:
    chosen = _pick_examples(recs)
    have = [c for c in SHOWCASE if c in recs]
    by_id = {c: {r["prompt_id"]: r for r in recs[c]} for c in have}
    out: list[str] = []
    for n, pid in enumerate(chosen, 1):
        r0 = by_id[have[0]][pid]
        out += [f"### Example {n}: `{pid}` ({r0['category']})", "", f"**Prompt:** {r0['instruction']}", ""]
        for c in have:
            r = by_id[c][pid]
            m = r["metrics"]
            flags = " · ".join(f"{lab}={'✔' if m[k] else '✘'}" for k, lab in
                               [("turn_cross", "Turn"), ("constraint", "Constraint"), ("clean_stop", "Stop"), ("fluency", "Fluency")])
            out += [f"**`{c}`** — {flags} · stop=`{r['stop_reason']}` · {r['n_new_tokens']} tokens", "",
                    "~~~text", r["response"] if r["response"] else "(empty)", "~~~", ""]
    return out


def _diag_table(diag: dict) -> list[str]:
    layers = diag.get("layers", [])
    if not layers:
        return []
    lines = ["| Layer | ‖h‖ | ‖v_init‖/‖h‖ | ‖v_sus⊥‖/‖h‖ | cos(v_sus,v_init) | stop AUC | τ_stop | active drift dirs | trap z (dir 1) "
             "| clean / trap touched v1 (κ=1) | v2 hinge κ (p99) | clean / trap touched v2 | M3 λ | M3 CV R² |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in layers:
        lam = d.get("predictor_lambda", 10.0)
        r2 = d["predictor_cv"][f"ridge_{lam:g}"]["cv_r2_vs_train_mean"]
        lines.append(f"| {d['layer']} | {d['hidden_norm']:.1f} | {d['v_init_rel']:.3f} | {d['v_sustain_perp_norm'] / d['hidden_norm']:.3f} | "
                     f"{d['cos_init_sustain']:.2f} | {d['stop_auc_final_vs_midpunct']:.2f} | {d['tau_stop']:.2f} | "
                     f"{d.get('drift_active_directions', '–')} | {d.get('drift_trap_tokens_mean_z', [float('nan')])[0]:.2f} | "
                     f"{100 * d.get('drift_frac_clean_tokens_touched', float('nan')):.0f}% / {100 * d.get('drift_frac_trap_tokens_touched', float('nan')):.0f}% | "
                     f"{', '.join(f'{k:.2f}' for k in d.get('hinge_kappa_p99', [])) or '–'} | "
                     f"{100 * d.get('hinge_frac_clean_tokens_touched', float('nan')):.0f}% / {100 * d.get('hinge_frac_trap_tokens_touched', float('nan')):.0f}% | "
                     f"{lam:g} | {r2:+.2f} |")
    return lines


def _hparam_table(run: dict) -> list[str]:
    meta = run.get("hparams_meta") or {}
    defaults = meta.get("spec_defaults") or {}
    hp = run.get("hparams") or {}
    if not hp:
        return ["Spec default coefficients were used (no tuned hparams file)."]
    lines = ["| Coefficient | Spec default | Used (frozen) |", "|---|---|---|"]
    for k in sorted(hp):
        lines.append(f"| `{k}` | {defaults.get(k, '–')} | {hp[k]} |")
    return lines


def _appendix(dirs: list[Path]) -> list[str]:
    out: list[str] = []
    for d in dirs:
        if not (d / "summary.json").exists():
            continue
        summary, recs = _load(d)
        run = summary["run"]
        ordered = {c: recs[c] for c in ALL_CONDITIONS if c in recs}
        summ = {c: summarize(rs) for c, rs in ordered.items()}
        diag = run.get("calibration_diagnostics", {})
        pins = diag.get("pin_heads", {}).get("selected", [])
        out += [f"### `{run['model']}` — `{d.relative_to(ROOT)}` ({run['n_prompts']} prompts from `{run['prompts_file']}`, "
                f"calibration {run['calibration_meta'].get('n_kept')}/{run['calibration_meta'].get('n_prompts')} prompts, "
                f"{len(pins)} pinned heads: " + ", ".join(f"L{p['layer']}H{p['head']} ({p['delta']:+.3f})" for p in pins) + ")", "",
                *_summary_table(summ), "", "#### Per-category Composite (%)", "", *_category_table(ordered, "composite"), "",
                "#### Per-category ConstraintScore (%)", "", *_category_table(ordered, "constraint"), "",
                "#### Module contributions", "", *_analysis(summ, ordered, diag), "", *_phase2_analysis(summ, ordered, run),
                *_phase3_analysis(summ, ordered, run), *_phase4_analysis(summ, ordered, _load_instruct(d), d)]
        ue = unit_economics(d, summ, ordered, _load_instruct(d), run)
        if ue:
            out += ["#### Unit economics", "", *ue, "", *[ln.replace("### ", "#### ", 1) for ln in _sensitivity(d)]]
    return out


def _methodology(run: dict, diag: dict) -> list[str]:
    pins = diag.get("pin_heads", {})
    sel = pins.get("selected", [])
    dmax = max((p["delta"] for p in sel), default=float("nan"))
    hpm = run.get("hparams_meta") or {}
    n_dev = 15
    return [
        "## Methodology notes and caveats", "",
        "- **Drift / turn markers used by CleanStop and TurnCross** (line-anchored, case-insensitive, applied identically to every "
        "condition): new-turn labels `Question[ n]:` / `Q:` / `Q1.` / `[Question]` / `[Answer]` / `User:` / `Human:` / `Assistant:` / `A:`; "
        "forum boilerplate (`Posted by`, `N comments`, `Share |`, `Related links/questions`, `Reply`, `Upvote`, `Tags:`, `Asked by`, "
        "`See also:` …); and instruction-dataset templates leaking from pretraining (`A single-select problem`, `Available options:`, "
        "`Options are:`, `Choose your answer`, `Pick your answer`, `(a). yes`-style option lines, hallucinated `You are a helpful assistant`). "
        "The template/bracket/`A:` markers were added after inspecting the calibration negatives (Y⁻) and the 1.5B smoke-test outputs, "
        "because the raw Qwen2.5 base model's dominant failure is appending FLAN-style option blocks. All metrics are re-scored from the "
        "logged response text whenever this report is generated, so every condition is judged by the same final detector. "
        "The TurnCross *echo* rule was also tightened after inspecting test outputs: a response fails only if its opening reproduces "
        "≥90% of the instruction's words and either starts with the same word (literal re-typing) or contains nothing beyond the "
        "restatement. The first, fuzzier version (first-line similarity > 0.8) mis-flagged assistant preambles such as \"Here is the SQL "
        "query to find …\" and math answers that restate the premise. The change only removed false positives, and it moved at most "
        "one prompt per condition.",
        "- **Latency** (`ms/token`, `tok/s`) is wall-clock for the whole generation divided by generated tokens, *including the prefill "
        "pass*. This charges the URIAL condition for its 768-token prefix, which is a real per-query cost, but it inflates its ms/token "
        "relative to a decode-only measurement. Runs were executed one at a time on an otherwise idle GPU.",
        f"- **Tuning resolution.** Coefficients were chosen by coordinate search on {n_dev} dev prompts with the 1.5B model and frozen "
        f"before any test-set run ({hpm.get('provenance', 'spec defaults')}). One dev prompt flipping one metric moves dev Composite by "
        f"{100 / n_dev / 4:.2f} pts, so each adopted change (λ_erase 0.7→0.35, β_attn 1.25→0.5, α_caa 1→2) reflects one or two prompts; "
        "treat them as \"not worse than the spec defaults\", not as precise optima. The dynamic-stop coefficients (EOS boost 0/2/5/10, "
        "α_stop 0/1/2) produced *identical* dev scores, i.e. the completion check rarely changed the outcome on dev.",
        "- **URIAL is a high-context reference, not a ceiling.** Its demonstrations are deliberately thorough (multi-paragraph, with "
        "explanations), so its answers are long and often hit the 160-token cap, which costs it CleanStop/Constraint points. The same "
        "teacher answers (generated with a 256-token budget) are what the engine is calibrated toward.",
        f"- **Module 4 evidence on this model:** {len(sel)} pinned heads out of {pins.get('candidates', '?')} candidates in the 0.30N–0.75N band; "
        f"head selection requires a positive instruction-attention delta (successful − drifted), max delta {dmax:+.3f} "
        f"(mean over all candidates {pins.get('delta_mean', float('nan')):+.4f}). Raw continuations often copy the question, so many heads "
        "attend to the instruction *more* during drift than during clean answers.",
        "",
    ]


def _methodology_v2(run: dict, diag: dict, recs: dict) -> list[str]:
    if not any(c in recs for c in ("7_engine_v2_hinge_erase", "8_engine_v2_signed_attn", "9_engine_v2_full")):
        return []
    hp = run.get("hparams") or {}
    p2 = ((run.get("hparams_meta") or {}).get("phase2") or {})
    layers = diag.get("layers", [])
    p99 = [k for d in layers for k in d.get("hinge_kappa_p99", [])]
    ct = [d.get("hinge_frac_clean_tokens_touched", float("nan")) for d in layers]
    tt = [d.get("hinge_frac_trap_tokens_touched", float("nan")) for d in layers]
    tt1 = [d.get("drift_frac_trap_tokens_touched", float("nan")) for d in layers]
    fired = [r["engine_trace"].get("unembed_projections", 0) for r in recs.get("9_engine_v2_full", []) if r.get("engine_trace")]
    return [
        "### Phase 2 (engine v2) methodology", "",
        "- **Condition 7's hinge vs. v1.** v1 Module 2 already applied a continuous one-sided excess hinge: "
        "`h − λ·relu(uᵀh − c − κσ)·u` equals `h − λ·max(0, z − κ)·σ·u` with z the clean-standardised coordinate. "
        "What changes in condition 7 is the threshold, which goes from a global κ = 1.0 to a per-direction clean "
        f"99th percentile with floor {hp.get('hinge_kappa_floor', 2.2)} (calibrated p99: "
        f"{min(p99, default=float('nan')):.2f}–{max(p99, default=float('nan')):.2f}σ), plus a separately tuned strength "
        f"λ_hinge = {hp.get('lambda_hinge', '–')}. On calibration tokens the hinge touches {100 * min(ct):.1f}–{100 * max(ct):.1f}% of "
        f"clean tokens, against about 14% for v1. It touches {100 * min(tt):.0f}–{100 * max(tt):.0f}% of trap tokens, against "
        f"{100 * min(tt1):.0f}–{100 * max(tt1):.0f}% for v1. The p99 lands close to the mean trap excursion, so roughly half of "
        "trap tokens fall below it.",
        "- **Negation parser** (`engine/negation.py`, deterministic regex). Triggers: without / not / never / avoid / "
        "excluding / except / don't / do not / no. Quantity limits (`no more than`, `not fewer than` …) are skipped. "
        "A clause runs from the trigger to the next clause punctuation outside quotes. Targets are the quoted strings in "
        "the clause, or else the single-word items after the filler verbs (`mentioning`, `using the word` …), split on "
        "or / and / commas. Multi-word items such as \"the Eiffel Tower\" count as part of the negated span but are not "
        f"lexical targets. Signed attention used `neg_span_mode = {hp.get('neg_span_mode', '–')}` (`clause` repels the "
        "whole \"without using the word 'X'\" clause, `target` repels only X) and β_neg = "
        f"{hp.get('beta_neg_attn', '–')}; positive pinning keeps the Phase-1 β = {hp.get('beta_attn', '–')}.",
        "- **Condition 9's projection is close to a soft lexical ban.** The pre-LM-head state is projected off the unit "
        "unembedding direction of every single-token surface, capitalised and singular/plural variant of each negated "
        "target. That is equivalent to scaling those tokens' logits by (1 − γ) when they are positive, plus spill-over "
        f"onto tokens with correlated unembeddings. Final setting: γ = {hp.get('gamma_unembed', '–')}, "
        f"centred = {hp.get('unembed_center', '–')}. Across the 40 test prompts the projection fired {sum(fired)} times. "
        "Its negative-constraint gains are partly true by construction, since the banned surface forms become hard to "
        "emit. The verifier still catches inflections outside the banned set (`forced`, `balancing`) and "
        "non-single-token spellings.",
        "- **Determinism fix (applied before the condition-9 numbers shown here).** The banned tokens are projected one "
        "after another, and with γ > 1 those projections do not commute. The first implementation took the token order "
        "from a Python set, so it depended on the per-process hash seed, and condition 9 outputs varied between "
        "processes (verified on dev under two seeds). Banned ids are now sorted by token id. Condition 9's dev tuning was "
        "re-run under that order, and both models were re-generated on test. The earlier, non-reproducible test logs are "
        "kept as `9_engine_v2_full.nondeterministic_order.jsonl.bak`. Under the deterministic order the dev search "
        "selected the brief's literal *uncentred* directions at γ = 1.5 (dev 91.7, against 90.0 centred at γ = 1.5 and "
        "88.3 for both forms at γ = 2.0). The earlier Phase-2 finding that centring is required (85.0 vs. 90.0) was an "
        "artifact of the random order and is withdrawn.",
        f"- **Dev resolution for Phase 2.** The dev set has only 3 negation prompts, so the choices for "
        "neg_span_mode, β_neg and γ rest on 3 prompts. Coordinate descent compared the two span modes at β_neg = 1.0 "
        "(both scored 90.0) and then swept β_neg in clause mode only; `target` mode at β_neg = 0.5 was never run, so the "
        "clause-vs-target choice is not a finding. "
        "Dev composites: " + ", ".join(f"`{k}` {v:.1f}" for k, v in p2.get("dev_composite", {}).items()) + ".",
        "",
    ]


def _methodology_v21(run: dict, recs: dict) -> list[str]:
    if "10_engine_v2_1" not in recs:
        return []
    hp = run.get("hparams") or {}
    p3 = ((run.get("hparams_meta") or {}).get("phase3") or {})
    return [
        "### Phase 3 (engine v2.1, condition 10) methodology", "",
        "- **Definition.** Condition 7 plus four changes, with no signed attention (pinning uses the Phase-1 "
        f"β = {hp.get('beta_attn', '–')}):",
        f"  1. **Boundary-coupled hinge** (κ_boundary = {hp.get('kappa_boundary', '–')} instead of the p99 threshold). It "
        "applies inside a boundary window: the last token contains `. ! ? \\n`; or the step is ≤ 2 tokens after a "
        "newline; or the previous step's completion score was ≥ τ_stop − 0.08. The score clause has to lag by one step, "
        "because the current step's score is only computed at the stop layer, after most hinge layers have run.",
        f"  2. **Drift-assisted dynamic stop.** It triggers if the stop-layer drift z ≥ {hp.get('tau_drift_stop', '–')} "
        "while the score is ≥ τ_stop − 0.12. The stop check now starts at t ≥ 6.",
        f"  3. **Sibling-orthogonalised unembedding projection.** Banned directions are mean-centred, as the brief "
        f"specifies, and orthogonalised against the top-{hp.get('sibling_k', '–')} favoured non-banned word tokens. A "
        "fallback applies when the remaining overlap is below 0.3.",
        f"  4. **Anti-echo guard.** It suppresses the negation trigger words (`without`, `mentioning` …) for the first "
        f"{hp.get('anti_echo_window', '–')} generated tokens.",
        f"- **Inherited and cost details.** Condition 10's projection strength γ = {hp.get('gamma_unembed', '–')} was "
        "inherited from the condition-9 re-tune, which used uncentred directions; it was not re-tuned for the centred "
        "directions condition 10 uses. Condition 10 decodes about 12% slower than condition 7, for two reasons: the "
        "boundary clause syncs the stop-layer score at every step, and on negation prompts the sibling basis needs an "
        "extra full-vocabulary matrix-vector product per step.",
        "- **Is the boundary-onset diagnosis right?** Teacher-forced on 24 *calibration* answers followed by a "
        "`[Question]` or FLAN continuation, the mean drift z is:",
        "  - −0.4 on clean mid-answer tokens and 1.1 on the last answer token;",
        "  - **3.9σ at the boundary newline** (3B; 3.65σ on 1.5B), against a mean p99 threshold of 3.1σ (2.9σ);",
        "  - 4.3–4.9σ on the first drift tokens.",
        "  So the onset is *not* below the p99 threshold, and the v2 hinge already fires there. It only removes 75% of "
        "the small excess above ~3.1σ, which leaves the state near 3.3σ. A lower boundary threshold helps because it "
        "removes much more of an excursion the hinge already sees.",
        "- **Is the sibling-suppression diagnosis right?** Yes.",
        "  - Centred unembedding cosines between ' Python' and ' Java' / ' Ruby' / ' JavaScript' are 0.45 / 0.38 / 0.36, "
        "and ' Assembly' is 0.02, the only language left in the withdrawn non-deterministic condition-9 answer to `t_ng_02` "
        "(the deterministic re-run answers it correctly, so that example no longer demonstrates the effect).",
        "  - On a real 1.5B state, the plain projection moved the top-8 clean siblings' relative logits by −3.6 to −13.7 "
        "(' violin' −13.7). The sibling-orthogonalised projection moved them by exactly 0.00, while the banned tokens "
        "fell 10–31 logits.",
        "- **Dev evidence for the new thresholds** (1.5B, 15 prompts; start = the brief's values):",
        "  - the drift-stop threshold (1.5 / 1.8 / 2.2) and the anti-echo window (8 / 12 / 16) produced *identical* dev "
        "scores, so they are untested by dev;",
        "  - κ_boundary 1.0 → 1.5 moved dev from 85.0 to 91.7, and κ_boundary 2.0 scored 90.0;",
        "  - sibling K = 4 beat K = 8 (93.3 vs. 91.7), a single-prompt difference.",
        "  Dev composites: " + ", ".join(f"`{k}` {v:.1f}" for k, v in p3.get("dev_composite", {}).items()) + ".",
        "",
    ]


def _methodology_v22(run: dict, recs: dict, irecs: dict) -> list[str]:
    if "11_engine_v2_2_lean" not in recs and not irecs:
        return []
    hp = run.get("hparams") or {}
    out = ["### Phase 4 (v2.2 lean engine and post-trained reference) methodology", ""]
    if "11_engine_v2_2_lean" in recs:
        out += [
            "- **Condition 11 (`11_engine_v2_2_lean`)** keeps the two v2.1 mechanisms that the leave-one-outs credit: the "
            f"boundary-coupled hinge (κ_boundary = {hp.get('kappa_boundary', '–')}) and the sibling-orthogonalised "
            f"projection (K = {hp.get('sibling_k', '–')}, γ = {hp.get('gamma_unembed', '–')}, centred). It drops the "
            "drift-assisted stop and the anti-echo guard. It has no new coefficients, and the stop check still starts at "
            "t ≥ 6 as in v2.1. **Its mechanism set was chosen from the test-set leave-one-outs, so its test score is not "
            "a held-out estimate.** Condition 10 remains the dev-selected headline.",
            "- **Lean execution (`boundary_trigger = \"text\"`, `sibling_fast = True`):**",
            "  - **Text-only boundary trigger.** The boundary window opens only when the last generated token contains "
            "`. ! ? \\n`, or within 2 steps after a newline. The v2.1 score clause is dropped, so the stop-layer "
            "completion score is computed only on stop-check steps (punctuation tokens), as in condition 7. No other step "
            "has an extra host sync.",
            "  - **Sync-free projection.** The sibling-orthogonalised projection computes the same arithmetic without host "
            "synchronisation: the fallback and Gram-Schmidt branches are `torch.where` selections, and the trace "
            "counters stay on device until the answer ends.",
            "  - **Smaller sibling search.** The sibling search multiplies only the pre-indexed word-token rows of the "
            "unembedding (tokens containing an ASCII letter; 61.8% of Qwen2.5's vocabulary) instead of the whole "
            "vocabulary.",
            "- **Why a second vocabulary product remains.**",
            "  - The siblings are the top-K tokens of the *unprojected* logits, while the emitted logits must come from "
            "the *projected* state, so an exact implementation reads the unembedding twice on negation prompts.",
            "  - Applying the update in logit space (`logits + W·Δh`) does not avoid this: `W·Δh` is itself a "
            "full-vocabulary product.",
            "  - Any re-association that is not bit-identical in bf16 changes greedy outputs at near-ties.",
            "  - The lean path therefore keeps the exact computation, makes the search product 38% smaller, and removes "
            "the synchronisation stalls, which `scripts/bench_speed.py` shows were most of the cost.",
        ]
    if irecs:
        out += [
            "- **Post-trained reference (conditions 12 and 13).** The official `-Instruct` checkpoints were run on the "
            "same 40 prompts with the same greedy loop, `max_new_tokens = 160`, stop strings and frozen verifiers.",
            "  - **Frames.** `_bare` uses the identical `Question: {instruction}\\nAnswer:` frame. `_chat` uses the "
            "checkpoint's chat template, which includes Qwen's default system prompt; its *Extra ctx tokens* column "
            "counts the template's overhead over the bare frame.",
            "  - **Turn endings.** Generation ends on either turn-ending id in the checkpoint's `generation_config` "
            "(`<|im_end|>`, `<|endoftext|>`).",
            "  - **Decoding.** Qwen's recommended sampling settings (temperature 0.7, top-p 0.8, repetition penalty "
            "1.05) are *not* used: every condition decodes greedily.",
            "- **Condition 13** attaches only condition 11's lexical operator to the Instruct checkpoint. It uses the "
            "same frozen base-dev coefficients and no calibration artifact, and nothing was tuned on Instruct. It can "
            "only change the 8 prompts that contain a parsed negation.",
            "- **Post-training recovery** is (C_engine − C_raw) / (C_Instruct − C_raw), with a 95% paired-bootstrap CI "
            "over prompts (10k resamples). The headline uses the raw base without stop strings (condition 1), as in "
            "every other comparison; the version over condition 2 (with stop strings) is listed too.",
            "- **Frozen verifiers.** The verifiers were written for base-model failure modes and were not changed for "
            "Instruct outputs. In particular, CleanStop fails answers that hit the 160-token cap mid-sentence, which "
            "penalises verbose answers.",
            "- **Sensitivity runs** (separate tags, never mixed with the protocol logs):",
            "  - `test40_512tok` reruns conditions 1, 10, 11, 12 and 13 (chat) with `max_new_tokens = 512`. It was "
            "planned after seeing Instruct's 160-token truncations and before any 512-token output existed. Its "
            "latencies are not comparable: part of it ran alongside the 1.5B KL job.",
            "  - `test40_sdpa` reruns conditions 1, 2, 12 and 13 (chat) on HF's SDPA attention. The engine conditions "
            "cannot run there: SDPA has no slot for the Module-4 bias, and `AssistantEngine` refuses `pin_attention` "
            "without `steer_eager`.",
            "- **Clean-token KL** (`eval/alignment_tax.py`). Each condition is teacher-forced along the same clean "
            "answers; engine conditions replay them one token per pass with the full per-step schedule, including the "
            "EOS boosts and the lexical ban. The reference is the raw base in the bare frame, decoded step by step. The "
            "answers are URIAL-teacher text in the base model's style, not Instruct-style text.",
        ]
    return out + [""]


def _instruct_cases(recs: dict, irecs: dict, k: int = 6) -> list[str]:
    """Verbatim engine-vs-Instruct cases, chosen by rule: every prompt where condition 13 changed the Instruct answer,
    then the first prompts (by id) where engine v2.1 and Instruct-chat Composite differ."""
    a, b, e = "12_official_instruct_chat", "13_instruct_plus_orth_ban_chat", "10_engine_v2_1"
    if a not in irecs or e not in recs:
        return []
    conds = [(e, recs), ("11_engine_v2_2_lean", recs), (a, irecs), (b, irecs)]
    conds = [(c, src) for c, src in conds if c in src]
    by = {c: {r["prompt_id"]: r for r in src[c]} for c, src in conds}
    changed = sorted(p for p in by[a] if b in by and by[b][p]["response"] != by[a][p]["response"])
    differ = sorted(p for p in by[a] if by[a][p]["metrics"]["composite"] != by[e][p]["metrics"]["composite"])
    picks = changed + [p for p in differ if p not in changed][:k]
    if not picks:
        return []
    out = ["## 8. Engine vs. post-trained Instruct (verbatim)", "",
           f"Selected by rule: the {len(changed)} prompts where the sibling-orthogonal ban changed the Instruct (chat) "
           f"answer, then the first {min(k, len(differ))} prompts (by id) whose Composite differs between `{e}` and "
           f"`{a}`. Outputs are shown after stop-string truncation.", ""]
    for pid in picks:
        r0 = by[a][pid]
        out += [f"### `{pid}` ({r0['category']})", "", f"**Prompt:** {r0['instruction']}", ""]
        for c, _ in conds:
            r = by[c][pid]
            m = r["metrics"]
            flags = " · ".join(f"{lab}={'✔' if m[key] else '✘'}" for key, lab in
                               [("turn_cross", "Turn"), ("constraint", "Constraint"), ("clean_stop", "Stop"), ("fluency", "Fluency")])
            out += [f"**`{c}`** — {flags} · stop=`{r['stop_reason']}` · {r['n_new_tokens']} tokens", "",
                    "~~~text", r["response"] if r["response"] else "(empty)", "~~~", ""]
    return out


TARGETED = [("t_ng_02", "v1 repetition loop (3B)"),
            ("t_qa_01", "v1 option-label loop; v2 FLAN leak (3B)"),
            ("t_ng_01", "lexical negative constraint 'force'"), ("t_ng_04", "lexical negative constraint 'balance'; v2 evasion"),
            ("t_rm_01", "v2 post-answer [Question] drift (3B)"), ("t_rm_05", "v2 post-answer [Question] drift (3B)"),
            ("t_rm_07", "v2 post-answer [Question] drift (3B)"), ("t_rm_08", "v2 post-answer [Question] drift (3B)")]


def _targeted_cases(recs: dict[str, list[dict]]) -> list[str]:
    conds = [c for c in ("1_raw_base", "6_full_assistant_engine", "9_engine_v2_full", "10_engine_v2_1") if c in recs]
    if "9_engine_v2_full" not in conds:
        return []
    by_id = {c: {r["prompt_id"]: r for r in recs[c]} for c in conds}
    out = ["## 6. Targeted failure cases (verbatim: raw base vs. v1 vs. v2 vs. v2.1)", ""]
    for pid, why in TARGETED:
        if not all(pid in by_id[c] for c in conds):
            continue
        r0 = by_id[conds[0]][pid]
        out += [f"### `{pid}`: {why}", "", f"**Prompt:** {r0['instruction']}", ""]
        for c in conds:
            r = by_id[c][pid]
            m = r["metrics"]
            flags = " · ".join(f"{lab}={'✔' if m[k] else '✘'}" for k, lab in
                               [("turn_cross", "Turn"), ("constraint", "Constraint"), ("clean_stop", "Stop"), ("fluency", "Fluency")])
            tr = r.get("engine_trace") or {}
            extra = ""
            if tr:
                extra = " · " + "; ".join(f"{k}: `{v!r}`" for k, v in tr.items())
            reasons = [e["reason"] for e in r.get("stop_events", []) if "reason" in e]
            if reasons:
                extra += f" · EOS boosts: `{reasons}`"
            out += [f"**`{c}`** — {flags} · stop=`{r['stop_reason']}` · {r['n_new_tokens']} tokens{extra}", "",
                    "~~~text", r["response"] if r["response"] else "(empty)", "~~~", ""]
    return out


def recovery_ci(recs: dict, raw: str, eng: str, inst: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """Post-training recovery (C_eng - C_raw) / (C_inst - C_raw) with a paired-bootstrap 95% CI over prompts."""
    import random
    by = {c: {r["prompt_id"]: 100.0 * r["metrics"]["composite"] for r in recs[c]} for c in (raw, eng, inst)}
    ids = [i for i in by[raw] if i in by[eng] and i in by[inst]]

    def ratio(sample: list[str]) -> float | None:
        n = len(sample)
        r, e, s = (sum(by[c][i] for i in sample) / n for c in (raw, eng, inst))
        return (e - r) / (s - r) if s - r > 0 else None

    rng = random.Random(seed)
    boots = [ratio([rng.choice(ids) for _ in ids]) for _ in range(n_boot)]
    ok = sorted(b for b in boots if b is not None)
    return {"point": ratio(ids), "lo": ok[int(0.025 * len(ok))] if ok else None,
            "hi": ok[int(0.975 * len(ok)) - 1] if ok else None, "undefined_frac": 1 - len(ok) / n_boot}


def _diff_ci(recs: dict, a: str, b: str) -> str:
    sa, sb = (100.0 * sum(r["metrics"]["composite"] for r in recs[c]) / len(recs[c]) for c in (a, b))
    lo, hi = paired_bootstrap_ci(recs, a, b)
    return f"{sb - sa:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def _recovery_table(rows: list[tuple[str, dict, str, str, str]]) -> list[str]:
    """rows: (stack label, records, raw, engine, instruct). A ratio is reported only if its denominator is resolved."""
    out = ["| Stack | Engine | Instruct reference | Instruct − raw base | Engine − Instruct | Post-training recovery |",
           "|---|---|---|---|---|---|"]
    for lab, recs, raw, eng, inst in rows:
        if not all(c in recs for c in (raw, eng, inst)):
            continue
        rc = recovery_ci(recs, raw, eng, inst)
        if rc["point"] is None:
            rec = "undefined (Instruct ≤ raw base)"
        elif rc["undefined_frac"] > 0.025:
            rec = f"not resolved (point {100 * rc['point']:.0f}%; denominator ≤ 0 in {100 * rc['undefined_frac']:.0f}% of resamples)"
        else:
            rec = f"**{100 * rc['point']:.0f}%** [{100 * rc['lo']:.0f}%, {100 * rc['hi']:.0f}%]"
        out.append(f"| {lab} | `{eng}` | `{inst}` | {_diff_ci(recs, raw, inst)} | {_diff_ci(recs, inst, eng)} | {rec} |")
    return out


def _negation_ids() -> set[str]:
    """Calibration and test prompt ids whose instruction contains a parsed negation target."""
    from engine.negation import parse_negations
    ids = set()
    for f in ("calibration_prompts.json", "test_prompts.json", "dev_prompts.json"):
        for p in json.loads((ROOT / "data" / f).read_text()):
            if any(n.targets for n in parse_negations(p["instruction"])):
                ids.add(p["id"])
    return ids


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _phase4_analysis(summ: dict, recs: dict, irecs: dict, out_dir: Path) -> list[str]:
    """Condition 11 (lean engine) and conditions 12-13 (post-trained reference) comparisons."""
    if "11_engine_v2_2_lean" not in recs and not irecs:
        return []
    allrecs = {**recs, **irecs}
    allsumm = {**summ, **{c: summarize(rs) for c, rs in irecs.items()}}
    out = ["### Phase 4 (v2.2 lean engine, post-trained reference) comparisons", ""]
    if "11_engine_v2_2_lean" in recs:
        for ref in ("1_raw_base", "3_urial_3shot_prefix", "10_engine_v2_1", "x9_v2_1_minus_anti_echo"):
            if ln := _cmp_line(allsumm, allrecs, f"v2.2 lean vs. `{ref}`", ref, "11_engine_v2_2_lean"):
                out.append(ln)
        if "x9_v2_1_minus_anti_echo" in recs:
            x9 = {r["prompt_id"]: r for r in recs["x9_v2_1_minus_anti_echo"]}
            same = [r["prompt_id"] for r in recs["11_engine_v2_2_lean"] if r["response"] == x9[r["prompt_id"]]["response"]]
            diff = [r["prompt_id"] for r in recs["11_engine_v2_2_lean"] if r["prompt_id"] not in set(same)]
            drift = {pid for pid, r in x9.items() if any(e.get("reason") == "drift" for e in r.get("stop_events", []))}
            out.append(f"- **Byte identity with `x9`:** {len(same)}/{len(same) + len(diff)} responses identical. "
                       f"{len(drift)} x9 prompts had a drift-stop EOS boost (the mechanism condition 11 drops); the "
                       f"{len(diff)} differing prompts are {', '.join(f'`{d}`' for d in diff) or 'none'}"
                       + (f", of which {len(set(diff) & drift)} had such a boost" if diff else "") + ".")
    for inst in ("12_official_instruct_chat", "12_official_instruct_bare"):
        if inst in allrecs:
            if ln := _cmp_line(allsumm, allrecs, f"Post-training's own gain ({inst.rsplit('_', 1)[1]} frame)", "1_raw_base", inst):
                out.append(ln)
    rows = [(f"protocol, raw = `{raw}`", allrecs, raw, eng, inst)
            for raw in ("1_raw_base", "2_raw_base_plus_stops")
            for eng in ("10_engine_v2_1", "11_engine_v2_2_lean")
            for inst in ("12_official_instruct_chat", "12_official_instruct_bare")]
    rt = _recovery_table(rows)
    if len(rt) > 2:
        out += ["", "**Engine vs. post-training** (Composite points, 95% paired-bootstrap CIs; recovery = "
                "(C_engine − C_raw) / (C_Instruct − C_raw), reported only when its denominator is resolved):", "", *rt, ""]
    for frame in ("chat", "bare"):
        a, b = f"12_official_instruct_{frame}", f"13_instruct_plus_orth_ban_{frame}"
        if a in irecs and b in irecs:
            if ln := _cmp_line(allsumm, allrecs, f"Sibling-orthogonal ban on Instruct ({frame} frame)", a, b):
                out.append(ln)
            na, nb = (_category_metric(allrecs, c, "negative_constraints", "constraint") for c in (a, b))
            ra = {r["prompt_id"]: r for r in irecs[a]}
            changed = [r["prompt_id"] for r in irecs[b] if r["response"] != ra[r["prompt_id"]]["response"]]
            out.append(f"  - negative_constraints ConstraintScore {na:.1f}% → {nb:.1f}%; responses changed on "
                       f"{len(changed)} prompts ({', '.join(f'`{c}`' for c in changed) or 'none'}).")
    bench = _read_json(out_dir.parent / "speed_bench.json")
    if bench:
        out += ["", f"**Same-session decode speed** (`{bench['model']}`, {bench['n_prompts']} test prompts, conditions "
                "interleaved prompt by prompt, `scripts/bench_speed.py`):", "",
                "| Condition | tok/s | tok/s, negation prompts | tok/s, other prompts | ms/token | identical to log |",
                "|---|---|---|---|---|---|"]
        for c, o in bench["conditions"].items():
            out.append(f"| `{c}` | {o['tok_per_s']:.2f} | {_fmt(o['tok_per_s_negation'], 2)} | {_fmt(o['tok_per_s_other'], 2)} "
                       f"| {o['ms_per_token']:.1f} | {o['identical_to_log']}/{len(o['per_prompt'])} |")
    return out + [""]


def _mb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.1f} MB"


def unit_economics(out_dir: Path, summ: dict, recs: dict, irecs: dict, run: dict) -> list[str]:
    """One row per way of getting assistant behaviour: what it costs and what it buys (also used by the blog post)."""
    if not irecs:
        return []
    slug = out_dir.parent.name
    tax = _read_json(out_dir.parent / "alignment_tax.json").get("summary", {})
    delta = _read_json(ROOT / "results" / (slug + INSTRUCT_SUFFIX) / "weight_delta.json")
    bench = _read_json(out_dir.parent / "speed_bench.json").get("conditions", {})
    long = {c: summarize(rs) for c, rs in _variant_recs(out_dir, "_512tok").items()}
    art = ROOT / "artifacts" / slug / "calib80" / "calibration.pt"
    secs = (run.get("calibration_meta") or {}).get("calibration_seconds")
    calib = f"{secs / 60:.0f} min of forward passes on 80 prompts" if secs else "not recorded for this model"
    isumm = {c: summarize(rs) for c, rs in irecs.items()}
    post = "SFT on > 1M examples + multi-stage RL; compute not disclosed"
    wchg = f"{100 * delta['params_changed_frac']:.1f}% of {delta['params'] / 1e9:.2f}B" if delta else "all"
    wsize = f"{_mb(delta['instruct_weights_bytes'])} (whole checkpoint)" if delta else "whole checkpoint"

    per = _read_json(out_dir.parent / "alignment_tax.json").get("per_text", {}).get("heldout", {})
    neg = _negation_ids()

    def kl(c: str) -> str:
        if c == "1_raw_base":
            return "0 (reference)"
        held = tax.get("heldout", {}).get(c)
        if not held:
            return "–"
        rs = [r for r in per.get(c, []) if r["id"] not in neg]
        n = sum(r["n"] for r in rs)
        return f"{held['mean_kl']:.3f} / {sum(r['kl_sum'] for r in rs) / n:.3f}" if n else f"{held['mean_kl']:.3f}"

    rows = [
        ("Raw base", "1_raw_base", summ, "0", "–", "none"),
        ("URIAL 3-shot prefix", "3_urial_3shot_prefix", summ, "0", "768-token prompt", "none"),
        ("Static CAA", "4_static_caa", summ, "0", f"{_mb(art.stat().st_size)} (same artifact)", calib),
        ("**Engine v2.1**", "10_engine_v2_1", summ, "0", _mb(art.stat().st_size), calib),
        ("**Engine v2.2 lean** (test-informed)", "11_engine_v2_2_lean", summ, "0", _mb(art.stat().st_size), calib),
        ("Official Instruct, chat template", "12_official_instruct_chat", isumm, wchg, wsize, post),
        ("Official Instruct, bare frame", "12_official_instruct_bare", isumm, wchg, wsize, post),
        ("Instruct + sibling-orth ban, chat", "13_instruct_plus_orth_ban_chat", isumm, wchg, wsize + " + 0", post),
    ]
    lines = ["| Approach | Weights changed | Artifact / weight delta | One-off cost | Extra context tokens | Clean-token KL, "
             "held-out: all / no-negation prompts (nats) | Composite, 160 tokens | Composite, 512 tokens | tok/s |",
             "|" + "---|" * 9]
    for label, c, src, w, size, cost in rows:
        if c not in src:
            continue
        s = src[c]
        speed = bench[c]["tok_per_s"] if c in bench else s["tok_per_s"]
        lines.append(f"| {label} (`{c}`) | {w} | {size} | {cost} | {s['extra_prompt_tokens']:.0f} | {kl(c)} "
                     f"| **{s['composite']:.1f}** | {_fmt(long[c]['composite']) if c in long else '–'} | {speed:.1f} |")
    return lines


def _economics_section(out_dir: Path, summ: dict, recs: dict, irecs: dict, run: dict) -> list[str]:
    ue = unit_economics(out_dir, summ, recs, irecs, run)
    if not ue:
        return []
    tax = _read_json(out_dir.parent / "alignment_tax.json")
    isumm = {c: summarize(rs) for c, rs in irecs.items()}
    out = ["## 7. Economics versus post-training (Phase 4)", "",
           "### Unit economics", "", *ue, "",
           "- **Weights changed / size.** The engine edits no weights; its artifact is the calibration file read by the "
           "hooks. The Instruct row compares every bf16 tensor of the two public checkpoints "
           "(`scripts/weight_delta.py`).",
           "- **Clean-token KL.** Mean D_KL(P_raw ‖ P_condition) in nats per token, teacher-forced at steps t > 4 on "
           "URIAL's clean answers to the test prompts (`eval/alignment_tax.py`; the calibration-answer version is in the "
           "detail table below). *no-negation* drops the 8 prompts with a parsed negation, where the lexical ban "
           "suppresses words the teacher answer uses by design. Lower means the condition leaves the base model's "
           "predictions on good assistant text more intact. It is a distance, not a quality score: Instruct is "
           "*meant* to move the distribution. The single-pass vs. step-by-step numerical floor of the raw base is "
           + (f"{tax['summary']['calibration']['raw_base_single_pass']['mean_kl']:.4f}" if tax else "–") + " nats.",
           "- **Composite, 512 tokens** reruns the same prompts with a 512-token budget (tag `test40_512tok`). "
           "Instruct writes longer answers, and the 160-token cap cuts many of them mid-sentence, which fails CleanStop.",
           "- **Stack.** Every row ran in bf16 on the steer_eager (HF-eager) attention, which Module 4 needs. The SDPA "
           "rows of the sensitivity table quantify what that costs the attention-agnostic conditions.",
           "- **tok/s** for conditions 2, 7, 10, x9 and 11 is the same-session benchmark; other rows are from their logs.",
           "- **One-off cost** for Instruct is the Qwen2.5 technical report's description of post-training (arXiv "
           "2412.15115); the engine's is measured wall time on the M4 Pro (trajectory generation and state collection).", ""]
    if isumm:
        out += ["### Post-trained reference conditions (all values in %)", "", *_summary_table(isumm), "",
                "Stop reasons (eos / stop_string / max_tokens): " + "; ".join(
                    f"`{c}` {s['stop_reasons']['eos']}/{s['stop_reasons']['stop_string']}/{s['stop_reasons']['max_tokens']}"
                    for c, s in isumm.items()), "",
                "### Composite by category, engine vs. Instruct (%)", "",
                *_category_table({**{c: recs[c] for c in ("1_raw_base", "10_engine_v2_1", "11_engine_v2_2_lean") if c in recs},
                                  **irecs}, "composite"), ""]
    out += _sensitivity(out_dir)
    if tax:
        neg = _negation_ids()

        def pooled(rows: list[dict], keep) -> float | None:
            rs = [r for r in rows if keep(r["id"])]
            n = sum(r["n"] for r in rs)
            return sum(r["kl_sum"] for r in rs) / n if n else None

        out += ["### Clean-token KL distortion in detail", "",
                "Mean KL is pooled over tokens; *median answer* is the median of per-answer means; *no negation* pools "
                "only answers to prompts without a parsed negation (on the others the engine's lexical ban suppresses "
                "words the teacher answer uses, by design).", "",
                "| Condition | KL, calib | KL, held-out | median answer, calib / held-out | no negation, calib / held-out | "
                "KL t ≤ 4, calib | top-1 agreement, calib / held-out |", "|---|---|---|---|---|---|---|"]
        for c, v in tax["summary"]["calibration"].items():
            h = tax["summary"].get("heldout", {}).get(c, {})
            pc, ph = tax["per_text"]["calibration"][c], tax["per_text"].get("heldout", {}).get(c, [])
            nn = f"{_fmt(pooled(pc, lambda i: i not in neg), 4)} / {_fmt(pooled(ph, lambda i: i not in neg), 4)}"
            out.append(f"| `{c}` | {v['mean_kl']:.4f} | {_fmt(h.get('mean_kl'), 4)} | {v['median_text_kl']:.4f} / "
                       f"{_fmt(h.get('median_text_kl'), 4)} | {nn} | {v['mean_kl_early']:.3f} | "
                       f"{100 * v['top1_agree']:.1f}% / {_fmt(100 * h['top1_agree'] if h else None)}% |")
        cal = tax["summary"]["calibration"]
        out += ["", f"{cal['raw_base_single_pass']['n_texts']} calibration answers ({cal['raw_base_single_pass']['n_tokens']} "
                f"tokens at t > 4) and {tax['summary'].get('heldout', {}).get('raw_base_single_pass', {}).get('n_texts', 0)} "
                f"held-out answers ({tax['summary'].get('heldout', {}).get('raw_base_single_pass', {}).get('n_tokens', 0)} tokens).", ""]
    return out


def write_report(out_dir: Path, path: Path, appendix_dirs: list[Path] | None = None) -> None:
    summary, recs = _load(out_dir)
    run = summary["run"]
    ordered = {c: recs[c] for c in ALL_CONDITIONS if c in recs}
    summ = {c: summarize(rs) for c, rs in ordered.items()}
    diag = run.get("calibration_diagnostics", {})
    meta = run.get("calibration_meta", {})
    hpm = run.get("hparams_meta") or {}
    pins = diag.get("pin_heads", {}).get("selected", [])
    irecs = _load_instruct(out_dir)

    L: list[str] = [
        f"# RESULTS — Zero-Weight-Update Assistant Engine on `{run['model']}`", "",
        f"_Generated automatically by `eval/report.py` on {date.today().isoformat()} from the logs in "
        f"`{out_dir.relative_to(ROOT)}/`. Greedy decoding, `max_new_tokens={run['max_new_tokens']}`, "
        f"{run['n_prompts']} held-out prompts (`data/{run['prompts_file']}`), device `{run['device']}` / `{run['dtype']}`, torch {run['torch']}._", "",
        "No weights were modified and no gradients were computed: every condition runs the same frozen checkpoint; "
        "steered conditions only add forward hooks (residual-stream edits and a pre-softmax attention bias).", "",
        "## Setup", "",
        f"- **Calibration:** {meta.get('n_kept', '?')}/{meta.get('n_prompts', '?')} calibration prompts kept after filtering URIAL teacher answers "
        f"(clean stop, >10 chars, distinct-4-gram ≥ 0.75); target layers {meta.get('target_layers')} of {meta.get('n_layers')}; "
        f"stop check at layer {diag.get('stop_layer')}; static-CAA layer {diag.get('caa_layer')}; "
        f"{len(pins)} instruction-pinning heads ({', '.join('L%dH%d' % (p['layer'], p['head']) for p in pins)}).",
        f"- **Hyperparameters:** `{run.get('hparams_file')}` — {hpm.get('provenance', 'spec defaults')}.",
        "- **Prompt frame for all steered conditions:** `Question: {instruction}\\nAnswer:` (0 extra tokens). "
        "Stop strings `[\"\\nQuestion:\", \"\\nUser:\"]` for conditions 2–6.",
        "- **Metric definitions:** TurnCross = starts answering (no new question / echo / empty / garbage / script switch); "
        "Constraint = every programmatic check attached to the prompt passes (for `direct_qa` this is ≥2 expected topic keywords; "
        "math = expected final answer; code = parses / required function & SQL clauses; format/negative = exact rule); "
        "CleanStop = no hallucinated turn/FAQ/forum markers and not truncated mid-sentence or looping at max tokens; "
        "Fluency = distinct-4-gram ratio > 0.75 and > 10 chars; Composite = mean of the four. "
        "**NaturalEOS** (extra) = generation ended with the model's own `<eos>` token rather than a stop string or the token cap — "
        "with stop strings active, CleanStop alone cannot distinguish \"model stopped\" from \"we cut it off\".", "",
        "## 1. Ablation summary (all values in %)", "", *_summary_table(summ), "",
        "Stop reasons (eos / stop_string / max_tokens): " + "; ".join(
            f"`{c}` {s['stop_reasons']['eos']}/{s['stop_reasons']['stop_string']}/{s['stop_reasons']['max_tokens']}" for c, s in summ.items()), "",
        "## 2. Per-category breakdown", "", "### Composite (%)", "", *_category_table(ordered, "composite"), "",
        "### ConstraintScore (%)", "", *_category_table(ordered, "constraint"), "",
        "## 3. Which modules contributed, and why", "", *_analysis(summ, ordered, diag), "",
        *_phase2_analysis(summ, ordered, run), *_phase3_analysis(summ, ordered, run),
        *_phase4_analysis(summ, ordered, irecs, out_dir),
    ]
    notes = out_dir / "analysis_notes.md"
    if notes.exists():
        L += [notes.read_text().strip(), ""]
    L += _methodology(run, diag)
    L += _methodology_v2(run, diag, ordered)
    L += _methodology_v21(run, ordered)
    L += _methodology_v22(run, ordered, irecs)
    L += ["## 4. Calibration diagnostics (per target layer)", "", *_diag_table(diag), "",
          "### Coefficients", "", *_hparam_table(run), "",
          "## 5. Qualitative side-by-side examples (verbatim)", "",
          "Outputs are shown exactly as returned after stop-string truncation and whitespace stripping.", "",
          *_examples(ordered), *_targeted_cases(ordered), *_economics_section(out_dir, summ, ordered, irecs, run),
          *_instruct_cases(ordered, irecs)]
    if appendix_dirs:
        app = _appendix(appendix_dirs)
        if app:
            L += ["## Appendix: other runs", "", *app]
    path.write_text("\n".join(L).rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--appendix", nargs="*", default=[], help="other result dirs to summarise in an appendix")
    ap.add_argument("--refresh-summary", action="store_true",
                    help="rewrite summary.json's per-condition block from the re-scored logs (no generation)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.refresh_summary:
        summary, recs = _load(out_dir)
        summary["summary"] = {c: summarize(rs) for c, rs in recs.items()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir, Path(args.out) if args.out else out_dir / "RESULTS.md", [Path(a).resolve() for a in args.appendix])


if __name__ == "__main__":
    main()
