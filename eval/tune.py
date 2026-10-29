"""Coordinate-search tuning of engine coefficients on the *dev* prompts (disjoint from the 40 test prompts).

Stage A tunes Module 1-3 coefficients on condition 5, stage B tunes beta_attn on condition 6 (Module 4),
stage C tunes the static-CAA multiplier for condition 4 so the baseline is not a strawman.
A candidate replaces the incumbent only if it improves dev Composite by more than ``--min-gain``.

    python -m eval.tune --model Qwen/Qwen2.5-1.5B --out configs/hparams.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from engine import EngineConfig, SteeringArtifact, load_model

from .conditions import load_prompts, run_condition, summarize

ROOT = Path(__file__).resolve().parents[1]

STAGE_A = [  # condition 5
    ("alpha_init", [0.5, 1.0, 2.0, 4.0]),
    ("alpha_sustain", [0.0, 0.35, 1.0, 2.0, 4.0]),
    ("lambda_erase", [0.0, 0.35, 0.7, 1.0]),
    ("erase_kappa", [0.0, 0.5, 1.0, 2.0]),
    ("predictor_weight", [0.0, 0.5, 1.0]),
    ("stop_logit_boost", [0.0, 2.0, 5.0, 10.0]),
    ("alpha_stop", [0.0, 1.0, 2.0]),
]
STAGE_B = [("beta_attn", [0.5, 1.25, 2.5, 4.0])]          # condition 6
STAGE_C = [("alpha_caa", [0.5, 1.0, 2.0, 4.0, 8.0])]      # condition 4

# Phase 2 (engine v2); v1 coefficients stay frozen at their Phase-1 values.
V2_DEFAULTS = {"hinge_kappa_floor": 2.2, "lambda_hinge": 0.75, "neg_span_mode": "clause", "beta_neg_attn": 1.0,
               "unembed_center": False, "gamma_unembed": 2.0}   # starting points = the Phase-2 spec's literal choices
STAGE_7 = [("hinge_kappa_floor", [2.2, 2.6, 3.0]), ("lambda_hinge", [0.5, 0.75, 1.0])]
STAGE_8 = [("neg_span_mode", ["clause", "target"]), ("beta_neg_attn", [0.5, 1.0, 2.0])]
STAGE_9 = [("unembed_center", [False, True]), ("gamma_unembed", [1.5, 2.0, 3.0])]

# Phase 3 (engine v2.1, condition 10); starting points = the v2.1 brief's values
V21_DEFAULTS = {"kappa_boundary": 1.0, "tau_drift_stop": 1.8, "sibling_k": 8, "anti_echo_window": 12}
STAGE_10 = [("tau_drift_stop", [1.5, 1.8, 2.2]), ("kappa_boundary", [1.0, 1.5, 2.0]),
            ("sibling_k", [4, 8, 16]), ("anti_echo_window", [8, 12, 16])]


def _key(cond: str, hp: dict) -> str:
    return hashlib.sha1(json.dumps([cond, sorted(hp.items())]).encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--artifact", default=None, help="calibration.pt (default: artifacts/<model>/calib80)")
    ap.add_argument("--dev", default=str(ROOT / "data" / "dev_prompts.json"))
    ap.add_argument("--out", default=str(ROOT / "configs" / "hparams.json"))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="2 = engine-v2 coefficients only; 3 = engine-v2.1 (condition 10) thresholds only")
    args = ap.parse_args()

    lm = load_model(args.model)
    art_path = Path(args.artifact) if args.artifact else ROOT / "artifacts" / lm.slug / "calib80" / "calibration.pt"
    art = SteeringArtifact.load(art_path)
    prompts = load_prompts(Path(args.dev))
    log_path = ROOT / "results" / lm.slug / "tuning_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            cache[r["key"]] = r

    def evaluate(cond: str, hp: dict) -> dict:
        k = _key(cond, hp)
        if k in cache:
            return cache[k]["summary"]
        t0 = time.time()
        s = summarize(run_condition(lm, art, cond, prompts, hp, args.max_new_tokens, desc=f"tune {cond}"))
        rec = {"key": k, "condition": cond, "hparams": hp, "summary": s, "seconds": round(time.time() - t0, 1)}
        cache[k] = rec
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {cond} {json.dumps({kk: hp[kk] for kk in sorted(hp)})} -> composite {s['composite']:.1f} "
              f"(T{s['turn_cross']:.0f} C{s['constraint']:.0f} S{s['clean_stop']:.0f} F{s['fluency']:.0f} "
              f"EOS{s['natural_eos']:.0f})", flush=True)
        return s

    defaults = asdict(EngineConfig())
    tunable = [n for st in (STAGE_A, STAGE_B, STAGE_C) for n, _ in st]
    hp = {n: defaults[n] for n in tunable}
    history = []
    if args.phase == 2:
        existing = json.loads(Path(args.out).read_text())
        hp = {**{k: v for k, v in existing["hparams"].items() if k in tunable}, **V2_DEFAULTS}
    if args.phase == 3:
        existing = json.loads(Path(args.out).read_text())
        hp = {**{k: v for k, v in existing["hparams"].items() if k not in V21_DEFAULTS}, **V21_DEFAULTS}

    def search(cond: str, stage: list, passes: int) -> None:
        best = evaluate(cond, hp)
        history.append({"condition": cond, "start": dict(hp), "composite": best["composite"]})
        for p in range(passes):
            changed = False
            for name, grid in stage:
                for v in grid:
                    if v == hp[name]:
                        continue
                    cand = {**hp, name: v}
                    s = evaluate(cond, cand)
                    if s["composite"] > best["composite"] + args.min_gain:
                        hp[name], best, changed = v, s, True
                print(f"[tune:{cond} pass {p + 1}] {name} = {hp[name]} (composite {best['composite']:.1f})", flush=True)
            if not changed:
                break
        history.append({"condition": cond, "end": dict(hp), "composite": best["composite"]})

    if args.phase == 2:
        phase2(args, hp, existing, evaluate, search, history, prompts, art_path)
        return
    if args.phase == 3:
        phase3(args, hp, existing, evaluate, search, history, prompts, art_path)
        return

    print("[tune] stage A: Modules 1-3 on condition 5")
    search("5_lifecycle_plus_nullspace", STAGE_A, args.passes)
    print("[tune] stage B: Module 4 on condition 6")
    search("6_full_assistant_engine", STAGE_B, 1)
    print("[tune] stage C: static CAA multiplier on condition 4")
    search("4_static_caa", STAGE_C, 1)

    out = {
        "hparams": hp,
        "meta": {
            "provenance": f"coordinate search on {len(prompts)} dev prompts (data/{Path(args.dev).name}, disjoint from the "
                          f"test set) with {args.model}; frozen afterwards",
            "tuned_on_model": args.model, "artifact": str(art_path.relative_to(ROOT)), "spec_defaults": {n: defaults[n] for n in tunable},
            "history": history, "min_gain": args.min_gain,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[tune] wrote {args.out}: {json.dumps(hp)}")


def phase2(args, hp: dict, existing: dict, evaluate, search, history: list, prompts: list, art_path: Path) -> None:
    v1 = dict(existing["hparams"])
    print("[tune:v2] x5: full engine with Module 2 off (no new coefficients)")
    s_x5 = evaluate("x5_full_minus_erase", hp)
    print("[tune:v2] condition 7: hinge threshold floor and strength")
    search("7_engine_v2_hinge_erase", STAGE_7, 1)
    s_7 = evaluate("7_engine_v2_hinge_erase", hp)
    rank = lambda s: (s["composite"], s["clean_stop"], s["fluency"])  # noqa: E731
    base = "7_engine_v2_hinge_erase" if rank(s_7) > rank(s_x5) else "x5_full_minus_erase"   # exact tie -> simpler x5
    hp["v2_base"] = base
    print(f"[tune:v2] base for 8/9: {base} (x5 {s_x5['composite']:.1f} vs 7 {s_7['composite']:.1f})")
    print("[tune:v2] condition 8: signed attention")
    search("8_engine_v2_signed_attn", STAGE_8, 1)
    print("[tune:v2] condition 9: unembedding projection")
    search("9_engine_v2_full", STAGE_9, 2)
    dev = {c: evaluate(c, hp)["composite"] for c in
           ("x5_full_minus_erase", "7_engine_v2_hinge_erase", "8_engine_v2_signed_attn", "9_engine_v2_full")}
    order = list(dev)                                         # increasing number of active mechanisms
    best = max(dev, key=lambda c: (dev[c], -order.index(c)))  # ties go to the simpler condition
    v2_keys = [*V2_DEFAULTS, "v2_base"]
    merged = {**v1, **{k: hp[k] for k in v2_keys}}
    assert all(merged[k] == v1[k] for k in v1 if k not in v2_keys), "phase 2 must not change v1 coefficients"
    existing["hparams"] = merged
    existing["meta"]["phase2"] = {
        "provenance": f"coordinate search on {len(prompts)} dev prompts with {args.model}; started from the Phase-2 spec "
                      f"values {V2_DEFAULTS}; a change was adopted only if dev Composite improved by > {args.min_gain}",
        "artifact": str(art_path.relative_to(ROOT)), "history": history, "dev_composite": dev,
        "dev_selected_best_v2": best, "v2_base_choice": {"x5": s_x5, "7": s_7, "chosen": base},
    }
    Path(args.out).write_text(json.dumps(existing, indent=2))
    print(f"[tune:v2] wrote {args.out}: {json.dumps({k: hp[k] for k in v2_keys})}; dev best v2 = {best} {dev}")


def phase3(args, hp: dict, existing: dict, evaluate, search, history: list, prompts: list, art_path: Path) -> None:
    frozen = dict(existing["hparams"])
    print("[tune:v2.1] condition 10: boundary hinge, drift stop, sibling-orthogonal projection, anti-echo")
    search("10_engine_v2_1", STAGE_10, 1)
    dev = {c: evaluate(c, hp)["composite"] for c in ("7_engine_v2_hinge_erase", "9_engine_v2_full", "10_engine_v2_1")}
    merged = {**frozen, **{k: hp[k] for k in V21_DEFAULTS}}
    assert all(merged[k] == frozen[k] for k in frozen if k not in V21_DEFAULTS), "phase 3 must not change earlier coefficients"
    existing["hparams"] = merged
    existing["meta"]["phase3"] = {
        "provenance": f"coordinate search on {len(prompts)} dev prompts with {args.model}; started from the v2.1 brief's "
                      f"values {V21_DEFAULTS}; a change was adopted only if dev Composite improved by > {args.min_gain}",
        "artifact": str(art_path.relative_to(ROOT)), "history": history, "dev_composite": dev,
    }
    Path(args.out).write_text(json.dumps(existing, indent=2))
    print(f"[tune:v2.1] wrote {args.out}: {json.dumps({k: hp[k] for k in V21_DEFAULTS})}; dev {dev}")


if __name__ == "__main__":
    main()
