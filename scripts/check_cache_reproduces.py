"""Regenerate a few cached test prompts and check they are byte-identical to the logged responses.

The cache guard in eval/conditions.py compares engine *configurations*; this script checks *code*: run it after any
change to engine/ or eval/conditions.py that could touch a code path an existing condition uses.

    python scripts/check_cache_reproduces.py --model Qwen/Qwen2.5-1.5B --conditions 6_full_assistant_engine 10_engine_v2_1

``--set key=value`` regenerates a condition with engine-config overrides and compares against *that condition's* log,
e.g. ``--conditions x9_v2_1_minus_anti_echo --set sibling_fast=true`` checks that the sync-free projection reproduces
x9 byte for byte. ``--timing`` prints same-session decode speed (run the conditions you compare in one invocation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import SteeringArtifact, load_model  # noqa: E402
from eval.conditions import ALL_CONDITIONS, load_hparams, load_prompts, run_condition, summarize  # noqa: E402


def _value(v: str):
    return {"true": True, "false": False}.get(v.lower(), int(v) if v.lstrip("-").isdigit() else v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--n-prompts", type=int, default=6, help="category-balanced subset of the test prompts")
    ap.add_argument("--tag", default="test40")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="engine-config overrides")
    ap.add_argument("--timing", action="store_true", help="print tok/s of each regenerated condition")
    args = ap.parse_args()
    overrides = {k: _value(v) for k, v in (kv.split("=", 1) for kv in args.set)}

    lm = load_model(args.model)
    art = SteeringArtifact.load(ROOT / "artifacts" / lm.slug / "calib80" / "calibration.pt")
    prompts = load_prompts(ROOT / "data" / "test_prompts.json", args.n_prompts)
    hp = load_hparams(ROOT / "configs" / "hparams.json")
    failures = 0
    for cond in args.conditions:
        log = ROOT / "results" / lm.slug / args.tag / f"{cond}.jsonl"
        cached = {json.loads(l)["prompt_id"]: json.loads(l)["response"] for l in log.read_text().splitlines() if l.strip()}
        table = None
        if overrides:
            urial, stops, toggles = ALL_CONDITIONS[cond]
            table = {cond: (urial, stops, {**toggles, **overrides})}
        new = run_condition(lm, art, cond, prompts, hp, 160, None, conditions=table)
        diff = [r["prompt_id"] for r in new if r["response"] != cached[r["prompt_id"]]]
        failures += len(diff)
        label = cond + (f" with {overrides}" if overrides else "")
        print(f"{label}: {len(new) - len(diff)}/{len(new)} byte-identical to {log.relative_to(ROOT)}" + (f"; differ: {diff}" if diff else ""))
        if args.timing:
            s = summarize(new)
            print(f"  timing: {s['tok_per_s']:.2f} tok/s, {s['ms_per_token']:.1f} ms/token, {s['avg_new_tokens']:.1f} tokens/prompt")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
