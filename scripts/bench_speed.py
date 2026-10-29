"""Same-session decode-speed benchmark: conditions interleaved prompt by prompt, outputs checked against their logs.

Cached logs were generated hours apart, so their latencies mix in thermal and background-load drift. This script
regenerates every prompt under each condition in rotating order within one process and reports tok/s, overall and
split into negation prompts (where the unembedding projection runs) and the rest.

    python scripts/bench_speed.py --model Qwen/Qwen2.5-3B --conditions 7_engine_v2_hinge_erase x9_v2_1_minus_anti_echo 11_engine_v2_2_lean

Writes results/<model>/speed_bench.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import AssistantEngine, SteeringArtifact, generate, load_model  # noqa: E402
from engine.prompts import EVAL_STOP_STRINGS  # noqa: E402
from eval.conditions import ALL_CONDITIONS, encode_for, engine_config, load_hparams, load_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--tag", default="test40")
    ap.add_argument("--n-prompts", type=int, default=None)
    args = ap.parse_args()

    lm = load_model(args.model)
    art = SteeringArtifact.load(ROOT / "artifacts" / lm.slug / "calib80" / "calibration.pt")
    hp = load_hparams(ROOT / "configs" / "hparams.json")
    prompts = load_prompts(ROOT / "data" / "test_prompts.json", args.n_prompts)
    engines = {c: AssistantEngine(lm, art, engine_config(ALL_CONDITIONS[c][2], hp)) if ALL_CONDITIONS[c][2] else None
               for c in args.conditions}
    cached = {}
    for c in args.conditions:
        log = ROOT / "results" / lm.slug / args.tag / f"{c}.jsonl"
        cached[c] = {json.loads(l)["prompt_id"]: json.loads(l)["response"] for l in log.read_text().splitlines() if l.strip()}
    runs: dict[str, list[dict]] = {c: [] for c in args.conditions}
    for i, p in enumerate(tqdm(prompts, desc="prompts")):
        order = args.conditions[i % len(args.conditions):] + args.conditions[: i % len(args.conditions)]
        for c in order:
            frame, stops, _ = ALL_CONDITIONS[c]
            eng = engines[c]
            if eng is not None:
                eng.attach()
            try:
                torch.manual_seed(0)
                res = generate(lm, encode_for(lm.tokenizer, p["instruction"], frame), 160,
                               EVAL_STOP_STRINGS if stops else None, eng, lm.eos_ids or None)
            finally:
                if eng is not None:
                    eng.detach()
            runs[c].append({"prompt_id": p["id"], "negation": bool(res.engine_trace.get("forbidden_tokens")),
                            "tokens": res.n_new_tokens, "seconds": res.latency_s,
                            "identical_to_log": res.text == cached[c][p["id"]]})

    def rate(rs: list[dict]) -> float | None:
        s = sum(r["seconds"] for r in rs)
        return sum(r["tokens"] for r in rs) / s if s else None

    neg_ids = {r["prompt_id"] for rs in runs.values() for r in rs if r["negation"]}
    out = {"model": lm.name, "n_prompts": len(prompts), "order": "rotating per prompt", "conditions": {}}
    for c, rs in runs.items():
        neg = [r for r in rs if r["prompt_id"] in neg_ids]
        rest = [r for r in rs if r["prompt_id"] not in neg_ids]
        out["conditions"][c] = {
            "tok_per_s": rate(rs), "tok_per_s_negation": rate(neg), "tok_per_s_other": rate(rest),
            "ms_per_token": 1000 * sum(r["seconds"] for r in rs) / max(sum(r["tokens"] for r in rs), 1),
            "tokens": sum(r["tokens"] for r in rs), "identical_to_log": sum(r["identical_to_log"] for r in rs),
            "per_prompt": rs,
        }
        o = out["conditions"][c]
        print(f"{c:32s} {o['tok_per_s']:.2f} tok/s (negation {o['tok_per_s_negation'] or 0:.2f}, other "
              f"{o['tok_per_s_other'] or 0:.2f}); {o['identical_to_log']}/{len(rs)} identical to log")
    path = ROOT / "results" / lm.slug / "speed_bench.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
