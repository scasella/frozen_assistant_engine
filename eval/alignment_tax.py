"""Clean-token KL distortion ("alignment tax") of assistant conditions relative to the raw base model.

Every condition is teacher-forced along the same clean assistant answers and compared with the raw base model in the
bare frame: mean per-token D_KL(P_raw || P_condition) in nats over answer steps t > 4 (t = 0 predicts the first answer
token), so the decaying init push of the lifecycle schedule is excluded. Engine conditions replay the answer one token
per pass with the engine's per-step schedule; engine-free conditions use one forward pass, whose numerical gap to the
step-by-step raw base is reported as ``raw_base_single_pass`` (the noise floor).

Text sets: ``calibration``, the calibration teacher answers that ended cleanly (the engine was calibrated on these
states, so this favours it); ``heldout``, the URIAL condition's test answers that pass the same filter (the teacher's
generation process on prompts the engine never saw).

    python -m eval.alignment_tax --model Qwen/Qwen2.5-3B
writes results/<model>/alignment_tax.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from engine import AssistantEngine, SteeringArtifact, encode_prompt, free_memory, load_model, teacher_force
from engine.calibration import ended_cleanly, keep_trajectory

from .conditions import ALL_CONDITIONS, INSTRUCT_CONDITIONS, INSTRUCT_SUFFIX, encode_for, engine_config, load_hparams

ROOT = Path(__file__).resolve().parents[1]
BASE_CONDITIONS = ["3_urial_3shot_prefix", "4_static_caa", "10_engine_v2_1", "11_engine_v2_2_lean"]
INSTRUCT_REFS = ["12_official_instruct_bare", "12_official_instruct_chat"]
T_MIN = 5


def _texts(lm, slug: str, tag: str) -> dict[str, list[dict]]:
    traj = json.loads((ROOT / "artifacts" / slug / "calib80" / "trajectories.json").read_text())
    calib = [{"id": t["id"], "instruction": t["instruction"], "ids": t["pos_ids"]}
             for t in traj if keep_trajectory(t) and ended_cleanly(t)]
    heldout = []
    log = ROOT / "results" / slug / tag / "3_urial_3shot_prefix.jsonl"
    for line in log.read_text().splitlines():
        r = json.loads(line)
        ids = lm.tokenizer(" " + r["response"], add_special_tokens=False)["input_ids"]
        tr = {"pos_text": r["response"], "pos_ids": ids, "pos_stop": r["stop_reason"]}
        if keep_trajectory(tr) and ended_cleanly(tr):
            heldout.append({"id": r["prompt_id"], "instruction": r["instruction"], "ids": ids})
    return {"calibration": calib, "heldout": heldout}


def _stats(ref: torch.Tensor, lp: torch.Tensor) -> dict:
    kl = (ref.exp() * (ref - lp)).sum(-1).to("cpu").to(torch.float64)
    agree = (ref.argmax(-1) == lp.argmax(-1)).cpu()
    late = slice(T_MIN, None)
    return {"kl_sum": float(kl[late].sum()), "n": int(kl[late].numel()), "top1_agree": int(agree[late].sum()),
            "kl_sum_early": float(kl[:T_MIN].sum()), "n_early": int(kl[:T_MIN].numel())}


def summarize_tax(per_text: list[dict]) -> dict:
    n = sum(r["n"] for r in per_text)
    ne = sum(r["n_early"] for r in per_text)
    per = sorted(r["kl_sum"] / r["n"] for r in per_text if r["n"])
    return {"mean_kl": sum(r["kl_sum"] for r in per_text) / max(n, 1), "n_tokens": n, "n_texts": len(per_text),
            "median_text_kl": per[len(per) // 2] if per else None,
            "top1_agree": sum(r["top1_agree"] for r in per_text) / max(n, 1),
            "mean_kl_early": sum(r["kl_sum_early"] for r in per_text) / max(ne, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--tag", default="test40")
    ap.add_argument("--max-texts", type=int, default=None, help="smoke runs only")
    args = ap.parse_args()

    lm = load_model(args.model)
    ilm = load_model(args.model + INSTRUCT_SUFFIX)
    art = SteeringArtifact.load(ROOT / "artifacts" / lm.slug / "calib80" / "calibration.pt")
    hp = load_hparams(ROOT / "configs" / "hparams.json")
    engines = {c: AssistantEngine(lm, art, engine_config(ALL_CONDITIONS[c][2], hp))
               for c in BASE_CONDITIONS if ALL_CONDITIONS[c][2] is not None}
    sets = _texts(lm, lm.slug, args.tag)
    per: dict[str, dict[str, list[dict]]] = {}
    t0 = time.time()
    for set_name, texts in sets.items():
        texts = texts[: args.max_texts]
        per[set_name] = {c: [] for c in ["raw_base_single_pass", *BASE_CONDITIONS, *INSTRUCT_REFS]}
        for tx in tqdm(texts, desc=set_name):
            ids, inst = tx["ids"], tx["instruction"]
            bare = encode_prompt(lm.tokenizer, inst)
            ref = teacher_force(lm, bare, ids)
            rows = {"raw_base_single_pass": teacher_force(lm, bare, ids, stepwise=False)}
            for c in BASE_CONDITIONS:
                frame, _, toggles = ALL_CONDITIONS[c]
                enc = encode_for(lm.tokenizer, inst, frame)
                if toggles is None:
                    rows[c] = teacher_force(lm, enc, ids, stepwise=False)
                else:
                    with engines[c] as eng:
                        rows[c] = teacher_force(lm, enc, ids, eng)
            for c in INSTRUCT_REFS:
                rows[c] = teacher_force(ilm, encode_for(ilm.tokenizer, inst, INSTRUCT_CONDITIONS[c][0]), ids,
                                        stepwise=False)
            for c, lp in rows.items():
                per[set_name][c].append({"id": tx["id"], **_stats(ref, lp)})
            del ref, rows
        free_memory()
    out = {
        "model": lm.name, "instruct_model": ilm.name, "direction": "KL(P_raw_base || P_condition), nats per token",
        "positions": f"answer steps t >= {T_MIN} (t = 0 predicts the first answer token); *_early = t < {T_MIN}",
        "reference": "raw base, bare frame, step-by-step decoding",
        "seconds": round(time.time() - t0, 1),
        "summary": {s: {c: summarize_tax(v) for c, v in d.items()} for s, d in per.items()},
        "per_text": per,
    }
    path = ROOT / "results" / lm.slug / "alignment_tax.json"
    path.write_text(json.dumps(out, indent=1))
    for s, d in out["summary"].items():
        for c, v in d.items():
            print(f"{s:12s} {c:28s} KL {v['mean_kl']:.4f} (t<{T_MIN}: {v['mean_kl_early']:.3f}) "
                  f"top1 {100 * v['top1_agree']:.1f}% over {v['n_tokens']} tokens / {v['n_texts']} texts")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
