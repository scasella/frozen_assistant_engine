"""Run calibration (if needed) and the 6-way ablation benchmark; write JSON logs and RESULTS.md.

Example:
    python -m eval.benchmark --model Qwen/Qwen2.5-3B --hparams configs/hparams.json --results-md RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from engine import CalibConfig, SteeringArtifact, calibrate, free_memory, load_model

from .conditions import (ALL_CONDITIONS, CONDITIONS, EXTRA_CONDITIONS, INSTRUCT_CONDITIONS, INSTRUCT_SUFFIX,
                         V2_CONDITIONS, load_hparams, load_prompts, run_condition, summarize)
from .report import write_report
from .verifiers import score_response

ROOT = Path(__file__).resolve().parents[1]


def get_artifact(lm, n_calib: int | None, recalibrate: bool, calib_cfg: CalibConfig) -> tuple[SteeringArtifact, Path]:
    prompts = load_prompts(ROOT / "data" / "calibration_prompts.json", n_calib)
    out_dir = ROOT / "artifacts" / lm.slug / f"calib{len(prompts)}"
    path = out_dir / "calibration.pt"
    if path.exists() and not recalibrate:
        print(f"[benchmark] loading calibration artifact {path.relative_to(ROOT)}")
        return SteeringArtifact.load(path), out_dir
    t0 = time.time()
    art = calibrate(lm, prompts, calib_cfg, out_dir)
    art.meta["calibration_seconds"] = round(time.time() - t0, 1)
    art.save(path)
    print(f"[benchmark] calibration done in {time.time() - t0:.0f}s -> {path.relative_to(ROOT)}")
    return art, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="steer_eager", choices=["steer_eager", "sdpa"],
                    help="sdpa = numerics sensitivity only (no Module 4); its tag must end in _sdpa")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "test_prompts.json"))
    ap.add_argument("--n-prompts", type=int, default=None, help="evaluate a category-balanced subset")
    ap.add_argument("--n-calib", type=int, default=None, help="calibrate on a category-balanced subset")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help=f"choices: {list(ALL_CONDITIONS)}; for an *{INSTRUCT_SUFFIX} model: {list(INSTRUCT_CONDITIONS)}")
    ap.add_argument("--extra-ablations", action="store_true", help="also run the x* single-module ablations")
    ap.add_argument("--v2", action="store_true", help="run only the Phase-2/3 conditions (x5, 7-10); others load from cache")
    ap.add_argument("--all", action="store_true", help="run every condition (1-10, x1-x9)")
    ap.add_argument("--hparams", default=str(ROOT / "configs" / "hparams.json"))
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--tag", default=None, help="results sub-directory name")
    ap.add_argument("--results-md", default=None, help="also write the report to this path (e.g. RESULTS.md)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached generations for this tag")
    ap.add_argument("--appendix", nargs="*", default=[], help="other result dirs to summarise in the report appendix")
    args = ap.parse_args()

    instruct = args.model.endswith(INSTRUCT_SUFFIX)   # post-trained reference: no calibration, no report of its own
    table = INSTRUCT_CONDITIONS if instruct else ALL_CONDITIONS
    if args.conditions is None:
        args.conditions = list(INSTRUCT_CONDITIONS if instruct else CONDITIONS)
    unknown = [c for c in args.conditions if c not in table]
    if unknown:
        raise SystemExit(f"unknown conditions for {args.model}: {unknown}")
    torch.manual_seed(0)
    lm = load_model(args.model, args.device, args.dtype, args.attn)
    print(f"[benchmark] {lm.name}: {lm.n_layers} layers, d={lm.d_model}, heads={lm.n_heads}, device={lm.device}")
    art, art_dir = (None, None) if instruct else get_artifact(lm, args.n_calib, args.recalibrate, CalibConfig())
    free_memory()

    hp_path = Path(args.hparams)
    hparams = load_hparams(hp_path)
    prompts = load_prompts(Path(args.prompts), args.n_prompts)
    conditions = list(args.conditions) + (list(EXTRA_CONDITIONS) if args.extra_ablations else [])
    if args.v2:
        conditions = ["x5_full_minus_erase", *V2_CONDITIONS]
    if args.all:
        conditions = list(ALL_CONDITIONS)
    tag = args.tag or f"test{len(prompts)}" + ("" if args.attn == "steer_eager" else f"_{args.attn}")
    if args.attn != "steer_eager" and not tag.endswith(f"_{args.attn}"):
        raise SystemExit(f"--attn {args.attn} runs must use a tag ending in _{args.attn} (the cache guard compares engine "
                         "configs only, so attention implementations must never share a log)")
    out_dir = ROOT / "results" / lm.slug / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, list[dict]] = {}
    for name in conditions:
        f = out_dir / f"{name}.jsonl"
        if args.fresh and f.exists():
            f.unlink()
        t0 = time.time()
        recs = run_condition(lm, art, name, prompts, hparams, args.max_new_tokens, f, conditions=table)
        all_records[name] = recs
        s = summarize(recs)
        print(f"[{name}] composite={s['composite']:.1f} turn={s['turn_cross']:.0f} cons={s['constraint']:.0f} "
              f"stop={s['clean_stop']:.0f} flu={s['fluency']:.0f} eos={s['natural_eos']:.0f} "
              f"{s['tok_per_s']:.1f} tok/s ({time.time() - t0:.0f}s)")
        free_memory()

    run_info = {
        "model": lm.name, "tag": tag, "n_prompts": len(prompts), "prompts_file": str(Path(args.prompts).name),
        "max_new_tokens": args.max_new_tokens, "hparams_file": str(hp_path.relative_to(ROOT)) if hp_path.exists() else None,
        "hparams": hparams, "hparams_meta": json.loads(hp_path.read_text()).get("meta") if hp_path.exists() else None,
        "artifact_dir": str(art_dir.relative_to(ROOT)) if art_dir else None, "calibration_meta": art.meta if art else None,
        "calibration_diagnostics": art.diagnostics if art else None, "device": str(lm.device), "dtype": args.dtype,
        "attn": lm.attn,
        "torch": torch.__version__,
    }
    for name in table:  # include every condition already logged under this tag (cached runs)
        f = out_dir / f"{name}.jsonl"
        if name not in all_records and f.exists():
            recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            by_id = {p["id"]: p for p in prompts}
            for r in recs:  # re-score with the current verifiers, as the report does
                r["metrics"] = score_response(r["response"], r["stop_reason"], by_id[r["prompt_id"]])
            all_records[name] = recs
    all_records = {c: all_records[c] for c in table if c in all_records}
    summary = {name: summarize(recs) for name, recs in all_records.items()}
    (out_dir / "summary.json").write_text(json.dumps({"run": run_info, "summary": summary}, indent=2))
    if instruct:
        print(f"[benchmark] wrote {out_dir / 'summary.json'}; the base model's report reads these logs")
        return
    if tag.endswith(("_sdpa", "_512tok")):   # sensitivity tags: summary only; the protocol report (test40) reads them
        (out_dir / "RESULTS.md").unlink(missing_ok=True)
        print(f"[benchmark] wrote {out_dir / 'summary.json'}; sensitivity tag, no report of its own")
        return
    report_paths = [out_dir / "RESULTS.md"] + ([Path(args.results_md)] if args.results_md else [])
    for p in report_paths:
        write_report(out_dir, p, [Path(a).resolve() for a in args.appendix])
        print(f"[benchmark] wrote {p}")


if __name__ == "__main__":
    main()
