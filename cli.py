"""Interactive side-by-side comparison: raw base model vs. the steered assistant engine.

    python cli.py --model Qwen/Qwen2.5-3B                       # interactive loop (raw vs. static CAA vs. engine v2.1)
    python cli.py --model Qwen/Qwen2.5-1.5B --prompt "Explain quantum computing in 3 bullet points."
    python cli.py --modes raw,urial,engine,v21                   # choose which columns to show
"""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

from engine import AssistantEngine, CalibConfig, SteeringArtifact, calibrate, encode_prompt, generate, load_model
from engine.prompts import EVAL_STOP_STRINGS
from eval.conditions import ALL_CONDITIONS, engine_config, load_hparams, load_prompts

ROOT = Path(__file__).resolve().parent
MODES = {
    "raw": ("raw base", "1_raw_base"),
    "stops": ("raw + stops", "2_raw_base_plus_stops"),
    "urial": ("URIAL 3-shot", "3_urial_3shot_prefix"),
    "caa": ("static CAA", "4_static_caa"),
    "m123": ("Modules 1+2+3", "5_lifecycle_plus_nullspace"),
    "engine": ("engine v1", "6_full_assistant_engine"),
    "v2": ("engine v2 (hinge)", "7_engine_v2_hinge_erase"),
    "v21": ("engine v2.1", "10_engine_v2_1"),
}


def _columns(blocks: list[tuple[str, str]]) -> str:
    width = shutil.get_terminal_size((160, 40)).columns
    n = len(blocks)
    col = max(30, (width - 3 * (n - 1)) // n)
    if col * n + 3 * (n - 1) > width or n == 1:
        return "\n\n".join(f"── {title} " + "─" * max(0, width - len(title) - 4) + f"\n{body}" for title, body in blocks)
    wrapped = []
    for title, body in blocks:
        lines = [title.center(col), "─" * col]
        for para in body.splitlines() or [""]:
            lines += textwrap.wrap(para, col, replace_whitespace=False, drop_whitespace=False) or [""]
        wrapped.append(lines)
    height = max(len(w) for w in wrapped)
    rows = []
    for i in range(height):
        rows.append(" │ ".join((w[i] if i < len(w) else "").ljust(col) for w in wrapped))
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--modes", default="raw,caa,v21", help=f"comma list of {list(MODES)}")
    ap.add_argument("--prompt", default=None, help="run once and exit")
    ap.add_argument("--hparams", default=str(ROOT / "configs" / "hparams.json"))
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    lm = load_model(args.model)
    art_path = Path(args.artifact) if args.artifact else ROOT / "artifacts" / lm.slug / "calib80" / "calibration.pt"
    if art_path.exists():
        art = SteeringArtifact.load(art_path)
    else:
        print(f"No calibration at {art_path}; calibrating on the 80 built-in prompts (one-off) ...")
        art = calibrate(lm, load_prompts(ROOT / "data" / "calibration_prompts.json"), CalibConfig(), art_path.parent)
    hparams = load_hparams(Path(args.hparams))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    engines = {}
    for m in modes:
        _, _, toggles = ALL_CONDITIONS[MODES[m][1]]
        engines[m] = AssistantEngine(lm, art, engine_config(toggles, hparams)) if toggles else None

    def answer(instruction: str) -> None:
        blocks = []
        for m in modes:
            title, cond = MODES[m]
            urial, stops, _ = ALL_CONDITIONS[cond]
            enc = encode_prompt(lm.tokenizer, instruction, urial=urial)
            eng = engines[m]
            if eng is not None:
                eng.attach()
            try:
                res = generate(lm, enc, args.max_new_tokens, EVAL_STOP_STRINGS if stops else None, eng)
            finally:
                if eng is not None:
                    eng.detach()
            blocks.append((f"{title} [{res.stop_reason}, {res.n_new_tokens} tok, {res.ms_per_token:.0f} ms/tok]", res.text))
        print(_columns(blocks) + "\n")

    if args.prompt:
        answer(args.prompt)
        return
    print(f"{lm.name} — modes: {', '.join(MODES[m][0] for m in modes)}. Empty line or Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q)


if __name__ == "__main__":
    main()
