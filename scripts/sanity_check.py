"""Pre-flight checks: custom attention == stock eager, tokenisation alignment, generation loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import encode_prompt, generate, load_model  # noqa: E402
from engine.prompts import URIAL_PREFIX, minimal_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()
    lm = load_model(args.model)
    tok = lm.tokenizer
    print(f"loaded {lm.name}: layers={lm.n_layers} d={lm.d_model} heads={lm.n_heads} eos={lm.eos_token_id} "
          f"attn={lm.model.config._attn_implementation} device={lm.device} dtype={lm.dtype}")

    inst = "Explain quantum computing in 3 bullet points."
    enc = encode_prompt(tok, inst)
    print("instruction tokens:", repr(tok.decode([enc.ids[i] for i in enc.inst_positions])))
    joint = tok(URIAL_PREFIX + minimal_prompt(inst))["input_ids"]
    enc_u = encode_prompt(tok, inst, urial=True)
    assert list(joint) == enc_u.ids, "URIAL concatenation differs from joint tokenisation"
    print(f"URIAL prefix tokens: {enc_u.extra_tokens}  (joint tokenisation matches)")

    ids = torch.tensor([enc_u.ids], device=lm.device)
    with torch.no_grad():
        out_custom = lm.model(input_ids=ids).logits[0, -8:].float()
        lm.model.config._attn_implementation = "eager"
        out_eager = lm.model(input_ids=ids).logits[0, -8:].float()
        lm.model.config._attn_implementation = "steer_eager"
    diff = (out_custom - out_eager).abs().max().item()
    print(f"max |logit diff| steer_eager vs eager: {diff:.3e}")
    assert diff < 1e-3, "custom attention is not equivalent to eager"

    res = generate(lm, enc, max_new_tokens=60, stop_strings=["\nQuestion:", "\nUser:"])
    print(f"raw+stops ({res.stop_reason}, {res.n_new_tokens} tok, {res.ms_per_token:.1f} ms/tok):\n{res.text}\n")
    res = generate(lm, enc_u, max_new_tokens=80, stop_strings=["\nQuestion:", "\nUser:"])
    print(f"urial ({res.stop_reason}, {res.n_new_tokens} tok, {res.ms_per_token:.1f} ms/tok):\n{res.text}")


if __name__ == "__main__":
    main()
