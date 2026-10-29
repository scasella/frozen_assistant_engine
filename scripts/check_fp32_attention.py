"""Pre-flight checks for the ``steer_fp32`` sensitivity stack (run before any ``*_fp32attn`` test run).

1. fp32 model: ``steer_fp32`` equals HF eager attention without a bias, and equals ``steer_eager`` with a Module-4 bias
   on a few heads (the bias path is the same, and the bias does change the output).
2. bf16 model: distance of ``steer_eager``, ``steer_fp32`` and SDPA to the fp32 eager reference (max |logit diff|,
   mean KL, top-1 agreement) on calibration prompts. ``steer_fp32`` should sit near SDPA, not near ``steer_eager``.
3. bf16 ``steer_fp32``: the raw base (condition 1, 160 tokens, nothing logged) on the 8 math test prompts, scored with
   the frozen ``math_answer`` check, next to the protocol and SDPA logs.

    python scripts/check_fp32_attention.py --model Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import encode_prompt, free_memory, load_model  # noqa: E402
from engine.hooks import STEER_ATTN, STEER_FP32_ATTN, attention_module, install_steerable_attention  # noqa: E402
from engine.prompts import URIAL_PREFIX, minimal_prompt  # noqa: E402
from eval.conditions import load_hparams, load_prompts, run_condition  # noqa: E402
from eval.verifiers import check_math_answer  # noqa: E402


@torch.no_grad()
def logits_under(lm, impl: str, ids: torch.Tensor) -> torch.Tensor:
    lm.model.config._attn_implementation = impl
    try:
        return lm.model(input_ids=ids).logits[0].float().cpu()
    finally:
        lm.model.config._attn_implementation = lm.attn


def fp32_checks(name: str, seqs: list[list[int]]) -> list[torch.Tensor]:
    """Equivalence checks on the fp32 model; returns its HF-eager logits for every sequence (the reference)."""
    lm = load_model(name, dtype="float32", attn="steer_fp32")
    install_steerable_attention(lm.model, STEER_ATTN)   # register steer_eager too, then restore steer_fp32
    lm.model.config._attn_implementation = STEER_FP32_ATTN
    refs = [logits_under(lm, "eager", torch.tensor([s], device=lm.device)) for s in seqs]
    ids, ref = seqs[0], refs[0]
    x = torch.tensor([ids], device=lm.device)
    for impl in (STEER_FP32_ATTN, STEER_ATTN):
        d = (logits_under(lm, impl, x) - ref).abs().max().item()
        print(f"[fp32 model, no bias] max |logit diff| {impl} vs HF eager: {d:.3e}")
        assert d < 1e-4, f"{impl} is not equivalent to eager attention in fp32"

    # Module-4-style bias: +1.25 on the first 16 positions for two heads of three layers.
    layers = [lm.n_layers // 4, lm.n_layers // 2, 3 * lm.n_layers // 4]
    for l in layers:
        bias = torch.zeros(1, lm.n_heads, 1, len(ids), dtype=lm.dtype)
        bias[0, [0, lm.n_heads // 2], 0, :16] = 1.25
        attention_module(lm.layers[l])._steer_bias = bias.to(lm.device)
    try:
        b_fp32, b_eager = logits_under(lm, STEER_FP32_ATTN, x), logits_under(lm, STEER_ATTN, x)
    finally:
        for l in layers:
            attention_module(lm.layers[l])._steer_bias = None
    d_bias = (b_fp32 - b_eager).abs().max().item()
    effect = (b_fp32 - ref).abs().max().item()
    print(f"[fp32 model, bias on layers {layers}] max |logit diff| steer_fp32 vs steer_eager: {d_bias:.3e}; "
          f"effect of the bias: {effect:.3e}")
    assert d_bias < 1e-4 and effect > 1e-2, "bias path differs between steer_fp32 and steer_eager, or has no effect"
    del lm
    free_memory()
    return refs


def bf16_distances(name: str, seqs: list[list[int]], refs: list[torch.Tensor]) -> None:
    lm = load_model(name, attn="steer_fp32")
    install_steerable_attention(lm.model, STEER_ATTN)
    lm.model.config._attn_implementation = STEER_FP32_ATTN
    print(f"[bf16 model] distance to the fp32 eager reference over {sum(len(s) for s in seqs)} positions "
          f"({len(seqs)} sequences):")
    for impl in (STEER_ATTN, STEER_FP32_ATTN, "sdpa"):
        dmax, kl_sum, agree, n = 0.0, 0.0, 0, 0
        for s, ref in zip(seqs, refs):
            out = logits_under(lm, impl, torch.tensor([s], device=lm.device))
            dmax = max(dmax, (out - ref).abs().max().item())
            lp_ref, lp = ref.log_softmax(-1), out.log_softmax(-1)
            kl_sum += (lp_ref.exp() * (lp_ref - lp)).sum(-1).sum().item()
            agree += (out.argmax(-1) == ref.argmax(-1)).sum().item()
            n += len(s)
        print(f"  {impl:12s} max |logit diff| {dmax:7.3f}   mean KL {kl_sum / n:.2e} nats   top-1 agreement {agree}/{n}")
    del lm
    free_memory()


def math_check(name: str) -> int:
    lm = load_model(name, attn="steer_fp32")
    prompts = [p for p in load_prompts(ROOT / "data" / "test_prompts.json") if p["category"] == "reasoning_math"]
    hp = load_hparams(ROOT / "configs" / "hparams.json")
    recs = run_condition(lm, None, "1_raw_base", prompts, hp, 160, None)
    by_id = {p["id"]: p for p in prompts}

    def solved(rs) -> set[str]:
        return {r["prompt_id"] for r in rs if r["prompt_id"] in by_id
                and check_math_answer(r["response"], by_id[r["prompt_id"]]["checks"][0]["answer"])}

    logged = {}
    for tag in ("test40", "test40_sdpa"):
        f = ROOT / "results" / lm.slug / tag / "1_raw_base.jsonl"
        if f.exists():
            logged[tag] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    mine = solved(recs)
    print(f"[bf16 steer_fp32] raw base solves {len(mine)}/{len(prompts)} math prompts; "
          + "; ".join(f"{t} log: {len(solved(rs))}/{len(prompts)}" for t, rs in logged.items()))
    if "test40_sdpa" in logged:
        sd = {r["prompt_id"]: r["response"] for r in logged["test40_sdpa"]}
        same = sum(r["response"] == sd[r["prompt_id"]] for r in recs)
        print(f"  identical to the SDPA log: {same}/{len(recs)}; solved under SDPA but not here: "
              f"{sorted(solved(logged['test40_sdpa']) - mine) or 'none'}")
    for r in recs:
        ok = "ok " if r["prompt_id"] in mine else "BAD"
        print(f"  {ok} {r['prompt_id']}: {r['response'][:110]!r}")
    del lm
    free_memory()
    return len(mine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n-calib", type=int, default=5, help="calibration prompts for the bf16 distance check")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    urial = tok(URIAL_PREFIX + minimal_prompt("Explain quantum computing in 3 bullet points."))["input_ids"]
    calib = load_prompts(ROOT / "data" / "calibration_prompts.json", args.n_calib)
    seqs = [list(urial)] + [encode_prompt(tok, p["instruction"]).ids for p in calib]

    refs = fp32_checks(args.model, seqs)
    bf16_distances(args.model, seqs, refs)
    math_check(args.model)


if __name__ == "__main__":
    main()
