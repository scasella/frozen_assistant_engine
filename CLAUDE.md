# CLAUDE.md

Read [AGENTS.md](AGENTS.md) before editing. It holds the repository layout, the experimental invariants and the exact
commands. The short version:

- Tune only on `data/dev_prompts.json` (1.5B), freeze into `configs/hparams.json`, and never select anything from test
  numbers.
- Never edit or regenerate cached `results/**/*.jsonl` logs. New behaviour gets a new condition name or a new `--tag`.
  After engine changes, run `scripts/check_cache_reproduces.py`.
- Iterate banned token ids in sorted order; no hash-order dependence anywhere.
- `eval/verifiers.py` is frozen; reports re-score from logged text.
- All benchmark conditions run bf16 on `steer_eager` attention; SDPA, fp32-score attention and 512-token runs are
  sensitivity tags only (`test40_sdpa`, `test40_fp32attn`, `test40_512tok`, `test40_512tok_fp32attn`).
- Verify with `.venv/bin/python -m pytest -q`, and regenerate `RESULTS.md` with `eval/report.py` (see AGENTS.md).
