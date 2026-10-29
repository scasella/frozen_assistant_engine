# CLAUDE.md

Read [AGENTS.md](AGENTS.md) before editing. It holds the repository layout, the experimental invariants and the exact
commands. The short version:

- Tune only on `data/dev_prompts.json` (1.5B), freeze into `configs/hparams.json`, and never select anything from test
  numbers.
- Never edit or regenerate cached `results/**/*.jsonl` logs. New behaviour gets a new condition name or a new `--tag`.
  After engine changes, run `scripts/check_cache_reproduces.py`.
- Iterate banned token ids in sorted order; no hash-order dependence anywhere.
- `eval/verifiers.py` is frozen; reports re-score from logged text.
- Verify with `.venv/bin/python -m pytest -q`, and regenerate `RESULTS.md` with `eval/report.py` (see AGENTS.md).
