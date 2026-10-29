# AGENTS.md: guide for coding agents and contributors

This repository is an experiment as much as it is software. Every number in [RESULTS.md](RESULTS.md) is regenerated from
logged generations. The rules below keep those numbers honest. Read them before changing anything under `engine/`,
`eval/`, `configs/`, `data/` or `results/`.

## Layout and responsibilities

| Path | Responsibility |
|---|---|
| `engine/model_loader.py` | Loads a HF causal LM on `mps` → `cuda` → `cpu` (bf16, `eval()`, `requires_grad_(False)`), finds the decoder layers, installs the `steer_eager` attention. |
| `engine/prompts.py` | The `Question: …\nAnswer:` frame, the 768-token URIAL prefix, stop strings, and the four trap templates used in calibration. |
| `engine/calibration.py` | Stage 1: teacher and raw trajectories (cached in `trajectories.json`). Stage 2: teacher-forced hidden-state statistics (cached in `states.pt`). Stage 3: closed-form solves on CPU float64 (mean differences, SVD, ridge, p99 thresholds). |
| `engine/artifact.py` | `SteeringArtifact`, the `.pt` container for all calibrated tensors, plus defaults for fields added later. |
| `engine/hooks.py` | All runtime interventions: the `steer_eager` attention function (Module 4 bias and the recorder), `EngineConfig`, and `AssistantEngine`. The engine covers the residual hooks (lifecycle, erasure, hinge, boundary window, stop checks), the pre-LM-head unembedding projection (plain or sibling-orthogonal) and the anti-echo guard. |
| `engine/negation.py` | Deterministic negation parser (clause spans, targets, echo words) and `forbidden_token_ids`, which is **sorted by token id**. |
| `engine/generator.py` | `encode_prompt` (instruction and negation token positions) and a greedy KV-cached decoding loop with stop strings and per-step engine state. |
| `eval/verifiers.py` | The four metrics (TurnCross, Constraint, CleanStop, Fluency) plus per-prompt rule checks. **Frozen.** |
| `eval/conditions.py` | The 19 conditions (`CONDITIONS`, `EXTRA_CONDITIONS`, `V2_CONDITIONS`, `V21_ABLATIONS`), `engine_config` / `resolve_toggles`, the cached-log guard `relevant_config`, `run_condition` and `summarize`. |
| `eval/tune.py` | Dev-set coordinate search: `--phase 1` (Modules 1–4, CAA), `--phase 2` (x5, 7–9), `--phase 3` (condition 10). Each phase asserts that earlier coefficients are unchanged. |
| `eval/benchmark.py` | Loads or calibrates an artifact, runs conditions on the test prompts (reusing cached logs), and writes `summary.json` and the report. |
| `eval/report.py` | Renders `RESULTS.md` from logs. Every response is re-scored with the current verifiers. `--refresh-summary` rewrites `summary.json`. |
| `configs/hparams.json` | Frozen coefficients, with provenance and tuning history for all three phases under `meta`. |
| `data/` | `calibration_prompts.json` (80), `dev_prompts.json` (15) and `test_prompts.json` (40, with `checks`). The three sets are disjoint. |
| `artifacts/<model>/calib80/` | `calibration.pt` (steering tensors, torch pickle), `trajectories.json`, `calibration_diagnostics.json`. `states.pt` is gitignored, 0.8–1.1 GB, and regenerable. |
| `results/<model>/test40/` | One JSONL log per condition (40 records each), `summary.json`, `RESULTS.md`, and the hand-written `analysis_notes.md` that the report embeds. |
| `results/<model>/tuning_log.jsonl` | Cached dev evaluations, keyed by (condition, hparams). |
| `results/*/test40/*.nondeterministic_order.jsonl.bak`, `results/Qwen__Qwen2.5-1.5B/tuning_log_with_nondeterministic_9_10.jsonl.bak` | Withdrawn logs from before the projection-order fix, kept as evidence. Never load them as results. |
| `results/Qwen__Qwen2.5-1.5B/smoke10_defaults/` | Phase-1 smoke run with the pre-fix Module 2, kept for provenance; its `analysis_notes.md` marks it superseded. |
| `scripts/` | `sanity_check.py` (`steer_eager` must equal eager attention), `resolve_calibration.py`, `check_cache_reproduces.py`, `check_links.py`. |
| `tests/` | Model-free pytest suite. |

## Invariants (do not break these)

1. **No test-set leakage.**
   - Coefficients are tuned *only* on `data/dev_prompts.json` with `Qwen/Qwen2.5-1.5B` via `eval/tune.py`, then frozen
     into `configs/hparams.json` before any test run.
   - Never pick a value, a condition or a "best" variant by looking at test numbers.
   - Leave-one-out rows (`x6`–`x9`) are attribution only. `x9`'s 96.2 is explicitly *not* a selectable result.
   - A new phase must not change earlier coefficients; the phase-2 and phase-3 tuners assert this.
2. **Cached logs are immutable.**
   - `run_condition` reuses `results/<model>/<tag>/<condition>.jsonl` by `prompt_id`. It raises if a stored record's
     engine config differs from the current one on any parameter the condition can actually use (`relevant_config`).
   - To change a condition's behaviour, add a **new condition name** or use a **new `--tag`**. Never edit, truncate or
     silently regenerate an existing log.
   - The guard compares configurations, not code. After touching any code path an existing condition uses, run
     `scripts/check_cache_reproduces.py`, which must print byte-identical for every condition you might have affected.
3. **Deterministic projection order.**
   - Sequential projections with γ > 1 do not commute, so banned token ids must be iterated in sorted order (as
     `forbidden_token_ids` returns them).
   - Never iterate a `set`, or a hash-ordered structure, where the order affects results.
   - Check new code paths under two `PYTHONHASHSEED` values.
4. **Verifiers are frozen.**
   - `eval/verifiers.py` defines the metric for all 19 conditions, which are always re-scored from logged text.
   - Do not "fix" a blind spot after looking at test outputs. Known blind spots: `[Equation]` / `[Reasoning: …]`
     labels, evasive meta-answers, repeated answer lines.
   - If a verifier change is unavoidable, it applies to every condition. Disclose it in the methodology section of
     `eval/report.py` and regenerate.
5. **Tuning-cache keys exclude toggles.**
   - `tuning_log.jsonl` keys are (condition name, hparams).
   - If you change a condition's *toggles* in `eval/conditions.py`, delete that condition's entries from the tuning log
     before re-tuning. Otherwise stale scores are reused.
6. **Numerics.**
   - `steer_eager` must stay bit-identical to HF eager attention when no bias is set (`scripts/sanity_check.py`).
   - On MPS, move to CPU *before* upcasting: `.to("cpu").to(torch.float64)`. MPS has no float64.
   - Solves run on CPU float64.
   - `scripts/resolve_calibration.py` asserts that a re-solve leaves every pre-existing artifact tensor bit-identical.
7. **No weight updates, no gradients.** Everything runs under `torch.no_grad()` with `requires_grad_(False)`.
   Interventions are forward hooks only.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q                                   # model-free tests (verifiers, parser, guard, report, links)
.venv/bin/python scripts/check_links.py                         # relative markdown links

# regenerate RESULTS.md from logs (seconds, no model); --refresh-summary also rewrites summary.json
.venv/bin/python -m eval.report results/Qwen__Qwen2.5-1.5B/test40 --refresh-summary
.venv/bin/python -m eval.report results/Qwen__Qwen2.5-3B/test40 --refresh-summary --out RESULTS.md --appendix results/Qwen__Qwen2.5-1.5B/test40

# model-dependent checks
.venv/bin/python scripts/sanity_check.py --model Qwen/Qwen2.5-1.5B
.venv/bin/python scripts/check_cache_reproduces.py --model Qwen/Qwen2.5-1.5B --conditions 6_full_assistant_engine 10_engine_v2_1

# calibration: re-solve from a local states.pt (no forward passes), or recalibrate (reuses cached trajectories)
.venv/bin/python scripts/resolve_calibration.py --model Qwen/Qwen2.5-3B
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --recalibrate --conditions 1_raw_base --tag recal
```

### Adding a new experimental condition

1. Add a new entry to `eval/conditions.py`: a new dict, or an existing one such as `V21_ABLATIONS`. It maps
   `name -> (use_urial, use_stop_strings, toggles)`. Toggles are `EngineConfig` fields; `"_base": "v2_base"` inherits
   the dev-selected v2 base. New mechanisms get new `EngineConfig` fields whose defaults leave every existing condition
   unchanged. Add their parameters to `_PARAMS_OF` so the cache guard knows when they matter.
2. Tune any new coefficients on dev only. Add a new `--phase` to `eval/tune.py` that starts from the brief's or spec's
   values, asserts earlier coefficients are unchanged, and records provenance in `configs/hparams.json` → `meta`.
3. Run `scripts/check_cache_reproduces.py` on the existing conditions that share code with the new one.
4. Run the new condition on test once:
   `python -m eval.benchmark --model <m> --conditions <name> --tag test40`.
5. Add comparison lines and methodology text to `eval/report.py`, record interpretation in
   `results/Qwen__Qwen2.5-3B/test40/analysis_notes.md`, regenerate `RESULTS.md`, and run `pytest`. Update
   `test_nineteen_conditions` as well.
