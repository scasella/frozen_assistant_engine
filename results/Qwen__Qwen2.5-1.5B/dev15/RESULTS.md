# RESULTS — Zero-Weight-Update Assistant Engine on `Qwen/Qwen2.5-1.5B`

_Generated automatically by `eval/report.py` on 2026-10-29 from the logs in `results/Qwen__Qwen2.5-1.5B/dev15/`. Greedy decoding, `max_new_tokens=160`, 15 held-out prompts (`data/dev_prompts.json`), device `mps` / `bfloat16`, torch 2.15.0._

No weights were modified and no gradients were computed: every condition runs the same frozen checkpoint; steered conditions only add forward hooks (residual-stream edits and a pre-softmax attention bias).

## Setup

- **Calibration:** 77/80 calibration prompts kept after filtering URIAL teacher answers (clean stop, >10 chars, distinct-4-gram ≥ 0.75); target layers [9, 11, 13, 15, 16, 18, 20, 22] of 28; stop check at layer 20; static-CAA layer 16; 2 instruction-pinning heads (L9H8, L8H6).
- **Hyperparameters:** `configs/hparams.json` — coordinate search on 15 dev prompts (data/dev_prompts.json, disjoint from the test set) with Qwen/Qwen2.5-1.5B; frozen afterwards.
- **Prompt frame for all steered conditions:** `Question: {instruction}\nAnswer:` (0 extra tokens). Stop strings `["\nQuestion:", "\nUser:"]` for conditions 2–6.
- **Metric definitions:** TurnCross = starts answering (no new question / echo / empty / garbage / script switch); Constraint = every programmatic check attached to the prompt passes (for `direct_qa` this is ≥2 expected topic keywords; math = expected final answer; code = parses / required function & SQL clauses; format/negative = exact rule); CleanStop = no hallucinated turn/FAQ/forum markers and not truncated mid-sentence or looping at max tokens; Fluency = distinct-4-gram ratio > 0.75 and > 10 chars; Composite = mean of the four. **NaturalEOS** (extra) = generation ended with the model's own `<eos>` token rather than a stop string or the token cap — with stop strings active, CleanStop alone cannot distinguish "model stopped" from "we cut it off".

## 1. Ablation summary (all values in %)

| Condition | TurnCross | Constraint | CleanStop | Fluency | **Composite** | NaturalEOS | Extra ctx tokens | ms/token | tok/s | avg new tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| `10_engine_v2_1` | 100.0 | 86.7 | 93.3 | 93.3 | **93.3** | 80.0 | 0 | 29.0 | 34.5 | 65 |
| `11_engine_v2_2_lean` | 100.0 | 86.7 | 100.0 | 93.3 | **95.0** | 80.0 | 0 | 26.7 | 37.4 | 57 |

Stop reasons (eos / stop_string / max_tokens): `10_engine_v2_1` 12/2/1; `11_engine_v2_2_lean` 12/3/0

## 2. Per-category breakdown

### Composite (%)

| Condition | direct_qa | format_constraints | negative_constraints | reasoning_math | code_and_sql |
|---|---|---|---|---|---|
| `10_engine_v2_1` | 83.3 | 100.0 | 83.3 | 100.0 | 100.0 |
| `11_engine_v2_2_lean` | 83.3 | 100.0 | 91.7 | 100.0 | 100.0 |

### ConstraintScore (%)

| Condition | direct_qa | format_constraints | negative_constraints | reasoning_math | code_and_sql |
|---|---|---|---|---|---|
| `10_engine_v2_1` | 66.7 | 100.0 | 66.7 | 100.0 | 100.0 |
| `11_engine_v2_2_lean` | 66.7 | 100.0 | 66.7 | 100.0 | 100.0 |

## 3. Which modules contributed, and why

Deltas are in Composite points on the same prompts; brackets give a 95% paired-bootstrap CI over prompts (10k resamples). With 40 prompts one prompt flipping one metric moves Composite by 0.625 pts, so intervals that straddle 0 should be read as "no detectable difference".

- Calibration diagnostics: completion-centroid AUC (final token vs. mid-answer punctuation) ranges 0.69–0.86 across target layers (stop check at layer 20); 1–1 drift direction(s) per layer pass the separation test, with trap tokens 2.2–3.1 clean-σ along the leading one; ‖v_init‖/‖h‖ = 0.26–0.36.
- Module 3 cross-validated R² of the predicted t=0 URIAL shift (beyond the mean shift v_init; CV-selected λ per layer: 100, 100, 100, 100, 100, 1000, 1000, 1000): +0.48 to +0.69 across layers — the input-dependent component carries real signal.

### Phase 3 (engine v2.1, condition 10) comparisons

- **reasoning_math Composite**: `10_engine_v2_1` 100.0%.
- **negative_constraints ConstraintScore**: `10_engine_v2_1` 66.7%.
- **Mechanism activity on these prompts (condition 10):** EOS boosts by reason centroid 8, drift 2, newturn 2; boundary-window steps 133; unembedding projections 629 (sibling fallbacks 0); prompts with an anti-echo set 3.

### Phase 4 (v2.2 lean engine, post-trained reference) comparisons

- **v2.2 lean vs. `10_engine_v2_1`** (`10_engine_v2_1` → `11_engine_v2_2_lean`): Composite **+1.7** [+0.0, +5.0] (TurnCross +0.0, Constraint +0.0, CleanStop +6.7, Fluency +0.0, NaturalEOS +0.0).

## Methodology notes and caveats

- **Drift / turn markers used by CleanStop and TurnCross** (line-anchored, case-insensitive, applied identically to every condition): new-turn labels `Question[ n]:` / `Q:` / `Q1.` / `[Question]` / `[Answer]` / `User:` / `Human:` / `Assistant:` / `A:`; forum boilerplate (`Posted by`, `N comments`, `Share |`, `Related links/questions`, `Reply`, `Upvote`, `Tags:`, `Asked by`, `See also:` …); and instruction-dataset templates leaking from pretraining (`A single-select problem`, `Available options:`, `Options are:`, `Choose your answer`, `Pick your answer`, `(a). yes`-style option lines, hallucinated `You are a helpful assistant`). The template/bracket/`A:` markers were added after inspecting the calibration negatives (Y⁻) and the 1.5B smoke-test outputs, because the raw Qwen2.5 base model's dominant failure is appending FLAN-style option blocks. All metrics are re-scored from the logged response text whenever this report is generated, so every condition is judged by the same final detector. The TurnCross *echo* rule was also tightened after inspecting test outputs: a response fails only if its opening reproduces ≥90% of the instruction's words and either starts with the same word (literal re-typing) or contains nothing beyond the restatement. The first, fuzzier version (first-line similarity > 0.8) mis-flagged assistant preambles such as "Here is the SQL query to find …" and math answers that restate the premise. The change only removed false positives, and it moved at most one prompt per condition.
- **Latency** (`ms/token`, `tok/s`) is wall-clock for the whole generation divided by generated tokens, *including the prefill pass*. This charges the URIAL condition for its 768-token prefix, which is a real per-query cost, but it inflates its ms/token relative to a decode-only measurement. Runs were executed one at a time on an otherwise idle GPU.
- **Tuning resolution.** Coefficients were chosen by coordinate search on 15 dev prompts with the 1.5B model and frozen before any test-set run (coordinate search on 15 dev prompts (data/dev_prompts.json, disjoint from the test set) with Qwen/Qwen2.5-1.5B; frozen afterwards). One dev prompt flipping one metric moves dev Composite by 1.67 pts, so each adopted change (λ_erase 0.7→0.35, β_attn 1.25→0.5, α_caa 1→2) reflects one or two prompts; treat them as "not worse than the spec defaults", not as precise optima. The dynamic-stop coefficients (EOS boost 0/2/5/10, α_stop 0/1/2) produced *identical* dev scores, i.e. the completion check rarely changed the outcome on dev.
- **URIAL is a high-context reference, not a ceiling.** Its demonstrations are deliberately thorough (multi-paragraph, with explanations), so its answers are long and often hit the 160-token cap, which costs it CleanStop/Constraint points. The same teacher answers (generated with a 256-token budget) are what the engine is calibrated toward.
- **Module 4 evidence on this model:** 2 pinned heads out of 168 candidates in the 0.30N–0.75N band; head selection requires a positive instruction-attention delta (successful − drifted), max delta +0.004 (mean over all candidates -0.0198). Raw continuations often copy the question, so many heads attend to the instruction *more* during drift than during clean answers.

### Phase 3 (engine v2.1, condition 10) methodology

- **Definition.** Condition 7 plus four changes, with no signed attention (pinning uses the Phase-1 β = 0.5):
  1. **Boundary-coupled hinge** (κ_boundary = 1.5 instead of the p99 threshold). It applies inside a boundary window: the last token contains `. ! ? \n`; or the step is ≤ 2 tokens after a newline; or the previous step's completion score was ≥ τ_stop − 0.08. The score clause has to lag by one step, because the current step's score is only computed at the stop layer, after most hinge layers have run.
  2. **Drift-assisted dynamic stop.** It triggers if the stop-layer drift z ≥ 1.8 while the score is ≥ τ_stop − 0.12. The stop check now starts at t ≥ 6.
  3. **Sibling-orthogonalised unembedding projection.** Banned directions are mean-centred, as the brief specifies, and orthogonalised against the top-4 favoured non-banned word tokens. A fallback applies when the remaining overlap is below 0.3.
  4. **Anti-echo guard.** It suppresses the negation trigger words (`without`, `mentioning` …) for the first 12 generated tokens.
- **Inherited and cost details.** Condition 10's projection strength γ = 1.5 was inherited from the condition-9 re-tune, which used uncentred directions; it was not re-tuned for the centred directions condition 10 uses. Condition 10 decodes about 12% slower than condition 7, for two reasons: the boundary clause syncs the stop-layer score at every step, and on negation prompts the sibling basis needs an extra full-vocabulary matrix-vector product per step.
- **Is the boundary-onset diagnosis right?** Teacher-forced on 24 *calibration* answers followed by a `[Question]` or FLAN continuation, the mean drift z is:
  - −0.4 on clean mid-answer tokens and 1.1 on the last answer token;
  - **3.9σ at the boundary newline** (3B; 3.65σ on 1.5B), against a mean p99 threshold of 3.1σ (2.9σ);
  - 4.3–4.9σ on the first drift tokens.
  So the onset is *not* below the p99 threshold, and the v2 hinge already fires there. It only removes 75% of the small excess above ~3.1σ, which leaves the state near 3.3σ. A lower boundary threshold helps because it removes much more of an excursion the hinge already sees.
- **Is the sibling-suppression diagnosis right?** Yes.
  - Centred unembedding cosines between ' Python' and ' Java' / ' Ruby' / ' JavaScript' are 0.45 / 0.38 / 0.36, and ' Assembly' is 0.02, the only language left in the withdrawn non-deterministic condition-9 answer to `t_ng_02` (the deterministic re-run answers it correctly, so that example no longer demonstrates the effect).
  - On a real 1.5B state, the plain projection moved the top-8 clean siblings' relative logits by −3.6 to −13.7 (' violin' −13.7). The sibling-orthogonalised projection moved them by exactly 0.00, while the banned tokens fell 10–31 logits.
- **Dev evidence for the new thresholds** (1.5B, 15 prompts; start = the brief's values):
  - the drift-stop threshold (1.5 / 1.8 / 2.2) and the anti-echo window (8 / 12 / 16) produced *identical* dev scores, so they are untested by dev;
  - κ_boundary 1.0 → 1.5 moved dev from 85.0 to 91.7, and κ_boundary 2.0 scored 90.0;
  - sibling K = 4 beat K = 8 (93.3 vs. 91.7), a single-prompt difference.
  Dev composites: `7_engine_v2_hinge_erase` 91.7, `9_engine_v2_full` 91.7, `10_engine_v2_1` 93.3.

### Phase 4 (v2.2 lean engine and post-trained reference) methodology

- **Condition 11 (`11_engine_v2_2_lean`)** keeps the two v2.1 mechanisms that the leave-one-outs credit: the boundary-coupled hinge (κ_boundary = 1.5) and the sibling-orthogonalised projection (K = 4, γ = 1.5, centred). It drops the drift-assisted stop and the anti-echo guard. It has no new coefficients, and the stop check still starts at t ≥ 6 as in v2.1. **Its mechanism set was chosen from the test-set leave-one-outs, so its test score is not a held-out estimate.** Condition 10 remains the dev-selected headline.
- **Lean execution (`boundary_trigger = "text"`, `sibling_fast = True`):**
  - **Text-only boundary trigger.** The boundary window opens only when the last generated token contains `. ! ? \n`, or within 2 steps after a newline. The v2.1 score clause is dropped, so the stop-layer completion score is computed only on stop-check steps (punctuation tokens), as in condition 7. No other step has an extra host sync.
  - **Sync-free projection.** The sibling-orthogonalised projection computes the same arithmetic without host synchronisation: the fallback and Gram-Schmidt branches are `torch.where` selections, and the trace counters stay on device until the answer ends.
  - **Smaller sibling search.** The sibling search multiplies only the pre-indexed word-token rows of the unembedding (tokens containing an ASCII letter; 61.8% of Qwen2.5's vocabulary) instead of the whole vocabulary.
- **Why a second vocabulary product remains.**
  - The siblings are the top-K tokens of the *unprojected* logits, while the emitted logits must come from the *projected* state, so an exact implementation reads the unembedding twice on negation prompts.
  - Applying the update in logit space (`logits + W·Δh`) does not avoid this: `W·Δh` is itself a full-vocabulary product.
  - Any re-association that is not bit-identical in bf16 changes greedy outputs at near-ties.
  - The lean path therefore keeps the exact computation, makes the search product 38% smaller, and removes the synchronisation stalls, which `scripts/bench_speed.py` shows were most of the cost.

## 4. Calibration diagnostics (per target layer)

| Layer | ‖h‖ | ‖v_init‖/‖h‖ | ‖v_sus⊥‖/‖h‖ | cos(v_sus,v_init) | stop AUC | τ_stop | active drift dirs | trap z (dir 1) | clean / trap touched v1 (κ=1) | v2 hinge κ (p99) | clean / trap touched v2 | M3 λ | M3 CV R² |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 47.3 | 0.285 | 0.095 | 0.52 | 0.69 | 0.82 | 1 | 2.24 | 15% / 77% | 2.48 | 1% / 53% | 100 | +0.64 |
| 11 | 51.6 | 0.318 | 0.102 | 0.41 | 0.70 | 0.81 | 1 | 2.58 | 15% / 80% | 2.69 | 1% / 55% | 100 | +0.68 |
| 13 | 55.6 | 0.328 | 0.109 | 0.47 | 0.74 | 0.80 | 1 | 2.81 | 15% / 83% | 2.99 | 1% / 55% | 100 | +0.69 |
| 15 | 58.1 | 0.337 | 0.121 | 0.56 | 0.76 | 0.80 | 1 | 3.03 | 15% / 88% | 2.82 | 1% / 62% | 100 | +0.66 |
| 16 | 60.9 | 0.342 | 0.127 | 0.59 | 0.78 | 0.79 | 1 | 3.07 | 14% / 88% | 3.05 | 1% / 58% | 100 | +0.60 |
| 18 | 70.9 | 0.358 | 0.129 | 0.58 | 0.81 | 0.75 | 1 | 3.03 | 14% / 88% | 3.12 | 1% / 57% | 1000 | +0.66 |
| 20 | 88.9 | 0.331 | 0.125 | 0.55 | 0.86 | 0.68 | 1 | 2.82 | 13% / 86% | 3.04 | 1% / 51% | 1000 | +0.61 |
| 22 | 130.5 | 0.263 | 0.100 | 0.53 | 0.85 | 0.65 | 1 | 2.46 | 15% / 85% | 2.77 | 1% / 46% | 1000 | +0.48 |

### Coefficients

| Coefficient | Spec default | Used (frozen) |
|---|---|---|
| `alpha_caa` | 1.0 | 2.0 |
| `alpha_init` | 1.0 | 1.0 |
| `alpha_stop` | 1.0 | 1.0 |
| `alpha_sustain` | 0.35 | 0.35 |
| `anti_echo_window` | – | 12 |
| `beta_attn` | 1.25 | 0.5 |
| `beta_neg_attn` | – | 0.5 |
| `erase_kappa` | 1.0 | 1.0 |
| `gamma_unembed` | – | 1.5 |
| `hinge_kappa_floor` | – | 2.6 |
| `kappa_boundary` | – | 1.5 |
| `lambda_erase` | 0.7 | 0.35 |
| `lambda_hinge` | – | 0.75 |
| `neg_span_mode` | – | clause |
| `predictor_weight` | 1.0 | 1.0 |
| `sibling_k` | – | 4 |
| `stop_logit_boost` | 5.0 | 5.0 |
| `tau_drift_stop` | – | 1.8 |
| `unembed_center` | – | False |
| `v2_base` | – | 7_engine_v2_hinge_erase |

## 5. Qualitative side-by-side examples (verbatim)

Outputs are shown exactly as returned after stop-string truncation and whitespace stripping.
