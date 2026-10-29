# RESULTS — Zero-Weight-Update Assistant Engine on `Qwen/Qwen2.5-1.5B`

_Generated automatically by `eval/report.py` on 2026-10-29 from the logs in `results/Qwen__Qwen2.5-1.5B/test40/`. Greedy decoding, `max_new_tokens=160`, 40 held-out prompts (`data/test_prompts.json`), device `mps` / `bfloat16`, torch 2.15.0._

No weights were modified and no gradients were computed: every condition runs the same frozen checkpoint; steered conditions only add forward hooks (residual-stream edits and a pre-softmax attention bias).

## Setup

- **Calibration:** 77/80 calibration prompts kept after filtering URIAL teacher answers (clean stop, >10 chars, distinct-4-gram ≥ 0.75); target layers [9, 11, 13, 15, 16, 18, 20, 22] of 28; stop check at layer 20; static-CAA layer 16; 2 instruction-pinning heads (L9H8, L8H6).
- **Hyperparameters:** `configs/hparams.json` — coordinate search on 15 dev prompts (data/dev_prompts.json, disjoint from the test set) with Qwen/Qwen2.5-1.5B; frozen afterwards.
- **Prompt frame for all steered conditions:** `Question: {instruction}\nAnswer:` (0 extra tokens). Stop strings `["\nQuestion:", "\nUser:"]` for conditions 2–6.
- **Metric definitions:** TurnCross = starts answering (no new question / echo / empty / garbage / script switch); Constraint = every programmatic check attached to the prompt passes (for `direct_qa` this is ≥2 expected topic keywords; math = expected final answer; code = parses / required function & SQL clauses; format/negative = exact rule); CleanStop = no hallucinated turn/FAQ/forum markers and not truncated mid-sentence or looping at max tokens; Fluency = distinct-4-gram ratio > 0.75 and > 10 chars; Composite = mean of the four. **NaturalEOS** (extra) = generation ended with the model's own `<eos>` token rather than a stop string or the token cap — with stop strings active, CleanStop alone cannot distinguish "model stopped" from "we cut it off".

## 1. Ablation summary (all values in %)

| Condition | TurnCross | Constraint | CleanStop | Fluency | **Composite** | NaturalEOS | Extra ctx tokens | ms/token | tok/s | avg new tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| `1_raw_base` | 100.0 | 55.0 | 55.0 | 92.5 | **75.6** | 77.5 | 0 | 25.4 | 39.4 | 76 |
| `2_raw_base_plus_stops` | 100.0 | 55.0 | 67.5 | 100.0 | **80.6** | 77.5 | 0 | 25.1 | 39.9 | 63 |
| `3_urial_3shot_prefix` | 100.0 | 60.0 | 62.5 | 100.0 | **80.6** | 40.0 | 768 | 30.2 | 33.1 | 118 |
| `4_static_caa` | 100.0 | 45.0 | 57.5 | 100.0 | **75.6** | 20.0 | 0 | 25.0 | 40.1 | 113 |
| `5_lifecycle_plus_nullspace` | 100.0 | 60.0 | 75.0 | 87.5 | **80.6** | 67.5 | 0 | 25.6 | 39.1 | 84 |
| `6_full_assistant_engine` | 100.0 | 62.5 | 80.0 | 95.0 | **84.4** | 67.5 | 0 | 25.7 | 39.0 | 79 |
| `x1_lifecycle_only` | 100.0 | 60.0 | 62.5 | 100.0 | **80.6** | 67.5 | 0 | 24.2 | 41.2 | 74 |
| `x2_lifecycle_plus_erase` | 100.0 | 57.5 | 57.5 | 90.0 | **76.2** | 57.5 | 0 | 24.6 | 40.6 | 82 |
| `x3_full_minus_dynamic_stop` | 100.0 | 62.5 | 77.5 | 95.0 | **83.8** | 62.5 | 0 | 25.4 | 39.3 | 86 |
| `x4_pin_attention_only` | 100.0 | 62.5 | 67.5 | 100.0 | **82.5** | 80.0 | 0 | 24.3 | 41.2 | 64 |
| `x5_full_minus_erase` | 100.0 | 57.5 | 67.5 | 100.0 | **81.2** | 62.5 | 0 | 23.5 | 42.6 | 81 |
| `7_engine_v2_hinge_erase` | 100.0 | 57.5 | 72.5 | 95.0 | **81.2** | 60.0 | 0 | 23.8 | 41.9 | 80 |
| `8_engine_v2_signed_attn` | 100.0 | 55.0 | 72.5 | 92.5 | **80.0** | 57.5 | 0 | 23.9 | 41.8 | 84 |
| `9_engine_v2_full` | 100.0 | 70.0 | 70.0 | 87.5 | **81.9** | 60.0 | 0 | 25.9 | 38.6 | 85 |
| `10_engine_v2_1` | 100.0 | 67.5 | 87.5 | 90.0 | **86.2** | 72.5 | 0 | 26.9 | 37.1 | 77 |
| `x6_v2_1_minus_boundary_hinge` | 100.0 | 67.5 | 77.5 | 92.5 | **84.4** | 67.5 | 0 | 27.3 | 36.6 | 78 |
| `x7_v2_1_minus_drift_stop` | 100.0 | 67.5 | 87.5 | 90.0 | **86.2** | 72.5 | 0 | 27.2 | 36.7 | 77 |
| `x8_v2_1_minus_sibling_orth` | 100.0 | 65.0 | 75.0 | 77.5 | **79.4** | 57.5 | 0 | 26.1 | 38.4 | 88 |
| `x9_v2_1_minus_anti_echo` | 100.0 | 70.0 | 85.0 | 90.0 | **86.2** | 70.0 | 0 | 26.9 | 37.2 | 77 |

Stop reasons (eos / stop_string / max_tokens): `1_raw_base` 31/0/9; `2_raw_base_plus_stops` 31/5/4; `3_urial_3shot_prefix` 16/8/16; `4_static_caa` 8/15/17; `5_lifecycle_plus_nullspace` 27/6/7; `6_full_assistant_engine` 27/6/7; `x1_lifecycle_only` 27/6/7; `x2_lifecycle_plus_erase` 23/6/11; `x3_full_minus_dynamic_stop` 25/7/8; `x4_pin_attention_only` 32/4/4; `x5_full_minus_erase` 25/5/10; `7_engine_v2_hinge_erase` 24/7/9; `8_engine_v2_signed_attn` 23/7/10; `9_engine_v2_full` 24/6/10; `10_engine_v2_1` 29/6/5; `x6_v2_1_minus_boundary_hinge` 27/6/7; `x7_v2_1_minus_drift_stop` 29/6/5; `x8_v2_1_minus_sibling_orth` 23/7/10; `x9_v2_1_minus_anti_echo` 28/6/6

## 2. Per-category breakdown

### Composite (%)

| Condition | direct_qa | format_constraints | negative_constraints | reasoning_math | code_and_sql |
|---|---|---|---|---|---|
| `1_raw_base` | 65.6 | 87.5 | 75.0 | 56.2 | 93.8 |
| `2_raw_base_plus_stops` | 65.6 | 87.5 | 81.2 | 68.8 | 100.0 |
| `3_urial_3shot_prefix` | 84.4 | 90.6 | 75.0 | 75.0 | 78.1 |
| `4_static_caa` | 90.6 | 71.9 | 68.8 | 65.6 | 81.2 |
| `5_lifecycle_plus_nullspace` | 90.6 | 84.4 | 71.9 | 65.6 | 90.6 |
| `6_full_assistant_engine` | 93.8 | 87.5 | 78.1 | 71.9 | 90.6 |
| `x1_lifecycle_only` | 68.8 | 87.5 | 81.2 | 68.8 | 96.9 |
| `x2_lifecycle_plus_erase` | 59.4 | 84.4 | 81.2 | 62.5 | 93.8 |
| `x3_full_minus_dynamic_stop` | 90.6 | 87.5 | 78.1 | 71.9 | 90.6 |
| `x4_pin_attention_only` | 65.6 | 93.8 | 81.2 | 71.9 | 100.0 |
| `x5_full_minus_erase` | 90.6 | 84.4 | 75.0 | 68.8 | 87.5 |
| `7_engine_v2_hinge_erase` | 87.5 | 84.4 | 75.0 | 68.8 | 90.6 |
| `8_engine_v2_signed_attn` | 87.5 | 84.4 | 68.8 | 68.8 | 90.6 |
| `9_engine_v2_full` | 87.5 | 84.4 | 78.1 | 68.8 | 90.6 |
| `10_engine_v2_1` | 93.8 | 84.4 | 90.6 | 71.9 | 90.6 |
| `x6_v2_1_minus_boundary_hinge` | 87.5 | 84.4 | 90.6 | 68.8 | 90.6 |
| `x7_v2_1_minus_drift_stop` | 93.8 | 84.4 | 90.6 | 71.9 | 90.6 |
| `x8_v2_1_minus_sibling_orth` | 93.8 | 84.4 | 56.2 | 71.9 | 90.6 |
| `x9_v2_1_minus_anti_echo` | 93.8 | 84.4 | 90.6 | 71.9 | 90.6 |

### ConstraintScore (%)

| Condition | direct_qa | format_constraints | negative_constraints | reasoning_math | code_and_sql |
|---|---|---|---|---|---|
| `1_raw_base` | 62.5 | 50.0 | 25.0 | 37.5 | 100.0 |
| `2_raw_base_plus_stops` | 62.5 | 50.0 | 25.0 | 37.5 | 100.0 |
| `3_urial_3shot_prefix` | 100.0 | 75.0 | 25.0 | 25.0 | 75.0 |
| `4_static_caa` | 100.0 | 25.0 | 12.5 | 12.5 | 75.0 |
| `5_lifecycle_plus_nullspace` | 100.0 | 50.0 | 25.0 | 50.0 | 75.0 |
| `6_full_assistant_engine` | 100.0 | 50.0 | 25.0 | 62.5 | 75.0 |
| `x1_lifecycle_only` | 75.0 | 50.0 | 37.5 | 37.5 | 100.0 |
| `x2_lifecycle_plus_erase` | 75.0 | 37.5 | 37.5 | 37.5 | 100.0 |
| `x3_full_minus_dynamic_stop` | 100.0 | 50.0 | 25.0 | 62.5 | 75.0 |
| `x4_pin_attention_only` | 62.5 | 75.0 | 25.0 | 50.0 | 100.0 |
| `x5_full_minus_erase` | 100.0 | 37.5 | 25.0 | 50.0 | 75.0 |
| `7_engine_v2_hinge_erase` | 100.0 | 37.5 | 25.0 | 50.0 | 75.0 |
| `8_engine_v2_signed_attn` | 100.0 | 37.5 | 12.5 | 50.0 | 75.0 |
| `9_engine_v2_full` | 100.0 | 37.5 | 87.5 | 50.0 | 75.0 |
| `10_engine_v2_1` | 100.0 | 37.5 | 75.0 | 50.0 | 75.0 |
| `x6_v2_1_minus_boundary_hinge` | 100.0 | 37.5 | 75.0 | 50.0 | 75.0 |
| `x7_v2_1_minus_drift_stop` | 100.0 | 37.5 | 75.0 | 50.0 | 75.0 |
| `x8_v2_1_minus_sibling_orth` | 100.0 | 37.5 | 62.5 | 50.0 | 75.0 |
| `x9_v2_1_minus_anti_echo` | 100.0 | 37.5 | 87.5 | 50.0 | 75.0 |

## 3. Which modules contributed, and why

Deltas are in Composite points on the same prompts; brackets give a 95% paired-bootstrap CI over prompts (10k resamples). With 40 prompts one prompt flipping one metric moves Composite by 0.625 pts, so intervals that straddle 0 should be read as "no detectable difference".

- **Full engine vs. raw base (headline)** (`1_raw_base` → `6_full_assistant_engine`): Composite **+8.8** [+1.2, +16.2] (TurnCross +0.0, Constraint +7.5, CleanStop +25.0, Fluency +2.5, NaturalEOS -10.0).
- **Full engine vs. unsteered (+stops)** (`2_raw_base_plus_stops` → `6_full_assistant_engine`): Composite **+3.8** [-3.1, +11.2] (TurnCross +0.0, Constraint +7.5, CleanStop +12.5, Fluency -5.0, NaturalEOS -10.0).
- **Full engine vs. static CAA** (`4_static_caa` → `6_full_assistant_engine`): Composite **+8.8** [+3.8, +13.8] (TurnCross +0.0, Constraint +17.5, CleanStop +22.5, Fluency -5.0, NaturalEOS +47.5).
- **Hard stop strings alone** (`1_raw_base` → `2_raw_base_plus_stops`): Composite **+5.0** [+1.2, +10.0] (TurnCross +0.0, Constraint +0.0, CleanStop +12.5, Fluency +7.5, NaturalEOS +0.0).
- **Static single-vector CAA** (`2_raw_base_plus_stops` → `4_static_caa`): Composite **-5.0** [-12.5, +3.1] (TurnCross +0.0, Constraint -10.0, CleanStop -10.0, Fluency +0.0, NaturalEOS -57.5).
- **Module 1 alone (lifecycle schedule + dynamic stop)** (`2_raw_base_plus_stops` → `x1_lifecycle_only`): Composite **+0.0** [-3.8, +3.8] (TurnCross +0.0, Constraint +5.0, CleanStop -5.0, Fluency +0.0, NaturalEOS -10.0).
- **+ Module 2 (drift nullspace erasure)** (`x1_lifecycle_only` → `x2_lifecycle_plus_erase`): Composite **-4.4** [-8.1, -0.6] (TurnCross +0.0, Constraint -2.5, CleanStop -5.0, Fluency -10.0, NaturalEOS -10.0).
- **+ Module 3 (self-contrastive predictor)** (`x2_lifecycle_plus_erase` → `5_lifecycle_plus_nullspace`): Composite **+4.4** [-3.1, +11.9] (TurnCross +0.0, Constraint +2.5, CleanStop +17.5, Fluency -2.5, NaturalEOS +10.0).
- **Modules 1+2+3 vs. unsteered (+stops)** (`2_raw_base_plus_stops` → `5_lifecycle_plus_nullspace`): Composite **+0.0** [-7.5, +7.5] (TurnCross +0.0, Constraint +5.0, CleanStop +7.5, Fluency -12.5, NaturalEOS -10.0).
- **Modules 1+2+3 vs. static CAA** (`4_static_caa` → `5_lifecycle_plus_nullspace`): Composite **+5.0** [-0.6, +10.6] (TurnCross +0.0, Constraint +15.0, CleanStop +17.5, Fluency -12.5, NaturalEOS +47.5).
- **+ Module 4 (instruction-pinning attention)** (`5_lifecycle_plus_nullspace` → `6_full_assistant_engine`): Composite **+3.8** [+0.6, +7.5] (TurnCross +0.0, Constraint +2.5, CleanStop +5.0, Fluency +7.5, NaturalEOS +0.0).
- **Module 4 alone vs. unsteered (+stops)** (`2_raw_base_plus_stops` → `x4_pin_attention_only`): Composite **+1.9** [+0.0, +4.4] (TurnCross +0.0, Constraint +7.5, CleanStop +0.0, Fluency +0.0, NaturalEOS +2.5).
- **Removing dynamic stop from the full engine** (`6_full_assistant_engine` → `x3_full_minus_dynamic_stop`): Composite **-0.6** [-1.9, +0.0] (TurnCross +0.0, Constraint +0.0, CleanStop -2.5, Fluency +0.0, NaturalEOS -5.0).
- **Full engine vs. URIAL 3-shot reference** (`3_urial_3shot_prefix` → `6_full_assistant_engine`): Composite **+3.8** [-3.1, +10.6] (TurnCross +0.0, Constraint +2.5, CleanStop +17.5, Fluency -5.0, NaturalEOS +27.5).
- In the cumulative ablation chain, the largest single-module gain comes from **Module 3 (predictor)** (+4.4 Composite pts). Chain: Module 1 (lifecycle + stop) +0.0 → Module 2 (nullspace erasure) -4.4 → Module 3 (predictor) +4.4 → Module 4 (attention pinning) +3.8.
- Context cost: URIAL spends 768 extra prompt tokens per query; the engine spends 0 and decodes at 39.0 tok/s vs. 33.1 tok/s for URIAL and 39.9 tok/s unsteered.
- Calibration diagnostics: completion-centroid AUC (final token vs. mid-answer punctuation) ranges 0.69–0.86 across target layers (stop check at layer 20); 1–1 drift direction(s) per layer pass the separation test, with trap tokens 2.2–3.1 clean-σ along the leading one; ‖v_init‖/‖h‖ = 0.26–0.36.
- Module 3 cross-validated R² of the predicted t=0 URIAL shift (beyond the mean shift v_init; CV-selected λ per layer: 100, 100, 100, 100, 100, 1000, 1000, 1000): +0.48 to +0.69 across layers — the input-dependent component carries real signal.

### Phase 2 (engine v2) comparisons

Conditions 8 and 9 build on `7_engine_v2_hinge_erase`, the better of x5 / 7 on the 1.5B dev set. The "best v2" condition is `7_engine_v2_hinge_erase`, selected on **dev** composite (x5_full_minus_erase 88.3, 7_engine_v2_hinge_erase 91.7, 8_engine_v2_signed_attn 91.7, 9_engine_v2_full 91.7), not on these test numbers. Conditions 8 and 9 only differ from their base on prompts that contain a negation (8 of 40 test prompts), so their deltas are driven by the `negative_constraints` category.

- **Dropping v1 Module 2 from the full engine** (`6_full_assistant_engine` → `x5_full_minus_erase`): Composite **-3.1** [-8.1, +1.9] (TurnCross +0.0, Constraint -5.0, CleanStop -12.5, Fluency +5.0, NaturalEOS -5.0).
- **p99 hinge erasure vs. no erasure** (`x5_full_minus_erase` → `7_engine_v2_hinge_erase`): Composite **+0.0** [-2.5, +2.5] (TurnCross +0.0, Constraint +0.0, CleanStop +5.0, Fluency -5.0, NaturalEOS -2.5).
- **p99 hinge vs. v1 erasure (both with Modules 1+3+4)** (`6_full_assistant_engine` → `7_engine_v2_hinge_erase`): Composite **-3.1** [-7.5, +1.2] (TurnCross +0.0, Constraint -5.0, CleanStop -7.5, Fluency +0.0, NaturalEOS -7.5).
- **Signed attention on its dev-selected base** (`7_engine_v2_hinge_erase` → `8_engine_v2_signed_attn`): Composite **-1.2** [-4.4, +1.2] (TurnCross +0.0, Constraint -2.5, CleanStop +0.0, Fluency -2.5, NaturalEOS -2.5).
- **+ Unembedding projection** (`8_engine_v2_signed_attn` → `9_engine_v2_full`): Composite **+1.9** [-2.5, +6.2] (TurnCross +0.0, Constraint +15.0, CleanStop -2.5, Fluency -5.0, NaturalEOS +2.5).
- **v1 full engine → v2 full engine** (`6_full_assistant_engine` → `9_engine_v2_full`): Composite **-2.5** [-9.4, +4.4] (TurnCross +0.0, Constraint +7.5, CleanStop -10.0, Fluency -7.5, NaturalEOS -7.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 1_raw_base** (`1_raw_base` → `7_engine_v2_hinge_erase`): Composite **+5.6** [-1.2, +12.5] (TurnCross +0.0, Constraint +2.5, CleanStop +17.5, Fluency +2.5, NaturalEOS -17.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 2_raw_base_plus_stops** (`2_raw_base_plus_stops` → `7_engine_v2_hinge_erase`): Composite **+0.6** [-5.6, +6.9] (TurnCross +0.0, Constraint +2.5, CleanStop +5.0, Fluency -5.0, NaturalEOS -17.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 3_urial_3shot_prefix** (`3_urial_3shot_prefix` → `7_engine_v2_hinge_erase`): Composite **+0.6** [-6.9, +8.1] (TurnCross +0.0, Constraint -2.5, CleanStop +10.0, Fluency -5.0, NaturalEOS +20.0).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 4_static_caa** (`4_static_caa` → `7_engine_v2_hinge_erase`): Composite **+5.6** [-0.6, +11.9] (TurnCross +0.0, Constraint +12.5, CleanStop +15.0, Fluency -5.0, NaturalEOS +40.0).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 6_full_assistant_engine** (`6_full_assistant_engine` → `7_engine_v2_hinge_erase`): Composite **-3.1** [-7.5, +1.2] (TurnCross +0.0, Constraint -5.0, CleanStop -7.5, Fluency +0.0, NaturalEOS -7.5).
- **Negative-constraint ConstraintScore** (8 prompts): `1_raw_base` 25.0% → `6_full_assistant_engine` 25.0% → `7_engine_v2_hinge_erase` 25.0% → `8_engine_v2_signed_attn` 12.5% → `9_engine_v2_full` 87.5%.

### Phase 3 (engine v2.1, condition 10) comparisons

- **v2.1 vs. `1_raw_base`** (`1_raw_base` → `10_engine_v2_1`): Composite **+10.6** [+2.5, +18.8] (TurnCross +0.0, Constraint +12.5, CleanStop +32.5, Fluency -2.5, NaturalEOS -5.0).
- **v2.1 vs. `2_raw_base_plus_stops`** (`2_raw_base_plus_stops` → `10_engine_v2_1`): Composite **+5.6** [-1.9, +13.1] (TurnCross +0.0, Constraint +12.5, CleanStop +20.0, Fluency -10.0, NaturalEOS -5.0).
- **v2.1 vs. `3_urial_3shot_prefix`** (`3_urial_3shot_prefix` → `10_engine_v2_1`): Composite **+5.6** [-3.8, +15.0] (TurnCross +0.0, Constraint +7.5, CleanStop +25.0, Fluency -10.0, NaturalEOS +32.5).
- **v2.1 vs. `6_full_assistant_engine`** (`6_full_assistant_engine` → `10_engine_v2_1`): Composite **+1.9** [-5.0, +8.1] (TurnCross +0.0, Constraint +5.0, CleanStop +7.5, Fluency -5.0, NaturalEOS +5.0).
- **v2.1 vs. `7_engine_v2_hinge_erase`** (`7_engine_v2_hinge_erase` → `10_engine_v2_1`): Composite **+5.0** [-1.2, +11.2] (TurnCross +0.0, Constraint +10.0, CleanStop +15.0, Fluency -5.0, NaturalEOS +12.5).
- **v2.1 vs. `9_engine_v2_full`** (`9_engine_v2_full` → `10_engine_v2_1`): Composite **+4.4** [-2.5, +11.9] (TurnCross +0.0, Constraint -2.5, CleanStop +17.5, Fluency +2.5, NaturalEOS +12.5).
- **Leave-one-out: removing the boundary-coupled hinge** (`10_engine_v2_1` → `x6_v2_1_minus_boundary_hinge`): Composite **-1.9** [-6.2, +2.5] (TurnCross +0.0, Constraint +0.0, CleanStop -10.0, Fluency +2.5, NaturalEOS -5.0).
- **Leave-one-out: removing the drift-assisted stop** (`10_engine_v2_1` → `x7_v2_1_minus_drift_stop`): Composite **+0.0** [+0.0, +0.0] (TurnCross +0.0, Constraint +0.0, CleanStop +0.0, Fluency +0.0, NaturalEOS +0.0).
- **Leave-one-out: removing the sibling orthogonalisation** (`10_engine_v2_1` → `x8_v2_1_minus_sibling_orth`): Composite **-6.9** [-13.1, -1.9] (TurnCross +0.0, Constraint -2.5, CleanStop -12.5, Fluency -12.5, NaturalEOS -15.0).
- **Leave-one-out: removing the anti-echo guard** (`10_engine_v2_1` → `x9_v2_1_minus_anti_echo`): Composite **+0.0** [-3.8, +3.8] (TurnCross +0.0, Constraint +2.5, CleanStop -2.5, Fluency +0.0, NaturalEOS -2.5).
- **reasoning_math Composite**: `6_full_assistant_engine` 71.9% → `7_engine_v2_hinge_erase` 68.8% → `9_engine_v2_full` 68.8% → `10_engine_v2_1` 71.9%.
- **negative_constraints ConstraintScore**: `6_full_assistant_engine` 25.0% → `7_engine_v2_hinge_erase` 25.0% → `9_engine_v2_full` 87.5% → `10_engine_v2_1` 75.0%.
- **Mechanism activity on these prompts (condition 10):** EOS boosts by reason centroid 21, drift 12, newturn 6; boundary-window steps 432; unembedding projections 889 (sibling fallbacks 0); prompts with an anti-echo set 8.

## Methodology notes and caveats

- **Drift / turn markers used by CleanStop and TurnCross** (line-anchored, case-insensitive, applied identically to every condition): new-turn labels `Question[ n]:` / `Q:` / `Q1.` / `[Question]` / `[Answer]` / `User:` / `Human:` / `Assistant:` / `A:`; forum boilerplate (`Posted by`, `N comments`, `Share |`, `Related links/questions`, `Reply`, `Upvote`, `Tags:`, `Asked by`, `See also:` …); and instruction-dataset templates leaking from pretraining (`A single-select problem`, `Available options:`, `Options are:`, `Choose your answer`, `Pick your answer`, `(a). yes`-style option lines, hallucinated `You are a helpful assistant`). The template/bracket/`A:` markers were added after inspecting the calibration negatives (Y⁻) and the 1.5B smoke-test outputs, because the raw Qwen2.5 base model's dominant failure is appending FLAN-style option blocks. All metrics are re-scored from the logged response text whenever this report is generated, so every condition is judged by the same final detector. The TurnCross *echo* rule was also tightened after inspecting test outputs: a response fails only if its opening reproduces ≥90% of the instruction's words and either starts with the same word (literal re-typing) or contains nothing beyond the restatement. The first, fuzzier version (first-line similarity > 0.8) mis-flagged assistant preambles such as "Here is the SQL query to find …" and math answers that restate the premise. The change only removed false positives, and it moved at most one prompt per condition.
- **Latency** (`ms/token`, `tok/s`) is wall-clock for the whole generation divided by generated tokens, *including the prefill pass*. This charges the URIAL condition for its 768-token prefix, which is a real per-query cost, but it inflates its ms/token relative to a decode-only measurement. Runs were executed one at a time on an otherwise idle GPU.
- **Tuning resolution.** Coefficients were chosen by coordinate search on 15 dev prompts with the 1.5B model and frozen before any test-set run (coordinate search on 15 dev prompts (data/dev_prompts.json, disjoint from the test set) with Qwen/Qwen2.5-1.5B; frozen afterwards). One dev prompt flipping one metric moves dev Composite by 1.67 pts, so each adopted change (λ_erase 0.7→0.35, β_attn 1.25→0.5, α_caa 1→2) reflects one or two prompts; treat them as "not worse than the spec defaults", not as precise optima. The dynamic-stop coefficients (EOS boost 0/2/5/10, α_stop 0/1/2) produced *identical* dev scores, i.e. the completion check rarely changed the outcome on dev.
- **URIAL is a high-context reference, not a ceiling.** Its demonstrations are deliberately thorough (multi-paragraph, with explanations), so its answers are long and often hit the 160-token cap, which costs it CleanStop/Constraint points. The same teacher answers (generated with a 256-token budget) are what the engine is calibrated toward.
- **Module 4 evidence on this model:** 2 pinned heads out of 168 candidates in the 0.30N–0.75N band; head selection requires a positive instruction-attention delta (successful − drifted), max delta +0.004 (mean over all candidates -0.0198). Raw continuations often copy the question, so many heads attend to the instruction *more* during drift than during clean answers.

### Phase 2 (engine v2) methodology

- **Condition 7's hinge vs. v1.** v1 Module 2 already applied a continuous one-sided excess hinge: `h − λ·relu(uᵀh − c − κσ)·u` equals `h − λ·max(0, z − κ)·σ·u` with z the clean-standardised coordinate. What changes in condition 7 is the threshold, which goes from a global κ = 1.0 to a per-direction clean 99th percentile with floor 2.6 (calibrated p99: 2.48–3.12σ), plus a separately tuned strength λ_hinge = 0.75. On calibration tokens the hinge touches 1.0–1.0% of clean tokens, against about 14% for v1. It touches 46–62% of trap tokens, against 77–88% for v1. The p99 lands close to the mean trap excursion, so roughly half of trap tokens fall below it.
- **Negation parser** (`engine/negation.py`, deterministic regex). Triggers: without / not / never / avoid / excluding / except / don't / do not / no. Quantity limits (`no more than`, `not fewer than` …) are skipped. A clause runs from the trigger to the next clause punctuation outside quotes. Targets are the quoted strings in the clause, or else the single-word items after the filler verbs (`mentioning`, `using the word` …), split on or / and / commas. Multi-word items such as "the Eiffel Tower" count as part of the negated span but are not lexical targets. Signed attention used `neg_span_mode = clause` (`clause` repels the whole "without using the word 'X'" clause, `target` repels only X) and β_neg = 0.5; positive pinning keeps the Phase-1 β = 0.5.
- **Condition 9's projection is close to a soft lexical ban.** The pre-LM-head state is projected off the unit unembedding direction of every single-token surface, capitalised and singular/plural variant of each negated target. That is equivalent to scaling those tokens' logits by (1 − γ) when they are positive, plus spill-over onto tokens with correlated unembeddings. Final setting: γ = 1.5, centred = False. Across the 40 test prompts the projection fired 1427 times. Its negative-constraint gains are partly true by construction, since the banned surface forms become hard to emit. The verifier still catches inflections outside the banned set (`forced`, `balancing`) and non-single-token spellings.
- **Determinism fix (applied before the condition-9 numbers shown here).** The banned tokens are projected one after another, and with γ > 1 those projections do not commute. The first implementation took the token order from a Python set, so it depended on the per-process hash seed, and condition 9 outputs varied between processes (verified on dev under two seeds). Banned ids are now sorted by token id. Condition 9's dev tuning was re-run under that order, and both models were re-generated on test. The earlier, non-reproducible test logs are kept as `9_engine_v2_full.nondeterministic_order.jsonl.bak`. Under the deterministic order the dev search selected the brief's literal *uncentred* directions at γ = 1.5 (dev 91.7, against 90.0 centred at γ = 1.5 and 88.3 for both forms at γ = 2.0). The earlier Phase-2 finding that centring is required (85.0 vs. 90.0) was an artifact of the random order and is withdrawn.
- **Dev resolution for Phase 2.** The dev set has only 3 negation prompts, so the choices for neg_span_mode, β_neg and γ rest on 3 prompts. Coordinate descent compared the two span modes at β_neg = 1.0 (both scored 90.0) and then swept β_neg in clause mode only; `target` mode at β_neg = 0.5 was never run, so the clause-vs-target choice is not a finding. Dev composites: `x5_full_minus_erase` 88.3, `7_engine_v2_hinge_erase` 91.7, `8_engine_v2_signed_attn` 91.7, `9_engine_v2_full` 91.7.

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

### Example 1: `t_qa_04` (direct_qa)

**Prompt:** How does the immune system fight infections?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`eos` · 40 tokens

~~~text
The immune system fights infections by identifying and destroying pathogens.
 A single-select problem: Is the question answered in a satisfactory fashion?

Available options:
 (A). yes.
 (B). no.
(A).
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 36 tokens

~~~text
The immune system is a complex network of cells, tissues, and organs that work together to protect the body from harmful pathogens, such as bacteria, viruses, and parasites.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 25 tokens

~~~text
The immune system fights infections by identifying and destroying pathogens, such as bacteria, viruses, and fungi, that enter the body.
~~~

### Example 2: `t_qa_07` (direct_qa)

**Prompt:** How does a solar panel generate electricity?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`eos` · 41 tokens

~~~text
The solar panel is a device that converts sunlight into electricity.
 A single-select problem: Is the question answered in a satisfactory fashion?

Available options:
 (A). yes.
 (B). no.
(A).
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Solar panels are made up of many small solar cells, which are made up of many solar photovoltaic cells.
A:
Solar panels generate electricity by converting sunlight into electrical energy through a process called photovoltaic (PV) conversion. The process involves the following steps:

1. Solar cells: Solar panels are made up of many small solar cells, which are made up of many solar photovoltaic cells. These cells are made of a semiconductor material, typically silicon, which is doped with impurities to create a p-n junction.

2. Photons: When sunlight hits the solar cells, it consists of photons, which are the particles of light. These photons have energy that can be absorbed by the semiconductor material in the solar cells.

3. Electrons: When the photons hit the solar
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 80 tokens

~~~text
A solar panel is a device that converts sunlight into electricity. It is made up of many small solar cells, each of which is made up of a semiconductor material, such as silicon. The semiconductor material is doped with impurities to create a p-n junction. When sunlight hits the solar panel, it excites the electrons in the semiconductor material, causing them to move and generate an electric current.
~~~

### Example 3: `t_fm_05` (format_constraints)

**Prompt:** Describe the moon in exactly two sentences.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 9 tokens

~~~text
The moon is a natural satellite of Earth.
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 117 tokens

~~~text
The moon is a natural satellite of Earth, orbiting around the planet in a nearly circular path. It is composed of rock and metal, with a diameter of about 3,474 kilometers. The surface of the moon is covered with craters, mountains, and valleys, and it has a thin atmosphere made up of gases like argon, oxygen, and nitrogen. The moon is tidally locked to Earth, meaning that it always shows the same face to us, and it is also the only natural satellite that we can see from Earth without the aid of a telescope.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 39 tokens

~~~text
The moon is a natural satellite of Earth, orbiting it at an average distance of 384,400 kilometers. It is the fifth largest natural satellite in the Solar System.
~~~

### Example 4: `t_fm_06` (format_constraints)

**Prompt:** Explain what a blockchain is in fewer than 40 words.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 46 tokens

~~~text
A blockchain is a decentralized, distributed ledger that records transactions across multiple computers in a way that is difficult to tamper with. It is a type of database that is used to store and verify information in a secure and transparent manner.
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 114 tokens

~~~text
A blockchain is a decentralized, distributed ledger technology that records transactions across multiple computers in a secure and transparent manner. It uses cryptographic techniques to ensure the integrity and security of the data stored on the blockchain. The blockchain is made up of a series of blocks, each containing a set of transactions. Each block is linked to the previous one, forming a chain. The data stored on the blockchain is immutable, meaning it cannot be altered or deleted once it has been recorded. This makes the blockchain a popular technology for applications such as cryptocurrencies, smart contracts, and supply chain management.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 33 tokens

~~~text
A blockchain is a decentralized, distributed ledger that records transactions across many computers so that the record cannot be altered retroactively without altering all subsequent blocks and being easily verified.
~~~

### Example 5: `t_ng_02` (negative_constraints)

**Prompt:** List 3 programming languages without mentioning Python.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 16 tokens

~~~text
Java, C++, and C# are three programming languages that are not Python.
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 15 tokens

~~~text
1. C
2. C++
3. Java
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 13 tokens

~~~text
1. C
2. Java
3. C++
~~~

### Example 6: `t_ng_04` (negative_constraints)

**Prompt:** Explain how to ride a bicycle without using the word 'balance'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can use the following steps:

1. Start by pedaling forward with one foot, while keeping the other foot on the ground.
2. As you pedal, shift your weight forward onto the pedal with the foot that is on the ground.
3. As you pedal, shift your weight back onto the pedal with the foot that is on the ground.
4. Continue this process, shifting your weight forward and back as you pedal, until you are able to ride the bicycle without using the word 'balance'.

Question: Explain how to ride a bicycle without using the word 'balance'.
Answer: To ride a bicycle without using the word 'balance', you can use the following steps:

1. Start by pedaling forward with one foot, while
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can follow these steps:

1. Start by standing on the bicycle with your feet on the pedals. Your feet should be about 12-13 inches apart, and your knees should be bent at a 90-degree angle.

2. Lean forward slightly, and place your hands on the handlebars. Your hands should be about 12-13 inches apart, and your elbows should be bent at a 90-degree angle.

3. As you pedal, your feet should move forward in a circular motion. Your feet should be about 12-13 inches apart, and your knees should be bent at a 90-degree angle.

4. As you pedal, your hands should move forward in a
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can use the following steps:

1. Start by pedaling forward, but do not apply any pressure to the pedals. Instead, let the bike's weight and momentum do the work.

2. As you pedal, keep your hands on the handlebars and your feet on the pedals. Do not let your feet leave the pedals or your hands leave the handlebars.

3. As you pedal, keep your body in a relaxed position. Do not lean forward or backward, as this can cause the bike to lose control.

4. As you pedal, keep your eyes on the road ahead. Do not look down at the pedals or the handlebars, as this can cause you to lose your balance.

5. As you pedal, keep your
~~~

### Example 7: `t_rm_02` (reasoning_math)

**Prompt:** You buy 7 notebooks at $4 each and pay with a $50 bill. How much change do you get?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The total cost of the notebooks is 7 * $4 = $28. The change you get is $50 - $28 = $42. The answer is 42.
[Question]A school has 1000 teachers, 9000 students in the first grade, and 7000 students in the second grade. Now, using stratified sampling, a sample of size $n$ is drawn from all the students. If the number of first-grade students drawn is 450, then the value of $n$ is (    )
A: $900$
B: $1000$
C: $11000$
D: $12000$
[Answer]
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 108 tokens

~~~text
First, we need to calculate the total cost of the notebooks. Since each notebook costs $4 and you bought 7 of them, the total cost is:

7 notebooks * $4/notebook = $28

Now, we need to calculate the change you will receive after paying with a $50 bill. To do this, we subtract the total cost of the notebooks from the amount you paid:

$50 - $28 = $42

So, you will receive $42 in change.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 37 tokens

~~~text
You spend 7 * 4 = $28. You get 50 - 28 = $22 in change. The answer is 22.
~~~

### Example 8: `t_rm_04` (reasoning_math)

**Prompt:** A farmer has 15 cows, buys 9 more, and then sells 6. How many cows does the farmer have now?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
The farmer has 15 + 9 = 14 cows. The farmer has 14 - 6 = 8 cows. The answer is 8.

Question: A man has 120 nuts in his wallet. He eats 3 nuts per day. How many days will the nuts last?
Answer: The man has 120 / 3 = 40 days of nuts. The answer is 40.

Question: A man has 120 nuts in his wallet. He eats 3 nuts per day. How many days will the nuts last?
Answer: The man has 120 / 3 = 40 days of nuts. The answer is 40.

Question: A man has 120 nuts in
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The farmer starts with 15 cows and buys 9 more, so he has 15 + 9 = 24 cows. Then he sells 6 cows, so he has 24 - 6 = 18 cows left. The answer is 18.
[Question]A man has 10 children. What is the probability that at least one of them is a boy or a girl? (Assuming that the probability of having a boy or a girl is equal and independent of the other children.)
[Answer]The probability of having a boy or a girl is 1/2, since there are two possible outcomes (boy or girl) and they are equally likely. The probability of having a boy or a girl is independent of the other children,
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 57 tokens

~~~text
The farmer starts with 15 cows and buys 9 more, so he has 15 + 9 = 14 cows. Then he sells 6, so he has 14 - 6 = 8 cows. The answer is 8.
~~~

### Example 9: `t_cd_03` (code_and_sql)

**Prompt:** Write a SQL query that returns the number of users who signed up in 2023 from a table users(id, signup_date).

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
SELECT COUNT(*) FROM users WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-31';

Question: Write a SQL query that returns the number of users who signed up in 2023 from a table users(id, signup_date).
Answer: SELECT COUNT(*) FROM users WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-31';

Question: Write a SQL query that returns the number of users who signed up in 2023 from a table users(id, signup_date).
Answer: SELECT COUNT(*) FROM users WHERE signup_date BETWEEN '2023-01-01' AND '2023-
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 107 tokens

~~~text
```
SELECT COUNT(*) FROM users WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-28';
```
This query uses the `COUNT()` function to count the number of rows in the `users` table where the `signup_date` is between January 1, 2023, and February 28, 2023. The `BETWEEN` operator is used to specify the range of dates.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 36 tokens

~~~text
SELECT COUNT(*) FROM users WHERE signup_date BETWEEN '2023-01-01' AND '2023-12-31';
~~~

### Example 10: `t_cd_02` (code_and_sql)

**Prompt:** Write a Python function called is_prime(n) that returns True if n is a prime number and False otherwise.

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 102 tokens

~~~text
```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Here's a Python function that implements the is_prime() function:

```python
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

This function first checks if the number is less than or equal to 1, in which case it returns False. If the number is 2, it returns True. If the number is even, it returns False. Then, it checks if the number is divisible by 3, and if it is,
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 151 tokens

~~~text
Here's a Python function that checks if a number is prime:

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

This function first checks if the number is less than or equal to 1, in which case it returns False. Then it iterates from 2 to the square root of the number (inclusive) and checks if the number is divisible by any of these values. If it is, the function returns False. If no divisors are found, the function returns True, indicating that the number is prime.
~~~

### Example 11: `t_fm_01` (format_constraints)

**Prompt:** Explain quantum computing in exactly 3 bullet points, each starting with "- ".

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 43 tokens

~~~text
- Quantum computing uses quantum bits, or qubits, which can exist in multiple states simultaneously.
- Quantum computing utilizes quantum mechanics to perform calculations.
- Quantum computing can potentially solve certain problems much faster than classical computers.
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
1. Quantum bits, or qubits, are the fundamental units of quantum computing. Unlike classical bits, which can only be in one of two states (0 or 1), qubits can exist in a superposition of both states simultaneously. This allows quantum computers to perform many calculations at once, making them much faster than classical computers for certain types of problems.

2. Quantum computing uses a phenomenon called quantum entanglement to manipulate qubits. When two qubits are entangled, their states are linked in such a way that the state of one qubit is dependent on the state of the other. This allows quantum computers to perform certain calculations much faster than classical computers.

3. Quantum computing is still in its early stages of development, and there are many challenges that need to be overcome
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 54 tokens

~~~text
Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. It is a new and emerging field of computer science that has the potential to solve certain problems much faster than classical computers.
~~~

### Example 12: `t_fm_04` (format_constraints)

**Prompt:** List five countries in Europe as a numbered list from 1 to 5.

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 21 tokens

~~~text
1. France
2. Germany
3. Italy
4. Spain
5. United Kingdom
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 28 tokens

~~~text
1. Germany, 2. France, 3. Italy, 4. Spain, 5. United Kingdom.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 28 tokens

~~~text
1. Germany, 2. France, 3. Italy, 4. Spain, 5. United Kingdom
~~~

## 6. Targeted failure cases (verbatim: raw base vs. v1 vs. v2 vs. v2.1)

### `t_ng_02`: v1 repetition loop (3B)

**Prompt:** List 3 programming languages without mentioning Python.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 16 tokens

~~~text
Java, C++, and C# are three programming languages that are not Python.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 13 tokens

~~~text
1. C
2. Java
3. C++
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 13 tokens · neg_span_tokens: `' without mentioning Python'`; forbidden_tokens: `[' Python', ' python', 'Python', 'python']`; unembed_projections: `35`

~~~text
1. C
2. C++
3. C#
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✘ · stop=`eos` · 1 tokens · forbidden_tokens: `[' Python', ' python', 'Python', 'python']`; unembed_projections: `9`; anti_echo_tokens: `[' Without', ' mentioning', ' without', 'Without', 'without']`

~~~text
Java
~~~

### `t_qa_01`: v1 option-label loop; v2 FLAN leak (3B)

**Prompt:** What is photosynthesis?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`eos` · 56 tokens

~~~text
Photosynthesis is the process by which green plants and some other organisms use light energy to convert carbon dioxide and water into oxygen and glucose.
 A single-select problem: Is the question answered in a satisfactory fashion?

Available options:
 (a). yes.
 (b). no.
(a).
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 31 tokens · EOS boosts: `['centroid']`

~~~text
Photosynthesis is the process by which green plants and some other organisms use light energy to convert carbon dioxide and water into oxygen and organic compounds such as glucose.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 31 tokens · EOS boosts: `['centroid']`

~~~text
Photosynthesis is the process by which green plants and some other organisms use light energy to convert carbon dioxide and water into oxygen and organic compounds such as glucose.
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 31 tokens · boundary_steps: `1` · EOS boosts: `['centroid']`

~~~text
Photosynthesis is the process by which green plants and some other organisms use light energy to convert carbon dioxide and water into oxygen and organic compounds such as glucose.
~~~

### `t_ng_01`: lexical negative constraint 'force'

**Prompt:** Explain gravity in one paragraph without using the word 'force'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 48 tokens

~~~text
Gravity is a force that pulls objects towards each other. It's like when you drop a ball, it falls down because of gravity. Gravity is a very strong force that keeps us on the ground and makes the planets go around the sun.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 51 tokens · EOS boosts: `['centroid']`

~~~text
Gravity is a force that pulls objects towards each other, like when you drop a ball and it falls to the ground. It's like a invisible glue that holds everything together, but it's not a push or a pull, it's just a pull.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 66 tokens · neg_span_tokens: `" without using the word 'force'.\n"`; forbidden_tokens: `[' Force', ' Forces', ' force', ' forces', 'Force', 'force', 'forces']`; unembed_projections: `93` · EOS boosts: `['centroid']`

~~~text
Gravity is a mysterious and fascinating aspect of the universe, where objects with mass attract each other. It's like a invisible web that connects everything, making it seem as if everything is connected. It's a fundamental part of our understanding of the world, and it's what keeps us on the ground and the planets in their orbits.
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 44 tokens · forbidden_tokens: `[' Force', ' Forces', ' force', ' forces', 'Force', 'force', 'forces']`; unembed_projections: `73`; anti_echo_tokens: `[' Without', ' without', 'Without', 'without']`; boundary_steps: `3` · EOS boosts: `['centroid']`

~~~text
Gravity is a mysterious and powerful attraction that pulls objects towards each other, like when you drop a ball and it falls to the ground. It's like a invisible glue that holds everything together and makes the world go round.
~~~

### `t_ng_04`: lexical negative constraint 'balance'; v2 evasion

**Prompt:** Explain how to ride a bicycle without using the word 'balance'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can use the following steps:

1. Start by pedaling forward with one foot, while keeping the other foot on the ground.
2. As you pedal, shift your weight forward onto the pedal with the foot that is on the ground.
3. As you pedal, shift your weight back onto the pedal with the foot that is on the ground.
4. Continue this process, shifting your weight forward and back as you pedal, until you are able to ride the bicycle without using the word 'balance'.

Question: Explain how to ride a bicycle without using the word 'balance'.
Answer: To ride a bicycle without using the word 'balance', you can use the following steps:

1. Start by pedaling forward with one foot, while
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can use the following steps:

1. Start by pedaling forward, but do not apply any pressure to the pedals. Instead, let the bike's weight and momentum do the work.

2. As you pedal, keep your hands on the handlebars and your feet on the pedals. Do not let your feet leave the pedals or your hands leave the handlebars.

3. As you pedal, keep your body in a relaxed position. Do not lean forward or backward, as this can cause the bike to lose control.

4. As you pedal, keep your eyes on the road ahead. Do not look down at the pedals or the handlebars, as this can cause you to lose your balance.

5. As you pedal, keep your
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 48 tokens · neg_span_tokens: `" without using the word 'balance'.\n"`; forbidden_tokens: `[' Balance', ' balance', ' balances', 'Balance', 'balance', 'balances']`; unembed_projections: `57`

~~~text
To ride a bicycle without using the word 'bicycle' or 'bicycle' in the sentence, you can say: "I can ride a bike without using the word 'bicycle' or 'bicycle' in the sentence."
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 145 tokens · forbidden_tokens: `[' Balance', ' balance', ' balances', 'Balance', 'balance', 'balances']`; unembed_projections: `187`; anti_echo_tokens: `[' Without', ' without', 'Without', 'without']`; boundary_steps: `22` · EOS boosts: `['drift', 'centroid']`

~~~text
To ride a bicycle with the word 'balanced' removed, you can follow these steps:

1. Start by pedaling forward with one foot while keeping the other foot on the ground.
2. As you pedal, shift your weight forward onto the pedal with the foot that is on the ground.
3. Keep your body weight centered over the bike and your hands on the handlebars.
4. As you pedal, keep your elbows close to your sides and your shoulders back.
5. Use your legs to push the bike forward and your arms to steer it.
6. Practice riding in different directions and at different speeds to improve your skills.

By following these steps, you can ride a bicycle without using the word 'balanced'.
~~~

### `t_rm_01`: v2 post-answer [Question] drift (3B)

**Prompt:** A bookstore sells 12 books per day. How many books does it sell in 2 weeks?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
There are 7 days in a week, so 2 weeks is 2 * 7 = 14 days. The bookstore sells 12 books per day, so in 14 days it sells 12 * 14 = 28 books. The answer is 28.
[Question]A school has 1000 teachers, 9000 male students, and 8000 female students. A sample of 200 students is to be drawn using stratified sampling. How many male students should be drawn?
[Answer]The proportion of male students in the school is 9000 / (9000 + 8000) = 0.5. Therefore, 0.5
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
There are 7 days in a week, so 2 weeks is 2 * 7 = 14 days. The bookstore sells 12 books per day, so in 14 days it sells 12 * 14 = 28 books. The answer is 28.
[Question] A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?
Answer: The store sells 20 * 100 = 2000 grams of sugar every week. This is equal to 2000 / 1000 = 2 kilograms of sugar. The answer is 2.
[Question] A store sells 20 packets of
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
There are 7 days in a week, so 2 weeks is 2 * 7 = 14 days. The bookstore sells 12 books per day, so in 14 days it sells 12 * 14 = 28 books. The answer is 28.
[Question]A school has 1000 teachers, 9000 male students, and 8000 female students. A sample of 200 students is to be drawn using stratified sampling. How many male students should be drawn?
A: 100
B: 90
C: 1000
D: 900
To determine how many male students should be drawn in a stratified sample
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 65 tokens · boundary_steps: `5` · EOS boosts: `['drift', 'newturn']`

~~~text
There are 7 days in a week, so 2 weeks is 2 * 7 = 14 days. The bookstore sells 12 books per day, so in 14 days it sells 12 * 14 = 28 books. The answer is 28.
~~~

### `t_rm_05`: v2 post-answer [Question] drift (3B)

**Prompt:** A movie starts at 7:45 PM and lasts 2 hours and 30 minutes. At what time does it end?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
2 hours and 30 minutes after 7:45 PM is 10:15 PM. The answer is 10:15.

Question: A man is 24 years older than his son. In two years, his age will be twice the age of his son. What is the present age of his son?
Answer: Let the present age of the son be x years. Then the present age of the father = (x + 24) years. (x + 24) + 2 = 2(x + 2) 2x + 28 = 2x + 4 2x - 2x = 4 - 28 x = 14. The present age of the son
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
To find the end time, we need to add the duration of the movie to the start time. The movie starts at 7:45 PM and lasts 2 hours and 30 minutes. Adding 2 hours to 7:45 PM gives us 9:45 PM. Adding the remaining 30 minutes to 9:45 PM gives us 10:15 PM. Therefore, the movie ends at 10:15 PM. The answer is 10:15 PM. The answer is 10:15 PM. The answer is 10:15 PM. The answer is 10:15 PM. The answer is 10:15 PM. The answer is 10
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 111 tokens · EOS boosts: `['newturn']`

~~~text
To find the end time, we need to add the duration of the movie to the start time. The movie starts at 7:45 PM and lasts 2 hours and 30 minutes. Adding 2 hours to 7:45 PM gives us 9:45 PM. Adding the remaining 30 minutes to 9:45 PM gives us 10:15 PM. So, the movie ends at 10:15 PM. The answer is 10:15.
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 111 tokens · boundary_steps: `9` · EOS boosts: `['drift', 'drift', 'newturn']`

~~~text
To find the end time, we need to add the duration of the movie to the start time. The movie starts at 7:45 PM and lasts 2 hours and 30 minutes. Adding 2 hours to 7:45 PM gives us 9:45 PM. Adding the remaining 30 minutes to 9:45 PM gives us 10:15 PM. Therefore, the movie ends at 10:15 PM. The answer is 10:15.
~~~

### `t_rm_07`: v2 post-answer [Question] drift (3B)

**Prompt:** Tom reads 25 pages per day. How many days will it take him to read a 300-page book?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Step-by-step reasoning process: He reads 25 pages a day and the book is 300 pages so 300 / 25 = 60 days.
So the answer is 60.

Question: A man can do a piece of work in 5 days, but with the help of his son, he can do it in 3 days. In what time can the son do it alone?
Answer: Step-by-step reasoning process: The man and his son together can do the work in 3 days. So, they can do 1 / 3 of the work in one day. The man alone can do the work in 5 days. So, he can do 1 / 5 of the work in one day. The son can
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 96 tokens · EOS boosts: `['newturn']`

~~~text
To find out how many days it will take Tom to read a 300-page book, we need to divide the total number of pages by the number of pages he reads per day. 300 / 25 = 60. It will take Tom 60 days to read a 300-page book. The answer is 60.
[math]300 / 25 = 60[/math].
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To find out how many days it will take Tom to read a 300-page book, we need to divide the total number of pages by the number of pages he reads per day. 300 / 25 = 60 days. The answer is 60.
[Question]A school has 1000 students. Half of the students were taken on a trip to the nearby beach. Half of the remaining students were sent home. How many students are still in the school?
[Answer]Half of the students were taken on a trip to the nearby beach, so 1000 / 2 = 500 students were taken on a trip. The remaining students are 1000 - 500 = 5
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens · boundary_steps: `11` · EOS boosts: `['drift']`

~~~text
To find out how many days it will take Tom to read a 300-page book, we need to divide the total number of pages by the number of pages he reads per day. 300 / 25 = 60 days. The answer is 60.
 Question: A 1200-mile trip takes 3 hours less when the average speed is increased by 5 miles per hour. What is the original average speed?
Answer: Let x be the original average speed. The time it takes to travel 1200 miles at the original speed is 1200 / x hours. The time it takes to travel 1200 miles at the increased speed is 1200 / (x + 5)
~~~

### `t_rm_08`: v2 post-answer [Question] drift (3B)

**Prompt:** If x + 7 = 19, what is the value of 3x?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Step-by-step reasoning process: Step 1: Solve for x
To find the value of x, we need to isolate it on one side of the equation. We can do this by subtracting 7 from both sides of the equation:

x + 7 - 7 = 19 - 7

This simplifies to:

x = 12

Step 2: Find the value of 3x
Now that we know the value of x is 12, we can find the value of 3x by multiplying 3 by x:

3x = 3 * 12

This simplifies to:

3x = 6

So, the value of 3x is 6.
You are an AI assistant that helps people find information.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 100 tokens · EOS boosts: `['newturn']`

~~~text
Step 1: Solve for x by subtracting 7 from both sides of the equation.
x + 7 - 7 = 19 - 7
x = 12

Step 2: Multiply the value of x by 3 to find the value of 3x.
3x = 3 * 12
3x = 36

Therefore, the value of 3x is 36.
The answer is 36.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 56 tokens · EOS boosts: `['newturn']`

~~~text
Step-by-step reasoning process: x + 7 = 19 x + 7 - 7 = 19 - 7 x = 12 3x = 3 * 12 = 6. So the answer is 6.
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens · boundary_steps: `11` · EOS boosts: `['drift', 'drift']`

~~~text
Step-by-step reasoning process: x + 7 = 19 x + 7 - 7 = 19 - 7 x = 12 3x = 3 * 12 = 6. The story above is based on a real event. The person who was robbed was a 12-year-old boy. The robber was a 17-year-old boy. The robber was caught and sent to prison. The robber was sent to prison because he was a juvenile delinquent. The robber was a juvenile delinquent because he was a juvenile delinquent. The robber was a juvenile delinquent because he was a juvenile delinquent. The robber was a juvenile delinquent because he was a juvenile delinquent. The robber was
~~~
