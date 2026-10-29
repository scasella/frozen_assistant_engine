# RESULTS — Zero-Weight-Update Assistant Engine on `Qwen/Qwen2.5-3B`

_Generated automatically by `eval/report.py` on 2026-10-29 from the logs in `results/Qwen__Qwen2.5-3B/test40/`. Greedy decoding, `max_new_tokens=160`, 40 held-out prompts (`data/test_prompts.json`), device `mps` / `bfloat16`, torch 2.15.0._

No weights were modified and no gradients were computed: every condition runs the same frozen checkpoint; steered conditions only add forward hooks (residual-stream edits and a pre-softmax attention bias).

## Setup

- **Calibration:** 79/80 calibration prompts kept after filtering URIAL teacher answers (clean stop, >10 chars, distinct-4-gram ≥ 0.75); target layers [12, 14, 17, 19, 21, 23, 26, 28] of 36; stop check at layer 28; static-CAA layer 21; 14 instruction-pinning heads (L27H1, L27H4, L16H5, L12H3, L25H14, L26H12, L14H0, L25H5, L18H3, L20H1, L15H4, L21H7, L26H11, L17H10).
- **Hyperparameters:** `configs/hparams.json` — coordinate search on 15 dev prompts (data/dev_prompts.json, disjoint from the test set) with Qwen/Qwen2.5-1.5B; frozen afterwards.
- **Prompt frame for all steered conditions:** `Question: {instruction}\nAnswer:` (0 extra tokens). Stop strings `["\nQuestion:", "\nUser:"]` for conditions 2–6.
- **Metric definitions:** TurnCross = starts answering (no new question / echo / empty / garbage / script switch); Constraint = every programmatic check attached to the prompt passes (for `direct_qa` this is ≥2 expected topic keywords; math = expected final answer; code = parses / required function & SQL clauses; format/negative = exact rule); CleanStop = no hallucinated turn/FAQ/forum markers and not truncated mid-sentence or looping at max tokens; Fluency = distinct-4-gram ratio > 0.75 and > 10 chars; Composite = mean of the four. **NaturalEOS** (extra) = generation ended with the model's own `<eos>` token rather than a stop string or the token cap — with stop strings active, CleanStop alone cannot distinguish "model stopped" from "we cut it off".

## 1. Ablation summary (all values in %)

| Condition | TurnCross | Constraint | CleanStop | Fluency | **Composite** | NaturalEOS | Extra ctx tokens | ms/token | tok/s | avg new tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| `1_raw_base` | 100.0 | 75.0 | 52.5 | 95.0 | **80.6** | 75.0 | 0 | 38.3 | 26.1 | 84 |
| `2_raw_base_plus_stops` | 100.0 | 75.0 | 55.0 | 97.5 | **81.9** | 75.0 | 0 | 38.2 | 26.2 | 81 |
| `3_urial_3shot_prefix` | 100.0 | 72.5 | 67.5 | 97.5 | **84.4** | 45.0 | 768 | 47.5 | 21.1 | 112 |
| `4_static_caa` | 100.0 | 62.5 | 55.0 | 97.5 | **78.8** | 42.5 | 0 | 38.0 | 26.3 | 119 |
| `5_lifecycle_plus_nullspace` | 100.0 | 75.0 | 77.5 | 85.0 | **84.4** | 75.0 | 0 | 39.1 | 25.6 | 81 |
| `6_full_assistant_engine` | 100.0 | 80.0 | 85.0 | 90.0 | **88.8** | 77.5 | 0 | 39.0 | 25.6 | 79 |
| `x1_lifecycle_only` | 100.0 | 77.5 | 65.0 | 97.5 | **85.0** | 77.5 | 0 | 38.8 | 25.8 | 72 |
| `x2_lifecycle_plus_erase` | 100.0 | 72.5 | 77.5 | 80.0 | **82.5** | 80.0 | 0 | 39.1 | 25.6 | 85 |
| `x3_full_minus_dynamic_stop` | 100.0 | 80.0 | 77.5 | 90.0 | **86.9** | 72.5 | 0 | 40.2 | 24.9 | 88 |
| `x4_pin_attention_only` | 100.0 | 75.0 | 57.5 | 92.5 | **81.2** | 75.0 | 0 | 40.7 | 24.6 | 83 |
| `x5_full_minus_erase` | 100.0 | 80.0 | 75.0 | 97.5 | **88.1** | 72.5 | 0 | 38.7 | 25.9 | 81 |
| `7_engine_v2_hinge_erase` | 100.0 | 80.0 | 80.0 | 100.0 | **90.0** | 72.5 | 0 | 39.1 | 25.6 | 75 |
| `8_engine_v2_signed_attn` | 100.0 | 82.5 | 77.5 | 97.5 | **89.4** | 70.0 | 0 | 39.0 | 25.6 | 78 |
| `9_engine_v2_full` | 100.0 | 87.5 | 77.5 | 97.5 | **90.6** | 70.0 | 0 | 41.9 | 23.9 | 78 |
| `10_engine_v2_1` | 100.0 | 87.5 | 92.5 | 97.5 | **94.4** | 80.0 | 0 | 44.2 | 22.6 | 74 |
| `x6_v2_1_minus_boundary_hinge` | 100.0 | 85.0 | 80.0 | 97.5 | **90.6** | 75.0 | 0 | 44.1 | 22.7 | 75 |
| `x7_v2_1_minus_drift_stop` | 100.0 | 87.5 | 92.5 | 97.5 | **94.4** | 80.0 | 0 | 44.3 | 22.6 | 72 |
| `x8_v2_1_minus_sibling_orth` | 100.0 | 87.5 | 87.5 | 90.0 | **91.2** | 70.0 | 0 | 42.7 | 23.4 | 79 |
| `x9_v2_1_minus_anti_echo` | 100.0 | 92.5 | 95.0 | 97.5 | **96.2** | 82.5 | 0 | 44.1 | 22.7 | 73 |
| `11_engine_v2_2_lean` | 100.0 | 92.5 | 92.5 | 97.5 | **95.6** | 82.5 | 0 | 43.9 | 22.8 | 71 |

Stop reasons (eos / stop_string / max_tokens): `1_raw_base` 30/0/10; `2_raw_base_plus_stops` 30/1/9; `3_urial_3shot_prefix` 18/9/13; `4_static_caa` 17/6/17; `5_lifecycle_plus_nullspace` 30/2/8; `6_full_assistant_engine` 31/4/5; `x1_lifecycle_only` 31/2/7; `x2_lifecycle_plus_erase` 32/2/6; `x3_full_minus_dynamic_stop` 29/4/7; `x4_pin_attention_only` 30/1/9; `x5_full_minus_erase` 29/3/8; `7_engine_v2_hinge_erase` 29/6/5; `8_engine_v2_signed_attn` 28/6/6; `9_engine_v2_full` 28/6/6; `10_engine_v2_1` 32/5/3; `x6_v2_1_minus_boundary_hinge` 30/5/5; `x7_v2_1_minus_drift_stop` 32/5/3; `x8_v2_1_minus_sibling_orth` 28/7/5; `x9_v2_1_minus_anti_echo` 33/5/2; `11_engine_v2_2_lean` 33/5/2

## 2. Per-category breakdown

### Composite (%)

| Condition | direct_qa | format_constraints | negative_constraints | reasoning_math | code_and_sql |
|---|---|---|---|---|---|
| `1_raw_base` | 65.6 | 84.4 | 87.5 | 71.9 | 93.8 |
| `2_raw_base_plus_stops` | 65.6 | 84.4 | 87.5 | 71.9 | 100.0 |
| `3_urial_3shot_prefix` | 87.5 | 84.4 | 75.0 | 96.9 | 78.1 |
| `4_static_caa` | 75.0 | 75.0 | 75.0 | 78.1 | 90.6 |
| `5_lifecycle_plus_nullspace` | 71.9 | 87.5 | 78.1 | 90.6 | 93.8 |
| `6_full_assistant_engine` | 81.2 | 90.6 | 78.1 | 100.0 | 93.8 |
| `x1_lifecycle_only` | 71.9 | 84.4 | 87.5 | 81.2 | 100.0 |
| `x2_lifecycle_plus_erase` | 62.5 | 87.5 | 87.5 | 78.1 | 96.9 |
| `x3_full_minus_dynamic_stop` | 78.1 | 84.4 | 78.1 | 100.0 | 93.8 |
| `x4_pin_attention_only` | 68.8 | 87.5 | 78.1 | 71.9 | 100.0 |
| `x5_full_minus_erase` | 93.8 | 84.4 | 84.4 | 78.1 | 100.0 |
| `7_engine_v2_hinge_erase` | 93.8 | 84.4 | 84.4 | 87.5 | 100.0 |
| `8_engine_v2_signed_attn` | 93.8 | 84.4 | 81.2 | 87.5 | 100.0 |
| `9_engine_v2_full` | 93.8 | 84.4 | 87.5 | 87.5 | 100.0 |
| `10_engine_v2_1` | 96.9 | 90.6 | 87.5 | 96.9 | 100.0 |
| `x6_v2_1_minus_boundary_hinge` | 93.8 | 84.4 | 87.5 | 87.5 | 100.0 |
| `x7_v2_1_minus_drift_stop` | 96.9 | 90.6 | 87.5 | 96.9 | 100.0 |
| `x8_v2_1_minus_sibling_orth` | 96.9 | 90.6 | 71.9 | 96.9 | 100.0 |
| `x9_v2_1_minus_anti_echo` | 96.9 | 90.6 | 96.9 | 96.9 | 100.0 |
| `11_engine_v2_2_lean` | 96.9 | 90.6 | 96.9 | 93.8 | 100.0 |

### ConstraintScore (%)

| Condition | direct_qa | format_constraints | negative_constraints | reasoning_math | code_and_sql |
|---|---|---|---|---|---|
| `1_raw_base` | 62.5 | 50.0 | 62.5 | 100.0 | 100.0 |
| `2_raw_base_plus_stops` | 62.5 | 50.0 | 62.5 | 100.0 | 100.0 |
| `3_urial_3shot_prefix` | 100.0 | 50.0 | 37.5 | 100.0 | 75.0 |
| `4_static_caa` | 100.0 | 37.5 | 25.0 | 75.0 | 75.0 |
| `5_lifecycle_plus_nullspace` | 87.5 | 50.0 | 50.0 | 100.0 | 87.5 |
| `6_full_assistant_engine` | 100.0 | 62.5 | 50.0 | 100.0 | 87.5 |
| `x1_lifecycle_only` | 75.0 | 50.0 | 62.5 | 100.0 | 100.0 |
| `x2_lifecycle_plus_erase` | 75.0 | 50.0 | 50.0 | 100.0 | 87.5 |
| `x3_full_minus_dynamic_stop` | 100.0 | 62.5 | 50.0 | 100.0 | 87.5 |
| `x4_pin_attention_only` | 62.5 | 50.0 | 62.5 | 100.0 | 100.0 |
| `x5_full_minus_erase` | 100.0 | 50.0 | 50.0 | 100.0 | 100.0 |
| `7_engine_v2_hinge_erase` | 100.0 | 50.0 | 50.0 | 100.0 | 100.0 |
| `8_engine_v2_signed_attn` | 100.0 | 50.0 | 62.5 | 100.0 | 100.0 |
| `9_engine_v2_full` | 100.0 | 50.0 | 87.5 | 100.0 | 100.0 |
| `10_engine_v2_1` | 100.0 | 62.5 | 75.0 | 100.0 | 100.0 |
| `x6_v2_1_minus_boundary_hinge` | 100.0 | 50.0 | 75.0 | 100.0 | 100.0 |
| `x7_v2_1_minus_drift_stop` | 100.0 | 62.5 | 75.0 | 100.0 | 100.0 |
| `x8_v2_1_minus_sibling_orth` | 100.0 | 62.5 | 75.0 | 100.0 | 100.0 |
| `x9_v2_1_minus_anti_echo` | 100.0 | 62.5 | 100.0 | 100.0 | 100.0 |
| `11_engine_v2_2_lean` | 100.0 | 62.5 | 100.0 | 100.0 | 100.0 |

## 3. Which modules contributed, and why

Deltas are in Composite points on the same prompts; brackets give a 95% paired-bootstrap CI over prompts (10k resamples). With 40 prompts one prompt flipping one metric moves Composite by 0.625 pts, so intervals that straddle 0 should be read as "no detectable difference".

- **Full engine vs. raw base (headline)** (`1_raw_base` → `6_full_assistant_engine`): Composite **+8.1** [+1.9, +14.4] (TurnCross +0.0, Constraint +5.0, CleanStop +32.5, Fluency -5.0, NaturalEOS +2.5).
- **Full engine vs. unsteered (+stops)** (`2_raw_base_plus_stops` → `6_full_assistant_engine`): Composite **+6.9** [+0.0, +13.8] (TurnCross +0.0, Constraint +5.0, CleanStop +30.0, Fluency -7.5, NaturalEOS +2.5).
- **Full engine vs. static CAA** (`4_static_caa` → `6_full_assistant_engine`): Composite **+10.0** [+3.1, +16.2] (TurnCross +0.0, Constraint +17.5, CleanStop +30.0, Fluency -7.5, NaturalEOS +35.0).
- **Hard stop strings alone** (`1_raw_base` → `2_raw_base_plus_stops`): Composite **+1.2** [+0.0, +3.8] (TurnCross +0.0, Constraint +0.0, CleanStop +2.5, Fluency +2.5, NaturalEOS +0.0).
- **Static single-vector CAA** (`2_raw_base_plus_stops` → `4_static_caa`): Composite **-3.1** [-8.8, +2.5] (TurnCross +0.0, Constraint -12.5, CleanStop +0.0, Fluency +0.0, NaturalEOS -32.5).
- **Module 1 alone (lifecycle schedule + dynamic stop)** (`2_raw_base_plus_stops` → `x1_lifecycle_only`): Composite **+3.1** [-1.2, +7.5] (TurnCross +0.0, Constraint +2.5, CleanStop +10.0, Fluency +0.0, NaturalEOS +2.5).
- **+ Module 2 (drift nullspace erasure)** (`x1_lifecycle_only` → `x2_lifecycle_plus_erase`): Composite **-2.5** [-8.1, +3.1] (TurnCross +0.0, Constraint -5.0, CleanStop +12.5, Fluency -17.5, NaturalEOS +2.5).
- **+ Module 3 (self-contrastive predictor)** (`x2_lifecycle_plus_erase` → `5_lifecycle_plus_nullspace`): Composite **+1.9** [-3.8, +8.1] (TurnCross +0.0, Constraint +2.5, CleanStop +0.0, Fluency +5.0, NaturalEOS -5.0).
- **Modules 1+2+3 vs. unsteered (+stops)** (`2_raw_base_plus_stops` → `5_lifecycle_plus_nullspace`): Composite **+2.5** [-3.1, +8.1] (TurnCross +0.0, Constraint +0.0, CleanStop +22.5, Fluency -12.5, NaturalEOS +0.0).
- **Modules 1+2+3 vs. static CAA** (`4_static_caa` → `5_lifecycle_plus_nullspace`): Composite **+5.6** [-0.6, +11.9] (TurnCross +0.0, Constraint +12.5, CleanStop +22.5, Fluency -12.5, NaturalEOS +32.5).
- **+ Module 4 (instruction-pinning attention)** (`5_lifecycle_plus_nullspace` → `6_full_assistant_engine`): Composite **+4.4** [+0.6, +8.8] (TurnCross +0.0, Constraint +5.0, CleanStop +7.5, Fluency +5.0, NaturalEOS +2.5).
- **Module 4 alone vs. unsteered (+stops)** (`2_raw_base_plus_stops` → `x4_pin_attention_only`): Composite **-0.6** [-4.4, +2.5] (TurnCross +0.0, Constraint +0.0, CleanStop +2.5, Fluency -5.0, NaturalEOS +0.0).
- **Removing dynamic stop from the full engine** (`6_full_assistant_engine` → `x3_full_minus_dynamic_stop`): Composite **-1.9** [-4.4, +0.0] (TurnCross +0.0, Constraint +0.0, CleanStop -7.5, Fluency +0.0, NaturalEOS -5.0).
- **Full engine vs. URIAL 3-shot reference** (`3_urial_3shot_prefix` → `6_full_assistant_engine`): Composite **+4.4** [-1.9, +10.6] (TurnCross +0.0, Constraint +7.5, CleanStop +17.5, Fluency -7.5, NaturalEOS +32.5).
- In the cumulative ablation chain, the largest single-module gain comes from **Module 4 (attention pinning)** (+4.4 Composite pts). Chain: Module 1 (lifecycle + stop) +3.1 → Module 2 (nullspace erasure) -2.5 → Module 3 (predictor) +1.9 → Module 4 (attention pinning) +4.4.
- Context cost: URIAL spends 768 extra prompt tokens per query; the engine spends 0 and decodes at 25.6 tok/s vs. 21.1 tok/s for URIAL and 26.2 tok/s unsteered.
- Calibration diagnostics: completion-centroid AUC (final token vs. mid-answer punctuation) ranges 0.56–0.84 across target layers (stop check at layer 28); 1–1 drift direction(s) per layer pass the separation test, with trap tokens 2.6–3.4 clean-σ along the leading one; ‖v_init‖/‖h‖ = 0.26–0.31.
- Module 3 cross-validated R² of the predicted t=0 URIAL shift (beyond the mean shift v_init; CV-selected λ per layer: 100, 100, 100, 100, 1000, 1000, 1000, 10000): +0.49 to +0.69 across layers — the input-dependent component carries real signal.

### Phase 2 (engine v2) comparisons

Conditions 8 and 9 build on `7_engine_v2_hinge_erase`, the better of x5 / 7 on the 1.5B dev set. The "best v2" condition is `7_engine_v2_hinge_erase`, selected on **dev** composite (x5_full_minus_erase 88.3, 7_engine_v2_hinge_erase 91.7, 8_engine_v2_signed_attn 91.7, 9_engine_v2_full 91.7), not on these test numbers. Conditions 8 and 9 only differ from their base on prompts that contain a negation (8 of 40 test prompts), so their deltas are driven by the `negative_constraints` category.

- **Dropping v1 Module 2 from the full engine** (`6_full_assistant_engine` → `x5_full_minus_erase`): Composite **-0.6** [-6.9, +5.6] (TurnCross +0.0, Constraint +0.0, CleanStop -10.0, Fluency +7.5, NaturalEOS -5.0).
- **p99 hinge erasure vs. no erasure** (`x5_full_minus_erase` → `7_engine_v2_hinge_erase`): Composite **+1.9** [+0.0, +4.4] (TurnCross +0.0, Constraint +0.0, CleanStop +5.0, Fluency +2.5, NaturalEOS +0.0).
- **p99 hinge vs. v1 erasure (both with Modules 1+3+4)** (`6_full_assistant_engine` → `7_engine_v2_hinge_erase`): Composite **+1.2** [-3.8, +6.9] (TurnCross +0.0, Constraint +0.0, CleanStop -5.0, Fluency +10.0, NaturalEOS -5.0).
- **Signed attention on its dev-selected base** (`7_engine_v2_hinge_erase` → `8_engine_v2_signed_attn`): Composite **-0.6** [-3.8, +1.9] (TurnCross +0.0, Constraint +2.5, CleanStop -2.5, Fluency -2.5, NaturalEOS -2.5).
- **+ Unembedding projection** (`8_engine_v2_signed_attn` → `9_engine_v2_full`): Composite **+1.2** [-4.4, +6.2] (TurnCross +0.0, Constraint +5.0, CleanStop +0.0, Fluency +0.0, NaturalEOS +0.0).
- **v1 full engine → v2 full engine** (`6_full_assistant_engine` → `9_engine_v2_full`): Composite **+1.9** [-5.6, +9.4] (TurnCross +0.0, Constraint +7.5, CleanStop -7.5, Fluency +7.5, NaturalEOS -7.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 1_raw_base** (`1_raw_base` → `7_engine_v2_hinge_erase`): Composite **+9.4** [+4.4, +15.0] (TurnCross +0.0, Constraint +5.0, CleanStop +27.5, Fluency +5.0, NaturalEOS -2.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 2_raw_base_plus_stops** (`2_raw_base_plus_stops` → `7_engine_v2_hinge_erase`): Composite **+8.1** [+3.1, +13.1] (TurnCross +0.0, Constraint +5.0, CleanStop +25.0, Fluency +2.5, NaturalEOS -2.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 3_urial_3shot_prefix** (`3_urial_3shot_prefix` → `7_engine_v2_hinge_erase`): Composite **+5.6** [-0.6, +11.9] (TurnCross +0.0, Constraint +7.5, CleanStop +12.5, Fluency +2.5, NaturalEOS +27.5).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 4_static_caa** (`4_static_caa` → `7_engine_v2_hinge_erase`): Composite **+11.2** [+6.2, +16.2] (TurnCross +0.0, Constraint +17.5, CleanStop +25.0, Fluency +2.5, NaturalEOS +30.0).
- **Best v2 (`7_engine_v2_hinge_erase`) vs. 6_full_assistant_engine** (`6_full_assistant_engine` → `7_engine_v2_hinge_erase`): Composite **+1.2** [-3.8, +6.9] (TurnCross +0.0, Constraint +0.0, CleanStop -5.0, Fluency +10.0, NaturalEOS -5.0).
- **Negative-constraint ConstraintScore** (8 prompts): `1_raw_base` 62.5% → `6_full_assistant_engine` 50.0% → `7_engine_v2_hinge_erase` 50.0% → `8_engine_v2_signed_attn` 62.5% → `9_engine_v2_full` 87.5%.

### Phase 3 (engine v2.1, condition 10) comparisons

- **v2.1 vs. `1_raw_base`** (`1_raw_base` → `10_engine_v2_1`): Composite **+13.8** [+7.5, +20.0] (TurnCross +0.0, Constraint +12.5, CleanStop +40.0, Fluency +2.5, NaturalEOS +5.0).
- **v2.1 vs. `2_raw_base_plus_stops`** (`2_raw_base_plus_stops` → `10_engine_v2_1`): Composite **+12.5** [+6.9, +18.1] (TurnCross +0.0, Constraint +12.5, CleanStop +37.5, Fluency +0.0, NaturalEOS +5.0).
- **v2.1 vs. `3_urial_3shot_prefix`** (`3_urial_3shot_prefix` → `10_engine_v2_1`): Composite **+10.0** [+3.8, +16.2] (TurnCross +0.0, Constraint +15.0, CleanStop +25.0, Fluency +0.0, NaturalEOS +35.0).
- **v2.1 vs. `6_full_assistant_engine`** (`6_full_assistant_engine` → `10_engine_v2_1`): Composite **+5.6** [+0.0, +11.2] (TurnCross +0.0, Constraint +7.5, CleanStop +7.5, Fluency +7.5, NaturalEOS +2.5).
- **v2.1 vs. `7_engine_v2_hinge_erase`** (`7_engine_v2_hinge_erase` → `10_engine_v2_1`): Composite **+4.4** [+0.6, +8.1] (TurnCross +0.0, Constraint +7.5, CleanStop +12.5, Fluency -2.5, NaturalEOS +7.5).
- **v2.1 vs. `9_engine_v2_full`** (`9_engine_v2_full` → `10_engine_v2_1`): Composite **+3.8** [-1.2, +8.8] (TurnCross +0.0, Constraint +0.0, CleanStop +15.0, Fluency +0.0, NaturalEOS +10.0).
- **Leave-one-out: removing the boundary-coupled hinge** (`10_engine_v2_1` → `x6_v2_1_minus_boundary_hinge`): Composite **-3.8** [-6.9, -1.2] (TurnCross +0.0, Constraint -2.5, CleanStop -12.5, Fluency +0.0, NaturalEOS -5.0).
- **Leave-one-out: removing the drift-assisted stop** (`10_engine_v2_1` → `x7_v2_1_minus_drift_stop`): Composite **+0.0** [+0.0, +0.0] (TurnCross +0.0, Constraint +0.0, CleanStop +0.0, Fluency +0.0, NaturalEOS +0.0).
- **Leave-one-out: removing the sibling orthogonalisation** (`10_engine_v2_1` → `x8_v2_1_minus_sibling_orth`): Composite **-3.1** [-7.5, +0.6] (TurnCross +0.0, Constraint +0.0, CleanStop -5.0, Fluency -7.5, NaturalEOS -10.0).
- **Leave-one-out: removing the anti-echo guard** (`10_engine_v2_1` → `x9_v2_1_minus_anti_echo`): Composite **+1.9** [+0.0, +5.0] (TurnCross +0.0, Constraint +5.0, CleanStop +2.5, Fluency +0.0, NaturalEOS +2.5).
- **reasoning_math Composite**: `6_full_assistant_engine` 100.0% → `7_engine_v2_hinge_erase` 87.5% → `9_engine_v2_full` 87.5% → `10_engine_v2_1` 96.9%.
- **negative_constraints ConstraintScore**: `6_full_assistant_engine` 50.0% → `7_engine_v2_hinge_erase` 50.0% → `9_engine_v2_full` 87.5% → `10_engine_v2_1` 75.0%.
- **Mechanism activity on these prompts (condition 10):** EOS boosts by reason centroid 14, drift 11, newturn 4; boundary-window steps 336; unembedding projections 1005 (sibling fallbacks 0); prompts with an anti-echo set 8.

### Phase 4 (v2.2 lean engine, post-trained reference) comparisons

- **v2.2 lean vs. `1_raw_base`** (`1_raw_base` → `11_engine_v2_2_lean`): Composite **+15.0** [+9.4, +21.2] (TurnCross +0.0, Constraint +17.5, CleanStop +40.0, Fluency +2.5, NaturalEOS +7.5).
- **v2.2 lean vs. `3_urial_3shot_prefix`** (`3_urial_3shot_prefix` → `11_engine_v2_2_lean`): Composite **+11.2** [+5.0, +17.5] (TurnCross +0.0, Constraint +20.0, CleanStop +25.0, Fluency +0.0, NaturalEOS +37.5).
- **v2.2 lean vs. `10_engine_v2_1`** (`10_engine_v2_1` → `11_engine_v2_2_lean`): Composite **+1.2** [-1.2, +4.4] (TurnCross +0.0, Constraint +5.0, CleanStop +0.0, Fluency +0.0, NaturalEOS +2.5).
- **v2.2 lean vs. `x9_v2_1_minus_anti_echo`** (`x9_v2_1_minus_anti_echo` → `11_engine_v2_2_lean`): Composite **-0.6** [-1.9, +0.0] (TurnCross +0.0, Constraint +0.0, CleanStop -2.5, Fluency +0.0, NaturalEOS +0.0).
- **Byte identity with `x9`:** 37/40 responses identical. 11 x9 prompts had a drift-stop EOS boost (the mechanism condition 11 drops); the 3 differing prompts are `t_qa_05`, `t_rm_05`, `t_rm_06`, of which 3 had such a boost.

### Interpretation (hand-written after inspecting the logged generations)

**Headline.** The full engine scores **88.8** Composite from the bare `Question/Answer` frame. That is +8.1 over the raw base,
with a 95% CI of [+1.9, +14.4]. It spends 0 extra context tokens, decodes at 25.6 tok/s against URIAL's 21.1, and comes
in +4.4 over the 768-token URIAL 3-shot reference, but that interval includes zero. The fair reading is that the engine
*matches* URIAL at no context cost; it does not reliably beat it. Hyperparameters were tuned on 1.5B dev prompts and
transferred to 3B unchanged; the only 3B check was a 15-prompt dev sanity run, with no re-tuning.

**Where the gain comes from: termination and genre drift, not turn-crossing.**
- **TurnCross is saturated.** Qwen2.5-3B base already starts an answer after `Answer:` on 40/40 prompts. Failure mode 1
  (question continuation *instead of* answering) essentially does not occur on this checkpoint with this frame.
- **The raw model's real failure is what comes after a correct answer.** 16/40 raw+stops responses are contaminated:
  - FLAN-style blocks like `A single-select problem: Is the question answered in a satisfactory fashion? Choose your
    answer from: (1) …`;
  - `[Question]` continuations, e.g. t_rm_01 answers "168" correctly and then writes `[Question]A store sells 20 packets…`;
  - hallucinated instruction-dataset system prompts.

  The full engine reduces contaminated responses to 2/40 and cap hits from 9 to 5. CleanStop rises 55% → 85%, while
  natural `<eos>` stays about the same (75% → 77.5%). So the clean endings are mostly the model's own `<eos>` arriving
  *before* it drifts, not stop strings cutting it off.
- **Static CAA makes termination worse, as the non-stationarity argument predicts.** One vector added at every position
  keeps the residual stream in "keep answering" mode: NaturalEOS falls 75% → 42.5% and 17/40 responses hit the token cap.
  Composite is −3.1 vs. raw+stops and −10.0 [−16.2, −3.1] vs. the full engine. The lifecycle schedule decays the
  initiation push after 4 steps and leaves the stop decision to the model, which preserves its native end-of-answer
  behaviour.

**Module-by-module (cumulative chain on 3B).** Individual steps move the score by 1–3 prompts, and most of their
intervals include zero. Only the endpoints and the Module-4 step are clearly resolved.
- **Module 1 (lifecycle + dynamic stop): +3.1.** It is mostly CleanStop (+10). Removing only the dynamic stop from the
  full engine costs −1.9 [−4.4, 0.0] (CleanStop −7.5), a small but consistently signed effect. On the 1.5B dev set the
  stop coefficients had no effect at all.
- **Module 2 (drift erasure): −2.5 net.** It is the most double-edged module: CleanStop +12.5 but Fluency −17.5. Erasing
  the leading drift direction removes the *surface signature* of the exam/forum genre (option labels, `Posted by`, turn
  labels), but not always the genre itself. Several x2 outputs fall into unlabeled multiple-choice option lists
  (`C. They use quantum bits… D. They use classical bits…`) or `The answer is 18. The final answer: 18.` loops, which
  Fluency catches. The near-zero-effect property holds on clean text: on 3B calibration data the dead zone touches 13–14%
  of clean tokens against 80–89% of trap tokens. The cost comes once the model has already left the clean distribution.
- **Module 3 (single-pass predictor): +1.9**, mostly Fluency +5. Its calibration signal is real: cross-validated R² of
  0.49–0.69 for the input-dependent part of the URIAL shift at the delimiter, beyond the mean vector. On 1.5B dev,
  dropping it cost 10 Composite points, the largest single-coefficient effect in the tuning log.
- **Module 4 (instruction pinning): +4.4 [+0.6, +8.8] on top of Modules 1–3**, the only cumulative step whose interval
  excludes zero. On its own it does nothing: −0.6 vs. raw+stops. The gain shows up as CleanStop +7.5, Fluency +5 and
  Constraint +5, i.e. fewer of the Module-2 option-list loops. A plausible reading is that once the register is steered,
  biasing 14 instruction-attending heads keeps generation anchored to *this* question rather than a templated one.
  That is a hypothesis, not something these ablations isolate. On 1.5B only 2 heads had positive selection evidence,
  and Module 4 was neutral-to-harmful there.

**What does not work.**
- **Lexical negative constraints** (`without using the word 'force'`) remain largely unsolved: 50% for the engine against
  62.5% for raw. Every Phase-1 condition, URIAL and static CAA included, uses the word in t_ng_01; most open with "Gravity is a force…". Neither more attention on
  the instruction span nor genre steering gives the model a mechanism for suppressing a specific token.
- **Verbosity inherited from the teacher.** The engine is calibrated toward URIAL-style answers, which are long. Five
  engine responses still hit the 160-token cap, among them a code answer that kept printing test cases and a numbered
  how-to list.

**Replication on Qwen2.5-1.5B** (same 40 test prompts, same frozen coefficients; 1.5B was the tuning model, but only on the
disjoint dev set; see Appendix):
- The full engine is again best: 84.4 Composite, against 75.6 raw, 80.6 raw+stops, 80.6 URIAL and 75.6 static CAA.
  The margin over raw+stops is smaller than on 3B and not resolved: +3.8.
- The two module-level effects that replicate are:
  - **Module 4 helps on top of Modules 1–3**: +3.8 [+0.6, +7.5] on 1.5B and +4.4 [+0.6, +8.8] on 3B.
  - **Module 2 erasure costs fluency**: −4.4 [−8.1, −0.6] Composite on 1.5B and −2.5 on 3B.
- Static CAA lowers NaturalEOS on both models (78% → 20% on 1.5B), which is the clearest evidence for the claim that a
  single static direction fights termination.
- TurnCross is 100% for every condition on both models, so the turn-crossing part of Module 1 is untested by this
  benchmark on Qwen2.5. These base checkpoints already answer after `Answer:`; their failures are drift and run-on
  after the answer.

### Phase 2 interpretation (engine v2, hand-written after inspecting the logged generations)

**Bottom line.** On 3B the v2 conditions score 88.1 (x5), **90.0 (7, the dev-selected v2)**, 89.4 (8) and 90.6 (9),
against 88.8 for v1. No v2-vs-v1 difference is statistically resolved: condition 7 is +1.2 [−3.8, +6.9] and condition 9
is +1.9 [−5.6, +9.4]. On 1.5B every v2 condition is at or below v1 (84.4): 81.2, 81.2, 80.0, 81.9. Condition 9's numbers
come from the deterministic re-run described in the methodology section; the first run had a hash-seed-dependent
projection order and is withdrawn. What v2 does buy is a slightly larger margin over the baselines: condition 7 is
+8.1 [+3.1, +13.1] over raw+stops on 3B, against +6.9 [+0.0, +13.8] for v1.

**1. Removing Module 2 from the full engine does not help (x5).** The brief expected it to, but the full engine scores
−0.6 on 3B and −3.1 on 1.5B without it. Module 2's negative delta in Phase 1 came from the Module-1-only chain (x1 → x2).
Alongside Modules 3 and 4 it trades fluency for clean stops: on 3B, x5 has Fluency 97.5 (v1: 90) but CleanStop 75
(v1: 85). The cached logs also show that the 3B `t_ng_02` list loop is not a Module-2 effect. It is absent in x2, and
it appears once Module 3 is added and with Module 4 alone.

**2. The p99 hinge recovers fluency on 3B (7 vs. x5: +1.9 [+0.0, +4.4]; Fluency 100%), but it gives back drift
suppression.**
- **Same operator, new threshold.** v1's operator was already a continuous excess hinge. Moving its threshold from
  κ=1 to each direction's clean p99 (2.6–3.3σ) cuts clean-token firing from about 14% to 1%, and trap-token firing from
  about 85% to about 57%.
- **What that buys.** The v1 loops disappear: the `t_qa_01` A:/B:/C: list, and `t_qa_03` / `t_qa_06`.
- **What it costs.** The FLAN templates come back:
  - `t_qa_01` now ends in the raw model's "A single-select problem…" block;
  - four of the eight 3B math answers (`t_rm_01`, `05`, `07`, `08`) now trail into `[Question]A store is selling…`-style
    continuations, against none under v1;
  - reasoning_math Composite falls 100 → 87.5 on 3B.

  The v1 setting (κ=1, λ=0.35) and the hinge (κ≈3, λ=0.75) sit at two ends of one trade-off. Neither dominates on
  both models.

**3. Polarity-aware signed attention has no measurable benefit (8 vs. 7).** It scores −0.6 [−3.8, +1.9] on 3B and −1.2
[−4.4, +1.2] on 1.5B. Negative-constraint ConstraintScore goes 50 → 62.5% on 3B and *drops* 25 → 12.5% on 1.5B. On dev,
the best β_neg (0.5) gave scores identical to condition 7's (91.7 overall, with the same per-metric breakdown), and any
stronger setting hurt. Repelling attention from the
negation clause also repels it from the instruction "without using the word…", which by itself removes nothing about
the forbidden word.

**4. The unembedding projection is what moves negative constraints (8 → 9).** Negative-constraint ConstraintScore goes
62.5 → 87.5% on 3B and 12.5 → 87.5% on 1.5B, a much larger shift than anything else in Phase 2, although Composite
moves only +1.2 [−4.4, +6.2] and +1.9 [−2.5, +6.2]. `t_ng_01` becomes a correct, fluent answer that never says
"force". Three caveats:
- **It is close to a lexical ban by construction.** It scales the banned surface forms' logits by (1 − γ) whenever they
  are positive. The verifier's main negative-constraint check is "banned word absent", so part of this gain is
  mechanical.
- **Some passes are evasions.** On `t_ng_04` both models answer "To ride a bicycle without using the word
  **'bicycle'**, you can say…". The echoed prompt phrase loses its banned word, the model substitutes the topic word,
  and it never explains how to ride. The verifier scores this as a full pass, because the negative-constraint checks
  only test the banned word and length, not task completion.
- **Bans can still derail.** On 1.5B, `t_ng_03` loops on "dynamic, dynamic, dynamic …".
- **Withdrawn claims.** Two Phase-2 claims came from the non-deterministic run and are withdrawn:
  - that centring the unembedding rows was required (85.0 vs. 90.0 dev);
  - that `t_ng_02` collapsed to "C++, Assembly Language, Assembly Language".

  Under the deterministic order the dev search picked the brief's literal uncentred form at γ = 1.5, and condition 9
  answers `t_ng_02` correctly ("1. C 2. C++ 3. C#") on both models.

**What these results support.** Negative lexical constraints respond to output-side suppression and not to attention
repulsion. The erasure threshold is a real fluency-vs-drift trade-off, not a bug. The headline engine gain still rests
on Phase 1's termination and drift effects.

### Phase 3 interpretation (engine v2.1, condition 10; hand-written after inspecting the logged generations)

**Bottom line.** Condition 10 scores **94.4** on 3B, the best dev-selected configuration in the study:
- +13.8 [+7.5, +20.0] over the raw base;
- +12.5 [+6.9, +18.1] over raw+stops;
- **+10.0 [+3.8, +16.2] over URIAL**, the first engine whose lead over URIAL is resolved;
- +5.6 [+0.0, +11.2] over v1 and +4.4 [+0.6, +8.1] over condition 7.

Over condition 9 it is +3.8 [−1.2, +8.8], which is not resolved. On 1.5B it scores 86.2, the best on that model, but
only the margin over the raw base is resolved (+10.6 [+2.5, +18.8]); over v1 it is +1.9 [−5.0, +8.1]. Its dev score was
93.3 against 91.7 for both 7 and 9, a one-prompt margin. Most of the test gain was therefore not visible on dev.

**What did the work (leave-one-out on test; attribution only, nothing was selected from these rows):**
- **Boundary-coupled hinge: the main effect.** Removing it costs −3.8 [−6.9, −1.2] on 3B (CleanStop −12.5) and −1.9 on
  1.5B (CleanStop −10). It repairs condition 7's drift leaks *as the verifier sees them*: 3B reasoning_math Composite
  goes from 87.5 back to 96.9, and `t_rm_05`, `07`, `08` and `t_qa_01` now pass CleanStop. The outputs are less clean
  than that suggests:
  - `t_rm_08` still continues with `[Equation] x + 7 = 19` / `[Step 1] Subtract 7 …`, a bracket-labelled continuation
    that the marker list (only `[Question]` / `[Answer]`) does not cover;
  - `t_rm_07` repeats "The answer is 12." three times without falling below the 4-gram fluency threshold;
  - `t_qa_01` is cut by the `Question:` stop string (the model still began a new turn) rather than ending on `<eos>`.

  These are known verifier blind spots, disclosed here rather than patched, because patching after seeing test
  outputs would be test-driven.

  The brief's mechanism ("the onset z of 1.5–2.5σ slips under p99") is not what calibration data show. The boundary
  newline sits at 3.9σ, above the 3.1σ threshold, so the v2 hinge already fired there but removed only part of a small
  excess. The boundary threshold (dev-tuned to 1.5, not the brief's 1.0) works because it removes most of an
  excursion that is already above threshold.
- **Sibling orthogonalisation: real, and largest on 1.5B.** Removing it costs −3.1 [−7.5, +0.6] on 3B and
  −6.9 [−13.1, −1.9] on 1.5B (Fluency −12.5, CleanStop −12.5). Its guarantee holds exactly: the top-K siblings'
  relative logits change by 0.00, against −3.6 to −13.7 under the plain projection, and there were no fallbacks.
- **Drift-assisted stop: inert.** It fired on 10 prompts per model but changed only 3/40 (3B) and 1/40 (1.5B)
  responses, with no metric change. On dev, its threshold (1.5 / 1.8 / 2.2) never changed a score either.
- **Anti-echo guard: not helpful.** Removing it scores *better* on 3B, +1.9 [+0.0, +5.0], and the same on 1.5B. The guard
  does stop the opening echo on `t_ng_04` ("To ride a bicycle effectively and safely…"), but:
  - the model restates the constraint ~25 tokens later, after the 12-token window, using the unbanned inflection
    "balancing";
  - on `t_ng_07` it lists two animals instead of three;
  - on 1.5B `t_ng_02` it collapses to "Java".

  Part of the "gain" from removing it is a verifier artefact: without the guard, `t_ng_04` echoes "without using the
  word **'balanc'**" (the banned token truncated), which the regex does not match. The 96.2 for the no-anti-echo
  variant is a test-set observation, not a selectable result.

**What these results support.** Coupling erasure strength to turn boundaries is the one v2.1 change with a resolved
effect: it recovers v1's drift suppression without v1's loops. Sibling orthogonalisation measurably removes the
collateral cost of output-side lexical suppression. The drift-coupled stop and the anti-echo guard add nothing
detectable here. Negative lexical constraints remain the weakest category at 75% for condition 10 on both models, and
the verifier cannot tell an evasive non-answer from a compliant one.

## Methodology notes and caveats

- **Drift / turn markers used by CleanStop and TurnCross** (line-anchored, case-insensitive, applied identically to every condition): new-turn labels `Question[ n]:` / `Q:` / `Q1.` / `[Question]` / `[Answer]` / `User:` / `Human:` / `Assistant:` / `A:`; forum boilerplate (`Posted by`, `N comments`, `Share |`, `Related links/questions`, `Reply`, `Upvote`, `Tags:`, `Asked by`, `See also:` …); and instruction-dataset templates leaking from pretraining (`A single-select problem`, `Available options:`, `Options are:`, `Choose your answer`, `Pick your answer`, `(a). yes`-style option lines, hallucinated `You are a helpful assistant`). The template/bracket/`A:` markers were added after inspecting the calibration negatives (Y⁻) and the 1.5B smoke-test outputs, because the raw Qwen2.5 base model's dominant failure is appending FLAN-style option blocks. All metrics are re-scored from the logged response text whenever this report is generated, so every condition is judged by the same final detector. The TurnCross *echo* rule was also tightened after inspecting test outputs: a response fails only if its opening reproduces ≥90% of the instruction's words and either starts with the same word (literal re-typing) or contains nothing beyond the restatement. The first, fuzzier version (first-line similarity > 0.8) mis-flagged assistant preambles such as "Here is the SQL query to find …" and math answers that restate the premise. The change only removed false positives, and it moved at most one prompt per condition.
- **Latency** (`ms/token`, `tok/s`) is wall-clock for the whole generation divided by generated tokens, *including the prefill pass*. This charges the URIAL condition for its 768-token prefix, which is a real per-query cost, but it inflates its ms/token relative to a decode-only measurement. Runs were executed one at a time on an otherwise idle GPU.
- **Tuning resolution.** Coefficients were chosen by coordinate search on 15 dev prompts with the 1.5B model and frozen before any test-set run (coordinate search on 15 dev prompts (data/dev_prompts.json, disjoint from the test set) with Qwen/Qwen2.5-1.5B; frozen afterwards). One dev prompt flipping one metric moves dev Composite by 1.67 pts, so each adopted change (λ_erase 0.7→0.35, β_attn 1.25→0.5, α_caa 1→2) reflects one or two prompts; treat them as "not worse than the spec defaults", not as precise optima. The dynamic-stop coefficients (EOS boost 0/2/5/10, α_stop 0/1/2) produced *identical* dev scores, i.e. the completion check rarely changed the outcome on dev.
- **URIAL is a high-context reference, not a ceiling.** Its demonstrations are deliberately thorough (multi-paragraph, with explanations), so its answers are long and often hit the 160-token cap, which costs it CleanStop/Constraint points. The same teacher answers (generated with a 256-token budget) are what the engine is calibrated toward.
- **Module 4 evidence on this model:** 14 pinned heads out of 288 candidates in the 0.30N–0.75N band; head selection requires a positive instruction-attention delta (successful − drifted), max delta +0.018 (mean over all candidates -0.0104). Raw continuations often copy the question, so many heads attend to the instruction *more* during drift than during clean answers.

### Phase 2 (engine v2) methodology

- **Condition 7's hinge vs. v1.** v1 Module 2 already applied a continuous one-sided excess hinge: `h − λ·relu(uᵀh − c − κσ)·u` equals `h − λ·max(0, z − κ)·σ·u` with z the clean-standardised coordinate. What changes in condition 7 is the threshold, which goes from a global κ = 1.0 to a per-direction clean 99th percentile with floor 2.6 (calibrated p99: 2.87–3.29σ), plus a separately tuned strength λ_hinge = 0.75. On calibration tokens the hinge touches 1.0–1.0% of clean tokens, against about 14% for v1. It touches 51–62% of trap tokens, against 80–89% for v1. The p99 lands close to the mean trap excursion, so roughly half of trap tokens fall below it.
- **Negation parser** (`engine/negation.py`, deterministic regex). Triggers: without / not / never / avoid / excluding / except / don't / do not / no. Quantity limits (`no more than`, `not fewer than` …) are skipped. A clause runs from the trigger to the next clause punctuation outside quotes. Targets are the quoted strings in the clause, or else the single-word items after the filler verbs (`mentioning`, `using the word` …), split on or / and / commas. Multi-word items such as "the Eiffel Tower" count as part of the negated span but are not lexical targets. Signed attention used `neg_span_mode = clause` (`clause` repels the whole "without using the word 'X'" clause, `target` repels only X) and β_neg = 0.5; positive pinning keeps the Phase-1 β = 0.5.
- **Condition 9's projection is close to a soft lexical ban.** The pre-LM-head state is projected off the unit unembedding direction of every single-token surface, capitalised and singular/plural variant of each negated target. That is equivalent to scaling those tokens' logits by (1 − γ) when they are positive, plus spill-over onto tokens with correlated unembeddings. Final setting: γ = 1.5, centred = False. Across the 40 test prompts the projection fired 1417 times. Its negative-constraint gains are partly true by construction, since the banned surface forms become hard to emit. The verifier still catches inflections outside the banned set (`forced`, `balancing`) and non-single-token spellings.
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
| 12 | 62.0 | 0.299 | 0.117 | 0.43 | 0.56 | 0.74 | 1 | 2.61 | 14% / 80% | 2.98 | 1% / 51% | 100 | +0.68 |
| 14 | 66.2 | 0.275 | 0.109 | 0.42 | 0.57 | 0.76 | 1 | 2.81 | 14% / 81% | 3.02 | 1% / 56% | 100 | +0.69 |
| 17 | 68.6 | 0.295 | 0.116 | 0.45 | 0.61 | 0.75 | 1 | 3.08 | 14% / 85% | 3.20 | 1% / 58% | 100 | +0.68 |
| 19 | 73.0 | 0.299 | 0.121 | 0.51 | 0.64 | 0.74 | 1 | 3.34 | 14% / 88% | 3.29 | 1% / 61% | 100 | +0.63 |
| 21 | 78.8 | 0.306 | 0.125 | 0.45 | 0.68 | 0.70 | 1 | 3.35 | 14% / 86% | 3.25 | 1% / 61% | 1000 | +0.65 |
| 23 | 84.3 | 0.296 | 0.126 | 0.50 | 0.72 | 0.70 | 1 | 3.41 | 13% / 88% | 3.26 | 1% / 62% | 1000 | +0.65 |
| 26 | 103.3 | 0.309 | 0.118 | 0.53 | 0.80 | 0.63 | 1 | 3.30 | 14% / 89% | 3.23 | 1% / 60% | 1000 | +0.62 |
| 28 | 137.8 | 0.259 | 0.095 | 0.53 | 0.84 | 0.60 | 1 | 2.83 | 14% / 89% | 2.87 | 1% / 53% | 10000 | +0.49 |

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

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`eos` · 51 tokens

~~~text
The immune system fights infections by recognizing and destroying pathogens, which are microorganisms that cause disease.
 A single-select problem: Is the question answered in a satisfactory fashion?

Choose your answer from:
 (1). yes.
 (2). no.
(1).
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
The immune system fights infections by recognizing and eliminating pathogens, such as bacteria, viruses, fungi, and parasites. It does this through a complex network of cells, tissues, and organs that work together to identify and neutralize foreign invaders.
```

Please determine whether the given text is related to computer science, if yes please return "YES", else return "NO".

Text:  The immune system fights infections by recognizing and eliminating pathogens, such as bacteria, viruses, fungi, and parasites. It does this through a complex network of cells, tissues, and organs that work together to identify and neutralize foreign invaders.

Based on the provided text, it is clear that the immune system plays a crucial role in fighting infections. The text describes how the immune system recognizes and eliminates pathogens, such as bacteria,
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 139 tokens

~~~text
The immune system fights infections by recognizing and destroying pathogens, which are microorganisms that cause disease. This process involves a complex network of cells, tissues, and organs that work together to identify and eliminate foreign invaders. The immune system has two main branches: the innate immune system and the adaptive immune system. The innate immune system is the first line of defense and includes physical barriers such as the skin and mucous membranes, as well as cells like macrophages and neutrophils that can engulf and destroy pathogens. The adaptive immune system, on the other hand, is more specific and involves the production of antibodies by B cells and the activation of T cells, which can recognize and target specific pathogens.
~~~

### Example 2: `t_qa_08` (direct_qa)

**Prompt:** What is the purpose of the United Nations?

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`eos` · 68 tokens

~~~text
The United Nations (UN) is an intergovernmental organization that was established on 24 October 1945 after the Second World War in order to prevent another such conflict.
 A single-select problem: Is the question answered in a satisfactory fashion?

Available choices:
 (1). yes
 (2). no
(1).
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The United Nations (UN) is an intergovernmental organization that was established after World War II with the aim of promoting international cooperation and maintaining international peace and security.
A more detailed answer would be:

The United Nations (UN) is an intergovernmental organization that was established on October 24, 1945, after World War II. Its primary purpose is to promote international cooperation and maintain international peace and security. The UN was created as a result of the devastation caused by the war and the desire to prevent future conflicts.

The UN is composed of 193 member states, including all of the world's major powers. Its main objectives are to:

1. Maintain international peace and security through diplomacy, mediation, and conflict resolution.
2. Promote human rights, democracy
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 149 tokens

~~~text
The United Nations (UN) is an intergovernmental organization that aims to promote international cooperation and maintain international peace and security. It was established after World War II to prevent future conflicts and address global issues such as human rights, economic development, and environmental protection. The UN has 193 member states and operates through various specialized agencies, programs, funds, and offices. Its main organs include the General Assembly, the Security Council, the Economic and Social Council, the International Court of Justice, and the Secretariat. The UN also has a number of specialized agencies, such as the World Health Organization, the United Nations Children's Fund, and the United Nations Development Programme, which work on specific issues such as health, education, and development.
~~~

### Example 3: `t_fm_05` (format_constraints)

**Prompt:** Describe the moon in exactly two sentences.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The moon is a natural satellite of Earth, orbiting around it at an average distance of about 384,400 kilometers. It is the fifth largest moon in the solar system and has a diameter of approximately 3,474 kilometers. The moon's surface is primarily composed of basaltic rock and is covered by a thin layer of dust and regolith. It has a complex geological history, with evidence of impact craters, volcanic activity, and tectonic movements. The moon's orbit around Earth is elliptical, with a period of approximately 27.3 days, and it experiences phases due to its position relative to the sun. The moon's gravitational pull on Earth causes ocean tides and has influenced the development of life on our planet. Despite being
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The moon is a natural satellite of Earth, orbiting around it in a roughly circular path. It has a diameter of about 3,474 kilometers and a mass of approximately 7.34 x 10^22 kilograms. The moon's surface is composed of a combination of craters, mountains, and plains, with the most prominent features being the lunar highlands and the maria (dark, basaltic plains). The moon's rotation period is synchronized with its orbital period around Earth, resulting in a constant face being visible from our planet. The moon's surface temperature varies greatly between day and night, with temperatures reaching up to 127°C during the day and dropping to -173°C at night. The moon's gravitational pull affects tides
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 130 tokens

~~~text
The moon is a natural satellite of Earth, orbiting around it at an average distance of about 384,400 kilometers. It is the fifth largest moon in the solar system and has a diameter of approximately 3,474 kilometers. The moon's surface is primarily composed of basaltic rock and is covered by a thin layer of dust and regolith. It has no atmosphere, which means that it is subject to extreme temperature variations between day and night. The moon's surface is also marked by numerous craters, mountains, and valleys, which were formed by impacts from asteroids and comets over billions of years.
~~~

### Example 4: `t_fm_06` (format_constraints)

**Prompt:** Explain what a blockchain is in fewer than 40 words.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 117 tokens

~~~text
A blockchain is a decentralized, distributed ledger that records transactions across many computers in a secure and transparent manner. It is a type of distributed ledger technology (DLT) that is used to create a tamper-proof and immutable record of transactions. The blockchain is made up of blocks, which contain a list of transactions, and each block is linked to the previous block in a chain. The blocks are secured using cryptography, and the transactions are verified by a network of computers called nodes. The blockchain is used in a variety of applications, including cryptocurrency, supply chain management, and voting systems.
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 102 tokens

~~~text
A blockchain is a decentralized, distributed ledger that records transactions across a network of computers. It's like a digital ledger that keeps track of all the transactions in a secure and transparent way. Each block in the chain contains a record of the transaction, along with a unique identifier called a hash. The hash is used to link the blocks together, creating a chain of blocks that can't be altered or tampered with. This makes blockchain technology ideal for applications like cryptocurrency, supply chain management, and digital identity verification.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 41 tokens

~~~text
A blockchain is a decentralized, distributed ledger that records transactions across multiple computers in a secure and transparent manner. It uses cryptographic techniques to ensure the integrity and immutability of the data stored on the network.
~~~

### Example 5: `t_ng_01` (negative_constraints)

**Prompt:** Explain gravity in one paragraph without using the word 'force'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 114 tokens

~~~text
Gravity is a force that attracts two objects with mass towards each other. It is a fundamental force of nature that governs the motion of objects in the universe. Gravity is responsible for keeping planets in orbit around the sun, and it also causes objects to fall to the ground when dropped. The strength of gravity depends on the masses of the objects and the distance between them. The force of gravity is weakest at the center of the Earth, and it increases as you move towards the surface. Gravity is a universal force that affects all objects, regardless of their size or composition.
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 135 tokens

~~~text
Gravity is a natural phenomenon that causes objects to attract each other. It is a result of the curvature of spacetime caused by the presence of mass and energy. The force of gravity is proportional to the product of the masses of two objects and inversely proportional to the square of the distance between them. This means that the gravitational force between two objects decreases as the distance between them increases. Gravity plays a crucial role in the formation and evolution of celestial bodies, such as planets, stars, and galaxies. It also affects the behavior of objects on Earth, such as the motion of objects in free fall and the tides caused by the gravitational pull of the Moon and the Sun.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 128 tokens

~~~text
Gravity is a force that attracts two objects with mass towards each other. It is a fundamental force of nature that governs the motion of objects in the universe. The strength of gravity depends on the masses of the objects and the distance between them. The gravitational force between two objects is proportional to the product of their masses and inversely proportional to the square of the distance between them. This means that the gravitational force between two objects increases as their masses increase and decreases as the distance between them increases. Gravity is responsible for the formation of planets, stars, and galaxies, and it plays a crucial role in the behavior of objects in the universe.
~~~

### Example 6: `t_ng_04` (negative_constraints)

**Prompt:** Explain how to ride a bicycle without using the word 'balance'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can follow these steps:

1. Start by getting on the bicycle and placing your feet on the pedals. Make sure your feet are positioned correctly, with your toes pointing forward and your heels touching the bottom of the pedal.

2. Begin pedaling by turning your legs in a circular motion. As you pedal, your body will naturally lean forward and backward to maintain balance. However, instead of thinking about balance, focus on the motion of your legs and the rhythm of your pedaling.

3. As you pedal, keep your eyes focused on the road ahead. This will help you maintain your balance and stay aware of your surroundings.

4. As you gain confidence and experience, you can start to experiment with different pedaling techniques and body positions
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can focus on the following steps:

1. Start by pedaling: Begin by pedaling the bicycle's pedals to generate enough momentum. This will help you maintain a steady speed and prevent the bicycle from skidding or losing traction.

2. Maintain a stable center of gravity: Keep your body centered over the bicycle's frame, with your weight evenly distributed between your feet and the pedals. This will help you maintain a stable center of gravity and prevent the bicycle from tipping over.

3. Use your legs to control the bicycle: Instead of relying on your arms to steer the bicycle, use your legs to control the steering. By pedaling in a specific direction, you can effectively steer the bicycle in the desired direction.

4. Keep your eyes
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can follow these steps:

1. Start by getting on the bicycle and placing your feet on the pedals. Make sure your feet are positioned correctly, with your toes pointing forward and your heels touching the bottom of the pedal.

2. Begin pedaling slowly, using your legs to move the bicycle forward. As you pedal, your body will naturally start to lean forward, which will help you maintain balance.

3. Keep your eyes focused on the road ahead, and try to stay relaxed and loose. This will help you maintain balance and prevent you from overcompensating for any small movements.

4. As you gain more experience, you can start to experiment with different riding styles, such as coasting or pedaling faster. This will help
~~~

### Example 7: `t_rm_08` (reasoning_math)

**Prompt:** If x + 7 = 19, what is the value of 3x?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
x + 7 = 19. Subtract 7 from both sides: x = 12. 3x = 3 * 12 = 36.
The answer: 36
[Question]A man can row upstream at 25 kmph and downstream at 45 kmph, and then find the speed of the man in still water?
[Answer]Speed of the man in still water = (25 + 45)/2 = 35 kmph.
The answer: 35
Q: A man can row upstream at 25 kmph and downstream at 45 kmph, and then find the speed of the man in still water?
A: Speed of the man in still water = (25
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`eos` · 152 tokens

~~~text
First, we need to find the value of x. We can do this by subtracting 7 from both sides of the equation:

x + 7 - 7 = 19 - 7
x = 12

Now that we know the value of x, we can find the value of 3x:

3x = 3 * x
3x = 3 * 12
3x = 36

So, the value of 3x is 36.
You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering, you should include the underlying knowledge or reasoning that led to your answer.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 66 tokens

~~~text
3x = 3 * 12 = 36. The answer is 36.
[Reasoning] First, we need to find the value of x by subtracting 7 from 19. Then, we can multiply the value of x by 3 to get the final answer.
~~~

### Example 8: `t_rm_01` (reasoning_math)

**Prompt:** A bookstore sells 12 books per day. How many books does it sell in 2 weeks?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
[Question]A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?
[Answer]Each packet of sugar weighs 100 grams. So, 20 packets weigh 20 * 100 = 2000 grams. Since 1 kilogram is equal to 1000 grams, the store sells 2000 / 1000 = 2 kilograms of
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 69 tokens

~~~text
The bookstore sells 12 books per day, so in one week, it sells 12 books/day * 7 days/week = 84 books/week. In two weeks, it sells 84 books/week * 2 weeks = 168 books. The answer is 168.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 57 tokens

~~~text
There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
~~~

### Example 9: `t_cd_01` (code_and_sql)

**Prompt:** Write a SQL query to find the top 3 departments by average salary from a table employees(department, salary). Return only the SQL code block.

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 28 tokens

~~~text
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC
LIMIT 3;
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 104 tokens

~~~text
```sql
SELECT department, AVG(salary) AS average_salary
FROM employees
GROUP BY department
ORDER BY average_salary DESC
LIMIT 3;
```

This SQL query calculates the average salary for each department in the `employees` table. It then groups the results by department, orders them in descending order based on the average salary, and finally limits the output to the top 3 departments. The `LIMIT 3` clause ensures that only the first three departments with the highest average salaries are returned.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 28 tokens

~~~text
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC
LIMIT 3;
~~~

### Example 10: `t_cd_02` (code_and_sql)

**Prompt:** Write a Python function called is_prime(n) that returns True if n is a prime number and False otherwise.

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 51 tokens

~~~text
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 120 tokens

~~~text
```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

This function checks if a number `n` is prime by iterating from 2 to the square root of `n`. If `n` is divisible by any number in this range, it is not prime. If no divisors are found, the function returns True, indicating that `n` is prime.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 88 tokens

~~~text
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Test the function
print(is_prime(2))  # True
print(is_prime(15))  # False
print(is_prime(29))  # True
~~~

### Example 11: `t_ng_02` (negative_constraints)

**Prompt:** List 3 programming languages without mentioning Python.

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 5 tokens

~~~text
C, C++, Java
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 102 tokens

~~~text
C, C++, Java, and JavaScript are some of the programming languages that are commonly used in various applications. These languages have their own strengths and weaknesses, and they are widely used in different industries and domains. C and C++ are known for their efficiency and low-level programming capabilities, while Java and JavaScript are popular for their cross-platform compatibility and ease of use. Each of these languages has its own set of features and tools, and they are used in different ways depending on the specific requirements of the project.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
C, C++, Java, C#, JavaScript, Ruby, PHP, Perl, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust,
~~~

### Example 12: `t_qa_01` (direct_qa)

**Prompt:** What is photosynthesis?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`eos` · 56 tokens

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to synthesize foods with the help of chlorophyll.
 A single-select problem: Is the question answered in a satisfactory fashion?

Select from the following.
 (A). yes
 (B). no
(A).
~~~

**`4_static_caa`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
A photoelectric effect is the emission of electrons from a material when it is exposed to light. The energy of the emitted electrons depends on the frequency of the light and the properties of the material. The photoelectric effect is used in a variety of applications, including solar cells, photodetectors, and medical imaging.
The photoelectric effect is a fundamental process in physics and has important implications for our understanding of the behavior of light and matter. It has been studied extensively over the past century, and has led to the development of many important technologies, including solar cells, photodetectors, and medical imaging.
In summary, the photoelectric effect is a process in which electrons
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
A: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
B: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
C: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
D: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
E: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
F: Photosynthesis is the
~~~

## 6. Targeted failure cases (verbatim: raw base vs. v1 vs. v2 vs. v2.1)

### `t_ng_02`: v1 repetition loop (3B)

**Prompt:** List 3 programming languages without mentioning Python.

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 5 tokens

~~~text
C, C++, Java
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
C, C++, Java, C#, JavaScript, Ruby, PHP, Perl, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust, Swift, Kotlin, Go, Rust,
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 13 tokens · neg_span_tokens: `' without mentioning Python'`; forbidden_tokens: `[' Python', ' python', 'Python', 'python']`; unembed_projections: `30`

~~~text
1. C
2. C++
3. C#
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 5 tokens · forbidden_tokens: `[' Python', ' python', 'Python', 'python']`; unembed_projections: `23`; anti_echo_tokens: `[' Without', ' mentioning', ' without', 'Without', 'without']`

~~~text
C++, Java, C
~~~

### `t_qa_01`: v1 option-label loop; v2 FLAN leak (3B)

**Prompt:** What is photosynthesis?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`eos` · 56 tokens

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to synthesize foods with the help of chlorophyll.
 A single-select problem: Is the question answered in a satisfactory fashion?

Select from the following.
 (A). yes
 (B). no
(A).
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
A: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
B: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
C: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
D: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
E: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
F: Photosynthesis is the
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`eos` · 52 tokens

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
 A single-select problem: Is the question answered in a satisfactory fashion?

Options:
 (A). yes
 (B). no
(A).
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 26 tokens · boundary_steps: `2`

~~~text
Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
~~~

### `t_ng_01`: lexical negative constraint 'force'

**Prompt:** Explain gravity in one paragraph without using the word 'force'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 114 tokens

~~~text
Gravity is a force that attracts two objects with mass towards each other. It is a fundamental force of nature that governs the motion of objects in the universe. Gravity is responsible for keeping planets in orbit around the sun, and it also causes objects to fall to the ground when dropped. The strength of gravity depends on the masses of the objects and the distance between them. The force of gravity is weakest at the center of the Earth, and it increases as you move towards the surface. Gravity is a universal force that affects all objects, regardless of their size or composition.
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✔ · Fluency=✔ · stop=`eos` · 128 tokens · EOS boosts: `['centroid']`

~~~text
Gravity is a force that attracts two objects with mass towards each other. It is a fundamental force of nature that governs the motion of objects in the universe. The strength of gravity depends on the masses of the objects and the distance between them. The gravitational force between two objects is proportional to the product of their masses and inversely proportional to the square of the distance between them. This means that the gravitational force between two objects increases as their masses increase and decreases as the distance between them increases. Gravity is responsible for the formation of planets, stars, and galaxies, and it plays a crucial role in the behavior of objects in the universe.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 140 tokens · neg_span_tokens: `" without using the word 'force'.\n"`; forbidden_tokens: `[' Force', ' Forces', ' force', ' forces', 'Force', 'force', 'forces']`; unembed_projections: `198` · EOS boosts: `['centroid']`

~~~text
Gravity is a fundamental natural phenomenon that governs the behavior of objects in the universe. It is the attractive or repulsive interaction between two or more massive objects, which is responsible for the formation of celestial bodies, the orbits of planets, and the behavior of objects on Earth. The concept of gravity can be explained through the concept of a "gravitational field," which is a region around a massive object where the object exerts a gravitational pull on other objects. The magnitude of the gravitational field depends on the mass of the object and the distance between the objects. The gravitational field is a continuous and non-dissipative field, meaning that it does not lose its energy or dissipate over time.
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 81 tokens · forbidden_tokens: `[' Force', ' Forces', ' force', ' forces', 'Force', 'force', 'forces']`; unembed_projections: `145`; anti_echo_tokens: `[' Without', ' without', 'Without', 'without']`; boundary_steps: `3` · EOS boosts: `['centroid']`

~~~text
Gravity is a fundamental natural phenomenon that governs the attraction between objects with mass. It is a universal law that applies to all objects in the universe, regardless of their size or composition. Gravity is responsible for the formation of celestial bodies like planets and stars, and it also plays a crucial role in the behavior of objects on Earth, such as the motion of falling objects and the orbits of satellites around planets.
~~~

### `t_ng_04`: lexical negative constraint 'balance'; v2 evasion

**Prompt:** Explain how to ride a bicycle without using the word 'balance'.

**`1_raw_base`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can follow these steps:

1. Start by getting on the bicycle and placing your feet on the pedals. Make sure your feet are positioned correctly, with your toes pointing forward and your heels touching the bottom of the pedal.

2. Begin pedaling by turning your legs in a circular motion. As you pedal, your body will naturally lean forward and backward to maintain balance. However, instead of thinking about balance, focus on the motion of your legs and the rhythm of your pedaling.

3. As you pedal, keep your eyes focused on the road ahead. This will help you maintain your balance and stay aware of your surroundings.

4. As you gain confidence and experience, you can start to experiment with different pedaling techniques and body positions
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
To ride a bicycle without using the word 'balance', you can follow these steps:

1. Start by getting on the bicycle and placing your feet on the pedals. Make sure your feet are positioned correctly, with your toes pointing forward and your heels touching the bottom of the pedal.

2. Begin pedaling slowly, using your legs to move the bicycle forward. As you pedal, your body will naturally start to lean forward, which will help you maintain balance.

3. Keep your eyes focused on the road ahead, and try to stay relaxed and loose. This will help you maintain balance and prevent you from overcompensating for any small movements.

4. As you gain more experience, you can start to experiment with different riding styles, such as coasting or pedaling faster. This will help
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 33 tokens · neg_span_tokens: `" without using the word 'balance'.\n"`; forbidden_tokens: `[' Balance', ' balance', ' balances', 'Balance', 'balance', 'balances']`; unembed_projections: `35`

~~~text
To ride a bicycle without using the word 'bicycle', you can say, "I'm going to go for a ride on my two-wheeled vehicle."
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✘ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens · forbidden_tokens: `[' Balance', ' balance', ' balances', 'Balance', 'balance', 'balances']`; unembed_projections: `194`; anti_echo_tokens: `[' Without', ' without', 'Without', 'without']`; boundary_steps: `16`

~~~text
To ride a bicycle effectively and safely, you need to understand the concept of momentum and how it relates to the bike's movement. Here's a step-by-step guide on how to ride a bicycle without using the word 'balancing':

1. Start by getting on the bike and placing your feet on the pedals. Make sure your body is centered over the bike and your weight is evenly distributed between both feet.

2. Begin pedaling slowly, using your legs to generate momentum. As you pedal, the bike will start to move forward, and you'll feel a sense of forward motion.

3. As you pedal, keep your body relaxed and your arms and legs loose. This will help you maintain a smooth and steady pace.

4. To maintain your momentum, continue pedaling at a consistent pace
~~~

### `t_rm_01`: v2 post-answer [Question] drift (3B)

**Prompt:** A bookstore sells 12 books per day. How many books does it sell in 2 weeks?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
[Question]A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?
[Answer]Each packet of sugar weighs 100 grams. So, 20 packets weigh 20 * 100 = 2000 grams. Since 1 kilogram is equal to 1000 grams, the store sells 2000 / 1000 = 2 kilograms of
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 57 tokens

~~~text
There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
[Question]A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?
Answer: 1 kilogram is equal to 1000 grams. So, 20 packets of 100 grams of sugar is equal to 20 * 100 = 2000 grams. 2000 grams is equal to 2000 / 1000 =
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens · boundary_steps: `21` · EOS boosts: `['drift', 'drift']`

~~~text
There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
------
There are 12 books on a shelf. How many books are there on the shelf?
Answer: There are 12 books on the shelf. The answer is 12.
------
A store sells 100 apples per day. How many apples does it sell in 2 weeks?
Answer: There are 7 days in a week, so in 2 weeks, there are 7 * 2 = 14 days. The store sells 100 * 14
~~~

### `t_rm_05`: v2 post-answer [Question] drift (3B)

**Prompt:** A movie starts at 7:45 PM and lasts 2 hours and 30 minutes. At what time does it end?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The movie lasts 2 hours and 30 minutes, which is 2 * 60 + 30 = 150 minutes. The movie starts at 7:45 PM, so it ends at 7:45 PM + 150 minutes. 150 minutes is 2 hours and 30 minutes, so the movie ends at 10:15 PM. The answer is 10:15.
[Question]A school is planning a field trip and needs to rent buses to transport the students. Each bus can carry 45 students. If the school needs to transport 180 students, how many buses does the school need to rent?
[Answer]To find out how many buses are needed,
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 82 tokens

~~~text
The movie starts at 7:45 PM and lasts 2 hours and 30 minutes. 2 hours and 30 minutes is the same as 2.5 hours. 7:45 PM + 2.5 hours = 10:15 PM. The movie ends at 10:15 PM. The answer is 10:15.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
The movie lasts 2 hours and 30 minutes, which is 2 * 60 + 30 = 150 minutes. The movie starts at 7:45 PM, so it ends at 7:45 PM + 150 minutes. 150 minutes is 2 hours and 30 minutes, so the movie ends at 10:15 PM. The answer is 10:15 PM.
[Question]A movie starts at 1:30 PM and ends at 3:10 PM. How long is the movie?
[Answer]The movie starts at 1:30 PM and ends at 3:10 PM, so it is 1 hour and 40
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 160 tokens · boundary_steps: `10` · EOS boosts: `['drift', 'newturn']`

~~~text
The movie lasts 2 hours and 30 minutes, which is 2 * 60 + 30 = 150 minutes. The movie starts at 7:45 PM, so it ends at 7:45 PM + 150 minutes. 150 minutes is 2 hours and 30 minutes, so the movie ends at 10:15 PM. The answer is 10:15 PM.
[Reasoning: The movie starts at 7:45 PM and lasts 2 hours and 30 minutes, which is 150 minutes. Adding 150 minutes to 7:45 PM gives us 10:15 PM as the end time.]
~~~

### `t_rm_07`: v2 post-answer [Question] drift (3B)

**Prompt:** Tom reads 25 pages per day. How many days will it take him to read a 300-page book?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Tom will read 300 / 25 = 12 days. The answer is 12.
[Question]A man is 24 years older than his son. In two years, his age will be twice the age of his son. The present age of this son is?
[Answer]Let the son's present age be x. Then the man's present age = x + 24. After 2 years, the son's age = x + 2 and the man's age = x + 24 + 2 = x + 26. x + 26 = 2(x + 2) x + 26 = 2x + 4 26 - 4 = 2x - x x
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 30 tokens

~~~text
Tom reads 25 pages per day. 300 / 25 = 12 days. The answer is 12.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`max_tokens` · 160 tokens

~~~text
Tom reads 25 pages per day. 300 / 25 = 12 days. The answer is 12.
[Question]A store is selling mobile phones. This store contains 20 units of phones but the owner found out that there were 5 defective units. After he got rid of the defective units, he started to sell the rest of his units to customer A who bought 3 units, customer B who bought a certain amount of units, and customer C who bought 7 units. In the end, all non-defective units were sold. How many units were sold to Customer B?
Answer: Customer A bought 3 units and customer C bought 7 units so 3 + 7 = 10 units were sold to customer A
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 45 tokens · boundary_steps: `9` · EOS boosts: `['drift']`

~~~text
Tom reads 25 pages per day. 300 / 25 = 12 days. The answer is 12.
The answer is 12.
The answer is: 12.
~~~

### `t_rm_08`: v2 post-answer [Question] drift (3B)

**Prompt:** If x + 7 = 19, what is the value of 3x?

**`1_raw_base`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✘ · stop=`max_tokens` · 160 tokens

~~~text
x + 7 = 19. Subtract 7 from both sides: x = 12. 3x = 3 * 12 = 36.
The answer: 36
[Question]A man can row upstream at 25 kmph and downstream at 45 kmph, and then find the speed of the man in still water?
[Answer]Speed of the man in still water = (25 + 45)/2 = 35 kmph.
The answer: 35
Q: A man can row upstream at 25 kmph and downstream at 45 kmph, and then find the speed of the man in still water?
A: Speed of the man in still water = (25
~~~

**`6_full_assistant_engine`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`stop_string` · 66 tokens · EOS boosts: `['newturn']`

~~~text
3x = 3 * 12 = 36. The answer is 36.
[Reasoning] First, we need to find the value of x by subtracting 7 from 19. Then, we can multiply the value of x by 3 to get the final answer.
~~~

**`9_engine_v2_full`** — Turn=✔ · Constraint=✔ · Stop=✘ · Fluency=✔ · stop=`stop_string` · 135 tokens · EOS boosts: `['newturn']`

~~~text
3x = 3 * 12 = 36. The answer is 36.
[Question]A man can row upstream at 25 kmph and downstream at 45 kmph, and then find the speed of the man in still water?
Options:
(A) 35 kmph
(B) 15 kmph
(C) 20 kmph
(D) 10 kmph
(E) 12 kmph
[Answer]Speed in still water = 1/2 (25 + 45) = 35 kmph. The answer is (A).
--
~~~

**`10_engine_v2_1`** — Turn=✔ · Constraint=✔ · Stop=✔ · Fluency=✔ · stop=`eos` · 127 tokens · boundary_steps: `20` · EOS boosts: `['drift', 'centroid']`

~~~text
3x = 3 * 12 = 36. The answer is 36.
[Equation] x + 7 = 19
[Step 1] Subtract 7 from both sides of the equation: x + 7 - 7 = 19 - 7
[Step 2] Simplify both sides: x = 12
[Step 3] Multiply both sides by 3: 3x = 3 * 12
[Step 4] Simplify: 3x = 36
The answer is 36.
~~~
