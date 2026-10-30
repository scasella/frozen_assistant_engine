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

### Phase 4 interpretation (v2.2 lean engine and the post-trained reference; hand-written after inspecting the logs)

**Condition 11 is an execution-cost result, not a new accuracy result.**
- **Exactness.** `results/lean_equivalence_checks.txt` records two checks:
  - the sync-free sibling projection reproduces x9 byte for byte on 40/40 prompts on both models;
  - the text-only boundary trigger matches x9 on 39/40 on both models. The exception on both is `t_rm_05`, where the
    v2.1 score clause had opened a boundary window.
- **Where 11 differs from x9.** Every response where condition 11 differs from x9 is on a prompt where x9's
  drift-assisted stop fired: 3/40 on 3B (`t_qa_05`, `t_rm_05`, `t_rm_06`) and 1/40 on 1.5B (`t_cd_01`). Only 3B
  `t_rm_05` changes a metric: x9 ended it with the `[Reasoning: …]` tag the verifier does not flag, while 11 runs to
  the token cap. Hence 95.6 against x9's 96.2. On 1.5B both score 86.2.
- **Speed.** In the same session (`speed_bench.json`, conditions interleaved prompt by prompt, every output
  byte-identical to its log), condition 11 decodes:
  - **24.30 tok/s on 3B**, against 24.81 for condition 7 (−2.1%) and 23.31 for condition 10 (−6.0%);
  - **39.02 tok/s on 1.5B**, against 39.82 (−2.0%) and 37.29 (−6.4%).

  On the 32 prompts without a negation it is indistinguishable from condition 7 (24.76 vs. 24.81; 39.81 vs. 39.81).
  On the 8 negation prompts the remaining cost is the second, word-only vocabulary product: 22.86 against 24.80 on
  3B, up from 20.47 under condition 10.
- **Where the "12%" came from.** The brief's 12% overhead compared logs generated hours apart. Measured in one session
  the v2.1 overhead was 6%, so about half of the logged gap was session drift.
- **Not a held-out estimate.** The mechanism set came from test leave-one-outs. On the 1.5B dev set condition 11
  scored 95.0, against condition 10's reproduced 93.3 (one prompt's CleanStop). Dev does not contradict the
  simplification, but nothing was selected from it.

**Against the official post-trained checkpoints: a token-budget effect at 160 tokens, parity at 512.**
- **At the protocol's 160 tokens (3B), the engine wins.**
  - Condition 10 beats Qwen2.5-3B-Instruct by +8.1 [+1.9, +14.4] with Instruct's chat template, and by +12.5
    [+6.2, +18.8] in the bare frame.
  - But Instruct's own gain over the raw base is not resolved (+5.6 [−1.2, +12.5]), so the brief's recovery ratio
    has no meaningful value at this budget.
- **Why Instruct scores low at 160 tokens.** All 14 of its non-negation failures are answers cut by the cap.
  Qwen2.5-Instruct writes thorough markdown (214 tokens per answer when allowed 512, against the engine's 82). With
  its chat template it shows zero contamination. In the bare frame even Instruct drifts like a base model on 2/40
  prompts (a FLAN `single-select problem` block, an `A:` turn), so its post-training is tied to the template.
- **With a 512-token budget (tag `test40_512tok`), the comparison is fair, and it is parity.**
  - Instruct-chat ends on its own on 37/40 prompts and scores 94.4; post-training's gain is now resolved, +15.6
    [+8.8, +22.5].
  - Condition 10 scores 96.2 (CleanStop 100%, no cap hits): engine − Instruct = +1.9 [−1.9, +5.6], and recovery is
    **112% [87%, 150%]**. Condition 11 recovers 108% [76%, 157%]. (Re-checked below with fp32 attention scores for
    all three conditions: still parity by a pre-registered rule, but narrowly. Engine − Instruct is −1.9 [−6.2, +2.5]
    and recovery 91% [70%, 111%].)
  - The raw base does not improve with room (78.8), because it uses the extra tokens to drift.
  - **On 1.5B, Instruct leads at 512 tokens.** Instruct-chat scores 90.0 against the engine's 87.5, so engine −
    Instruct = −2.5 [−10.0, +4.4] and recovery is 85% [40%, 135%]. That comparison runs on the eager-bf16 stack,
    which depresses 1.5B arithmetic (see Numerics below). It is not parity. (Corrected below: the stack depresses
    Instruct at least as much as the engine, so this comparison is not pessimistic for the engine. With fp32 scores
    for all three conditions the gap is −6.9 [−12.5, −1.9].)
  - On these 40 prompts and four programmatic checks, the engine buys essentially all of what post-training buys,
    with 0 changed weights, a 3.0 MB artifact, 0 extra prompt tokens and 2.6× shorter answers.
  - This is not a claim about general assistant quality. The checks do not measure helpfulness, depth, safety or
    multi-turn behaviour, where post-training does work the engine never attempts.
- **The 512-token budget** was planned as a sensitivity check of the cap once Instruct's truncations were seen,
  before any 512-token output existed. No coefficient or condition was chosen from it.

**Condition 13: the lexical operator adds value on top of post-training.**
- **3B, chat template.**
  - The ban raises negative_constraints ConstraintScore from 37.5% to 87.5% at 160 tokens (Composite +2.5
    [+0.6, +5.0]), and Instruct + ban is the best condition at 512 tokens: 97.5, +3.1 [+0.6, +6.2] over Instruct.
  - It fixes `t_ng_01`, `t_ng_02`, `t_ng_06` and `t_ng_07`, and breaks nothing.
  - Instruct's characteristic failure is echoing the banned word while restating the constraint ("Here are three
    programming languages without mentioning Python:").
  - Two "fixes" are evasions rather than better answers: "without mentioning any language that includes the word
    "Py"" and "without mentioning the specific ones you requested to be avoided".
- **1.5B, chat template.** The ban raises ConstraintScore from 37.5% to 75.0%, but Composite only by +0.6 [−4.4, +4.4]:
  - `t_ng_02` collapses into a loop ("…not related to the question of not being related to the question…");
  - `t_ng_03` gains a factual error the verifier passes: the ocean as a "salt-and-sugar-rich liquid".
- **Untouched prompts.** The 32 prompts without a parsed negation are byte-identical to condition 12, as they must be.
  Nothing was tuned on Instruct: γ, K and the fallback are the frozen base-model dev values.

**Alignment tax: the engine barely moves the base distribution except where it bans a word.**
- **Measured as:** clean-token KL D_KL(P_raw ‖ P_condition) at t > 4 along URIAL's clean held-out test answers.
- **3B.**
  - The engine's mean is 0.124 nats/token, but its median answer is 0.008. About two thirds of the total comes from
    three negation prompts, where the ban suppresses words the teacher answer uses ("trees", "computers").
  - Without the 8 negation prompts the engine is at **0.014 nats/token** with 97% top-1 agreement. That compares with
    URIAL 0.064, static CAA 0.081, Instruct in the bare frame 0.186 and **Instruct with its chat template 0.576** (81%
    top-1 agreement).
- **1.5B.** Same ordering: engine 0.023–0.024 vs. Instruct-chat 0.213 without negations.
- **What the measure is and is not.** It is a distance, not a quality score: post-training is *supposed* to move the
  distribution. The claim is only that the engine reaches the same scores while leaving the base model's next-token
  predictions on good assistant text essentially intact (calibration answers, in-sample for the engine: 0.014 vs.
  0.432).

**Numerics: the eager-bf16 attention that Module 4 needs costs Qwen2.5-1.5B its arithmetic.**
- **The finding.** On 1.5B the raw base solves 3/8 math prompts on the protocol stack and **8/8** with HF's SDPA
  attention (Composite 75.6 → 81.2). Errors look like "15 + 9 = 14" and "$50 − $28 = $42".
- **Cause.**
  - HF's own eager attention gives the identical wrong digits in bf16 on CPU, and `steer_eager` is bit-identical to
    it on MPS.
  - SDPA in bf16 and eager attention in fp32 are both correct.
  - So the eager path's bf16 rounding of the pre-softmax scores is the cause.
- **Scope.**
  - Every 1.5B number in this report, the v2.1.0 release included, is depressed by this. The comparisons stay fair
    among themselves (one shared stack), but the 1.5B *level* is not representative.
  - 3B is not visibly affected: math is 8/8 under both stacks, and the raw base scores 80.6 vs. 78.8.
- **Pre-registered rule.** Fixed before the SDPA runs: if 3B Instruct-chat moved by more than 2 points under SDPA, the
  SDPA Instruct would become the recovery denominator.
  - It moved 0.6 (86.2 → 85.6), so the protocol comparison stays primary.
  - On 1.5B, Instruct-chat did not move (84.4) while the raw base gained 5.6. The 1.5B recovery figures are
    confounded by numerics and should not be quoted. (Its Composite did not move at 160 tokens; its arithmetic did.
    See the fp32 follow-up below.)
- **Throughput.** SDPA is also faster: the raw base runs at 29.5 vs. 26.1 tok/s on 3B, and 48.9 vs. 39.4 on 1.5B. Part
  of the engine's throughput gap to Instruct is the eager attention its Module-4 bias needs, not the hooks. A fused
  attention kernel with an additive bias would remove it; that is untested here.

**Numerics, follow-up (sensitivity run): with fp32 attention scores, the 1.5B gap to Instruct at 512 tokens does not
close.** The bf16 eager stack had clearly depressed 1.5B-Instruct (+6.9 [+3.1, +10.6] with fp32 scores). For the
engine the cost is smaller and not resolved (+2.5 [−3.1, +8.1]). This is a labelled sensitivity run; the protocol
results above stay primary.
- **What was run.**
  - `steer_fp32` is `steer_eager` with query and key upcast before Q·Kᵀ, and the scaling, causal mask, Module-4 bias
    and softmax in fp32. The attention weights are cast back to bf16 for the value matmul.
  - Unlike SDPA it keeps Module 4, so it can run the engine. Everything else is bf16, as in the protocol.
  - The coefficients are the frozen ones, tuned on the 1.5B dev set on the bf16 `steer_eager` stack, so on
    `steer_fp32` the engine runs away from its tuning point. Nothing was re-tuned.
  - Tags: `test40_fp32attn` (160 tokens) and `test40_512tok_fp32attn` (512 tokens). The protocol logs are untouched.
- **Checks before any test output** (`scripts/check_fp32_attention.py`, `scripts/check_cache_reproduces.py`):
  - In an fp32 model, `steer_fp32` is bit-identical to HF eager (max |logit diff| 0). With a Module-4 bias it is
    bit-identical to `steer_eager`; the bias itself moves the logits by up to 3.7.
  - In bf16, against fp32 eager over 869 positions, mean KL is 2.7e-2 for `steer_eager`, 7.0e-4 for `steer_fp32`
    and 6.7e-4 for SDPA. Top-1 agreement is 821, 856 and 859.
  - The 1.5B raw base solves 8/8 math prompts on it (protocol 3/8, SDPA 8/8).
  - Conditions 1, 7, 10 and 11 still reproduce their protocol logs byte for byte on both models (on the script's
    6-prompt check subset).
  - The new fp32 logs reproduce byte for byte on the same subset: 1.5B conditions 1, 10 and 11 under
    `PYTHONHASHSEED` 0 and 1, and 3B conditions 10 and 11 under seed 1.
- **Fixed before any fp32 output existed.**
  - The verdict compares condition 10 with Instruct-chat at 512 tokens, with raw base, engine and Instruct all on
    `steer_fp32`.
  - **Closes** if engine − Instruct ≥ 0, **narrows** if it lies in (−2.5, 0), **stays** if ≤ −2.5. A move inside
    the ±7-point CI counts as unresolved.
  - The brief named the existing Instruct logs (`test40_sdpa`, `test40_512tok`) as the reference. Before any run, those
    logs showed the premise behind that choice ("Instruct moved by 0.0 under SDPA") held for Composite only:
    - 1.5B-Instruct's text changed on 34/40 prompts between eager and SDPA at 160 tokens;
    - its math went from 2/8 to 4/8, with the other four answers truncated;
    - in the eager 512-token log all six math failures end on `eos` with wrong arithmetic ("$50 − $28 = $42",
      "25 × 4 = 10", "300 / 25 = 30").
  - So a 1.5B-Instruct run on `steer_fp32` at 512 tokens (conditions 12 chat and bare, 13 chat) was added as the
    primary reference. The eager `test40_512tok` Instruct is kept as a secondary, continuity row.
  - At 160 tokens the reference is Instruct on SDPA (`test40_sdpa`).
- **1.5B, 160 tokens.**
  - The raw base gains +5.6 [−0.6, +11.2] and lands on SDPA's 81.2; its math goes from 3/8 to 8/8.
  - The engine gains less: v2.1 +1.9 [−5.0, +8.8] (88.1) and lean +2.5 [−3.1, +8.1] (88.8); math goes from 4/8 to 7/8.
  - The engine's margin over the raw base shrinks from +10.6 [+2.5, +18.8] to +6.9 [+0.0, +13.8].
  - Against Instruct on SDPA (84.4) the engine is +3.8 [−3.1, +10.0]. Recovery is not resolved: the denominator,
    Instruct − raw base, is only +3.1 [−3.8, +10.6].
- **1.5B, 512 tokens: the verdict is "stays".** On one stack the deficit is now resolved; its widening is a point
  estimate.
  - On one stack the raw base scores 78.8, engine v2.1 90.0 and Instruct-chat 96.9.
  - Engine − Instruct = **−6.9 [−12.5, −1.9]**, against −2.5 [−10.0, +4.4] on the protocol stack.
  - Recovery is **62% [28%, 90%]**, against 85% [40%, 135%].
  - The lean engine: −5.6 [−10.6, −0.6], recovery 69% [38%, 96%].
  - For the first time the CI excludes parity on 1.5B.
  - The change in the gap itself, paired per prompt, is −4.4 [−11.9, +2.5] (lean: −2.5 [−9.4, +4.4]). The move away
    from −2.5 is therefore not resolved. What the fp32 run adds is a deficit that excludes zero, not a resolved
    widening.
- **Why: Instruct-chat was hurt more than the engine (point estimates), and its change is the only resolved one.**
  - Instruct-chat gains +6.9 [+3.1, +10.6] (90.0 → 96.9); its math goes from 2/8 to 8/8. Ten prompts improve and none
    gets worse.
  - Engine v2.1 gains +2.5 [−3.1, +8.1].
  - Instruct-bare gains +11.2 (63.8 → 75.0), and Instruct + ban reaches 98.8.
  - Against the eager 512-token Instruct, the fp32 engine would read 0.0 [−6.2, +5.6] with 100% recovery, i.e.
    "closes". That is the artefact of comparing a precise-attention engine with a bf16-eager Instruct.
- **Where the engine still falls short on 1.5B** (512 tokens, fp32 scores):
  - Instruct leads on format_constraints (96.9 vs. 84.4), reasoning_math (100 vs. 84.4) and code_and_sql (100 vs.
    93.8). The engine leads on negative_constraints (90.6 vs. 87.5).
  - Its one remaining math miss is `t_rm_05`. The engine takes a minutes detour ("7:45 PM + 150 minutes = 8:35 PM"),
    while the fp32 raw base adds directly ("7:45 + 2:30 = 10:15").
  - The engine coefficients were tuned on the bf16 stack. Re-tuning them on fp32 scores would be a new dev-set phase.
    Whether it would recover some of the gap is untested.
- **3B check** (160 tokens, engine conditions only):
  - v2.1 −0.6 [−2.5, +1.2] (94.4 → 93.8), lean +0.0 [−1.9, +1.9] (95.6); 19–20 of 40 responses are byte-identical.
  - The only math flip is `t_rm_05` again ("10:30").
  - The 3B engine is unaffected, as before. Not re-checked: 3B Instruct and raw base on fp32 scores at 512 tokens, so
    the 3B parity claim has not been repeated on a precise stack. 3B Instruct is already 8/8 on math at 512 tokens on
    eager. (Re-checked below.)
- **Corrections to the notes above.**
  - "On 1.5B, Instruct-chat did not move (84.4)" is true of its Composite at 160 tokens only.
  - "Pessimistic for the engine" (1.5B at 512 tokens) points the wrong way. The eager stack depresses every 1.5B
    condition, but not equally. At 512 tokens it costs Instruct 6.9 [3.1, 10.6] points and the engine 2.5
    [−3.1, 8.1]. Their difference is not resolved (−4.4 [−11.9, +2.5]), but its point estimate says the
    protocol-stack 1.5B comparison flattered the engine, not that it was pessimistic.
  - The 3B rule (Instruct moved 0.6 at 160 tokens) stands as it was defined. A stable 160-token Composite turned out
    to be a weak test on 1.5B.
- **Throughput.** fp32 scores cost speed. The 1.5B raw base runs at 33.4 tok/s against 39.4 on the protocol stack, and
  engine v2.1 at 29.7 against 37.1 (160 tokens).

**Numerics, 3B at 512 tokens (sensitivity run): does 3B parity with Instruct survive fp32 attention scores?** This is
a labelled sensitivity run; the protocol results above stay primary.
- **Pre-registered** (written 2026-10-29T23:38Z on c1cc4fb, before any 3B or 3B-Instruct `steer_fp32` log at 512 tokens
  existed).
  - Runs, on `steer_fp32` at 512 tokens with the frozen coefficients (tag `test40_512tok_fp32attn`): 3B conditions 1, 2,
    10 and 11; 3B-Instruct conditions 12 chat, 12 bare and 13 chat. Nothing is re-tuned.
  - Verdict: condition 10 against `12_official_instruct_chat`, raw base = `1_raw_base`, all three on that stack.
    Protocol-stack baseline (eager, 512 tokens): engine − Instruct = +1.9 [−1.9, +5.6], recovery 112% [87%, 150%].
  - **Holds** if the paired CI of engine − Instruct includes 0 and the point estimate is ≥ −2.5. A CI entirely above 0
    also counts as holding and is reported as "engine ahead".
  - **Weakens** if the point estimate is < −2.5 while the CI still includes 0.
  - **Fails** if the CI's upper bound is < 0.
  - Reported, not used for the verdict: each condition's paired change from fp32 scores; the paired CI of the change
    in the engine − Instruct gap (per-prompt gap records, the same method that gave −4.4 [−11.9, +2.5] on 1.5B);
    recovery with `recovery_ci`; condition 11; the eager 512-token Instruct as a continuity row.
  - Expected, stated before the runs: little room for the 1.5B mechanism. On eager at 512 tokens 3B Instruct-chat
    already solves 8/8 math. Its nine imperfect prompts are three `direct_qa` truncations and six
    `negative_constraints` violations (it uses the banned word). If all nine flipped, Instruct would gain 5.6 points.
    The constraint misses are a behaviour that fp32 scores have no obvious reason to fix.
  - Confound, as on 1.5B: the coefficients were tuned on the 1.5B dev set on the bf16 stack.
- **Checks before any test output** (`scripts/check_fp32_attention.py --model Qwen/Qwen2.5-3B`):
  - In an fp32 model, `steer_fp32` is bit-identical to HF eager (max |logit diff| 0). With a Module-4 bias it is
    bit-identical to `steer_eager`; the bias itself moves the logits by up to 2.6.
  - In bf16, against fp32 eager over 869 positions, mean KL is 1.2e-3 for `steer_eager`, 5.9e-4 for `steer_fp32` and
    6.2e-4 for SDPA (1.5B: 2.7e-2, 7.0e-4, 6.7e-4). Top-1 agreement is 853, 856 and 856.
  - The 3B raw base solves 7/8 math prompts on it at 160 tokens (protocol and SDPA 8/8). The miss is `t_rm_05`.
  - After the runs, the new logs reproduced byte for byte at 512 tokens under `PYTHONHASHSEED` 1, on the 6-prompt
    check subset: 3B conditions 1, 10 and 11, and 3B-Instruct 12 chat.
- **Outcome: the verdict is "holds", narrowly. The protocol stack's lead does not survive.**
  - On one stack (`steer_fp32`, 512 tokens) the raw base scores 73.8, engine v2.1 93.8 and Instruct-chat 95.6.
  - Engine − Instruct = **−1.9 [−6.2, +2.5]**, against +1.9 [−1.9, +5.6] on the protocol stack. The CI includes 0 and
    the point estimate is above −2.5, so parity holds by the rule above. Two more prompt-metric flips against the
    engine (0.625 points each) would have made it "weakens".
  - Recovery is **91% [70%, 111%]**, against 112% [87%, 150%]. Its denominator grew (Instruct − raw base +21.9
    [+15.0, +28.8], from +15.6) because the raw base fell. With the eager raw base held fixed the ratio is 89%, so the
    drop is not a denominator artefact.
  - The lean engine: −0.6 [−5.6, +3.8], recovery 97% [76%, 121%].
  - Against the eager 512-token Instruct (continuity row), the fp32 engine reads −0.6 [−5.6, +4.4], recovery 97%
    [73%, 125%].
  - Where the two differ on this stack: Instruct leads on format_constraints (100 vs. 87.5) and reasoning_math (96.9
    vs. 90.6). The engine leads on negative_constraints (90.6 vs. 84.4) and direct_qa (100 vs. 96.9).
- **Paired changes from fp32 scores** (512 tokens, same prompts and frozen coefficients):
  - Instruct-chat +1.2 [−1.2, +3.8] (94.4 → 95.6). It gains three prompts: two `direct_qa` answers no longer end
    mid-sentence, and `t_ng_08` stops using the banned word. It loses `t_rm_05` ("the movie ends at 11:00 PM").
  - Engine v2.1 −2.5 [−5.6, +0.0] (96.2 → 93.8). This is not resolved: the CI touches 0. It loses three prompts and
    gains none: `t_fm_06` (a format rule), `t_rm_01` (after the right answer it drifts into a `------` worksheet
    continuation until the cap) and `t_rm_05` ("10:30").
  - The lean engine −0.6 [−2.5, +1.2], Instruct-bare −0.6 [−5.6, +4.4] and Instruct + ban +0.6 [−1.2, +3.1] (98.1).
  - The raw base −5.0 [−9.4, −1.2] is the only resolved change of a single condition.
  - **The change in the engine − Instruct gap**, paired per prompt, is **−3.8 [−7.5, −0.6]** (lean −1.9 [−4.4,
    +0.6]). It is resolved although neither condition's own change is: the two changes point in opposite directions,
    and every one of the five prompts whose gap changed moved against the engine (`t_rm_05`, lost by both, cancels).
    The direction matches 1.5B (−4.4 [−11.9, +2.5]), and this time the change is resolved. Its CI ends one prompt-metric flip
    from 0.
- **Reading** (interpretation, not a measurement).
  - On 3B the bf16 eager stack was already close to precise: its KL to fp32 is 1.2e-3, against 2.7e-2 on 1.5B.
  - The more precise stack does not make 3B more correct across the board. `t_rm_05`, solved by every condition on
    eager at 512 tokens, fails on `steer_fp32` for the raw base, both engines, Instruct-chat and Instruct + ban.
  - So on 3B this run behaves more like a small numerical perturbation than like the removal of a handicap, as on
    1.5B. Under it Instruct's level barely moves, but the engine's margin over Instruct moves by 3.8 points.
  - The 3B margin is therefore more fragile than the 3B Instruct level. One candidate cause is that the coefficients
    were tuned on the bf16 stack (1.5B dev set); this run cannot separate that from anything else.
  - The 160-token SDPA rule ("Instruct moved 0.6") was right about 3B Instruct, which moves +1.2 here. It never tested
    the engine or the raw base, and those are what moved.
- **What survives.**
  - 3B parity with Instruct at 512 tokens holds by the pre-registered rule on both stacks.
  - The engine's lead does not survive: on the precise stack Instruct is ahead by 1.9 points (not resolved).
  - The 112% recovery does not survive either: it is 91% [70%, 111%] on the precise stack.
  - The protocol numbers stay the primary results; this is a sensitivity run.
