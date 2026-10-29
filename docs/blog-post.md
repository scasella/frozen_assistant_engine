# Zero-Weight-Update Assistant Steering: How Non-Stationary Activation Lifecycle & Boundary-Coupled Erasure Beat 3-Shot In-Context Learning at Zero Token Cost

*A four-phase engineering log on steering frozen Qwen2.5 base models into assistants with forward hooks alone: what
worked, what didn't, and what we had to take back.*

> **TL;DR.** On `Qwen/Qwen2.5-3B` (base) and 40 held-out prompts, a set of forward hooks with no training, no gradients
> and no prompt prefix scores **94.4%** Composite on four deterministic checks. That is **+13.8 [+7.5, +20.0]** points
> over the raw base model, and **+10.0 [+3.8, +16.2]** over a 768-token URIAL 3-shot prompt, at zero extra tokens and
> 39% less wall time per query. Two mechanisms carry the final gain: erasing a "worksheet-continuation" direction harder
> *at sentence boundaries* than mid-sentence, and banning words through an unembedding projection that leaves
> neighbouring words untouched. On the smaller 1.5B model the engine is again best (86.2%), but most of its margins are
> within noise. Several things we tried did nothing, and two of our own earlier claims turned out to be artifacts of a
> hash-seed bug. **Phase 4:** against the official Qwen2.5-3B-Instruct, given a 512-token budget, the hooks recover
> **112% [87%, 150%]** of post-training's gain on these checks, with zero changed weights, a 3 MB artifact and 2.6×
> shorter answers. Bolted onto Instruct, the lexical operator adds +3.1 [+0.6, +6.2].

## 1. The problem: base models already answer, then keep going

Give a pretrained base model the frame

```text
Question: How does the immune system fight infections?
Answer:
```

and it starts answering. On our test set, Qwen2.5-3B and 1.5B cross into answering on **100%** of prompts. The common
story that base models "write more questions instead of answering" simply did not happen with this frame. The real
failures come *after* a correct answer:

```text
The immune system fights infections by recognizing and destroying pathogens, which are microorganisms that cause disease.
 A single-select problem: Is the question answered in a satisfactory fashion?

Choose your answer from:
 (1). yes.
 (2). no.
(1).
```

That is the raw 3B base model, verbatim. Instruction-tuning datasets such as FLAN leaked into pretraining, and the model
has learned that a Q/A pair is often followed by an exam-style rating block, a new `[Question]`, or a hallucinated
system prompt ("You are an AI assistant that helps people find information…"). **16 of 40** raw responses (with stop
strings on) contain such a continuation.

The two standard fixes each have a cost:

- **Few-shot prompting** (URIAL-style 3-shot demonstrations) works reasonably well (84.4%), but spends **768 extra
  prompt tokens on every query**, which dropped decode throughput to 21.1 tok/s in our runs.
- **Static activation steering** (Contrastive Activation Addition: add one "assistant" vector at every position) makes
  things *worse*: 78.8%. The vector keeps the model in "keep answering" mode, so the model stops finding its own end of
  answer. Natural `<eos>` falls from **75% to 42.5%** on 3B and from **77.5% to 20%** on 1.5B, and 17 of 40 3B responses
  run into the 160-token cap.

The design question became this: can we get the in-context teacher's behaviour from forward hooks alone, and keep the
model's brakes working?

## 2. Setup, briefly

- **Models.** `Qwen/Qwen2.5-3B` (primary) and `Qwen/Qwen2.5-1.5B`, both genuine base checkpoints. Runs used bfloat16 on
  Apple Silicon (M4 Pro, 24 GB). No weight is ever modified.
- **Data.** There are three disjoint prompt sets. 80 calibration prompts fit the engine. 15 dev prompts are the *only*
  ones used to choose coefficients. 40 test prompts, 8 each in direct QA, format constraints, negative constraints,
  arithmetic and code/SQL, are run once with frozen coefficients.
- **Metrics.** All four are programmatic, with no LLM judge:
  - **TurnCross:** the response starts answering.
  - **Constraint:** every machine-checkable rule passes (JSON keys, bullet count, banned word absent, final number,
    SQL clauses, …).
  - **CleanStop:** no hallucinated turns, forum text or exam templates, and not cut off mid-sentence or looping.
  - **Fluency:** distinct-4-gram ratio > 0.75.

  **Composite** is their mean. We also log **natural `<eos>`**: whether the model ended on its own token rather than a
  hard stop string.
- **Statistics.** Brackets are 95% paired-bootstrap intervals over the 40 prompts. One prompt flipping one metric moves
  Composite by 0.625 points, so anything whose interval straddles zero is "no detectable difference".
- **Calibration.** For each calibration prompt, the frozen model writes a teacher answer under the URIAL prefix, and a
  raw continuation without it. Teacher-forced passes then record hidden states for the teacher answer *with* and
  *without* the prefix, for the raw continuation, and for four hand-written "traps":
  - question-list continuation;
  - forum boilerplate;
  - a hallucinated user turn;
  - a FLAN option block.

  States go to CPU float64 at once, and all solves (mean differences, SVD, ridge, projections) run there. The 3B
  calibration takes ~21 minutes and yields a ~3 MB artifact.

## 3. Phase 1: a non-stationary engine (88.8%)

The first engine has four modules, each an intervention on the residual stream or the attention logits of a frozen
model. At step *t* = 0 the model processes the prompt's final `Answer:` token; *t* ≥ 1 are the generated tokens.

**Module 1: a lifecycle schedule instead of a constant push.**
- *Initiation.* `v_init` is the mean difference between the URIAL-conditioned and unframed states at the delimiter. It
  is added with a weight that decays as γᵗ (γ = 0.4) and vanishes after four tokens.
- *Sustain.* A smaller vector, from the same contrast on answer tokens 4–40 and orthogonalised against `v_init`, is added
  from *t* = 1.
- *Termination.* A completion check at a late layer compares the current state with the running mean of the answer so
  far. Measuring against the running mean cancels the steering's own constant offset; a naive centroid cosine never
  fired under steering. When the check fires at a punctuation token, it boosts `<eos>`.

The point of the schedule is to push hard exactly once, then let the model's own end-of-answer machinery work. Natural
`<eos>` stays at 77.5% under the Phase-1 engine and 80% under v2.1, against 42.5% under static CAA.

**Module 2: drift erasure.**
- Trap-minus-clean mean differences define a drift direction per layer: one clean direction survives the separation test
  at every layer, with trap tokens 2.6–3.4σ out.
- At each step the component of the state beyond a dead zone (κ standard deviations of the clean distribution) is
  subtracted.
- Our first version projected per-token trap directions out at every token and produced `.\n.\n.\n` loops. Per-token
  rows mostly capture *which token* is present rather than *which genre* you are in. Per-(prompt, trap) mean rows plus a
  one-sided dead zone fixed this.

**Module 3: a single-pass shift predictor.** Classifier-free guidance needs two forward passes, one with the teacher
context and one without. Instead, a rank-16 ridge regression predicts the input-dependent part of the URIAL shift from
the *unframed* delimiter state. On 3B its cross-validated R² beyond the mean shift is **+0.49 to +0.69** across layers
12–28, so the shift really does depend on the instruction, and it is predictable.

**Module 4: instruction-pinning attention.** Some heads attend to the instruction more while writing teacher answers
than raw continuations: 14 on 3B, 2 on 1.5B. Their pre-softmax logits get +β at the instruction's key positions. On its
own this does nothing (−0.6 on 3B). On top of Modules 1–3 it is the one Phase-1 step whose interval excludes zero:
**+4.4 [+0.6, +8.8]** on 3B and **+3.8 [+0.6, +7.5]** on 1.5B.

The full Phase-1 engine reached **88.8%** on 3B: +8.1 [+1.9, +14.4] over the raw base, and a statistical tie with
URIAL (+4.4 [−1.9, +10.6]) at zero tokens. Almost all of the gain was CleanStop (55% → 85%).

It also exposed a trade-off. With κ = 1σ, erasure touches about **14% of clean tokens**, which is enough to suppress
every math `[Question]` continuation (reasoning_math scored 100%). But it occasionally knocks the model into a loop:

```text
Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
A: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
B: Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and water into glucose and oxygen.
C: Photosynthesis is the proces…
```

## 4. Phase 2: a gentler threshold, and a lexical ban (90.0–90.6%)

**A p99 hinge.** v1's erasure was already a continuous excess hinge, `h − λ·relu(z − κ)·σ·u`. Raising κ from 1σ to each
layer's clean 99th percentile (about 3.1σ on 3B) changes how often it acts:
- clean tokens touched: 14% → **1%**;
- trap tokens touched: ~85% → ~57%.

Fluency went to **100%**, and condition 7 scored 90.0. But the math `[Question]` continuations came back, on 4 of 8
arithmetic prompts:

```text
… The bookstore sells 12 * 14 = 168 books in 2 weeks. The answer is 168.
[Question]A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?
```

So v1's κ = 1 and v2's κ ≈ 3.1 sit at two ends of one fluency-vs-drift trade-off, and neither dominates.

**Signed attention was inert.** We gave pinned heads −β on the negation clause ("without using the word 'force'") and +β
elsewhere. The best dev setting produced scores identical to not doing it, and on test it was −0.6 [−3.8, +1.9].
Repelling attention from the clause also repels it from the word "without", which does not help a model avoid "force".

**An unembedding projection does move negative constraints.**
- A deterministic parser extracts negated targets ('force', 'Python', 'lions or elephants').
- The pre-LM-head state is projected off those tokens' unembedding directions.
- Negative-constraint accuracy on 3B went from **50% to 87.5%**.

This is close to a lexical ban by construction, and some passes are evasions. Asked to explain bicycle riding without
the word 'balance', both models wrote "To ride a bicycle without using the word **'bicycle'**, you can say…". The
verifier only checks that 'balance' is absent, so this counts as a pass.

**A bug we caught in ourselves.** The banned tokens are projected one after another, and with γ > 1 those steps do not
commute. Their order came from a Python `set`, so it depended on the process's hash seed, and two runs of "the same"
condition gave different answers. We sorted by token id, re-ran the condition-9 dev search and test runs, and withdrew
two claims from the earlier write-up:
- that mean-centring the unembedding rows was necessary;
- that banning 'Python' collapsed the answer to "C++, Assembly Language, Assembly Language". In the deterministic run it
  answers "1. C 2. C++ 3. C#".

## 5. Phase 3: boundary physics and sibling geometry (94.4%)

### 5.1 Drift is born at the boundary

Where exactly does post-answer drift start? We teacher-forced 24 calibration answers followed by a `[Question]` or FLAN
continuation, and read the drift coordinate z, in clean-token standard deviations, at each position (3B):

| Position | mean z |
|---|---|
| mid-answer tokens | −0.4 |
| last answer token (`.`) | 1.1 |
| **the boundary newline** | **3.9** |
| first drift tokens | 4.3–4.9 |
| p99 hinge threshold | 3.1 |

The drift spike *starts at the boundary newline*, the token whose state predicts the `[`. The p99 hinge already fires
there, but it removes only 75% of the small excess above 3.1σ, leaving the state near **3.3σ**: still deep in drift
territory. (The brief we worked from assumed the onset sat *below* the threshold. It does not.)

The fix is a **boundary-coupled hinge**. Mid-sentence keeps κ = p99 (1% clean firing). Inside a boundary window, meaning
after `. ! ? \n` or up to two tokens past a newline, κ drops to **1.5σ** (dev-tuned; the brief's 1.0 was worse on dev).
On 3B, reasoning_math recovers from 87.5% to **96.9%** and CleanStop rises to **92.5%**. The v1 loops do not come back:
Fluency is 97.5%.

### 5.2 Banning 'Python' without hurting 'Java'

A naive projection `h ← h − γ⟨h, ŵ_b⟩ŵ_b` lowers every token whose unembedding correlates with the banned one. After mean
centring, ' Python' has cosine **0.45** with ' Java', 0.38 with ' Ruby' and 0.36 with ' JavaScript', but only 0.02 with
' Assembly'. Banning a word pushes down its whole neighbourhood.

The sibling-orthogonalised projection fixes this in three steps:
1. Take the K = 4 currently most-favoured non-banned word tokens, and their mean-centred unembedding rows.
2. Orthogonalise the banned direction against them: `w̃ = normalize((I − P)ŵ_b)`.
3. Step along `w̃` by `γ⟨h, ŵ_b⟩/⟨w̃, ŵ_b⟩`.

The banned alignment still shrinks by exactly (1 − γ), but the protected siblings' relative logits do not move. We
checked this on a real state, banning 'piano' and 'guitar' with K = 8:

| | relative logit change on the top clean candidates (' violin', ' trumpet', ' drum', …) | banned tokens |
|---|---|---|
| naive projection | −3.6 to −13.7 | −10 to −31 |
| sibling-orthogonalised | **0.00** | −10 to −31 |

### 5.3 What actually did the work

Condition 10 combines the boundary hinge, the sibling projection, a drift-assisted stop and an anti-echo guard (which
bans "without" in the first 12 tokens). Leave-one-out ablations on test, used for attribution only:

| Removing… | 3B | 1.5B |
|---|---|---|
| boundary-coupled hinge | **−3.8 [−6.9, −1.2]** | −1.9 [−6.2, +2.5] |
| sibling orthogonalisation | −3.1 [−7.5, +0.6] | **−6.9 [−13.1, −1.9]** |
| drift-assisted stop | 0.0 (fired on 10 prompts, changed 3 responses, no metric) | 0.0 |
| anti-echo guard | **+1.9 [+0.0, +5.0]** (better without it) | 0.0 |

The boundary hinge and the sibling projection carry the gain. The drift-assisted stop is inert. The anti-echo guard is
unhelpful, and part of its apparent harm is a verifier artifact: without it, the model echoes "without using the word
'**balanc**'", a truncated ban that the regex misses.

Condition 10 without the anti-echo guard scores **96.2%**. We do not report that as a result, because it was chosen by
looking at test numbers. The simpler "boundary hinge + sibling projection only" engine was later run as condition 11
(§7.1, 95.6%). It is just as test-informed, so both remain hypotheses for a fresh held-out set.

**Final numbers, condition 10:**

| | 3B | 1.5B |
|---|---|---|
| Composite | **94.4** | **86.2** |
| vs. raw base | **+13.8 [+7.5, +20.0]** | **+10.6 [+2.5, +18.8]** |
| vs. raw + stop strings | **+12.5 [+6.9, +18.1]** | +5.6 [−1.9, +13.1] |
| vs. URIAL 3-shot (768 tokens) | **+10.0 [+3.8, +16.2]** | +5.6 [−3.8, +15.0] |
| vs. Phase-1 engine | +5.6 [+0.0, +11.2] | +1.9 [−5.0, +8.1] |
| mean wall time per query (URIAL) | 3.25 s (5.30 s) | 2.08 s (3.57 s) |

The title's claim, beating 3-shot in-context learning, holds with a resolved interval on 3B. On 1.5B it is a
non-significant +5.6.

## 6. Side by side (verbatim, 3B)

**`t_qa_04`: "How does the immune system fight infections?"**
- *Raw base:* answers in one sentence, then "A single-select problem: Is the question answered in a satisfactory
  fashion? Choose your answer from: (1). yes. (2). no. (1)."
- *Static CAA:* answers, then writes a code fence and drifts into a new task: "Please determine whether the given text is
  related to computer science, if yes please return "YES"…". It hits the token cap.
- *Engine v2.1:* "The immune system fights infections by recognizing and destroying pathogens… The immune system has two
  main branches: the innate immune system and the adaptive immune system…". It ends on its own `<eos>`.

**`t_qa_01`: "What is photosynthesis?"**
- *v1:* the A:/B:/C: loop shown above.
- *v2 hinge:* the FLAN block leaks back in.
- *v2.1:* "Photosynthesis is the process by which plants and other organisms use sunlight to convert carbon dioxide and
  water into glucose and oxygen." This is clean but only partly earned: the model still started a new `Question:` turn,
  and the hard stop string cut it.

**`t_ng_01`: "Explain gravity in one paragraph without using the word 'force'."** Every condition without the
unembedding projection uses the word, URIAL included ("Gravity is a fundamental force of nature…"). Most open with
"Gravity is a force…". v2.1: "Gravity is a fundamental natural phenomenon that governs the attraction between
objects with mass. It is a universal law that applies to all objects in the universe…"

**`t_rm_05`: "A movie starts at 7:45 PM and lasts 2 hours and 30 minutes. At what time does it end?"**
- *v2 hinge:* "…the movie ends at 10:15 PM. The answer is 10:15 PM.\n[Question]A movie starts at 1:30 PM and ends at 3:10
  PM…"
- *v2.1:* the same correct answer, followed by "[Reasoning: The movie starts at 7:45 PM and lasts 2 hours and 30
  minutes…]". The verifier scores this as clean because it only knows `[Question]`/`[Answer]` labels. It is a blind spot,
  not a win.

**`t_rm_01`: "A bookstore sells 12 books per day. How many books does it sell in 2 weeks?"** This one is a regression.
v1 answered "…168 books in 2 weeks. The answer is 168." and stopped. v2.1 gives the same answer, then writes `------`
separators and new word problems until the token cap. Boundary-coupled erasure does not catch every drift genre.

## 7. Phase 4: what is it worth against real post-training?

The obvious question is economic. Qwen ships an official post-trained `Qwen2.5-3B-Instruct`. Is a 3 MB hook file a
substitute for it, a complement to it, or neither? We ran the official Instruct checkpoints through the same 40 prompts,
the same greedy loop and the same frozen verifiers, both with Qwen's native chat template and with our bare
`Question: … Answer:` frame.

### 7.1 First, a leaner engine

The leave-one-outs in §5.3 said only two v2.1 mechanisms matter. Condition 11 keeps exactly those, with no new
coefficients, and fixes how they run:

- **Text-only boundary trigger.** v2.1 also opened a boundary window when the *previous* step's completion score was
  high, which cost one host sync at the stop layer on every token. The lean trigger uses only the token text (`. ! ?
  \n`, or ≤ 2 steps after a newline).
- **Sync-free sibling projection.** The projection's branches become `torch.where` selections, and its trace counters
  stay on the GPU until the answer ends.
- **Word-only search.** The sibling search multiplies only the 62% of the vocabulary that are word tokens.

A single-read logit-space update is not possible here. The siblings come from the unprojected logits and the output
must come from the projected state, so an exact version reads the unembedding twice; `logits + W·Δh` is itself a
full-vocabulary product.

The sync-free projection reproduces x9 **byte for byte on 40/40 prompts** on both models. Every response where
condition 11 differs from x9 (3/40 on 3B) is on a prompt where the dropped drift-stop had fired.

Measured in one session with the conditions interleaved, condition 11 decodes at **24.3 tok/s on 3B, within 2% of
condition 7** (24.8), against 23.3 for v2.1. On prompts without a negation it is indistinguishable from condition 7.
The brief's "12% overhead" came from comparing logs recorded hours apart; in one session v2.1's overhead was 6%. The
mechanism set was read off the test leave-one-outs, so condition 11's 95.6 is not a held-out number; its claim is
cost, not accuracy.

### 7.2 The engine against Qwen2.5-3B-Instruct

| 3B, 40 test prompts | Composite, 160 tokens | Composite, 512 tokens | tokens per answer (512) |
|---|---|---|---|
| raw base | 80.6 | 78.8 | 155 |
| **engine v2.1** (`10_engine_v2_1`) | **94.4** | **96.2** | **82** |
| Qwen2.5-3B-Instruct, chat template | 86.2 | 94.4 | 214 |
| Qwen2.5-3B-Instruct, bare frame | 81.9 | 88.8 | 179 |
| Instruct + our sibling-orthogonal ban | 88.8 | **97.5** | 211 |

At the protocol's 160-token budget the engine beats Instruct by **+8.1 [+1.9, +14.4]**. We did not believe that
number, and reading the outputs showed why: Instruct writes thorough markdown answers. 17 of 40 hit the cap, and 15 of
those failed CleanStop as mid-sentence truncations. Instruct's chat-template answers contain zero contamination; post-training really did
remove the drift. At 160 tokens even post-training's own gain over the raw base is unresolved (+5.6 [−1.2, +12.5]),
so the "recovery ratio" the brief asked for is meaningless there.

Given 512 tokens, Instruct ends on its own on 37 of 40 prompts, and post-training's gain becomes clear:
**+15.6 [+8.8, +22.5]**. The engine scores 96.2 with no cap hits. Engine − Instruct is **+1.9 [−1.9, +5.6]**, and
the engine recovers **112% [87%, 150%]** of post-training's gain. The honest summary is *parity* on this benchmark,
with answers 2.6× shorter. On 1.5B it is not parity: at 512 tokens 1.5B-Instruct leads, 90.0 against 87.5 (engine −
Instruct −2.5 [−10.0, +4.4], recovery 85% [40%, 135%]), on a stack that handicaps 1.5B arithmetic (§7.5).

This benchmark measures four things: starting an answer, stopping cleanly, following machine-checkable rules, and not
looping. It says nothing about helpfulness, depth, safety, refusals or multi-turn behaviour, where post-training does
work the engine never attempts.

### 7.3 Unit economics

| 3B | Weights changed | Artifact | One-off cost | Extra prompt tokens | Clean-token KL from the base (nats/token) | Composite, 512 tokens |
|---|---|---|---|---|---|---|
| Engine v2.1 | 0 | 3.0 MB | 21 min of forward passes on 80 prompts, one laptop | 0 | **0.014** | 96.2 |
| URIAL 3-shot prefix | 0 | 768-token prompt | none | 768 | 0.064 | – |
| Qwen2.5-3B-Instruct | 86.8% of 3.09B parameters | 6.17 GB (the whole checkpoint) | SFT on > 1M examples + multi-stage RL | 25 (chat template) | 0.576 | 94.4 |

The KL column needs care:
- **How it is measured.** We teacher-force every condition along URIAL's clean held-out answers and measure
  D_KL(P_raw ‖ P_condition) per token, excluding the 8 prompts where the engine deliberately bans a word the answer
  uses.
- **What the engine does to it.** It changes the base model's next-token distribution by 0.014 nats per token and
  agrees with its top-1 prediction 97% of the time. Instruct moves it 40× further, as it is meant to.
- **How to read it.** Low KL is not "better". It means the engine gets its score by intervening at a handful of
  decisions, mostly where to stop and which words to avoid, while leaving everything else the base model knows alone.

### 7.4 The lexical operator transfers to post-trained models

Instruct's characteristic failure on negative constraints is echoing the banned word while restating the request:
"Here are three programming languages without mentioning Python:". We attached condition 11's sibling-orthogonal
projection to the Instruct checkpoint with zero re-tuning: the same γ and K from base-model dev, no calibration
artifact.

On 3B it raised negative-constraint compliance from **37.5% to 87.5%** and broke nothing. At 512 tokens, Instruct +
ban is the best condition we have measured, **+3.1 [+0.6, +6.2]** over Instruct alone. Two of its "fixes" are
evasions ("without mentioning the specific ones you requested to be avoided"). On 1.5B-Instruct it also caused one
loop ("…not related to the question of not being related to the question…").

### 7.5 A surprise: bf16 eager attention breaks 1.5B arithmetic

While reading Instruct outputs we saw "15 + 9 = 14". It was not Instruct's fault. Module 4 needs eager attention to
inject its bias, and in bf16 the eager path rounds pre-softmax scores. HF's own eager attention reproduces the same
wrong digits.

Under SDPA the 1.5B raw base solves **8/8** math prompts instead of 3/8, and its Composite rises from 75.6 to 81.2.
Every 1.5B number in this project, the v2.1.0 release included, ran on that stack. Comparisons among them remain fair,
but their level is depressed. 3B is unaffected: 8/8 under both stacks.

Before rerunning Instruct on SDPA we fixed a rule: if 3B Instruct moved by more than 2 points, the SDPA run would
become the reference. It moved 0.6, so the matched-stack comparison above stands. The fix for the engine, a fused
attention kernel with an additive bias, is future work.

## 8. Caveats we would want a reviewer to raise

- **Small samples.** 40 test prompts, and 15 dev prompts for tuning, of which only 3 contain a negation. Condition 10's
  dev advantage over conditions 7 and 9 was one prompt, so most of its test gain was not visible during tuning.
- **One model family, greedy decoding only.** Nothing here is shown for other architectures or sampling. The only
  post-trained models tested are Qwen2.5-Instruct, and only with the lexical operator (§7.4).
- **The protocol stack depresses 1.5B.** bf16 eager attention costs the 1.5B model its arithmetic (§7.5); every
  1.5B level is pessimistic.
- **The token budget matters.** At 160 tokens the comparison with Instruct is dominated by truncation (§7.2). The
  512-token rerun was planned after seeing that, and it is the fair comparison.
- **The verifier has blind spots.**
  - Bracket-labelled continuations other than `[Question]`/`[Answer]` are not flagged.
  - Evasive meta-answers to negative-constraint prompts pass.
  - Repeated "The answer is 12." lines pass the 4-gram fluency check.

  We disclose these rather than patch them, because patching after reading test outputs would be test-driven.
- **The negative-constraint gains are partly mechanical.** The unembedding projection makes banned surface forms hard to
  emit. Inflections outside the banned set ("balancing") still get through, and so do sub-word spellings: on `t_ng_04`
  the lean engine wrote "balanc" as `bal` + `anc`, which the verifier misses. The live studio's opt-in stem mode
  closes both. It bans stem-initial tokens and masks stem-completing continuations. The model then restates the rule
  as "balace", and short stems over-ban (`water` also bans "waterfall"). Stem mode is a demo setting, not a benchmark
  condition, and it was motivated by a test prompt.
- **URIAL is a reference, not a ceiling.** Its demonstrations are deliberately thorough, so its answers are long and hit
  the 160-token cap on 13 of 40 prompts. That costs it CleanStop.
- **Throughput.** In one session, the lean engine runs at 24.3 tok/s against 25.3 for the unsteered model on 3B. The
  eager attention that Module 4 needs costs more than the hooks: SDPA runs the raw base at 29.5 tok/s. The per-query
  time advantage over URIAL comes from skipping the prefix *and* from shorter answers.

## 9. Reproduce it

Every number above is regenerated from logged generations in seconds, without a model:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m eval.report results/Qwen__Qwen2.5-3B/test40 --out RESULTS.md --appendix results/Qwen__Qwen2.5-1.5B/test40
.venv/bin/python cli.py --model Qwen/Qwen2.5-3B --modes raw,caa,v21     # try it live
./run_demo.sh                                                            # live studio: 4-way arena + circuit telemetry
```

The [README](../README.md) has the architecture and commands, [RESULTS.md](../RESULTS.md) has every table, interval and
verbatim example, and the [ELI5 guide](eli5-and-use-cases.md) is the plain-English version.
