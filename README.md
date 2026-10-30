# Zero-Weight-Update Assistant Engine

**Make a raw pretrained base LLM answer like an instruction-tuned assistant, without training, without gradients, and
without spending a single extra prompt token.** The engine is a set of lightweight PyTorch forward hooks on a frozen
base model:
- non-stationary residual-stream steering;
- a single-pass shift predictor at the `Answer:` delimiter;
- boundary-coupled drift erasure;
- an instruction-pinning attention bias;
- a sibling-orthogonalised unembedding projection.

Every coefficient comes from forward passes on 80 calibration prompts plus closed-form linear algebra (mean differences,
SVD, ridge regression, projections). The prompt is just `Question: {instruction}\nAnswer:`.

**[Full experimental report (RESULTS.md)](RESULTS.md)** · **[Technical blog post](docs/blog-post.md)** ·
**[ELI5 explainer & use cases](docs/eli5-and-use-cases.md)** · **[Agent & developer guide (AGENTS.md)](AGENTS.md)**

## Headline results

`Qwen/Qwen2.5-3B` base, 40 held-out test prompts across 5 categories, greedy decoding, 160 new tokens. All metrics are
deterministic programmatic checks, with no LLM judge. Composite is the mean of TurnCross, Constraint, CleanStop and
Fluency. Brackets are 95% paired-bootstrap CIs over prompts.

| Condition (3B) | Composite | CleanStop | Fluency | Natural `<eos>` | Extra prompt tokens | tok/s | 1.5B Composite |
|---|---|---|---|---|---|---|---|
| `1_raw_base` | 80.6 | 52.5 | 95.0 | 75.0 | 0 | 26.1 | 75.6 |
| `2_raw_base_plus_stops` | 81.9 | 55.0 | 97.5 | 75.0 | 0 | 26.2 | 80.6 |
| `4_static_caa` (standard activation addition) | 78.8 | 55.0 | 97.5 | **42.5** | 0 | 26.3 | 75.6 (natural `<eos>` **20.0**) |
| `3_urial_3shot_prefix` (in-context teacher) | 84.4 | 67.5 | 97.5 | 45.0 | **768** | 21.1 | 80.6 |
| `6_full_assistant_engine` (Phase 1) | 88.8 | 85.0 | 90.0 | 77.5 | 0 | 25.6 | 84.4 |
| `7_engine_v2_hinge_erase` (Phase 2, dev-selected) | 90.0 | 80.0 | 100.0 | 72.5 | 0 | 25.6 | 81.2 |
| **`10_engine_v2_1` (Phase 3, dev-selected headline)** | **94.4** | **92.5** | 97.5 | **80.0** | **0** | 22.6 | **86.2** |
| `x9_v2_1_minus_anti_echo` † | 96.2 | 95.0 | 97.5 | 82.5 | 0 | 22.7 | 86.2 |
| `11_engine_v2_2_lean` (Phase 4) † | 95.6 | 92.5 | 97.5 | 82.5 | 0 | 24.3 ‡ | 86.2 |

On 3B, **condition 10** gains:
- **+13.8 [+7.5, +20.0]** over the raw base;
- **+10.0 [+3.8, +16.2]** over the 768-token URIAL prefix, the first engine whose lead over URIAL is statistically
  resolved;
- +5.6 [+0.0, +11.2] over the Phase-1 engine.

It answers a query in 3.25 s on average, against 5.30 s for URIAL (an M4 Pro, including prefill).

† `x9` is a leave-one-out ablation (condition 10 without the anti-echo guard), and it was found *on the test set*. It
is reported for attribution only. Treating 96.2 as a result would be test-set selection; it needs fresh held-out
validation first. Condition 11 keeps only the two mechanisms those leave-one-outs credit, so it is equally
test-informed; its claim is execution cost (‡ same-session benchmark: within 2% of condition 7, against 6% for
condition 10).

### Versus official post-training (Phase 4)

The official `Qwen/Qwen2.5-3B-Instruct` ran on the same 40 prompts, with the same greedy loop and the same frozen
verifiers. At the protocol's 160-token budget, Instruct's thorough answers hit the cap on 17/40 prompts, so the fair
comparison is the 512-token rerun:

| 3B | Weights changed | Artifact | Extra prompt tokens | Clean-token KL from base (held-out, nats) | Composite, 160 tok | Composite, 512 tok | Tokens per answer (512) |
|---|---|---|---|---|---|---|---|
| raw base | 0 | – | 0 | 0 | 80.6 | 78.8 | 155 |
| **engine v2.1** | **0** | **3.0 MB** | **0** | **0.014** | **94.4** | **96.2** | **82** |
| Qwen2.5-3B-Instruct (chat template) | 86.8% of 3.09B | 6.17 GB | 25 | 0.576 | 86.2 | 94.4 | 214 |
| Instruct + our sibling-orthogonal ban | 86.8% of 3.09B | 6.17 GB + 0 | 25 | – | 88.8 | **97.5** | 211 |

- **Parity with post-training on these checks.** At 512 tokens, post-training gains +15.6 [+8.8, +22.5] over the raw
  base. The engine is within noise of Instruct (+1.9 [−1.9, +5.6]) and recovers **112% [87%, 150%]** of that gain.
  With fp32 attention scores for all three conditions (a sensitivity run), parity still holds by its pre-registered
  rule, but narrowly: −1.9 [−6.2, +2.5], recovery 91% [70%, 111%]. The paired change in the gap is −3.8 [−7.5, −0.6],
  so the engine's lead on the protocol stack does not survive. The checks cover starting, stopping, rule-following and not looping, not helpfulness, safety or multi-turn quality.
  On 1.5B, Instruct leads at 512 tokens (90.0 vs. 87.5; recovery 85% [40%, 135%]). A sensitivity run with fp32
  attention scores for all three conditions puts it at 90.0 vs. 96.9: −6.9 [−12.5, −1.9], which excludes parity, and
  recovery 62% [28%, 90%]. The change from −2.5 is itself not resolved (−4.4 [−11.9, +2.5]).
- **Complementary, not only a substitute.** Bolting the engine's lexical ban onto Instruct, with no re-tuning, raises
  negative-constraint compliance from 37.5% to 87.5% and adds +3.1 [+0.6, +6.2] at 512 tokens.
- **Minimal distortion.** The KL column excludes the 8 prompts where the ban deliberately suppresses a word. There the
  engine moves the base model's next-token distribution 40× less than post-training does, with 97% top-1 agreement.
- **Numerics caveat.** Module 4 needs eager attention, and in bf16 eager attention breaks Qwen2.5-1.5B arithmetic, base
  and Instruct alike (raw base 3/8 math vs. 8/8 under SDPA or fp32 scores). All 1.5B levels in this repository are
  depressed, but not equally. At 512 tokens, fp32 scores lift Instruct by +6.9 [+3.1, +10.6] and the engine by +2.5
  [−3.1, +8.1]. So the protocol-stack 1.5B comparison is not pessimistic for the engine; it more likely flatters it.
  The 3B engine is unaffected at 160 tokens (−0.6 and +0.0). At 512 tokens on 3B, fp32 scores move Instruct by +1.2
  [−1.2, +3.8] and the engine by −2.5 [−5.6, +0.0], neither resolved; the gap between them moves by a resolved −3.8.

Details, CIs, the SDPA and 512-token sensitivity tables, and verbatim Instruct cases are in
[RESULTS.md §7–8](RESULTS.md).

**What the numbers do and do not say**

- **The gain is termination and genre drift, not turn-crossing.** Every condition scores TurnCross = 100%: Qwen2.5
  base models already start answering after `Answer:`. What they do wrong is keep going after a correct answer, with
  FLAN-style "A single-select problem… Available options: (a). yes" blocks, new `[Question]` problems and hallucinated
  system prompts. 16 of 40 raw+stops responses are contaminated.
- **Static steering breaks the brakes.** A single vector added at every token lowers natural `<eos>` from 75% to 42.5%
  (3B) and from 77.5% to 20% (1.5B). The engine's decaying schedule keeps it at 80%.
- **On 1.5B the picture is weaker.** Condition 10 is the best there too (86.2), but only its margin over the raw base is
  resolved (+10.6 [+2.5, +18.8]). Over the Phase-1 engine it is +1.9 [−5.0, +8.1]. All 1.5B levels are depressed by
  the bf16 eager-attention numerics described above, and the raw base more than the engine. With fp32 scores, the
  margin over the raw base is +6.9 [+0.0, +13.8] at 160 tokens.
- **Small samples.** Coefficients were tuned on 15 dev prompts, where one prompt equals 1.7 Composite points. Condition
  10's dev margin over conditions 7 and 9 was a single prompt.
- **Verifier blind spots** are documented in [RESULTS.md](RESULTS.md):
  - bracket-labelled continuations other than `[Question]` (`[Equation]`, `[Reasoning: …]`) are not flagged;
  - evasive meta-answers to negative-constraint prompts can pass;
  - repeated final-answer lines can pass the 4-gram fluency check.

## Architecture: five mechanisms on a frozen model

```text
  "Question: {instruction}\nAnswer:"          (0 extra tokens)
                 │
   embed ─► layers 0…11 ─► ┌─────────────── target band: 8 layers in 0.35N–0.80N (3B: 12,14,17,19,21,23,26,28) ──────────────┐
                           │ residual hooks, step t (t = 0 is the prefill pass ending at the "Answer:" delimiter)              │
                           │   M1  t<4  : h += α·γᵗ·v_init / K          (t = 0: v_init + M3's predicted, input-dependent shift) │
                           │   M1  t≥1  : h += α_s·v_sus⊥ / K                                                                   │
                           │   M2  t≥1  : h −= λ·relu(z − κ_t)·σ·u      κ_t = 1.5 in boundary windows, else per-layer p99       │
                           │   M1  stop layer: completion score vs. running mean ─► boost <eos>                               │
                           └───────────────────────────────────────────────────────────────────────────────────────────────────┘
   attention hooks (pinned heads in 0.30N–0.75N): M4  logits[q, instruction keys] += β
                 │
   final norm ─► M5  h −= γ·(⟨h,ŵ_b⟩/⟨w̃_b,ŵ_b⟩)·w̃_b   (banned targets only; w̃_b ⟂ top-K clean candidates) ─► lm_head ─► greedy
```

**Calibration.** For each of the 80 calibration prompts, the frozen model writes a teacher answer Y⁺ under a 768-token
URIAL 3-shot prefix, and a raw continuation Y⁻ without it. Teacher-forced passes record hidden states for:
- URIAL + prompt + Y⁺;
- prompt + Y⁺ (the same tokens without the prefix);
- prompt + Y⁻;
- four negative traps: question-list continuation, forum boilerplate, a multi-turn hallucination, and a FLAN option
  template.

States go to CPU float64 immediately, and all solves run there. On an M4 Pro the whole calibration takes about 21 min
for 3B and yields a ~3 MB artifact.

1. **Module 1: non-stationary lifecycle schedule and dynamic stop.**
   - `v_init = E[h_URIAL − h_raw]` at the delimiter is applied with a decaying weight γᵗ for t < 4.
   - A sustain vector from the same contrast on answer tokens t ∈ [4, 40], orthogonalised against `v_init`, is applied
     from t ≥ 1.
   - A completion check compares the state against the running mean of the answer and boosts `<eos>` above the
     calibrated τ.

   The initiation push fades after four tokens, so the model's own end-of-answer behaviour survives. That is why
   natural `<eos>` stays at 80% while static CAA drops it to 42.5%.
2. **Module 2: boundary-coupled excess-hinge erasure** (v2.1).
   - The drift directions u come from one row per (prompt, trap) mean difference, oriented toward the traps.
   - The update is `h −= λ·relu(z − κ)·σ·u`, with z the clean-standardised coordinate.
   - Mid-sentence, κ is each layer's clean-token 99th percentile (≈3.1σ on average), so only 1% of clean tokens are
     touched.
   - In a *boundary window* (after `. ! ? \n`, or up to 2 tokens past a newline), κ drops to 1.5σ.

   Teacher-forcing calibration answers into a drift continuation shows why the boundary matters. The boundary newline
   sits at **+3.9σ** (mid-answer tokens: −0.4σ), so a stationary 3.1σ hinge leaves about 3.3σ of the drift behind.
   Removing the boundary coupling costs **−3.8 [−6.9, −1.2]** Composite on 3B (leave-one-out `x6`).
3. **Module 3: single-pass delimiter shift predictor.** A rank-16 ridge regression predicts the input-dependent part of
   the URIAL shift from the unframed delimiter state. It is CFG-like steering without a second forward pass. On 3B
   layers 12–28 its cross-validated R² beyond the mean shift is +0.49 to +0.69.
4. **Module 4: instruction-pinning attention bias.** It adds +β to the pre-softmax logits of selected heads at the
   instruction's key positions. The heads are selected by their instruction-attention delta between teacher and raw
   continuations: 14 heads on 3B, 2 on 1.5B. Module 4 helps in combination with Modules 1–3 (+4.4 [+0.6, +8.8] on 3B,
   +3.8 [+0.6, +7.5] on 1.5B) and not on its own (−0.6 on 3B).
5. **Module 5: sibling-orthogonalised unembedding projection** (v2.1, negation prompts only).
   - A deterministic parser extracts negated targets ("without using the word 'force'").
   - Each banned token's mean-centred unembedding direction is orthogonalised against the top-K (K = 4) currently
     favoured non-banned word tokens, then removed with a rescaled step. The banned alignment still shrinks by exactly
     (1 − γ), while those siblings' relative logits do not move.
   - On a measured state, a naive projection lowered siblings by 3.6–13.7 logits (' violin' when banning ' piano'); the
     orthogonalised one moved them by **0.00**, while the banned tokens fell 10–31 logits.
   - Removing it costs **−6.9 [−13.1, −1.9]** on 1.5B (leave-one-out `x8`) and −3.1 [−7.5, +0.6] on 3B.

Two further v2.1 components, a drift-assisted stop and an anti-echo guard, are included in condition 10 but measured as
inert and unhelpful respectively. See [RESULTS.md §3](RESULTS.md) for every ablation.

## Quickstart

Requires **Python ≥ 3.10**. Tested with Python 3.12, torch 2.15.0 and transformers 5.21.0. The code uses the
transformers 5.x `AttentionInterface` / `masking_utils` APIs.

```bash
git clone https://github.com/scasella/frozen_assistant_engine.git && cd frozen_assistant_engine
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q                       # model-free tests: verifiers, parser, cache guard, report, links
```

Regenerate `RESULTS.md` from the logged generations. This takes seconds, needs no model, and re-scores every response
with the current verifiers:

```bash
.venv/bin/python -m eval.report results/Qwen__Qwen2.5-3B/test40 --out RESULTS.md --appendix results/Qwen__Qwen2.5-1.5B/test40
```

Try it live with the cached calibration artifact. The model downloads from Hugging Face, ungated:

```bash
.venv/bin/python cli.py --model Qwen/Qwen2.5-3B --modes raw,caa,v21 --prompt "List 3 programming languages without mentioning Python."
.venv/bin/python scripts/sanity_check.py --model Qwen/Qwen2.5-1.5B   # steer_eager == eager attention, tokenisation, generation
```

Re-run conditions on the 40 test prompts. Cached conditions are loaded from `results/` and only missing ones generate:

```bash
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --conditions 10_engine_v2_1 11_engine_v2_2_lean --tag test40
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B-Instruct --tag test40     # post-trained reference (12-13)
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --tag mynewrun --all    # all 20 base-model conditions; a new tag = fresh run
```

Recalibrate. Teacher and raw trajectories are cached in `artifacts/`, so only the forward passes re-run. This
overwrites `artifacts/<model>/calib80/calibration.pt` in place:

```bash
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --recalibrate --conditions 1_raw_base --tag recal
.venv/bin/python scripts/resolve_calibration.py --model Qwen/Qwen2.5-3B   # re-solve from a local states.pt cache (no forward passes)
```

Devices: `engine/model_loader.py` picks `mps`, then `cuda`, then `cpu`; override with `--device`. All reported runs used
an Apple M4 Pro (24 GB) with `mps` and bfloat16. CUDA and CPU paths are untested.

## Live studio and Circuit X-Ray

A local web app (`demo/`, FastAPI + vanilla JS, no build step) that runs the comparison live and explores the logged
data:

```bash
.venv/bin/pip install -r requirements-demo.txt
./run_demo.sh                        # http://127.0.0.1:8000  (python -m demo.server --instruct swap on < 16 GB)
.venv/bin/python scripts/verify_demo.py
```

- **Arena.** Four columns stream token by token on one GPU:
  - raw base (condition 1);
  - static CAA (condition 4);
  - the engine;
  - Qwen2.5-Instruct, with a switch that attaches the sibling-orthogonal ban (condition 13).

  The first three share the same frozen weights. Their decoders take turns (`engine/stream.py`), and each engine's
  hooks are live only during its own forward pass.
- **Circuit X-Ray.** Every engine token carries read-only telemetry:
  - the initiation pulse and the sustain norm;
  - drift z(t) against the hinge threshold κ(t) at the stop layer, and per layer;
  - cosine to the completion centroid;
  - the ban's suppressed tokens and the four protected siblings (their centred logits move by ≈ 0.00);
  - the attention the pinned heads pay to the instruction, with and without Module 4's bias;
  - optionally the **exact** per-token KL against the unsteered model, using one teacher-forced shadow pass per token.

  A drawer toggles M1–M4, the ban and stem morphology, and sets κ_boundary, γ_unembed and β_attn.
- **Alignment tax & economics.** The KL-vs-Composite chart, unit economics, recovery CIs and the 24-row ablation
  waterfall. Everything is read from `results/` through `eval/report.py`'s own loaders, so the numbers match RESULTS.md.
- **Benchmark browser.** All logged generations (protocol, 512-token and SDPA tags, both models) side by side, with
  filters for category and for prompts where the conditions disagree.

The studio's default engine is **"v2.3 demo"**: condition 11 plus `ban_morphology="stem"`. It is not a benchmark
condition; every logged condition keeps the exact surface-form ban.
- **The loophole it closes.** On `t_ng_04` ("…without using the word 'balance'") every logged condition broke the ban
  while restating the rule. The v2.2 lean engine wrote "balanc" as `bal`+`anc`, which the frozen verifier misses.
- **What stem mode adds.**
  - It bans every token that starts a word with the target's stem: `balance` → `balanc`, which covers ` balancing`,
    ` balanced` and `Balancer`.
  - It hard-masks tokens that would complete the stem from an unfinished word: after `bal`, it masks `anc`, `ancing`
    and the like. A sub-word spelling has no single unembedding direction to project out.
- **Result on `t_ng_04` (3B).** The output contains no "balanc" and passes all four checks. It restates the rule as
  "balace" instead.
- **Known over-breadth.** A short stem bans unrelated words. `water` bans ` waterfall`; `apples` → `appl` would ban
  ` apply`.

Verification results are in [results/demo_verification.json](results/demo_verification.json).

## Experimental history

The repository records three rounds, all on the same frozen checkpoints, calibration and held-out test set. The
[blog post](docs/blog-post.md) tells the story; [RESULTS.md](RESULTS.md) has every number.

- **Phase 1** covers Modules 1–4 and the 6-way ablation matrix (conditions 1–6, x1–x4). It reached 88.8 on 3B, a tie
  with URIAL at zero tokens. It also found the fluency-vs-drift trade-off in drift erasure.
- **Phase 2** covers conditions x5 and 7–9:
  - a p99 hinge threshold (Fluency 100%, but `[Question]` continuations leak back);
  - signed attention, which was inert;
  - unembedding projection, which moved negative-constraint ConstraintScore from 50% to 87.5% but is close to a lexical
    ban by construction.
- **Phase 3** covers condition 10 and the leave-one-out ablations x6–x9: the boundary-coupled hinge and sibling
  orthogonalisation.
- **Phase 4** covers the economics:
  - condition 11, the lean engine, which reproduces x9's projection byte for byte and decodes within 2% of
    condition 7;
  - conditions 12–13, the official Qwen2.5-Instruct checkpoints with and without the engine's lexical ban;
  - clean-token KL distortion, a base-vs-Instruct weight diff, and SDPA-attention and 512-token sensitivity runs.

Corrections recorded along the way (details in [RESULTS.md](RESULTS.md)):
- **bf16 eager attention costs Qwen2.5-1.5B its arithmetic** (found in Phase 4). Module 4 needs eager attention, which
  in bf16 rounds pre-softmax scores. Under it the 1.5B raw base solves 3/8 math prompts, against 8/8 under SDPA
  (Composite 75.6 vs. 81.2). 3B is unaffected (8/8 under both).
  - An earlier version of this note said the effect was the same for every condition. A follow-up with fp32 scores
    and Module 4 kept (`steer_fp32`, tags `test40_fp32attn` and `test40_512tok_fp32attn`) shows it is not.
  - At 512 tokens it costs 1.5B-Instruct 6.9 [3.1, 10.6] points (math 2/8 → 8/8) and the engine 2.5
    [−3.1, 8.1].
  - On a matched fp32 stack, Instruct leads the 1.5B engine by 6.9 [1.9, 12.5] at 512 tokens.
  - On 3B at 512 tokens the matched fp32 stack keeps parity by a pre-registered rule, but the engine's +1.9 lead
    becomes −1.9 [−6.2, +2.5] (paired change −3.8 [−7.5, −0.6]), and recovery falls from 112% to 91% [70%, 111%].
  - The protocol results stay primary; the coefficients were tuned on the bf16 stack.
- **Hash-seed bug.** The projection order in conditions 9 and 10 originally depended on the Python hash seed. It is now
  sorted by token id. Condition 9 was re-tuned and re-generated, and the earlier non-reproducible logs are kept as
  `results/*/test40/9_engine_v2_full.nondeterministic_order.jsonl.bak`.
- **Two withdrawn Phase-2 claims.** Both came from that run: that centring the unembedding was required, and the
  `t_ng_02` "Assembly Language" example.
- **Diagnoses in the briefs that the data did not support.** These are reported rather than adopted. For example, drift
  onset does not sit below the p99 threshold: the boundary newline is at 3.9σ, against a 3.1σ threshold.

## Repository layout

```text
frozen_assistant_engine/
├── README.md · RESULTS.md · AGENTS.md · CLAUDE.md · LICENSE · requirements*.txt
├── docs/                 blog-post.md, eli5-and-use-cases.md
├── data/                 calibration (80), dev (15) and held-out test (40) prompts with machine-checkable rules
├── engine/               model_loader, prompts (URIAL prefix, traps), calibration (fp64 solves), artifact,
│                         hooks (all runtime interventions + telemetry), negation (parser, stem ban),
│                         generator (greedy KV-cached loop), stream (token-by-token, interleavable decoder)
├── demo/                 live studio: server.py (FastAPI + SSE), data.py, static/ (HTML/CSS/JS); ./run_demo.sh
├── eval/                 verifiers (metrics), conditions (20 base + 4 Instruct), tune (dev search), benchmark, report,
│                         alignment_tax (clean-token KL)
├── configs/hparams.json  frozen coefficients with full tuning provenance (Phase 1–3; Phase 4 added none)
├── artifacts/<model>/    calibration.pt (steering tensors), trajectories.json, calibration_diagnostics.json
├── results/<model>/      per-condition JSONL generation logs, summary.json, analysis notes, tuning log,
│                         alignment_tax.json, speed_bench.json; <model>-Instruct/ holds conditions 12–13
├── scripts/              sanity_check.py, resolve_calibration.py, check_cache_reproduces.py, bench_speed.py,
│                         weight_delta.py, check_links.py, verify_demo.py
├── tests/                model-free pytest suite
└── cli.py                interactive side-by-side comparison
```

`calibration.pt` files are PyTorch pickles loaded with `weights_only=False`. Load only artifacts you trust.

## License

[MIT](LICENSE)
