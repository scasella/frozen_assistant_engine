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

On 3B, **condition 10** gains:
- **+13.8 [+7.5, +20.0]** over the raw base;
- **+10.0 [+3.8, +16.2]** over the 768-token URIAL prefix, the first engine whose lead over URIAL is statistically
  resolved;
- +5.6 [+0.0, +11.2] over the Phase-1 engine.

It answers a query in 3.25 s on average, against 5.30 s for URIAL (an M4 Pro, including prefill).

† `x9` is a leave-one-out ablation (condition 10 without the anti-echo guard), and it was found *on the test set*. It
is reported for attribution only. Treating 96.2 as a result would be test-set selection; it needs fresh held-out
validation first.

**What the numbers do and do not say**

- **The gain is termination and genre drift, not turn-crossing.** Every condition scores TurnCross = 100%: Qwen2.5
  base models already start answering after `Answer:`. What they do wrong is keep going after a correct answer, with
  FLAN-style "A single-select problem… Available options: (a). yes" blocks, new `[Question]` problems and hallucinated
  system prompts. 16 of 40 raw+stops responses are contaminated.
- **Static steering breaks the brakes.** A single vector added at every token lowers natural `<eos>` from 75% to 42.5%
  (3B) and from 77.5% to 20% (1.5B). The engine's decaying schedule keeps it at 80%.
- **On 1.5B the picture is weaker.** Condition 10 is the best there too (86.2), but only its margin over the raw base is
  resolved (+10.6 [+2.5, +18.8]). Over the Phase-1 engine it is +1.9 [−5.0, +8.1].
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
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --conditions 10_engine_v2_1 --tag test40
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --tag mynewrun --all    # all 19 conditions; a new tag = fresh run
```

Recalibrate. Teacher and raw trajectories are cached in `artifacts/`, so only the forward passes re-run. This
overwrites `artifacts/<model>/calib80/calibration.pt` in place:

```bash
.venv/bin/python -m eval.benchmark --model Qwen/Qwen2.5-3B --recalibrate --conditions 1_raw_base --tag recal
.venv/bin/python scripts/resolve_calibration.py --model Qwen/Qwen2.5-3B   # re-solve from a local states.pt cache (no forward passes)
```

Devices: `engine/model_loader.py` picks `mps`, then `cuda`, then `cpu`; override with `--device`. All reported runs used
an Apple M4 Pro (24 GB) with `mps` and bfloat16. CUDA and CPU paths are untested.

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

Corrections recorded along the way (details in [RESULTS.md](RESULTS.md)):
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
│                         hooks (all runtime interventions), negation (parser), generator (greedy KV-cached loop)
├── eval/                 verifiers (metrics), conditions (19 conditions), tune (dev search), benchmark, report
├── configs/hparams.json  frozen coefficients with full tuning provenance (Phase 1–3)
├── artifacts/<model>/    calibration.pt (steering tensors), trajectories.json, calibration_diagnostics.json
├── results/<model>/      per-condition JSONL generation logs, summary.json, analysis notes, tuning log
├── scripts/              sanity_check.py, resolve_calibration.py, check_links.py
├── tests/                model-free pytest suite
└── cli.py                interactive side-by-side comparison
```

`calibration.pt` files are PyTorch pickles loaded with `weights_only=False`. Load only artifacts you trust.

## License

[MIT](LICENSE)
