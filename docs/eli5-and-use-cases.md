# ELI5 Explainer & Practical Use Cases

The plain-English version of what this repository does, followed by where it could be useful. Each use case says
whether we actually measured it here or whether it is an untested extrapolation. For the numbers, see the
[README](../README.md) and [RESULTS.md](../RESULTS.md); for the full story, see the [blog post](blog-post.md).

## Part 1: Explain like I'm five

### Why base models ramble

A raw "base" language model has read a huge slice of the internet, but nobody has taught it to be an assistant. It is
like a brilliant student who answers your math question correctly in one sentence. Then, out of habit from reading
thousands of worksheets, it keeps writing: *Question 2…*, *Question 3…*, or a little multiple-choice quiz at the bottom
of the page ("Is the question answered in a satisfactory fashion? (a) yes (b) no").

The student already knows *how* to answer; in our tests it started answering every single time. What it doesn't know is
*when to put the pen down*.

### Why the usual "steering" trick breaks the brakes

One popular trick is activation steering: find a direction inside the model that means "sound like an assistant", and
add it at every word. That is like taping down the gas pedal. The model starts off nicely, but it can no longer find
the brake (the end-of-text token) when it is done. In our tests, its natural stopping rate fell from 75% to 42.5% on the
3B model and to 20% on the 1.5B model.

The other usual trick is to paste three example conversations in front of every question (few-shot prompting). It
works, but it is like making the student re-read three worked examples before every single question: 768 extra words of
reading each time.

### How the engine fixes this, without retraining and without extra words

1. **Tap the gas, then coast** (the lifecycle schedule). For the first four words we push the model firmly into
   "assistant mode", then let go. Because the push fades, the model's natural brakes still work, and it stops on its own
   about 80% of the time.
2. **Guard the full stop** (the boundary-coupled hinge). The worksheet habit almost always kicks in right after a period
   or a line break, the moment it is tempted to write *Question 2*. We measured this inside the model: mid-sentence the
   "worksheet" signal sits near zero, and at that line break it spikes. So we leave the model alone mid-sentence and
   erase the worksheet habit hard only at sentence and line boundaries. That one change gave the largest single
   improvement in the final round.
3. **Ban a word without banning its friends** (the sibling-orthogonalised projection). Say you ask for "programming
   languages without saying Python". Pushing "Python" away normally pushes away "Java" and "Ruby" too, because inside
   the model they live in the same neighbourhood. We first find which words the model wants to say next, then push
   "Python" down only in a direction that is at right angles to those words. Python drops, and its neighbours don't move
   at all: we measured exactly zero change.

Everything is computed from about 80 example questions, with no training. The result is a ~3 MB file of directions and
thresholds that the hooks read at run time.

### What it does not do

- **It is not magic for "don't say X" prompts.** On "explain bicycle riding without the word *balance*", the model
  sometimes dodges the task ("…without using the word *bicycle*, you can say…"). Our automatic checker even counts some
  of those dodges as passes.
- **It was tested on only two base models** (Qwen2.5 3B and 1.5B) and 40 test questions. On the smaller model most of
  the improvements are within statistical noise.
- **It is not a replacement for real post-training in general.** On our four checks, the 3B engine matches the
  official Qwen2.5-3B-Instruct when both get 512 tokens to answer (96.2 vs. 94.4). On 1.5B, Instruct is ahead (90.0 vs.
  87.5). The checks measure starting, stopping and following rules, not helpfulness or safety.

## Part 2: Practical use cases

| Use case | What we measured here | What is untested |
|---|---|---|
| **Zero-prefix inference.** Drop a few-shot or system prefix and keep assistant behaviour. | On 3B: 94.4% Composite against 84.4% for the 768-token URIAL prefix (+10.0 [+3.8, +16.2]); 3.25 s against 5.30 s per query (−39% wall time, which includes shorter answers); and no prefix in the KV cache, about **28 MB per query** saved in bf16 for this model. | Batched serving, other prompt formats, long multi-turn conversations. |
| **Training-free assistant behaviour on local hardware.** | Everything ran on an Apple M4 Pro (24 GB) in bf16. Calibration used 80 prompts and took ~21 min for 3B, producing a ~3 MB artifact. No gradients, no fine-tuning data. | Quantised checkpoints, domain-pretrained base models, CUDA/CPU runs, other architectures. |
| **Word bans without collateral damage, including on chat models.** | The sibling-orthogonalised projection leaves the top candidate words' relative logits unchanged (0.00) where a naive ban moved them by −3.6 to −13.7. Removing it cost −6.9 [−13.1, −1.9] Composite on 1.5B. Bolted onto the official Qwen2.5-3B-**Instruct** with no re-tuning, it raised "don't say X" compliance from 37.5% to 87.5%, breaking nothing on 3B. On 1.5B-Instruct it collapsed one answer into a loop. | Larger chat models, sampling, and bans the parser does not recognise. |
| **Stopping post-answer drift in math and code generation.** | On 3B arithmetic prompts, the boundary-coupled hinge took Composite from 87.5% (p99 hinge alone) to 96.9%, by erasing FLAN `[Question]` continuations at line boundaries. | Specialised math or code base models and longer outputs. Some drift genres still slip through (`------` separators, `[Reasoning: …]` tags). |

## Where to go next

- **Run it yourself:** `python cli.py --model Qwen/Qwen2.5-3B --modes raw,caa,v21` (see the [README](../README.md) for
  setup).
- **See every example output** in [RESULTS.md](../RESULTS.md) §5–6.
- **Hacking on the code:** read [AGENTS.md](../AGENTS.md) first. It lists the invariants that keep the experiments
  honest.
