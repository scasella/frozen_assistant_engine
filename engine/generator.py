"""Greedy decoding loop with lifecycle step tracking, stop strings and dynamic turn-stopping."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch
from transformers import DynamicCache

from .hooks import AssistantEngine
from .negation import char_spans_to_tokens, parse_negations
from .prompts import ANSWER_DELIM, QUESTION_PREFIX, URIAL_PREFIX


@dataclass
class EncodedPrompt:
    ids: list[int]
    inst_positions: list[int]   # token positions overlapping the {instruction} span
    extra_tokens: int           # tokens spent on any static prefix (URIAL)
    neg_clause_positions: list[int] = field(default_factory=list)  # tokens of negation clauses ("without ... 'force'")
    neg_target_positions: list[int] = field(default_factory=list)  # tokens of the negated targets themselves
    neg_targets: list[str] = field(default_factory=list)           # negated lexical targets ("force", "Python")
    neg_echo_words: list[str] = field(default_factory=list)        # negation trigger words ("without", "mentioning")


@dataclass
class GenerationResult:
    text: str                   # response after stop-string truncation, stripped
    raw_text: str               # everything the model generated before truncation
    response_ids: list[int]     # generated ids kept after truncation (trailing whitespace tokens dropped)
    n_new_tokens: int
    stop_reason: str            # eos | stop_string | max_tokens
    stop_string: str | None
    prompt_tokens: int
    latency_s: float
    ms_per_token: float
    stop_events: list[dict] = field(default_factory=list)
    engine_trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("response_ids")
        return d


_urial_cache: dict[int, list[int]] = {}


def encode_prompt(tokenizer, instruction: str, urial: bool = False) -> EncodedPrompt:
    """Tokenise ``Question: {instruction}\\nAnswer:`` naturally and locate the instruction tokens.

    The URIAL prefix ends in a blank line and the question starts with a letter, so tokenising the
    two pieces separately and concatenating ids is identical to joint tokenisation
    (asserted in ``scripts/sanity_check.py``).
    """
    text = f"{QUESTION_PREFIX}{instruction}{ANSWER_DELIM}"
    enc = tokenizer(text, add_special_tokens=not urial, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    start, end = len(QUESTION_PREFIX), len(QUESTION_PREFIX) + len(instruction)
    inst = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end and b > a]
    negs = parse_negations(instruction)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    neg_clause = char_spans_to_tokens(offsets, [n.clause for n in negs], start)
    neg_target = char_spans_to_tokens(offsets, [(a, b) for n in negs for a, b, _ in n.targets], start)
    targets = [t for n in negs for _, _, t in n.targets]
    echo_words = sorted({w for n in negs for w in n.echo_words})
    extra = 0
    if urial:
        key = id(tokenizer)
        if key not in _urial_cache:
            _urial_cache[key] = list(tokenizer(URIAL_PREFIX, add_special_tokens=True)["input_ids"])
        prefix = _urial_cache[key]
        extra = len(prefix)
        ids = prefix + ids
        inst = [i + extra for i in inst]
        neg_clause = [i + extra for i in neg_clause]
        neg_target = [i + extra for i in neg_target]
    return EncodedPrompt(ids=ids, inst_positions=inst, extra_tokens=extra, neg_clause_positions=neg_clause,
                         neg_target_positions=neg_target, neg_targets=targets, neg_echo_words=echo_words)


def _kept_ids(tokenizer, ids: list[int], char_limit: int | None) -> list[int]:
    """Longest prefix of ``ids`` whose decoded text fits within ``char_limit``; trailing blanks dropped."""
    j = len(ids)
    if char_limit is not None:
        while j > 0 and len(tokenizer.decode(ids[:j])) > char_limit:
            j -= 1
    kept = ids[:j]
    while kept and tokenizer.decode(kept[-1:]).strip() == "":
        kept.pop()
    return kept


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate(
    lm,
    prompt: EncodedPrompt,
    max_new_tokens: int = 160,
    stop_strings: list[str] | None = None,
    engine: AssistantEngine | None = None,
) -> GenerationResult:
    model, tok, dev = lm.model, lm.tokenizer, lm.device
    stop_strings = stop_strings or []
    cache = DynamicCache(config=model.config)
    input_ids = torch.tensor([prompt.ids], device=dev)
    if engine is not None:
        engine.begin(prompt, len(prompt.ids) + max_new_tokens + 1)

    generated: list[int] = []
    stop_reason, matched, cut = "max_tokens", None, None
    _sync(dev)
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    step = 0
    while True:
        logits = out.logits[0, -1]
        if engine is not None:
            logits = engine.process_logits(logits)
        next_id = int(torch.argmax(logits).item())
        if next_id == lm.eos_token_id:
            stop_reason = "eos"
            break
        generated.append(next_id)
        if stop_strings:
            text = tok.decode(generated)
            hits = [(text.find(s), s) for s in stop_strings if s in text]
            if hits:
                cut, matched = min(hits)
                stop_reason = "stop_string"
                break
        if len(generated) >= max_new_tokens:
            break
        step += 1
        if engine is not None:
            engine.set_step(step, tok.decode([next_id]))
        out = model(input_ids=torch.tensor([[next_id]], device=dev), past_key_values=cache, use_cache=True)
    _sync(dev)
    elapsed = time.perf_counter() - t0
    if engine is not None:
        engine.end()

    raw = tok.decode(generated)
    text = raw[:cut] if cut is not None else raw
    n = max(len(generated), 1)
    return GenerationResult(
        text=text.strip(),
        raw_text=raw,
        response_ids=_kept_ids(tok, generated, cut),
        n_new_tokens=len(generated),
        stop_reason=stop_reason,
        stop_string=matched,
        prompt_tokens=len(prompt.ids),
        latency_s=elapsed,
        ms_per_token=1000.0 * elapsed / n,
        stop_events=list(engine.stop_events) if engine is not None else [],
        engine_trace=engine.trace() if engine is not None else {},
    )
