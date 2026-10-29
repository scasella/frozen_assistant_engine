"""Token-by-token ``generate`` for live demos: several decoders can take turns on one frozen model.

``stream_generate`` runs exactly the arithmetic of ``generator.generate`` (same prefill, same per-step engine calls, same
stop rules) but yields after every token. An engine's hooks are live only around this decoder's own forward passes
(``engine.resume()`` / ``engine.suspend()``), so a raw decoder, a static-CAA decoder and the full engine can be
interleaved on the same model without seeing each other's steering. ``scripts/verify_demo.py`` checks the outputs
against the cached benchmark logs, both alone and interleaved.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch
from transformers import DynamicCache

from .generator import EncodedPrompt, GenerationResult, _kept_ids
from .hooks import AssistantEngine

_LOG_FLOOR = -27.6  # log(1e-12): a masked token has probability 0, so its KL term is reported against this floor


def step_kl(ref_logits: torch.Tensor, logits: torch.Tensor) -> tuple[float, bool]:
    """D_KL(P_ref || P) in nats for one next-token distribution; flags whether a masked (-inf) token carried P_ref mass."""
    lp_ref = torch.log_softmax(ref_logits.float(), -1)
    lp = torch.log_softmax(logits.float(), -1)
    floored = bool(torch.isinf(lp).any())
    lp = lp.clamp(min=_LOG_FLOOR)
    return float((lp_ref.exp() * (lp_ref - lp)).sum()), floored


@torch.no_grad()
def stream_generate(
    lm,
    prompt: EncodedPrompt,
    max_new_tokens: int = 160,
    stop_strings: list[str] | None = None,
    engine: AssistantEngine | None = None,
    eos_ids: tuple[int, ...] | None = None,
    reference_kl: bool = False,
) -> Iterator[dict]:
    """Yield ``{"type": "token", ...}`` per chosen token (the EOS choice included, flagged ``eos``), then
    ``{"type": "done", "result": GenerationResult}``.

    ``reference_kl`` teacher-forces the same tokens through the model with the engine suspended (its own KV cache) and
    reports D_KL(P_unsteered || P_steered) for every step: the exact per-token distortion, at one extra forward pass per
    token. ``latency_s`` counts only this decoder's own forward passes and token choice.
    """
    model, tok, dev = lm.model, lm.tokenizer, lm.device
    stop_strings = stop_strings or []
    stop_ids = set(eos_ids) if eos_ids else {lm.eos_token_id}
    cache = DynamicCache(config=model.config)
    ref_cache = DynamicCache(config=model.config) if reference_kl else None
    if engine is not None:
        engine.begin(prompt, len(prompt.ids) + max_new_tokens + 1)
    telemetry = engine is not None and engine.collect_telemetry

    def forward(ids: list[int], kv: DynamicCache, steer: bool, **kw):
        if steer and engine is not None:
            engine.resume()
        try:
            return model(input_ids=torch.tensor([ids], device=dev), past_key_values=kv, use_cache=True, **kw)
        finally:
            if steer and engine is not None:
                engine.suspend()

    generated: list[int] = []
    stop_reason, matched, cut = "max_tokens", None, None
    busy = 0.0
    t0 = time.perf_counter()
    out = forward(prompt.ids, cache, True, logits_to_keep=1)
    ref_ids: list[int] = list(prompt.ids)
    step = 0
    while True:
        model_logits = out.logits[0, -1]
        logits = engine.process_logits(model_logits) if engine is not None else model_logits
        next_id = int(torch.argmax(logits).item())
        busy += time.perf_counter() - t0
        ev: dict = {"type": "token", "step": step, "token_id": next_id, "token_str": tok.decode([next_id])}
        if telemetry:
            ev["telemetry"] = engine.pop_telemetry(model_logits, logits)
        if ref_cache is not None:
            ref = forward(ref_ids, ref_cache, False, logits_to_keep=1)
            kl, floored = step_kl(ref.logits[0, -1], logits)
            ev.setdefault("telemetry", {}).update(step_kl_nats=kl, **({"step_kl_floored": True} if floored else {}))
            ref_ids = [next_id]
        if next_id in stop_ids:
            stop_reason = "eos"
            ev["eos"] = True
            yield ev
            break
        generated.append(next_id)
        if stop_strings:
            text = tok.decode(generated)
            hits = [(text.find(s), s) for s in stop_strings if s in text]
            if hits:
                cut, matched = min(hits)
                stop_reason = "stop_string"
                ev["stop_string"] = matched
                yield ev
                break
        if len(generated) >= max_new_tokens:
            yield ev
            break
        yield ev
        t0 = time.perf_counter()
        step += 1
        if engine is not None:
            engine.set_step(step, tok.decode([next_id]))
        out = forward([next_id], cache, True)
    if engine is not None:
        engine.end()

    raw = tok.decode(generated)
    text = raw[:cut] if cut is not None else raw
    n = max(len(generated), 1)
    yield {"type": "done", "result": GenerationResult(
        text=text.strip(),
        raw_text=raw,
        response_ids=_kept_ids(tok, generated, cut),
        n_new_tokens=len(generated),
        stop_reason=stop_reason,
        stop_string=matched,
        prompt_tokens=len(prompt.ids),
        latency_s=busy,
        ms_per_token=1000.0 * busy / n,
        stop_events=list(engine.stop_events) if engine is not None else [],
        engine_trace=engine.trace() if engine is not None else {},
    ), "cut": cut}
