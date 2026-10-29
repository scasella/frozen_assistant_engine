"""Zero-Weight-Update Assistant Studio: live 4-way arena with circuit telemetry, plus the Phase 1–4 data explorers.

    python -m demo.server                       # http://127.0.0.1:8000
    python -m demo.server --instruct swap       # never hold Base and Instruct in memory together

The four arena columns share one frozen base model (raw, static CAA, engine) plus the official Instruct checkpoint. Their
decoders take turns token by token (``engine.stream``); each engine's hooks are live only during its own forward pass.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import AssistantEngine, EngineConfig, SteeringArtifact, load_model
from engine import generator as _generator
from engine import hooks as _hooks
from engine import negation as _negation
from engine.prompts import EVAL_STOP_STRINGS
from engine.stream import stream_generate
from eval.conditions import ALL_CONDITIONS, INSTRUCT_CONDITIONS, encode_for, engine_config, load_hparams, relevant_config
from eval.verifiers import score_response

from . import data

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
HP = load_hparams(ROOT / "configs" / "hparams.json")
GB = 1024 ** 3


# ---------------------------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------------------------

def _checkpoint_bytes(name: str) -> int:
    """bf16 weight bytes of a checkpoint (from the local HF cache; parameter-count fallback)."""
    snap = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{name.replace('/', '--')}" / "snapshots"
    sizes = [f.stat().st_size for f in snap.glob("*/*.safetensors")] if snap.exists() else []
    return sum(sizes) or (6_200_000_000 if "3B" in name else 3_100_000_000)


def _family(name: str) -> str:
    return name.removesuffix("-Instruct")


class ModelManager:
    """Keeps one model family (Base + Instruct) resident when memory allows; otherwise swaps Instruct in on demand."""

    def __init__(self, instruct_policy: str = "auto"):
        self.models: dict[str, object] = {}
        self.artifacts: dict[str, SteeringArtifact] = {}
        self.requested = instruct_policy        # auto | resident | swap | cached
        self.policy: dict[str, str] = {}        # family -> resident | swap | cached
        self.loading: str | None = None
        self.events: list[str] = []
        self.lock = threading.RLock()

    def _note(self, msg: str) -> None:
        self.events = (self.events + [f"{time.strftime('%H:%M:%S')} {msg}"])[-20:]

    def get(self, name: str, keep: tuple[str, ...] = ()):
        """Load ``name`` if needed, evicting the other family and, if memory is short, models not in ``keep``."""
        with self.lock:
            if name in self.models:
                self.models[name] = self.models.pop(name)          # LRU order: most recent last
                return self.models[name]
            for other in [n for n in self.models if _family(n) != _family(name)]:
                self.unload(other)
            need = _checkpoint_bytes(name) * 1.08 + 1.0 * GB
            for other in [n for n in self.models if n != name and n not in keep]:
                if psutil.virtual_memory().available >= need:
                    break
                self.unload(other)
            self.loading = name
            self._note(f"loading {name}")
            try:
                lm = load_model(name)
            finally:
                self.loading = None
            self.models[name] = lm
            self._note(f"loaded {name} ({lm.n_layers} layers, {lm.device})")
            return lm

    def unload(self, name: str) -> None:
        with self.lock:
            lm = self.models.pop(name, None)
            if lm is None:
                return
            _forget_caches(lm)
            del lm
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            self._note(f"unloaded {name}")

    def artifact(self, base: str) -> SteeringArtifact:
        if base not in self.artifacts:
            self.artifacts[base] = SteeringArtifact.load(ROOT / "artifacts" / data.slug(base) / "calib80" / "calibration.pt")
        return self.artifacts[base]

    def instruct_policy(self, base: str) -> str:
        """resident: hold Base + Instruct together; swap: load Instruct after the base columns finish; cached: replay
        the logged Instruct generations (benchmark prompts only)."""
        if self.requested != "auto":
            return self.requested
        # re-evaluated on every request: right after a load, macOS reports less available memory than it can reclaim
        inst = base + "-Instruct"
        resident = sum(_checkpoint_bytes(n) for n in self.models if n != inst)
        need = _checkpoint_bytes(inst) * 1.08 + 1.5 * GB
        room = psutil.virtual_memory().available + (_checkpoint_bytes(inst) if inst in self.models else 0)
        policy = "resident" if inst in self.models or room >= need else "swap"
        if self.policy.get(base) != policy:
            self._note(f"Instruct policy for {base}: {policy} ({room / GB:.1f} GB available, "
                       f"{need / GB:.1f} GB needed beside {resident / GB:.1f} GB resident)")
        self.policy[base] = policy
        return policy

    def status(self) -> dict:
        vm = psutil.virtual_memory()
        mps = {}
        if torch.backends.mps.is_available():
            mps = {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}
        return {"ram": {"total": vm.total, "available": vm.available, "used": vm.total - vm.available,
                        "percent": vm.percent, "process_rss": psutil.Process().memory_info().rss},
                "mps": mps, "resident": [{"name": n, "bytes": _checkpoint_bytes(n)} for n in self.models],
                "loading": self.loading, "policy": dict(self.policy), "requested_policy": self.requested,
                "events": self.events[-8:]}


def _forget_caches(lm) -> None:
    """Drop every cache keyed by a tensor address or tokenizer id: a new model can reuse a freed address."""
    w = lm.model.get_output_embeddings().weight
    ptr, tid = w.data_ptr(), id(lm.tokenizer)
    _hooks._VOCAB_MEAN.pop(ptr, None)
    for d in (_hooks._WORD_MASK, _hooks._WORD_ROWS, _negation._CONTINUATIONS):
        for k in [k for k in d if k[0] == tid or ptr in k]:
            d.pop(k, None)
    _negation._VOCAB_TEXT.pop(tid, None)
    _generator._urial_cache.pop(tid, None)


MANAGER = ModelManager()
GPU = threading.Lock()
_current = {"cancel": threading.Event()}


# ---------------------------------------------------------------------------------------------
# Request schema and column construction
# ---------------------------------------------------------------------------------------------

class Modules(BaseModel):
    lifecycle: bool = True          # M1: initiation + sustain schedule, completion-centroid stop
    boundary_hinge: bool = True     # M2: hinge drift erasure with the boundary-coupled threshold
    predictor: bool = True          # M3: rank-16 input-dependent initiation shift
    pin_attention: bool = True      # M4: instruction-pinning attention bias
    unembed_ban: bool = True        # sibling-orthogonal unembedding projection
    stem: bool = True               # ban_morphology="stem"


class StreamRequest(BaseModel):
    model: str = "3B"
    prompt_id: str | None = None
    instruction: str | None = None
    columns: list[str] = Field(default_factory=lambda: ["raw", "caa", "engine", "instruct"])
    instruct_ban: bool = False
    instruct_frame: str = "chat"     # chat | bare
    modules: Modules = Field(default_factory=Modules)
    kappa_boundary: float = HP.get("kappa_boundary", 1.5)
    gamma_unembed: float = HP.get("gamma_unembed", 1.5)
    beta_attn: float = HP.get("beta_attn", 0.5)
    max_new_tokens: int = 160
    reference_kl: bool = True        # exact per-token KL against the unsteered model (one extra forward per token)
    replay: bool = False             # stream every column from the cached logs (benchmark prompts only; no GPU)


COND11 = engine_config(ALL_CONDITIONS["11_engine_v2_2_lean"][2], HP)
ORTH_BAN = engine_config(INSTRUCT_CONDITIONS["13_instruct_plus_orth_ban_chat"][2], HP)


def engine_column_config(req: StreamRequest) -> EngineConfig:
    m = req.modules
    return replace(COND11, lifecycle=m.lifecycle, dynamic_stop=m.lifecycle, predictor=m.predictor,
                   erase=m.boundary_hinge, boundary_hinge=m.boundary_hinge, pin_attention=m.pin_attention,
                   unembed_project=m.unembed_ban, ban_morphology="stem" if m.stem else "exact",
                   kappa_boundary=req.kappa_boundary, gamma_unembed=req.gamma_unembed, beta_attn=req.beta_attn)


def _same(cfg: EngineConfig, ref: EngineConfig) -> bool:
    return relevant_config(asdict(cfg)) == relevant_config(asdict(ref))


def _diff(cfg: EngineConfig, ref: EngineConfig) -> dict:
    a, b = relevant_config(asdict(cfg)), relevant_config(asdict(ref))
    return {k: [b.get(k), a.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def build_columns(req: StreamRequest) -> list[dict]:
    base = data.MODELS[req.model]
    inst = base + "-Instruct"
    cols = []
    for c in req.columns:
        if c == "raw":
            cols.append({"id": "raw", "label": "Raw base (unsteered)", "model": base, "frame": False, "stops": False,
                         "cfg": None, "condition": "1_raw_base"})
        elif c == "caa":
            cols.append({"id": "caa", "label": "Static steering (CAA)", "model": base, "frame": False, "stops": True,
                         "cfg": engine_config(ALL_CONDITIONS["4_static_caa"][2], HP), "condition": "4_static_caa"})
        elif c == "engine":
            cfg = engine_column_config(req)
            stem_only = replace(cfg, ban_morphology="exact")
            if _same(cfg, COND11):
                label, cond = "Engine v2.2 lean (condition 11)", "11_engine_v2_2_lean"
            elif _same(stem_only, COND11):
                label, cond = "Engine v2.3 demo (11 + stem ban)", None
            else:
                label, cond = "Engine (custom ablation)", None
            cols.append({"id": "engine", "label": label, "model": base, "frame": False, "stops": True, "cfg": cfg,
                         "condition": cond, "telemetry": True, "config_diff": _diff(cfg, COND11)})
        elif c == "instruct":
            frame = "chat" if req.instruct_frame == "chat" else False
            suffix = "chat" if frame == "chat" else "bare"
            if req.instruct_ban:
                cfg = replace(ORTH_BAN, ban_morphology="stem" if req.modules.stem else "exact",
                              gamma_unembed=req.gamma_unembed)
                cond = f"13_instruct_plus_orth_ban_{suffix}" if _same(cfg, ORTH_BAN) else None
                label = "Official Instruct + our ban" + ("" if cond else " (stem)")
            else:
                cfg, cond, label = None, f"12_official_instruct_{suffix}", "Official Instruct (post-trained)"
            cols.append({"id": "instruct", "label": label, "model": inst, "frame": frame, "stops": True, "cfg": cfg,
                         "condition": cond, "telemetry": cfg is not None})
    return cols


def prompt_record(req: StreamRequest) -> dict:
    if req.prompt_id:
        p = data.test_prompts().get(req.prompt_id)
        if p is None:
            raise HTTPException(404, f"unknown prompt {req.prompt_id}")
        if not req.instruction or req.instruction.strip() == p["instruction"]:
            return p
    if not req.instruction or not req.instruction.strip():
        raise HTTPException(400, "give a prompt_id or an instruction")
    text = req.instruction.strip()
    return {"id": "custom", "category": "custom", "instruction": text, "checks": data.infer_checks(text)}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, default=_json_default)}\n\n"


def _json_default(o):
    return str(o)


def _clean(obj):
    """Replace NaN/inf with None so the browser's JSON.parse accepts the payload."""
    if isinstance(obj, float):
        return obj if obj == obj and abs(obj) != float("inf") else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _prompt_view(lm, enc, cfg: EngineConfig | None, art: SteeringArtifact | None) -> dict:
    tok = lm.tokenizer
    inst = set(enc.inst_positions)
    pinned = bool(cfg and cfg.pin_attention and art is not None)
    return {"tokens": [tok.decode([i]) for i in enc.ids], "inst_positions": sorted(inst),
            "neg_clause_positions": enc.neg_clause_positions, "neg_target_positions": enc.neg_target_positions,
            "beta_attn": cfg.beta_attn if pinned else 0.0,
            "pin_heads": [list(h) for h in art.pin_heads] if pinned else [], "extra_tokens": enc.extra_tokens}


def _finish(col: dict, res, cut: int | None, prompt: dict, logged: dict | None, stems: list[str]) -> dict:
    m = score_response(res.text, res.stop_reason, prompt)
    out = {"type": "done", "col": col["id"], "text": res.text, "raw_text": res.raw_text,
           "cut_char": cut, "stop_reason": res.stop_reason, "stop_string": res.stop_string,
           "n_new_tokens": res.n_new_tokens, "latency_s": res.latency_s,
           "tok_s": res.n_new_tokens / res.latency_s if res.latency_s else None,
           "metrics": {k: m[k] for k in ("turn_cross", "constraint", "clean_stop", "fluency", "composite", "natural_eos")}
                      | {"constraint_detail": m["constraint_detail"], "contamination": m["contamination"],
                         "n_checks": len(prompt["checks"])},
           "contamination_spans": data.contamination_spans(res.raw_text),
           "stem_strict": data.stem_strict(res.text, stems), "engine_trace": res.engine_trace,
           "stop_events": res.stop_events}
    if logged is not None:
        out["logged"] = {"response": logged["response"], "composite": logged["metrics"]["composite"],
                         "identical": logged["response"] == res.text, "condition": col["condition"]}
    return out


def _replay(col: dict, rec: dict, tok, delay: float = 0.012) -> Iterator[dict]:
    """Stream a logged generation token by token (no telemetry), for cached mode."""
    ids = tok.encode(rec.get("raw_text", rec["response"]), add_special_tokens=False)
    for i, t in enumerate(ids):
        time.sleep(delay)
        yield {"type": "token", "step": i, "token_id": t, "token_str": tok.decode([t])}
    yield {"type": "replay_done", "rec": rec}


def run_arena(req: StreamRequest, cancel: threading.Event | None = None) -> Iterator[str]:
    prev = _current["cancel"]
    prev.set()                                     # a new run pre-empts the one in flight
    cancel = cancel or threading.Event()
    _current["cancel"] = cancel
    yield _sse({"type": "hello", "request": req.model_dump()})
    waited = 0.0
    while not GPU.acquire(timeout=0.25):
        waited += 0.25
        if cancel.is_set():
            yield _sse({"type": "cancelled"})
            return
        if waited % 2 < 0.25:
            yield _sse({"type": "status", "msg": "waiting for the previous generation to stop…"})
    try:
        yield from _run_locked(req, cancel)
    except HTTPException as e:
        yield _sse({"type": "error", "msg": e.detail})
    except Exception as e:  # surfaced in the UI; the server keeps running
        import traceback
        traceback.print_exc()
        yield _sse({"type": "error", "msg": f"{type(e).__name__}: {e}"})
    finally:
        GPU.release()


def _run_locked(req: StreamRequest, cancel: threading.Event) -> Iterator[str]:
    if req.model not in data.MODELS:
        raise HTTPException(400, f"model must be one of {list(data.MODELS)}")
    prompt = prompt_record(req)
    stems = data.negation_info(prompt["instruction"])["stems"]
    base = data.MODELS[req.model]
    cols = build_columns(req)
    max_new = int(min(max(req.max_new_tokens, 8), 512))
    tag = {160: "test40", 512: "test40_512tok"}.get(max_new)
    logs = data.logs(req.model, tag) if (tag and prompt["id"] != "custom") else {}
    policy = MANAGER.instruct_policy(base) if any(c["id"] == "instruct" for c in cols) else "resident"

    def logged(col: dict) -> dict | None:
        return logs.get(col["condition"], {}).get(prompt["id"]) if col["condition"] else None

    # cached mode: replay what can be replayed; columns without a log cannot run
    replay_ids = set()
    if req.replay or policy == "cached":
        replay_ids = {c["id"] for c in cols if (req.replay or c["id"] == "instruct") and logged(c) is not None}
        missing = [c["label"] for c in cols if (req.replay or c["id"] == "instruct") and c["id"] not in replay_ids]
        if missing:
            yield _sse({"type": "status", "msg": "no cached log for " + ", ".join(missing) +
                        " (custom prompt or non-logged settings): running live"})
    tok_src = None
    phases: list[list[dict]] = [[c for c in cols if c["id"] != "instruct" or policy == "resident" or c["id"] in replay_ids],
                                [c for c in cols if c["id"] == "instruct" and policy == "swap" and c["id"] not in replay_ids]]
    for phase in phases:
        if not phase or cancel.is_set():
            continue
        gens, meta = {}, {}
        for col in phase:
            if col["id"] in replay_ids:
                from transformers import AutoTokenizer
                tok_src = tok_src or AutoTokenizer.from_pretrained(col["model"])
                gens[col["id"]] = _replay(col, logged(col), tok_src)
                meta[col["id"]] = {"col": col, "lm": None}
                yield _sse({"type": "start", "col": col["id"], "label": col["label"], "model": col["model"],
                            "condition": col["condition"], "source": "cached log", "telemetry": False})
                continue
            if MANAGER.loading or col["model"] not in MANAGER.models:
                yield _sse({"type": "status", "msg": f"loading {col['model']} into unified memory…"})
            lm = MANAGER.get(col["model"], keep=tuple(c["model"] for c in phase))
            art = MANAGER.artifact(base) if col["model"] == base else None
            cfg = col["cfg"]
            eng = None
            if cfg is not None:
                cfg = replace(cfg)
                eng = AssistantEngine(lm, art, cfg, collect_telemetry=bool(col.get("telemetry")))
            enc = encode_for(lm.tokenizer, prompt["instruction"], col["frame"])
            ref_kl = bool(req.reference_kl and col.get("telemetry") and eng is not None)
            gens[col["id"]] = stream_generate(lm, enc, max_new, EVAL_STOP_STRINGS if col["stops"] else None, eng,
                                              lm.eos_ids or None, reference_kl=ref_kl)
            meta[col["id"]] = {"col": col, "lm": lm, "engine": eng}
            yield _sse({"type": "start", "col": col["id"], "label": col["label"], "model": col["model"],
                        "condition": col["condition"], "source": "live", "telemetry": bool(col.get("telemetry")),
                        "reference_kl": ref_kl, "config": relevant_config(asdict(cfg)) if cfg else None,
                        "config_diff": col.get("config_diff"), "prompt_view": _prompt_view(lm, enc, cfg, art),
                        "frame": col["frame"] or "bare", "forbidden": (eng.trace().get("forbidden_tokens") if eng else None),
                        "artifact": {"layers": art.layers, "stop_layer": art.layers[art.stop_k], "tau_stop": art.tau_stop,
                                     "hinge_kappa": [round(max(float(p[0]), cfg.hinge_kappa_floor), 3) for p in art.drift_p99],
                                     "kappa_boundary": cfg.kappa_boundary} if (art is not None and cfg is not None) else None})
        text = {k: "" for k in gens}
        active = list(gens)
        while active:
            if cancel.is_set():
                yield _sse({"type": "cancelled"})
                return
            for cid in list(active):
                ev = next(gens[cid])
                if ev["type"] == "token":
                    if not ev.get("eos"):
                        text[cid] += ev["token_str"]
                    spans = data.contamination_spans(text[cid])
                    payload = {"type": "token", "col": cid, "i": ev["step"], "id": ev["token_id"], "s": ev["token_str"],
                               "eos": bool(ev.get("eos")), "contam_from": spans[0][0] if spans else None}
                    if "stop_string" in ev:
                        payload["stop_string"] = ev["stop_string"]
                    if "telemetry" in ev:
                        payload["tel"] = _clean(ev["telemetry"])
                    yield _sse(payload)
                elif ev["type"] == "done":
                    yield _sse(_clean(_finish(meta[cid]["col"], ev["result"], ev["cut"], prompt,
                                              logged(meta[cid]["col"]), stems)))
                    active.remove(cid)
                elif ev["type"] == "replay_done":
                    rec = ev["rec"]
                    m = score_response(rec["response"], rec["stop_reason"], prompt)
                    raw = rec.get("raw_text", rec["response"])
                    cut = raw.find(rec["stop_string"]) if rec.get("stop_string") else None
                    yield _sse(_clean({"type": "done", "col": cid, "text": rec["response"], "raw_text": raw,
                                       "cut_char": cut, "stop_reason": rec["stop_reason"], "stop_string": rec.get("stop_string"),
                                       "n_new_tokens": rec["n_new_tokens"], "latency_s": rec.get("latency_s"),
                                       "tok_s": rec["n_new_tokens"] / rec["latency_s"] if rec.get("latency_s") else None,
                                       "metrics": {k: m[k] for k in ("turn_cross", "constraint", "clean_stop", "fluency",
                                                                     "composite", "natural_eos")}
                                                  | {"constraint_detail": m["constraint_detail"],
                                                     "contamination": m["contamination"], "n_checks": len(prompt["checks"])},
                                       "contamination_spans": data.contamination_spans(raw),
                                       "stem_strict": data.stem_strict(rec["response"], stems), "source": "cached log",
                                       "engine_trace": rec.get("engine_trace") or {}}))
                    active.remove(cid)
        if phase is phases[0] and phases[1] and policy == "swap":
            yield _sse({"type": "status", "msg": "swapping the Instruct checkpoint in (memory policy: swap)…"})
    yield _sse({"type": "end"})


# ---------------------------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------------------------

app = FastAPI(title="Zero-Weight-Update Assistant Studio")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    eco = {}
    for mk in data.MODELS:
        try:
            e = data.economics(mk)
            eco[mk] = {"artifact_bytes": e["artifact_bytes"], "weight_delta": e["weight_delta"],
                       "calibration_seconds": e["calibration_seconds"]}
        except FileNotFoundError:
            pass
    return {**MANAGER.status(), "busy": GPU.locked(), "models": data.MODELS, "footprint": eco,
            "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")}


@app.get("/api/prompts")
def prompts():
    return data.prompts_payload()


@app.get("/api/summary")
def summary(model: str = "3B"):
    if model not in data.MODELS:
        raise HTTPException(400, "unknown model")
    return data.economics(model)


@app.get("/api/ablation_data")
def ablation(model: str = "3B"):
    if model not in data.MODELS:
        raise HTTPException(400, "unknown model")
    return data.ablation_data(model)


@app.get("/api/logs")
def logs(model: str = "3B", tag: str = "test40"):
    if model not in data.MODELS or tag not in data.TAGS:
        raise HTTPException(400, "unknown model or tag")
    return {"model": data.MODELS[model], "tag": tag, "tag_label": data.TAGS[tag], "conditions": data.logs_lite(model, tag),
            "labels": {c: v[0] for c, v in data.LABELS.items()}}


@app.get("/api/config")
def config(model: str = "3B"):
    art = MANAGER.artifact(data.MODELS[model])
    return {"engine_defaults": relevant_config(asdict(replace(COND11, ban_morphology="stem"))),
            "condition_11": relevant_config(asdict(COND11)), "hparams_provenance": "frozen on the 1.5B dev set",
            "artifact": {"layers": art.layers, "stop_layer": art.layers[art.stop_k], "tau_stop": art.tau_stop,
                         "pin_heads": [list(h) for h in art.pin_heads], "caa_layer": art.caa_layer,
                         "hinge_p99": [round(float(p[0]), 3) for p in art.drift_p99],
                         "hinge_kappa_floor": COND11.hinge_kappa_floor, "pred_rank": int(art.pred_a.shape[-1])},
            "ranges": {"kappa_boundary": [0.5, 3.5, 0.1], "gamma_unembed": [0.0, 3.0, 0.1], "beta_attn": [0.0, 2.0, 0.05]}}


@app.post("/api/stream")
async def stream(req: StreamRequest):
    """The arena runs in its own thread to completion (or to its cancel flag); the response only drains a queue. A
    client that disconnects mid-stream sets the flag, so the GPU lock is always released (a suspended generator
    driven by the response itself would hold it until garbage collection)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def put(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:                       # event loop closed (server shutdown)
            cancel.set()

    def produce() -> None:
        try:
            for chunk in run_arena(req, cancel):
                put(chunk)
        finally:
            put(None)

    threading.Thread(target=produce, daemon=True, name="arena").start()

    async def body():
        try:
            while (chunk := await queue.get()) is not None:
                yield chunk
        finally:
            cancel.set()

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/cancel")
def cancel():
    _current["cancel"].set()
    return {"ok": True}


def preload(models: list[str]) -> None:
    for name in models:
        try:
            MANAGER.get(name)
            time.sleep(3)                                          # let the page cache settle before measuring
            if not name.endswith("-Instruct") and MANAGER.instruct_policy(name) == "resident":
                MANAGER.get(name + "-Instruct")
        except Exception as e:  # noqa: BLE001 - reported in /api/status
            MANAGER._note(f"preload of {name} failed: {e}")
    data.warm()


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--instruct", choices=["auto", "resident", "swap", "cached"], default="auto",
                    help="how the Instruct column gets memory (auto: resident if it fits, else swap)")
    ap.add_argument("--no-preload", action="store_true", help="load models on the first arena request")
    args = ap.parse_args()
    MANAGER.requested = args.instruct
    if not args.no_preload:
        threading.Thread(target=preload, args=(["Qwen/Qwen2.5-3B"],), daemon=True).start()
    else:
        threading.Thread(target=data.warm, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
