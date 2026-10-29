"""Offline calibration of the 4-module engine.

Forward passes run on the model's device (MPS) under ``torch.no_grad``; every collected hidden state is
immediately moved to CPU float64 and all covariance / SVD / ridge / projection math happens there.
Nothing here computes a gradient or touches a weight.

Stages
  1. trajectories : URIAL-framed teacher answers Y+ and raw unframed continuations Y- (cached as JSON)
  2. states       : teacher-forced passes A (URIAL + prompt + Y+), B (prompt + Y+), N (prompt + Y-) and
                    three negative traps; compact per-prompt statistics are accumulated on CPU
  3. solve        : v_init, v_sustain^perp, completion centroid / v_stop / tau, drift subspace U (SVD),
                    low-rank ridge predictor (A, B), static-CAA vector, instruction-pinning heads
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .artifact import SteeringArtifact
from .generator import encode_prompt, generate
from .hooks import ResidualRecorder, attention_module
from .model_loader import LoadedModel, free_memory
from .prompts import (CALIB_STOP_STRINGS, trap_exam_template, trap_forum_boilerplate, trap_multi_turn,
                      trap_question_continuation)

F64 = torch.float64
_PUNCT_ENDINGS = (".", "!", "?", "\n", "`")
CONSTRAINT_CATEGORIES = {"format", "negative"}


@dataclass
class CalibConfig:
    n_target_layers: int = 8
    layer_band: tuple[float, float] = (0.35, 0.80)
    head_band: tuple[float, float] = (0.30, 0.75)
    sustain_window: tuple[int, int] = (4, 40)
    attn_window: tuple[int, int] = (1, 32)
    caa_window: tuple[int, int] = (0, 40)
    pos_max_new_tokens: int = 256
    neg_max_new_tokens: int = 64
    trap_tokens_cap: int = 16
    clean_tokens_cap: int = 24
    trap_b_prefix_tokens: int = 24
    drift_rank: int = 8
    erase_kappa: float = 1.0      # dead zone: drift coordinates within mean + kappa*std of clean tokens are untouched
    drift_min_z: float = 0.5      # a drift direction is active only if trap tokens sit >= this many clean stds away
    ridge_lambda: float = 10.0
    ridge_select_cv: bool = True  # pick lambda per layer from RIDGE_GRID by 5-fold CV R^2 on the calibration prompts
    pred_rank: int = 16
    pin_frac: float = 0.05
    stop_min_step: int = 8
    stop_quantile: float = 0.95
    sustain_source: str = "ctx"   # ctx: URIAL vs unframed on the same Y+ tokens; fail: Y+ (URIAL) vs Y- (raw)
    seed: int = 0


def select_layers(n_layers: int, k: int, lo: float, hi: float) -> list[int]:
    a, b = math.floor(lo * n_layers), math.floor(hi * n_layers)
    return sorted({round(a + (b - a) * i / (k - 1)) for i in range(k)})


def head_band_layers(n_layers: int, lo: float, hi: float) -> list[int]:
    return list(range(math.floor(lo * n_layers), math.floor(hi * n_layers) + 1))


def distinct_ngram_ratio(text: str, n: int = 4) -> float:
    words = text.lower().split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 if not grams else len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------------------------
# Stage 1: trajectories
# ---------------------------------------------------------------------------------------------

def collect_trajectories(lm: LoadedModel, prompts: list[dict], cfg: CalibConfig, path: Path) -> list[dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        cache = {r["id"]: r for r in json.loads(path.read_text())}
    todo = [p for p in prompts if p["id"] not in cache]
    for n, p in enumerate(tqdm(todo, desc="trajectories", disable=not todo)):
        inst = p["instruction"]
        pos = generate(lm, encode_prompt(lm.tokenizer, inst, urial=True), cfg.pos_max_new_tokens, CALIB_STOP_STRINGS)
        neg = generate(lm, encode_prompt(lm.tokenizer, inst), cfg.neg_max_new_tokens, None)
        cache[p["id"]] = {
            "id": p["id"], "category": p["category"], "instruction": inst,
            "pos_text": pos.text, "pos_ids": pos.response_ids, "pos_stop": pos.stop_reason,
            "neg_text": neg.raw_text, "neg_ids": neg.response_ids,
        }
        if n % 8 == 7 or n == len(todo) - 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(list(cache.values()), indent=1))
    return [cache[p["id"]] for p in prompts]


def keep_trajectory(tr: dict) -> bool:
    """Usable teacher answer: non-trivial and not degenerate (it may still have hit the token cap)."""
    return len(tr["pos_text"]) > 10 and len(tr["pos_ids"]) >= 8 and distinct_ngram_ratio(tr["pos_text"]) >= 0.75


def ended_cleanly(tr: dict) -> bool:
    """Only answers the teacher finished on its own contribute completion-centroid / stop statistics."""
    return tr["pos_stop"] in ("stop_string", "eos")


# ---------------------------------------------------------------------------------------------
# Stage 2: hidden-state statistics
# ---------------------------------------------------------------------------------------------

class AttnMassRecorder:
    """Per-head attention mass from response query positions onto the instruction key span."""

    def __init__(self, lm: LoadedModel, layer_ids: list[int]):
        self.lm, self.layer_ids = lm, layer_ids
        self.q: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self.mass: dict[int, torch.Tensor] = {}

    def arm(self, q_positions: list[int], k_positions: list[int]) -> None:
        dev = self.lm.device
        self.q = torch.as_tensor(q_positions, device=dev, dtype=torch.long)
        self.k = torch.as_tensor(k_positions, device=dev, dtype=torch.long)
        self.mass = {}
        for l in self.layer_ids:
            attention_module(self.lm.layers[l])._attn_recorder = self._make(l)

    def _make(self, layer_id: int):
        def rec(_module, w: torch.Tensor) -> None:
            sub = w[0].index_select(1, self.q).index_select(2, self.k)        # [H, |q|, |inst|]
            self.mass[layer_id] = sub.float().sum(-1).mean(-1).to("cpu").to(F64)  # [H]
        return rec

    def disarm(self) -> None:
        for l in self.layer_ids:
            attention_module(self.lm.layers[l])._attn_recorder = None


def _window_mean(h: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Mean of rows t in [lo, hi] (row index == step t); falls back to all rows >= 1 if empty."""
    hi = min(hi, h.shape[0] - 1)
    if hi >= lo:
        return h[lo:hi + 1].mean(0)
    return h[1:].mean(0) if h.shape[0] > 1 else h[0]


def _spaced(positions: list[int], cap: int) -> list[int]:
    if len(positions) <= cap:
        return positions
    step = len(positions) / cap
    return [positions[int(i * step)] for i in range(cap)]


@torch.no_grad()
def collect_states(lm: LoadedModel, trajs: list[dict], layers: list[int], head_layers: list[int],
                   cfg: CalibConfig) -> dict:
    tok = lm.tokenizer
    decoder = lm.decoder()
    urial_ids = encode_prompt(tok, "x", urial=True)
    urial_ids = urial_ids.ids[: urial_ids.extra_tokens]
    off = len(urial_ids)
    rng = random.Random(cfg.seed)
    all_insts = [t["instruction"] for t in trajs]
    rec = ResidualRecorder(lm.layers, layers)
    attn = AttnMassRecorder(lm, head_layers)

    per = {l: defaultdict(list) for l in layers}   # per-prompt vectors (float64)
    rows = {l: defaultdict(list) for l in layers}  # variable-length row sets (float32)
    head_delta: list[torch.Tensor] = []

    def run(ids: list[int]) -> None:
        decoder(input_ids=torch.tensor([ids], device=lm.device), use_cache=False)

    def toks(text: str) -> list[int]:
        return list(tok(text, add_special_tokens=False)["input_ids"])

    for tr in tqdm(trajs, desc="states"):
        enc = encode_prompt(tok, tr["instruction"])
        P, R, N = enc.ids, tr["pos_ids"], tr["neg_ids"]
        p0, T, Tn = len(P) - 1, len(tr["pos_ids"]), len(tr["neg_ids"])
        constraint = tr["category"] in CONSTRAINT_CATEGORIES
        a_lo, a_hi = cfg.attn_window

        # Pass B: unframed prompt + clean answer (and head attention for constraint prompts)
        if constraint:
            attn.arm([p0 + t for t in range(a_lo, min(a_hi, T) + 1)], enc.inst_positions)
        run(P + R)
        hB = {l: rec.grab(l, range(p0, p0 + T + 1)) for l in layers}
        mass_b = dict(attn.mass)
        attn.disarm()
        # Pass N: unframed prompt + raw base continuation
        use_n = constraint and Tn >= a_lo
        if use_n:
            attn.arm([p0 + t for t in range(a_lo, min(a_hi, Tn) + 1)], enc.inst_positions)
        run(P + N)
        n_rows = min(Tn, cfg.caa_window[1])
        hN = {l: rec.grab(l, range(p0, p0 + n_rows + 1)) for l in layers}
        mass_n = dict(attn.mass)
        attn.disarm()
        if use_n and mass_b and mass_n:
            head_delta.append(torch.stack([mass_b[l] - mass_n[l] for l in head_layers]))
        # Pass A: URIAL + prompt + clean answer
        run(urial_ids + P + R)
        hA = {l: rec.grab(l, range(off + p0, off + p0 + T + 1)) for l in layers}

        # Traps (all in the unframed context the engine runs in)
        others = rng.sample([q for q in all_insts if q != tr["instruction"]], 3)
        trap_ids = {
            "A": (P, toks(trap_question_continuation(others, rng))),
            "B": (P + R[: min(T, cfg.trap_b_prefix_tokens)], toks(trap_forum_boilerplate(others, rng))),
            "C": (P + R, toks(trap_multi_turn(others))),
            "D": (P + R, toks(trap_exam_template(others, rng))),
        }
        trap_h: dict[str, dict[int, torch.Tensor]] = {}
        for name, (ctx, tail) in trap_ids.items():
            run(ctx + tail)
            pos = list(range(len(ctx), len(ctx) + len(tail)))
            pos = _spaced(pos, cfg.trap_tokens_cap)
            trap_h[name] = {l: rec.grab(l, pos) for l in layers}

        # Non-final punctuation steps inside the clean answer (negatives for the stop threshold)
        finished = ended_cleanly(tr)
        punct_t = [t for t in range(cfg.stop_min_step, T)
                   if tok.decode(R[t - 1: t]).rstrip(" \t").endswith(_PUNCT_ENDINGS)] if finished else []

        s_lo, s_hi = cfg.sustain_window
        c_lo, c_hi = cfg.caa_window
        for l in layers:
            a, b, n = hA[l], hB[l], hN[l]
            d = per[l]
            d["t0_a"].append(a[0])
            d["t0_b"].append(b[0])
            d["sus_a"].append(_window_mean(a, s_lo, s_hi))
            d["sus_b"].append(_window_mean(b, s_lo, s_hi))
            d["sus_n"].append(_window_mean(n, s_lo, s_hi))
            d["caa_a"].append(a[c_lo: min(c_hi, T) + 1].mean(0))
            d["caa_n"].append(n[c_lo:].mean(0))
            clean = b[1:].mean(0)
            d["clean_b"].append(clean)
            if finished:
                d["final_b"].append(b[T])
                # self-centred completion features: state minus the running mean of the answer so far
                d["final_b_rc"].append(b[T] - b[1:T].mean(0))
            d["trap_c"].append(trap_h["C"][l].mean(0))
            d["norm_b"].append(b[1:].norm(dim=-1).mean())
            if punct_t:
                rows[l]["punct_b"].append(b[punct_t].float())
                rows[l]["punct_b_rc"].append(torch.stack([b[t] - b[1:t].mean(0) for t in punct_t]).float())
            rows[l]["clean_tok"].append(b[_spaced(list(range(1, T + 1)), cfg.clean_tokens_cap)].float())
            for name in ("A", "B", "C", "D"):
                # one row per (prompt, trap): token-identity variance averages out, register shift remains
                rows[l]["drift"].append((trap_h[name][l].mean(0) - clean)[None].float())
                rows[l]["trap_tok"].append(trap_h[name][l].float())
        free_memory()

    rec.remove()
    stacked = {l: {k: torch.stack(v) for k, v in per[l].items()} for l in layers}
    flat = {l: {k: torch.cat(v).to(F64) for k, v in rows[l].items()} for l in layers}
    heads = torch.stack(head_delta).mean(0) if head_delta else None
    return {"per": stacked, "rows": flat, "head_delta": heads, "n_head_prompts": len(head_delta)}


# ---------------------------------------------------------------------------------------------
# Stage 3: closed-form solves (CPU float64)
# ---------------------------------------------------------------------------------------------

def _unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _cos(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return (_unit(x) * _unit(e)).sum(-1)


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).double() + 0.5 * (diff == 0).double()).mean())


def ridge_low_rank(x: torch.Tensor, y: torch.Tensor, lam: float, rank: int):
    """W = (Xc^T Xc + lam I)^-1 Xc^T Yc, computed in the equivalent dual form Xc^T (Xc Xc^T + lam I)^-1 Yc,
    then truncated to rank k by SVD:  W ~= A B^T  with A = U_k S_k, B = V_k."""
    xbar, ybar = x.mean(0), y.mean(0)
    xc, yc = x - xbar, y - ybar
    n = xc.shape[0]
    alpha = torch.linalg.solve(xc @ xc.T + lam * torch.eye(n, dtype=F64), yc)   # [N, d]
    q, r = torch.linalg.qr(xc.T)                                                # W = Q (R alpha)
    um, s, vh = torch.linalg.svd(r @ alpha, full_matrices=False)
    k = min(rank, s.shape[0])
    a = (q @ um[:, :k]) * s[:k]
    b = vh[:k].T
    return xbar, ybar, a, b, s


def ridge_cv(x: torch.Tensor, y: torch.Tensor, lams: list[float], rank: int, folds: int = 5) -> dict:
    n = x.shape[0]
    folds = min(folds, n)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    out = {}
    for lam in [None, *lams]:
        cos_sum, sse, sst = 0.0, 0.0, 0.0
        for f in range(folds):
            te = idx[f::folds]
            held = set(te.tolist())
            tr = torch.tensor([i for i in idx.tolist() if i not in held])
            ybar = y[tr].mean(0)
            if lam is None:
                pred = ybar.expand(len(te), -1)
            else:
                xbar, ybar, a, b, _ = ridge_low_rank(x[tr], y[tr], lam, rank)
                pred = ybar + ((x[te] - xbar) @ a) @ b.T
            cos_sum += float(_cos(pred, y[te]).sum())
            sse += float(((pred - y[te]) ** 2).sum())
            sst += float(((y[te] - ybar) ** 2).sum())
        out["mean_only" if lam is None else f"ridge_{lam:g}"] = {
            "cv_cosine": cos_sum / n, "cv_r2_vs_train_mean": 1.0 - sse / max(sst, 1e-12)}
    return out


RIDGE_GRID = [10.0, 1e2, 1e3, 1e4, 1e5]


def solve(states: dict, layers: list[int], head_layers: list[int], n_heads: int, cfg: CalibConfig,
          meta: dict) -> SteeringArtifact:
    K = len(layers)
    v_init, v_sus, v_stop, mu_tok, eos_dir = [], [], [], [], []
    u_drift, centers, scales, p99s, xbars, pas, pbs, norms = [], [], [], [], [], [], [], []
    stop_auc, stop_tau, layer_diag = [], [], []

    for l in layers:
        s, r = states["per"][l], states["rows"][l]
        # Module 1a: turn-crossing initiation vector at the delimiter (t = 0)
        vi = (s["t0_a"] - s["t0_b"]).mean(0)
        # Module 1b: sustain vector over t in [4, 40], orthogonalised against v_init
        vs_ctx = (s["sus_a"] - s["sus_b"]).mean(0)
        vs_fail = s["sus_a"].mean(0) - s["sus_n"].mean(0)
        vs = vs_ctx if cfg.sustain_source == "ctx" else vs_fail
        vs_perp = vs - (vs @ vi) / (vi @ vi + 1e-8) * vi
        # Module 1c: completion centroid. Features are self-centred (state minus the running mean of the
        # answer so far) so that constant steering offsets cancel at run time; a global-mean variant is
        # reported for comparison.
        mu = s["clean_b"].mean(0)
        e = _unit(_unit(s["final_b_rc"]).mean(0))
        pos_scores = _cos(s["final_b_rc"], e)
        neg_scores = _cos(r["punct_b_rc"], e) if "punct_b_rc" in r else torch.empty(0, dtype=F64)
        auc = _auc(pos_scores, neg_scores)
        e_g = _unit(_unit(s["final_b"] - mu).mean(0))
        auc_global = _auc(_cos(s["final_b"] - mu, e_g), _cos(r["punct_b"] - mu, e_g)) if "punct_b" in r else float("nan")
        tau = float(torch.quantile(neg_scores, cfg.stop_quantile)) if len(neg_scores) else 0.5
        vstop = s["final_b"].mean(0) - s["trap_c"].mean(0)
        # Module 2: drift subspace from (prompt, trap) mean-difference rows, clean directions projected out
        q, _ = torch.linalg.qr(torch.stack([vi, vs_perp], dim=1))
        drift = r["drift"]
        drift = drift - (drift @ q) @ q.T
        _, sv, vh = torch.linalg.svd(drift, full_matrices=False)
        U = vh[: cfg.drift_rank].T.contiguous()                      # [d, r], orthonormal columns
        sign = torch.sign((drift @ U).mean(0))
        U = U * torch.where(sign == 0, torch.ones_like(sign), sign)  # orient: traps lie on the + side
        clean_coord = r["clean_tok"] @ U
        c_mean, c_std = clean_coord.mean(0), clean_coord.std(0) + 1e-6
        trap_z = (r["trap_tok"] @ U - c_mean) / c_std
        active = trap_z.mean(0) >= cfg.drift_min_z               # keep only directions that actually separate
        c_std = torch.where(active, c_std, torch.full_like(c_std, 1e6))   # inactive -> dead zone never reached
        clean_z = (clean_coord - c_mean) / c_std
        p99 = torch.quantile(clean_z, 0.99, dim=0)                # v2 hinge threshold per direction
        hinge_k = torch.clamp(p99, min=2.2)
        # Module 3: low-rank ridge predictor of the URIAL shift at t = 0
        cv = ridge_cv(s["t0_b"], s["t0_a"] - s["t0_b"], RIDGE_GRID, cfg.pred_rank)
        lam = cfg.ridge_lambda
        if cfg.ridge_select_cv:
            lam = max(RIDGE_GRID, key=lambda g: cv[f"ridge_{g:g}"]["cv_r2_vs_train_mean"])
        xbar, ybar, pa, pb, psv = ridge_low_rank(s["t0_b"], s["t0_a"] - s["t0_b"], lam, cfg.pred_rank)

        hn = float(s["norm_b"].mean())
        v_init.append(vi); v_sus.append(vs_perp); v_stop.append(vstop); mu_tok.append(mu); eos_dir.append(e)
        u_drift.append(U); centers.append(c_mean); scales.append(c_std); p99s.append(p99)
        xbars.append(xbar); pas.append(pa); pbs.append(pb)
        norms.append(torch.tensor(hn, dtype=F64)); stop_auc.append(auc); stop_tau.append(tau)
        layer_diag.append({
            "layer": l,
            "hidden_norm": hn,
            "v_init_norm": float(vi.norm()), "v_init_rel": float(vi.norm()) / hn,
            "v_sustain_norm": float(vs.norm()), "v_sustain_perp_norm": float(vs_perp.norm()),
            "cos_init_sustain": float(_cos(vs, vi)), "cos_sustain_ctx_vs_fail": float(_cos(vs_ctx, vs_fail)),
            "v_stop_norm": float(vstop.norm()),
            "stop_auc_final_vs_midpunct": auc, "stop_auc_global_centering": auc_global, "tau_stop": tau,
            "stop_tpr_at_tau": float((pos_scores > tau).double().mean()),
            "drift_rows": int(drift.shape[0]),
            "drift_explained_var_top_r": float((sv[: cfg.drift_rank] ** 2).sum() / (sv ** 2).sum()),
            "drift_clean_overlap": float(torch.linalg.norm(U.T @ q)),
            "drift_trap_tokens_mean_z": [round(float(x), 2) for x in trap_z.mean(0)],
            "drift_active_directions": int(active.sum()),
            "drift_frac_clean_tokens_touched": float((clean_z > cfg.erase_kappa).any(-1).double().mean()),
            "drift_frac_trap_tokens_touched": float((trap_z[:, active] > cfg.erase_kappa).any(-1).double().mean()),
            "hinge_kappa_p99": [round(float(x), 2) for x in p99[active]],
            "hinge_frac_clean_tokens_touched": float((clean_z[:, active] > hinge_k[active]).any(-1).double().mean()),
            "hinge_frac_trap_tokens_touched": float((trap_z[:, active] > hinge_k[active]).any(-1).double().mean()),
            "predictor_top_singular_values": [float(x) for x in psv[:4]],
            "predictor_cv": cv, "predictor_lambda": lam,
        })

    stop_k = max(range(K), key=lambda k: (stop_auc[k] if not math.isnan(stop_auc[k]) else -1, k))
    caa_k = K // 2
    sc = states["per"][layers[caa_k]]
    v_caa = sc["caa_a"].mean(0) - sc["caa_n"].mean(0)

    pin_heads: list[tuple[int, int]] = []
    head_diag: dict = {}
    if states["head_delta"] is not None:
        hd = states["head_delta"]                                   # [n_head_layers, H]
        n_pick = max(1, round(cfg.pin_frac * hd.numel()))
        vals, flat_idx = torch.topk(hd.flatten(), n_pick)
        pin_heads = [(head_layers[int(i) // n_heads], int(i) % n_heads) for i, v in zip(flat_idx, vals) if v > 0]
        head_diag = {
            "n_prompts": states["n_head_prompts"], "candidates": hd.numel(),
            "selected": [{"layer": l, "head": h, "delta": float(hd[head_layers.index(l), h])} for l, h in pin_heads],
            "delta_mean": float(hd.mean()), "delta_std": float(hd.std()),
        }

    return SteeringArtifact(
        layers=layers,
        v_init=torch.stack(v_init), v_sustain=torch.stack(v_sus), v_stop=torch.stack(v_stop),
        mu_tok=torch.stack(mu_tok), eos_dir=torch.stack(eos_dir),
        u_drift=torch.stack(u_drift), drift_center=torch.stack(centers), drift_scale=torch.stack(scales),
        drift_p99=torch.stack(p99s),
        pred_xbar=torch.stack(xbars), pred_a=torch.stack(pas), pred_b=torch.stack(pbs),
        hidden_norm=torch.stack(norms), stop_k=stop_k, tau_stop=stop_tau[stop_k],
        caa_layer=layers[caa_k], v_caa=v_caa, pin_heads=pin_heads,
        diagnostics={"layers": layer_diag, "stop_layer": layers[stop_k], "caa_layer": layers[caa_k],
                     "caa_norm": float(v_caa.norm()), "pin_heads": head_diag},
        meta=meta,
    )


def calibrate(lm: LoadedModel, prompts: list[dict], cfg: CalibConfig, out_dir: Path) -> SteeringArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = select_layers(lm.n_layers, cfg.n_target_layers, *cfg.layer_band)
    head_layers = head_band_layers(lm.n_layers, *cfg.head_band)
    trajs = collect_trajectories(lm, prompts, cfg, out_dir / "trajectories.json")
    free_memory()
    kept = [t for t in trajs if keep_trajectory(t)]
    n_end = sum(ended_cleanly(t) for t in kept)
    print(f"[calibration] kept {len(kept)}/{len(trajs)} teacher trajectories ({n_end} ended cleanly); layers={layers}")
    # Cache the collected statistics so solver changes never require new forward passes.
    collect_keys = ("n_target_layers", "layer_band", "head_band", "sustain_window", "attn_window", "caa_window",
                    "pos_max_new_tokens", "neg_max_new_tokens", "trap_tokens_cap", "clean_tokens_cap",
                    "trap_b_prefix_tokens", "stop_min_step", "seed")
    sig = json.dumps({k: getattr(cfg, k) for k in collect_keys} | {"ids": [t["id"] for t in kept]}, default=str)
    cache_path = out_dir / "states.pt"
    states = None
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("sig") == sig:
            states = cached["states"]
            print("[calibration] reusing cached hidden-state statistics")
    if states is None:
        states = collect_states(lm, kept, layers, head_layers, cfg)
        torch.save({"sig": sig, "states": states}, cache_path)
    free_memory()
    meta = {
        "model": lm.name, "n_prompts": len(trajs), "n_kept": len(kept), "n_ended_cleanly": n_end, "n_layers": lm.n_layers,
        "d_model": lm.d_model, "n_heads": lm.n_heads, "target_layers": layers, "head_band": head_layers,
        "config": asdict(cfg),
    }
    art = solve(states, layers, head_layers, lm.n_heads, cfg, meta)
    del states
    free_memory()
    art.save(out_dir / "calibration.pt")
    return art
