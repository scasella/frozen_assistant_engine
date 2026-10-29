"""Runtime interventions on a frozen model.

* ``steer_eager`` attention: numerically identical to HF eager attention, plus an optional
  per-query-head pre-softmax bias (Module 4) and an optional attention-weight recorder (head discovery).
* ``AssistantEngine``: residual-stream forward hooks implementing Module 1 (lifecycle schedule +
  dynamic stop), Module 2 (drift nullspace erasure), Module 3 (single-pass self-contrastive predictor),
  and the static-CAA baseline.
* ``ResidualRecorder``: captures decoder-layer outputs during calibration forward passes.

All weights stay frozen; every hook only edits activations in the forward pass.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_utils import AttentionInterface

from .artifact import SteeringArtifact
from .negation import forbidden_token_ids

STEER_ATTN = "steer_eager"
_PUNCT_ENDINGS = (".", "!", "?", "\n", "`")


# ---------------------------------------------------------------------------------------------
# Attention: eager + pre-softmax bias (Module 4) + recorder
# ---------------------------------------------------------------------------------------------

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def steer_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rep = query.shape[1] // key.shape[1]
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
    bias: torch.Tensor | None = getattr(module, "_steer_bias", None)
    if bias is not None:
        # bias: [1, n_query_heads, 1, max_len] -> broadcast over query rows, sliced to kv length
        attn_weights = attn_weights + bias[..., : attn_weights.shape[-1]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    recorder: Callable | None = getattr(module, "_attn_recorder", None)
    if recorder is not None:
        recorder(module, attn_weights)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def install_steerable_attention(model: nn.Module) -> None:
    """Route every attention layer through ``steer_eager`` (eager math, with bias/recorder slots)."""
    AttentionInterface.register(STEER_ATTN, steer_eager_attention_forward)
    # Without a registered mask function HF silently skips the causal mask for unknown backends.
    AttentionMaskInterface.register(STEER_ATTN, eager_mask)
    model.config._attn_implementation = STEER_ATTN


def attention_module(layer: nn.Module) -> nn.Module:
    for name in ("self_attn", "attn", "attention", "self_attention"):
        mod = getattr(layer, name, None)
        if isinstance(mod, nn.Module):
            return mod
    raise RuntimeError(f"No attention submodule found on {type(layer).__name__}")


_VOCAB_MEAN: dict[int, torch.Tensor] = {}
_WORD_MASK: dict[tuple[int, int], torch.Tensor] = {}


def _word_mask(tokenizer, vocab_rows: int, device) -> torch.Tensor:
    """True for vocabulary entries whose decoded text contains an ASCII letter (word tokens, not punctuation)."""
    key = (id(tokenizer), vocab_rows)
    if key not in _WORD_MASK:
        mask = torch.zeros(vocab_rows, dtype=torch.bool)
        for i in range(min(len(tokenizer), vocab_rows)):
            if re.search(r"[A-Za-z]", tokenizer.decode([i])):
                mask[i] = True
        _WORD_MASK[key] = mask.to(device)
    return _WORD_MASK[key]


@torch.no_grad()
def _vocab_mean(weight: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Mean unembedding row in float32, computed in chunks (no full-size upcast copy) and cached per weight."""
    key = weight.data_ptr()
    if key not in _VOCAB_MEAN:
        acc = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
        for i in range(0, weight.shape[0], chunk):
            acc += weight[i:i + chunk].float().sum(0)
        _VOCAB_MEAN[key] = acc / weight.shape[0]
    return _VOCAB_MEAN[key]


def _hidden_of(output: object) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output  # type: ignore[index,return-value]


# ---------------------------------------------------------------------------------------------
# Calibration recorder
# ---------------------------------------------------------------------------------------------

class ResidualRecorder:
    """Keeps the most recent output of each watched decoder layer (batch 0) on-device."""

    def __init__(self, layers: nn.ModuleList, layer_ids: list[int]):
        self.out: dict[int, torch.Tensor] = {}
        self._handles = [layers[l].register_forward_hook(self._make(l)) for l in layer_ids]

    def _make(self, layer_id: int):
        def hook(_module, _args, output):
            self.out[layer_id] = _hidden_of(output)[0].detach()
        return hook

    def grab(self, layer_id: int, positions: list[int] | torch.Tensor) -> torch.Tensor:
        """Selected positions -> CPU float64 [n, d] (all linear algebra happens on CPU in fp64)."""
        h = self.out[layer_id]
        idx = torch.as_tensor(positions, device=h.device, dtype=torch.long)
        return h.index_select(0, idx).to("cpu").to(torch.float64)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.out.clear()


# ---------------------------------------------------------------------------------------------
# Runtime engine
# ---------------------------------------------------------------------------------------------

@dataclass
class EngineConfig:
    # module toggles
    lifecycle: bool = True        # Module 1: init (t<init_steps) + sustain (t>=1)
    dynamic_stop: bool = True     # Module 1: completion-centroid stop check + EOS boost
    erase: bool = True            # Module 2: drift nullspace erasure
    predictor: bool = True        # Module 3: input-dependent init shift at t=0
    pin_attention: bool = False   # Module 4: instruction-pinning attention bias
    static_caa: bool = False      # baseline: single-vector CAA at every position
    # Module 1
    alpha_init: float = 1.0
    gamma: float = 0.4
    init_steps: int = 4
    alpha_sustain: float = 0.35
    split_budget: bool = True     # divide additive vectors by #target layers (residual stream accumulates)
    # dynamic stop
    alpha_stop: float = 1.0
    stop_min_step: int = 8
    stop_logit_boost: float = 5.0
    newturn_logprob: float = -1.5  # after a newline, stop if logp("Question"/"User"/...) exceeds this
    # Module 2
    lambda_erase: float = 0.7
    erase_kappa: float = 1.0      # dead zone (in clean-token std units) before erasure engages
    erase_mode: str = "v1"        # v1: global kappa=erase_kappa, lambda_erase | hinge: per-direction p99 kappa (>= floor), lambda_hinge
    lambda_hinge: float = 0.75
    hinge_kappa_floor: float = 2.2
    # Module 3
    predictor_weight: float = 1.0
    predictor_clip: float = 1.0   # ||predicted correction|| <= clip * ||v_init||
    # Module 4
    beta_attn: float = 1.25
    signed_attention: bool = False  # v2: +beta on task tokens, -beta_neg_attn on the negated span
    beta_neg_attn: float = 1.0
    neg_span_mode: str = "clause"   # clause: trigger..end of clause ("without using the word 'force'") | target: target tokens only
    # v2: one-sided projection of the pre-LM-head state off negated targets' unembedding directions
    unembed_project: bool = False
    gamma_unembed: float = 2.0
    unembed_center: bool = True     # subtract the vocabulary-mean unembedding before normalising (see README)
    # v2.1: boundary-coupled hinge, drift-assisted stop, sibling-orthogonal projection, anti-echo guard
    boundary_hinge: bool = False    # lower the hinge threshold to kappa_boundary inside sentence/line boundary windows
    kappa_boundary: float = 1.0
    boundary_after_newline: int = 2  # steps after a newline that still count as the boundary window
    boundary_score_margin: float = 0.08  # ... or when the (previous step's) completion score >= tau_stop - margin
    drift_stop: bool = False        # stop also when the stop-layer drift z >= tau_drift_stop and score >= tau_stop - margin
    tau_drift_stop: float = 1.8
    drift_stop_score_margin: float = 0.12
    sibling_orth: bool = False      # orthogonalise banned directions against the top-K favoured clean word tokens
    sibling_k: int = 8
    sibling_min_overlap: float = 0.3
    anti_echo: bool = False         # also suppress the negation trigger words during the first anti_echo_window tokens
    anti_echo_window: int = 12
    # static CAA
    alpha_caa: float = 1.0

    def any_residual(self) -> bool:
        return self.lifecycle or self.dynamic_stop or self.erase or self.predictor or self.static_caa


class AssistantEngine:
    """Attaches forward hooks to a frozen model and steers it according to a per-step schedule.

    Step convention: t = 0 is the prefill pass whose last position is the prompt delimiter
    (``Answer:``); t >= 1 are single-token decode passes processing generated token t.
    """

    def __init__(self, lm, artifact: SteeringArtifact, cfg: EngineConfig):
        self.lm = lm
        self.cfg = cfg
        self.art = artifact
        dev = lm.device
        f32 = torch.float32
        self.layer_ids = list(artifact.layers)
        self.k_of = {l: k for k, l in enumerate(self.layer_ids)}
        self.v_init = artifact.v_init.to(dev, f32)
        self.v_sus = artifact.v_sustain.to(dev, f32)
        self.v_stop = artifact.v_stop.to(dev, f32)
        self.mu_tok = artifact.mu_tok.to(dev, f32)
        self.eos_dir = artifact.eos_dir.to(dev, f32)
        self.u = artifact.u_drift.to(dev, f32)
        self.u_t = self.u.transpose(1, 2).contiguous()
        self.c = artifact.drift_center.to(dev, f32)
        self.c_sd = artifact.drift_scale.to(dev, f32)
        self.hinge_kappa = torch.clamp(artifact.drift_p99.to(dev, f32), min=cfg.hinge_kappa_floor)
        self.xbar = artifact.pred_xbar.to(dev, f32)
        self.pa = artifact.pred_a.to(dev, f32)
        self.pb_t = artifact.pred_b.to(dev, f32).transpose(1, 2).contiguous()
        self.v_init_norm = self.v_init.norm(dim=-1)
        self.v_caa = artifact.v_caa.to(dev, lm.dtype)
        self.scale = 1.0 / len(self.layer_ids) if cfg.split_budget else 1.0
        self.pin_by_layer: dict[int, list[int]] = {}
        for l, h in artifact.pin_heads:
            self.pin_by_layer.setdefault(int(l), []).append(int(h))

        tok = lm.tokenizer
        self.newturn_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Question", "User", "Q", "Human")})
        self._handles: list = []
        self._attn_bias: dict[int, torch.Tensor] = {}
        self.t = 0
        self.check_stop = False
        self.stop_triggered = False
        self.last_token_text = ""
        self.stop_events: list[dict] = []
        self._run_sum: torch.Tensor | None = None   # running sum of stop-layer states (self-centring)
        self._run_n = 0
        self._forbid: torch.Tensor | None = None    # [n, d] unit unembedding rows of negated target tokens
        self._forbid_ids: list[int] = []
        self._echo: torch.Tensor | None = None      # [m, d] unit rows of negation trigger words (anti-echo)
        self._echo_ids: list[int] = []
        self._trace: dict = {}
        self._since_newline = 10 ** 9               # steps since the last generated token containing a newline
        self._last_score: float | None = None       # stop-layer completion score at the previous step
        self.in_boundary = False

    # -- lifecycle -----------------------------------------------------------------------------
    def attach(self) -> "AssistantEngine":
        layers = self.lm.layers
        if self.cfg.static_caa:
            self._handles.append(layers[self.art.caa_layer].register_forward_hook(self._caa_hook))
        if self.cfg.lifecycle or self.cfg.erase or self.cfg.predictor or self.cfg.dynamic_stop:
            for l in self.layer_ids:
                self._handles.append(layers[l].register_forward_hook(self._make_hook(self.k_of[l])))
        if self.cfg.unembed_project:
            head = self.lm.model.get_output_embeddings()
            self._handles.append(head.register_forward_pre_hook(self._unembed_hook))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._set_attn_bias(False)
        self._attn_bias.clear()

    def __enter__(self) -> "AssistantEngine":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def begin(self, prompt, max_len: int) -> None:
        """Reset per-generation state; pre-build Module-4 bias tensors and the negated-target directions."""
        cfg = self.cfg
        self.t = 0
        self.check_stop = False
        self.stop_triggered = False
        self.stop_events = []
        self._run_sum, self._run_n = None, 0
        self._attn_bias.clear()
        self._forbid, self._forbid_ids, self._echo, self._echo_ids = None, [], None, []
        self._trace = {}
        self._since_newline, self._last_score, self.in_boundary = 10 ** 9, None, False
        inst = list(prompt.inst_positions)
        neg = prompt.neg_clause_positions if cfg.neg_span_mode == "clause" else prompt.neg_target_positions
        neg = [i for i in neg if i in set(inst)] if cfg.signed_attention else []
        pos_set = [i for i in inst if i not in set(neg)]
        if cfg.pin_attention and self.pin_by_layer and inst:
            for l, heads in self.pin_by_layer.items():
                bias = torch.zeros(1, self.lm.n_heads, 1, max_len, dtype=self.lm.dtype)
                h_idx = torch.as_tensor(heads)[:, None]
                if pos_set:
                    bias[0, h_idx, 0, torch.as_tensor(pos_set)[None, :]] = cfg.beta_attn
                if neg:
                    bias[0, h_idx, 0, torch.as_tensor(neg)[None, :]] = -cfg.beta_neg_attn
                self._attn_bias[l] = bias.to(self.lm.device)
            if neg:
                self._trace["neg_span_tokens"] = self.lm.tokenizer.decode([prompt.ids[i] for i in neg])
        if cfg.unembed_project and prompt.neg_targets:
            ids = forbidden_token_ids(self.lm.tokenizer, prompt.neg_targets)
            if ids:
                self._forbid_ids = list(ids)
                self._forbid = self._unit_rows(self._forbid_ids)
                self._trace["forbidden_tokens"] = sorted(ids.values())
                self._trace["unembed_projections"] = 0
        if cfg.unembed_project and cfg.anti_echo and prompt.neg_echo_words:
            echo = forbidden_token_ids(self.lm.tokenizer, prompt.neg_echo_words)
            echo = {i: s for i, s in echo.items() if i not in set(self._forbid_ids)}
            if echo:
                self._echo_ids = list(echo)
                self._echo = self._unit_rows(self._echo_ids)
                self._trace["anti_echo_tokens"] = sorted(echo.values())
        self._set_attn_bias(False)

    def _unit_rows(self, ids: list[int]) -> torch.Tensor:
        weight = self.lm.model.get_output_embeddings().weight
        w = weight[ids].float()
        if self.cfg.unembed_center:
            w = w - _vocab_mean(weight)
        return w / (w.norm(dim=-1, keepdim=True) + 1e-8)

    def trace(self) -> dict:
        return dict(self._trace)

    def set_step(self, t: int, last_token_text: str) -> None:
        self.t = t
        self.last_token_text = last_token_text
        self.stop_triggered = False
        stripped = last_token_text.rstrip(" \t")
        self.check_stop = self.cfg.dynamic_stop and t >= self.cfg.stop_min_step and stripped.endswith(_PUNCT_ENDINGS)
        self._set_attn_bias(t >= 1)
        self._since_newline = 0 if "\n" in last_token_text else self._since_newline + 1
        if self.cfg.boundary_hinge:
            # the completion score of the *current* step is only known at the stop layer, after most hinge
            # layers have run, so the score clause uses the previous step's score
            near_end = self._last_score is not None and self._last_score >= self.art.tau_stop - self.cfg.boundary_score_margin
            self.in_boundary = (any(ch in last_token_text for ch in ".!?\n")
                                or self._since_newline <= self.cfg.boundary_after_newline or near_end)
            if self.in_boundary:
                self._trace["boundary_steps"] = self._trace.get("boundary_steps", 0) + 1

    def end(self) -> None:
        self._set_attn_bias(False)
        self._attn_bias.clear()
        self._forbid = None

    def _set_attn_bias(self, on: bool) -> None:
        for l in self.pin_by_layer:
            attention_module(self.lm.layers[l])._steer_bias = self._attn_bias.get(l) if on else None

    # -- logits --------------------------------------------------------------------------------
    def process_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Dynamic turn-stopping: boost EOS when the completion check fired at this step."""
        if not self.check_stop:
            return logits
        reason = (self.stop_triggered if isinstance(self.stop_triggered, str) else "centroid") if self.stop_triggered else None
        if reason is None and self.last_token_text.endswith("\n"):
            logp = torch.log_softmax(logits.float(), dim=-1)
            newturn = logp[self.newturn_ids].max().item()
            if newturn > self.cfg.newturn_logprob:
                reason = "newturn"
        if reason is None:
            return logits
        logits = logits.clone()
        logits[self.lm.eos_token_id] += self.cfg.stop_logit_boost
        self.stop_events.append({"t": self.t, "reason": reason})
        return logits

    # -- pre-LM-head projection (v2) -------------------------------------------------------------
    def _unembed_hook(self, module, args):
        """For each banned token v with <h, w_v> > 0: h <- h - gamma <h, w_v> w_v (last position only).

        v2.1 (sibling_orth): the removal direction is w_v orthogonalised against the top-K currently favoured clean
        word tokens, rescaled so <h, w_v> still shrinks by exactly (1 - gamma) while those tokens' centred logits
        are unchanged. anti_echo adds the negation trigger words during the first anti_echo_window steps.
        """
        cfg = self.cfg
        banned, banned_ids = self._forbid, list(self._forbid_ids)
        if cfg.anti_echo and self._echo is not None and self.t < cfg.anti_echo_window:
            banned = self._echo if banned is None else torch.cat([banned, self._echo])
            banned_ids += self._echo_ids
        if banned is None:
            return None
        x = args[0].clone()
        h = x[:, -1, :].float()
        q = self._sibling_basis(module.weight, h, banned_ids) if cfg.sibling_orth else None
        fired, fallback = 0, 0
        for w in banned:
            a = h @ w                                              # [B]
            on = a > 0
            d, scale = w, torch.ones_like(a)
            if q is not None:
                wt = w - q @ (q.T @ w)
                n = wt.norm()
                if float(n) >= cfg.sibling_min_overlap:            # <w~/|w~|, w> = |w~| for unit w
                    d, scale = wt / n, 1.0 / n.expand_as(a)
                else:
                    fallback += 1
            h = h - cfg.gamma_unembed * torch.where(on, a * scale, torch.zeros_like(a))[:, None] * d[None]
            fired += int(on.sum())
        x[:, -1, :] = h.to(x.dtype)
        self._trace["unembed_projections"] = self._trace.get("unembed_projections", 0) + fired
        if fallback:
            self._trace["sibling_fallbacks"] = self._trace.get("sibling_fallbacks", 0) + fallback
        return (x, *args[1:])

    def _sibling_basis(self, weight: torch.Tensor, h: torch.Tensor, banned_ids: list[int]) -> torch.Tensor:
        """Orthonormal basis [d, K] of the centred unembedding rows of the top-K favoured clean word tokens."""
        logits = (h[:1].to(weight.dtype) @ weight.T).float()[0]
        logits = logits.masked_fill(~_word_mask(self.lm.tokenizer, weight.shape[0], weight.device), float("-inf"))
        logits[torch.as_tensor(banned_ids, device=logits.device)] = float("-inf")
        top = torch.topk(logits, self.cfg.sibling_k).indices
        rows = weight[top].float() - _vocab_mean(weight)
        basis = []
        for r in rows:                                             # modified Gram-Schmidt (stays on device)
            for b in basis:
                r = r - (r @ b) * b
            n = r.norm()
            if float(n) > 1e-6:
                basis.append(r / n)
        return torch.stack(basis, dim=1)

    # -- residual hooks ------------------------------------------------------------------------
    def _caa_hook(self, _module, _args, output):
        h = _hidden_of(output)
        h.add_(self.cfg.alpha_caa * self.v_caa)
        return output

    def _make_hook(self, k: int):
        def hook(_module, _args, output):
            h = _hidden_of(output)
            h[:, -1, :] = self._steer_last(k, h[:, -1, :].float()).to(h.dtype)
            return output
        return hook

    def _steer_last(self, k: int, x: torch.Tensor) -> torch.Tensor:
        cfg, t, s = self.cfg, self.t, self.scale
        if k == self.art.stop_k and t >= 1:
            every_step = cfg.boundary_hinge or cfg.drift_stop
            if (self.check_stop or every_step) and self._run_n > 0:
                z = x - self._run_sum / self._run_n              # state minus running mean of this answer
                score = torch.nn.functional.cosine_similarity(z, self.eos_dir[k][None], dim=-1)[0].item()
                self._last_score = score
                if self.check_stop:
                    tau = self.art.tau_stop
                    self.stop_triggered = "centroid" if score > tau else False
                    ev = {"t": t, "score": round(score, 4)}
                    if cfg.drift_stop and not self.stop_triggered:
                        zd = float((((x @ self.u[k]) - self.c[k]) / self.c_sd[k]).max())
                        ev["drift_z"] = round(zd, 2)
                        if zd >= cfg.tau_drift_stop and score >= tau - cfg.drift_stop_score_margin:
                            self.stop_triggered = "drift"
                    self.stop_events.append(ev)
            self._run_sum = x.clone() if self._run_sum is None else self._run_sum + x
            self._run_n += 1
        delta = torch.zeros_like(x)
        if cfg.erase and t >= 1:
            z = x @ self.u[k] - self.c[k]                        # [B, r] drift coordinates vs clean centroid
            if cfg.erase_mode == "hinge":
                # v2: z-scored, per-direction threshold at the clean p99 (>= floor); subtract only the excess
                zs = z / self.c_sd[k]
                kappa = self.hinge_kappa[k]
                if cfg.boundary_hinge and self.in_boundary:
                    kappa = torch.clamp(kappa, max=cfg.kappa_boundary)  # v2.1: catch drift onsets at boundaries
                excess = torch.relu(zs - kappa) * self.c_sd[k]
                delta -= cfg.lambda_hinge * (excess @ self.u_t[k])
            else:
                excess = torch.relu(z - cfg.erase_kappa * self.c_sd[k])  # one-sided dead zone: only drift-ward excursions
                delta -= cfg.lambda_erase * (excess @ self.u_t[k])
        if cfg.lifecycle and t < cfg.init_steps:
            v = self.v_init[k].expand_as(x)
            if cfg.predictor and t == 0:
                corr = ((x - self.xbar[k]) @ self.pa[k]) @ self.pb_t[k]
                n = corr.norm(dim=-1, keepdim=True)
                cap = cfg.predictor_clip * self.v_init_norm[k]
                corr = corr * torch.clamp(cap / (n + 1e-8), max=1.0)
                v = v + cfg.predictor_weight * corr
            delta += cfg.alpha_init * (cfg.gamma ** t) * s * v
        if cfg.lifecycle and t >= 1:
            delta += cfg.alpha_sustain * s * self.v_sus[k]
        if self.stop_triggered and k >= self.art.stop_k:
            delta += cfg.alpha_stop * s * self.v_stop[k]
        return x + delta
