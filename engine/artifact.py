"""Container for everything calibration produces; saved as a single .pt file per model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

_TENSOR_FIELDS = (
    "v_init", "v_sustain", "v_stop", "mu_tok", "eos_dir", "u_drift", "drift_center", "drift_scale", "drift_p99",
    "pred_xbar", "pred_a", "pred_b", "hidden_norm", "v_caa",
)


@dataclass
class SteeringArtifact:
    layers: list[int]                 # target layer indices (K of them)
    v_init: torch.Tensor              # [K, d]  turn-crossing initiation vectors
    v_sustain: torch.Tensor           # [K, d]  sustain vectors, orthogonalised against v_init
    v_stop: torch.Tensor              # [K, d]  termination vectors
    mu_tok: torch.Tensor              # [K, d]  mean clean response-token state (unframed)
    eos_dir: torch.Tensor             # [K, d]  unit completion-centroid direction (self-centred features)
    u_drift: torch.Tensor             # [K, d, r] orthonormal drift subspace, oriented so traps are on the + side
    drift_center: torch.Tensor        # [K, r]  mean clean-token coordinate in the drift subspace
    drift_scale: torch.Tensor         # [K, r]  std of clean-token coordinates (dead-zone width unit)
    drift_p99: torch.Tensor           # [K, r]  99th percentile of z-scored clean-token coordinates (v2 hinge threshold)
    pred_xbar: torch.Tensor           # [K, d]  Module-3 input mean
    pred_a: torch.Tensor              # [K, d, k]
    pred_b: torch.Tensor              # [K, d, k]
    hidden_norm: torch.Tensor         # [K]     mean ||h|| over response tokens
    stop_k: int                       # index into `layers` where the stop check runs
    tau_stop: float
    caa_layer: int
    v_caa: torch.Tensor               # [d]
    pin_heads: list[tuple[int, int]]  # (layer, query-head)
    diagnostics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: getattr(self, name).float().cpu() for name in _TENSOR_FIELDS}
        payload.update(
            layers=self.layers, stop_k=self.stop_k, tau_stop=self.tau_stop, caa_layer=self.caa_layer,
            pin_heads=[list(h) for h in self.pin_heads], diagnostics=self.diagnostics, meta=self.meta,
        )
        torch.save(payload, path)
        (path.parent / "calibration_diagnostics.json").write_text(
            json.dumps({"meta": self.meta, "diagnostics": self.diagnostics}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> "SteeringArtifact":
        p = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            layers=list(p["layers"]), stop_k=int(p["stop_k"]), tau_stop=float(p["tau_stop"]),
            caa_layer=int(p["caa_layer"]), pin_heads=[tuple(h) for h in p["pin_heads"]],
            diagnostics=p["diagnostics"], meta=p["meta"],
            **{name: p[name] if name in p else _default(name, p) for name in _TENSOR_FIELDS},
        )


def _default(name: str, payload: dict) -> torch.Tensor:
    """Fill fields added after an artifact was written (pre-v2 files lack the hinge percentiles)."""
    if name == "drift_p99":
        return torch.full_like(payload["drift_scale"], 2.2)
    raise KeyError(f"calibration artifact is missing '{name}'; re-run scripts/resolve_calibration.py")
