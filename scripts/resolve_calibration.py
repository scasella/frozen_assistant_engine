"""Re-solve a calibration artifact from its cached trajectories + hidden-state statistics (no forward passes).

Used when the closed-form solver gains new outputs (e.g. the v2 hinge percentiles). Every tensor that already
existed is checked to be bit-identical, so generations logged with the old artifact stay valid.

    python scripts/resolve_calibration.py --model Qwen/Qwen2.5-3B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import CalibConfig, SteeringArtifact, calibrate, load_model  # noqa: E402
from engine.artifact import _TENSOR_FIELDS  # noqa: E402
from eval.conditions import load_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", default=None)
    args = ap.parse_args()
    lm = load_model(args.model)
    out_dir = Path(args.calib_dir) if args.calib_dir else ROOT / "artifacts" / lm.slug / "calib80"
    path = out_dir / "calibration.pt"
    old = SteeringArtifact.load(path)
    assert (out_dir / "states.pt").exists(), "no cached states; this would trigger new forward passes"
    path.rename(out_dir / "calibration_prev.pt")
    new = calibrate(lm, load_prompts(ROOT / "data" / "calibration_prompts.json"), CalibConfig(), out_dir)
    for k in ("calibration_seconds", "calibration_seconds_note"):
        if k in old.meta:
            new.meta[k] = old.meta[k]
    new.save(path)
    old_payload = torch.load(out_dir / "calibration_prev.pt", map_location="cpu", weights_only=False)
    for name in _TENSOR_FIELDS:
        if name in old_payload:
            assert torch.equal(old_payload[name], getattr(new, name).float()), f"{name} changed"
    assert (new.layers, new.stop_k, new.tau_stop, new.caa_layer, new.pin_heads) == \
        (old.layers, old.stop_k, old.tau_stop, old.caa_layer, old.pin_heads), "scalar fields changed"
    (out_dir / "calibration_prev.pt").unlink()
    print(f"[resolve] {path.relative_to(ROOT)}: all pre-existing tensors identical; added drift_p99")
    for d in new.diagnostics["layers"]:
        print(f"  layer {d['layer']}: p99={d['hinge_kappa_p99']} clean touched {d['hinge_frac_clean_tokens_touched']:.3f} "
              f"(v1 {d['drift_frac_clean_tokens_touched']:.3f}), trap touched {d['hinge_frac_trap_tokens_touched']:.3f} "
              f"(v1 {d['drift_frac_trap_tokens_touched']:.3f})")


if __name__ == "__main__":
    main()
