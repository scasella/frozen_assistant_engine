"""How much post-training changed the checkpoint: compare base and Instruct safetensors tensor by tensor on CPU.

    python scripts/weight_delta.py --model Qwen/Qwen2.5-3B

Writes results/<instruct slug>/weight_delta.json: parameter counts, the fraction of parameters whose bf16 value differs,
and the relative Frobenius norm of the update. A weight-modifying "adapter" equivalent to Instruct is the full set of
changed tensors, i.e. essentially the whole checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.conditions import INSTRUCT_SUFFIX  # noqa: E402


def _tensors(repo: str) -> dict[str, tuple[Path, str]]:
    d = Path(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"]))
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            out.update({k: (f, k) for k in st.keys()})
    return out


def _get(loc: tuple[Path, str]) -> torch.Tensor:
    with safe_open(str(loc[0]), "pt") as st:
        return st.get_tensor(loc[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    args = ap.parse_args()
    inst = args.model + INSTRUCT_SUFFIX
    a, b = _tensors(args.model), _tensors(inst)
    common = sorted(set(a) & set(b))
    n = changed = 0
    tensors_changed = 0
    sq_w = sq_d = 0.0
    nbytes = 0
    for k in common:
        x, y = _get(a[k]), _get(b[k])
        if x.shape != y.shape:
            raise SystemExit(f"shape mismatch for {k}: {x.shape} vs {y.shape}")
        diff = x != y
        c = int(diff.sum())
        n += x.numel()
        changed += c
        tensors_changed += c > 0
        sq_w += float(x.double().pow(2).sum())
        sq_d += float((y.double() - x.double()).pow(2).sum())
        nbytes += y.numel() * y.element_size()
    out = {
        "base": args.model, "instruct": inst, "tensors_compared": len(common),
        "only_in_base": sorted(set(a) - set(b)), "only_in_instruct": sorted(set(b) - set(a)),
        "params": n, "params_changed": changed, "params_changed_frac": changed / n,
        "tensors_changed": tensors_changed, "relative_frobenius_update": (sq_d / sq_w) ** 0.5,
        "instruct_weights_bytes": nbytes,
    }
    path = ROOT / "results" / inst.replace("/", "__") / "weight_delta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("only_")}, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
