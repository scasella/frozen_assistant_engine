"""Model / tokenizer loading with strict MPS + no-grad discipline."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .hooks import install_steerable_attention

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class LoadedModel:
    name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    layers: nn.ModuleList
    n_layers: int
    d_model: int
    n_heads: int
    eos_token_id: int
    eos_ids: tuple[int, ...] = ()   # every id that ends a turn (generation_config); base Qwen2.5: (eos_token_id,)
    attn: str = "steer_eager"       # steer_eager (all benchmark conditions) | sdpa (numerics sensitivity only)

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")

    def decoder(self) -> nn.Module:
        """The transformer trunk without the LM head (used for logit-free calibration passes)."""
        for attr in ("model", "transformer", "gpt_neox"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        return self.model.base_model


def pick_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the stack of decoder blocks for any standard HF causal LM."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj: object = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    best: nn.ModuleList | None = None
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and (best is None or len(module) > len(best)):
            best = module
    if best is None:
        raise RuntimeError("Could not locate decoder layers")
    return best


def load_model(name: str, device: str | None = None, dtype: str = "bfloat16", attn: str = "steer_eager") -> LoadedModel:
    """``attn="sdpa"`` keeps HF's fused attention (no Module-4 bias). In bf16 the eager path rounds attention scores to
    bf16 before the softmax, which measurably hurts Qwen2.5-1.5B arithmetic; sdpa exists to quantify that."""
    if attn not in ("steer_eager", "sdpa"):
        raise ValueError(f"unknown attention implementation {attn!r}")
    dev = pick_device(device)
    torch_dtype = _DTYPES[dtype]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype,
                                                 attn_implementation="eager" if attn == "steer_eager" else "sdpa")
    model.to(dev)
    model.eval()
    model.requires_grad_(False)
    if attn == "steer_eager":
        install_steerable_attention(model)

    cfg = model.config
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else cfg.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    gen_eos = [gen_eos] if isinstance(gen_eos, int) else list(gen_eos or [])
    eos_ids = tuple(dict.fromkeys([int(eos), *gen_eos]))
    layers = find_decoder_layers(model)
    return LoadedModel(
        name=name,
        model=model,
        tokenizer=tokenizer,
        device=dev,
        dtype=torch_dtype,
        layers=layers,
        n_layers=len(layers),
        d_model=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        eos_token_id=int(eos),
        eos_ids=eos_ids,
        attn=attn,
    )


def free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
