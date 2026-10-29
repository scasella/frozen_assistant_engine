"""The attention functions on random tensors (CPU, no model), and the sensitivity-tag rule."""

import pytest
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

from engine.hooks import steer_eager_attention_forward, steer_fp32_attention_forward
from eval.benchmark import results_tag


class _Attn(nn.Module):
    num_key_value_groups = 2


def _inputs(dtype, seed=0, q_len=24, kv_len=24, heads=4, kv_heads=2, dim=64):
    g = torch.Generator().manual_seed(seed)
    q = 3.0 * torch.randn(1, heads, q_len, dim, generator=g)   # large scores, where bf16 rounding matters
    k = 3.0 * torch.randn(1, kv_heads, kv_len, dim, generator=g)
    v = torch.randn(1, kv_heads, kv_len, dim, generator=g)
    causal = torch.ones(q_len, kv_len, dtype=torch.bool).tril(kv_len - q_len)
    mask = torch.zeros(1, 1, q_len, kv_len).masked_fill(~causal, torch.finfo(dtype).min)
    return q.to(dtype), k.to(dtype), v.to(dtype), mask.to(dtype)


def _bias(dtype, heads=4, kv_len=24):
    bias = torch.zeros(1, heads, 1, kv_len + 8, dtype=dtype)   # longer than kv: sliced like the engine's buffer
    bias[0, [0, 3], 0, :6] = 1.25
    bias[0, 1, 0, 6:9] = -1.0
    return bias


def test_steer_eager_is_bit_identical_to_hf_eager_in_bf16():
    q, k, v, mask = _inputs(torch.bfloat16)
    mod = _Attn().eval()
    ours, _ = steer_eager_attention_forward(mod, q, k, v, mask, scaling=0.125)
    ref, _ = eager_attention_forward(mod, q, k, v, mask, scaling=0.125)
    assert torch.equal(ours, ref)


def test_fp32_path_equals_hf_eager_in_fp32():
    q, k, v, mask = _inputs(torch.float32)
    mod = _Attn().eval()
    ours, w = steer_fp32_attention_forward(mod, q, k, v, mask, scaling=0.125)
    ref, w_ref = eager_attention_forward(mod, q, k, v, mask, scaling=0.125)
    assert torch.allclose(ours, ref, atol=1e-6) and torch.allclose(w, w_ref, atol=1e-7)


def test_fp32_path_applies_the_bias_like_steer_eager_in_fp32():
    q, k, v, mask = _inputs(torch.float32)
    mod = _Attn().eval()
    plain, _ = steer_fp32_attention_forward(mod, q, k, v, mask, scaling=0.125)
    mod._steer_bias = _bias(torch.float32)
    ours, _ = steer_fp32_attention_forward(mod, q, k, v, mask, scaling=0.125)
    ref, _ = steer_eager_attention_forward(mod, q, k, v, mask, scaling=0.125)
    assert torch.allclose(ours, ref, atol=1e-6)
    assert (ours - plain).abs().max() > 1e-2


@pytest.mark.parametrize("with_bias", [False, True])
def test_fp32_path_in_bf16_is_closer_to_fp32_than_steer_eager(with_bias):
    q, k, v, mask = _inputs(torch.bfloat16)
    mod32, mod16 = _Attn().eval(), _Attn().eval()
    if with_bias:
        mod32._steer_bias, mod16._steer_bias = _bias(torch.float32), _bias(torch.bfloat16)
    ref, _ = steer_eager_attention_forward(mod32, q.float(), k.float(), v.float(), mask.float(), scaling=0.125)
    fp32, w = steer_fp32_attention_forward(mod16, q, k, v, mask, scaling=0.125)
    eager, _ = steer_eager_attention_forward(mod16, q, k, v, mask, scaling=0.125)
    assert fp32.dtype == w.dtype == torch.bfloat16
    err_fp32, err_eager = ((o.float() - ref).abs().max().item() for o in (fp32, eager))
    assert err_fp32 < 0.5 * err_eager


def test_fp32_path_feeds_the_recorder():
    q, k, v, mask = _inputs(torch.bfloat16)
    mod, seen = _Attn().eval(), []
    mod._attn_recorder = lambda m, w: seen.append(w)
    steer_fp32_attention_forward(mod, q, k, v, mask, scaling=0.125)
    assert len(seen) == 1 and seen[0].shape == (1, 4, 24, 24) and seen[0].dtype == torch.bfloat16


def test_sensitivity_tags():
    assert results_tag("steer_eager", None, 40) == "test40"
    assert results_tag("sdpa", None, 40) == "test40_sdpa"
    assert results_tag("steer_fp32", None, 40) == "test40_fp32attn"
    assert results_tag("steer_fp32", "test40_512tok_fp32attn", 40) == "test40_512tok_fp32attn"
    for attn, tag in [("steer_fp32", "test40"), ("steer_fp32", "test40_512tok"), ("steer_fp32", "test40_sdpa"),
                      ("sdpa", "test40_fp32attn"), ("steer_eager", "test40_fp32attn"), ("steer_eager", "test40_sdpa")]:
        with pytest.raises(SystemExit):
            results_tag(attn, tag, 40)
