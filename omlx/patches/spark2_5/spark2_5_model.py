# Vendored from mlx-optiq 0.5.8 (MIT), optiq/mlx_lm_patches/spark2_5.py.
# SPDX-License-Identifier: MIT
"""``spark2_5`` for mlx-lm: XHToken Spark-X2.5 (4B), ported from the checkpoint's
``modeling_spark.py``.

A dense decoder with a 3:1 hybrid attention layout (three sliding-window
layers, one full-attention layer, repeated), and three things stock mlx-lm
models do not have, each of which the port has to get exactly right:

* **Per-layer-type RoPE.** Full-attention layers rotate only the first
  quarter of each head (``partial_rotary_factor`` 0.25, theta 5e6); sliding
  layers rotate the whole head at theta 1e4. The rotation is the split-halves
  ("rotate_half") convention, which is ``traditional=False`` in MLX, and
  ``nn.RoPE(dims=...)`` rotates the first ``dims`` features and passes the
  rest -- the same ``x_rot`` / ``x_pass`` split the reference makes.
* **A head-wise output gate.** ``g_proj`` maps the layer input to one logit per
  head; ``sigmoid`` of it scales that head's attention output before
  ``out_proj``. Drop it and every layer's attention is a different function.
* **A float32 residual stream.** The reference embeds, then keeps the residual
  in float32 and casts to the weight dtype only at the input of each attention
  and MLP block. Cheap on a 4B, and it is what the checkpoint was trained
  against; the port does the same so the bf16 MLX model matches the reference
  and the sensitivity sweep measures the quantization, not the port.

Also: a fused ``q_k_v_proj`` (one Linear, split q/k/v), exact-erf GELU, plain
RMSNorm (no ``1 + w``), tied embeddings named ``model.embedding``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "spark2_5"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 10240
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 131072
    max_position_embeddings: int = 1048576
    sliding_window: Optional[int] = 512
    layer_types: List[str] = field(default_factory=list)
    rope_parameters: Dict[str, Any] = field(default_factory=dict)
    headwise_attn_output_gate: bool = True
    gate_attn_act_mode: str = "sigmoid"
    hidden_act: str = "gelu"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"layer_types has {len(self.layer_types)} entries for "
                             f"{self.num_hidden_layers} layers")
        if self.hidden_act != "gelu":
            raise ValueError(f"spark2_5: only hidden_act='gelu' is implemented, got {self.hidden_act!r}")

    def rope_for(self, layer_type: str) -> tuple[float, float]:
        p = self.rope_parameters.get(layer_type, {})
        return float(p.get("rope_theta", 10000)), float(p.get("partial_rotary_factor", 1.0))


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x):
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.layer_type = layer_type
        self.q_k_v_proj = nn.Linear(args.hidden_size, self.q_dim + 2 * self.kv_dim, bias=args.attention_bias)
        self.g_proj = (nn.Linear(args.hidden_size, self.n_heads, bias=args.attention_bias)
                       if args.headwise_attn_output_gate else None)
        self.out_proj = nn.Linear(self.q_dim, args.hidden_size, bias=args.attention_bias)
        self.gate_mode = args.gate_attn_act_mode
        theta, factor = args.rope_for(layer_type)
        self.rope = nn.RoPE(int(self.head_dim * factor), traditional=False, base=theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        qkv = self.q_k_v_proj(x)
        q = qkv[..., : self.q_dim]
        k = qkv[..., self.q_dim: self.q_dim + self.kv_dim]
        v = qkv[..., self.q_dim + self.kv_dim:]
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        if self.g_proj is not None:
            g = self.g_proj(x).reshape(B, L, self.n_heads, 1).transpose(0, 2, 1, 3)
            g = mx.sigmoid(g.astype(mx.float32)) if self.gate_mode == "sigmoid" else nn.silu(g.astype(mx.float32))
            out = out * g.astype(out.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.self_attn = Attention(args, self.layer_type)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, h: mx.array, mask, cache, act_dtype) -> mx.array:
        # `h` is the float32 residual stream; blocks see the weight dtype.
        r = self.self_attn(self.input_layernorm(h).astype(act_dtype), mask, cache)
        h = h + r.astype(mx.float32)
        r = self.mlp(self.post_attention_layernorm(h).astype(act_dtype))
        return h + r.astype(mx.float32)


class SparkModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def _act_dtype(self) -> mx.Dtype:
        # The norm weights keep the checkpoint dtype through quantization
        # (Linear weights become packed uint32), so they say what the blocks
        # should compute in.
        return self.norm.weight.dtype

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        h = input_embeddings if input_embeddings is not None else self.embedding(inputs)
        act = self._act_dtype()
        h = h.astype(mx.float32)
        if cache is None:
            cache = [None] * len(self.layers)
        types = self.args.layer_types
        full_i = next((i for i, t in enumerate(types) if t == "full_attention"), None)
        slide_i = next((i for i, t in enumerate(types) if t == "sliding_attention"), None)
        full_mask = create_attention_mask(h, cache[full_i]) if full_i is not None else None
        slide_mask = (create_attention_mask(h, cache[slide_i], window_size=self.args.sliding_window)
                      if slide_i is not None else None)
        for layer, c in zip(self.layers, cache):
            mask = slide_mask if layer.layer_type == "sliding_attention" else full_mask
            h = layer(h, mask, c, act)
        return self.norm(h).astype(act)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SparkModel(args)
        self.tie_word_embeddings = True
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
            self.tie_word_embeddings = False

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        out = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            return self.model.embedding.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if "lm_head.weight" not in weights and not self.tie_word_embeddings:
            self.tie_word_embeddings = True
            self.pop("lm_head")
        elif "lm_head.weight" in weights and self.tie_word_embeddings:
            weights.pop("lm_head.weight")           # tied: the embedding is the head
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [RotatingKVCache(max_size=self.args.sliding_window) if t == "sliding_attention" else KVCache()
                for t in self.args.layer_types]
