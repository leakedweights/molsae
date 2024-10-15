import jax

import flax.linen as nn
import jax.numpy as jnp

from transformer_utils import WeightedEinsum, RoPE

class GroupedQueryAttention(nn.Module):
    """Adapted from https://arxiv.org/abs/2408.00118"""
    d_model: int
    hidden_dim: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    logit_cap: float
    f_embed: int
    apply_causal_mask: bool = True

    def _grouped_attn_inner(self, q_proj, k_proj):
        b, t, kg, h = q_proj.shape
        k = self.num_kv_heads
        g = kg // k
        s = k_proj.shape[-3]
        q_proj = q_proj.reshape((b, t, k, g, h))
        logits = jnp.einsum("BTKGH,BSKH->BTKGS", q_proj, k_proj)
        logits = logits.reshape((b, t, kg, s))
        return logits

    def _grouped_attn_outer(self, probs, v_proj):
        b, t, kg, h = probs.shape
        k = self.num_kv_heads
        g = kg // k
        probs = probs.reshape((b, t, k, g, h))
        enc = jnp.einsum("BTKGS,BSKH->BTKGH", probs, v_proj)
        b, t, k, g, h = enc.shape
        enc = enc.reshape((b, t, k*g, h))
        return enc

    def _vanilla_attn_inner(self, q_proj, k_proj):
        logits = jnp.einsum("BTNH,BSNH->BTNS", q_proj, k_proj)
        return logits

    def _vanilla_attn_outer(self, probs, v_proj):
        enc = jnp.einsum("BTNS,BSNH->BTNH", probs, v_proj)
        return enc

    @nn.compact
    def __call__(self, x, pos, mask):
        gqa = self.num_kv_heads != self.num_heads and self.num_kv_heads > 1
        wq_shape = (self.num_heads, self.d_model, self.head_dim)
        wkv_shape = (2, self.num_kv_heads, self.d_model, self.head_dim)
        wout_shape = (self.num_heads, self.head_dim, self.d_model)

        if gqa:
            attn_inner = self._grouped_attn_inner
            attn_outer = self._grouped_attn_outer
        else:
            attn_inner = self._vanilla_attn_inner
            attn_outer = self._vanilla_attn_outer

        q_proj = WeightedEinsum(dim=wq_shape)("BTD,NDH->BTNH", x)
        q_proj = RoPE(q_proj, pos, self.head_dim, self.f_embed) / jnp.sqrt(self.head_dim)

        k_proj, v_proj = WeightedEinsum(dim=wkv_shape)("BSD,CKDH->CBSKH", x)
        k_proj = RoPE(k_proj, pos, self.head_dim, self.f_embed)

        logits = attn_inner(q_proj, k_proj)
        logits = jnp.tanh(logits / self.logit_cap) * self.logit_cap

        logits = jnp.where((jnp.expand_dims(~mask, -2)), logits, -100) # keep values where mask is False

        probs = jax.nn.softmax(logits, axis=-1).astype(k_proj.dtype)

        enc = attn_outer(probs, v_proj)
        out = WeightedEinsum(wout_shape)("BTNH,NHD->BTD", enc)
        return out