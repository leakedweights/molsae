import jax

import jax.numpy as jnp
import flax.linen as nn

from functools import partial
from typing import Sequence

def RoPE(inputs, positions, head_dim, max_wavelength):
    """From https://arxiv.org/abs/2408.00118"""
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class WeightedEinsum(nn.Module):
    dim: int | Sequence

    @nn.compact
    def __call__(self, literal: str, x: jax.Array):
        w = self.param("w", nn.initializers.glorot_normal(), self.dim)
        return jnp.einsum(literal, x, w)
    
class Embedder(nn.Module):
    """From https://arxiv.org/abs/2408.00118"""
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)
    
class FeedForward(nn.Module):
    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        in_dims = (2, self.features, self.hidden_dim)
        out_dims = (self.hidden_dim, self.features)
        w_init = jax.nn.initializers.glorot_normal()

        w1, w2 = self.param("weights_in", w_init, in_dims)
        w3 = self.param("weights_out", w_init, out_dims)
        y = nn.silu(jnp.dot(x, w1))
        x = y * jnp.dot(x, w2)
        output = jnp.dot(x, w3)
        return output
    
@partial(jax.jit, static_argnums=(1,))
def causal_mask(input_tokens, pad_token_id):
    batch_size, seq_length = input_tokens.shape
    padding_mask = (input_tokens == pad_token_id)
    causal_mask = jnp.triu(jnp.ones((seq_length, seq_length), dtype=jnp.bool), k=1)

    padding_mask_expanded = padding_mask[:, None, :]
    causal_mask_expanded = causal_mask[None, :, :]
    combined_mask = causal_mask_expanded | padding_mask_expanded

    return combined_mask