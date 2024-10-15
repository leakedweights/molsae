import jax.numpy as jnp
import flax.linen as nn

from attention import GroupedQueryAttention
from transformer_utils import Embedder, FeedForward

class TransformerBlock(nn.Module):
    d_model: int
    hidden_dim: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    logit_cap: float
    f_embed: int

    @nn.compact
    def __call__(self, x, pos, mask):
        y = nn.RMSNorm()(x)
        y = GroupedQueryAttention(self.d_model,
                      self.hidden_dim,
                      self.head_dim,
                      self.num_heads,
                      self.num_kv_heads,
                      self.logit_cap,
                      self.f_embed)(y, pos, mask)

        y = nn.RMSNorm()(y)
        x = x + y

        y = nn.RMSNorm()(x)
        y = FeedForward(self.d_model, self.hidden_dim)(y)
        y = nn.RMSNorm()(x)
        x = x + y
        return x


class Decoder(nn.Module):
    vocab_size: int
    num_layers: int = 6
    d_model: int = 256
    hidden_dim: int = 512
    head_dim: int = 64
    num_heads: int = 4
    num_kv_heads: int = 4
    layer_logit_cap: float = 50.0
    final_logit_cap: float = 30.0
    f_embed: int = 10_000

    @nn.compact
    def __call__(self, x, pos, mask):

        x = Embedder(self.vocab_size, self.d_model).encode(x)

        for i in range(self.num_layers):
            x = TransformerBlock(self.d_model,
                                 self.hidden_dim,
                                 self.head_dim,
                                 self.num_heads,
                                 self.num_kv_heads,
                                 self.layer_logit_cap,
                                 self.f_embed)(x, pos, mask)

        x = nn.RMSNorm()(x)

        logits = nn.Dense(self.vocab_size, use_bias=False)(x)
        logits = self.final_logit_cap * jnp.tanh(logits / self.final_logit_cap)

        return logits