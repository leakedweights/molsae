import jax.numpy as jnp
import flax.linen as nn

from .attention import GroupedQueryAttention
from .transformer_utils import Embedder, FeedForward

class TransformerBlock(nn.Module):
    d_model: int
    hidden_dim: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    logit_cap: float
    f_embed: int
    tracked: bool = False

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
        mlp_post_norm = nn.RMSNorm()(x)

        residual = x + mlp_post_norm
        
        return residual, ((mlp_post_norm, residual) if self.track else ())


class Decoder(nn.Module):
    vocab_size: int
    num_layers: int = 6
    d_model: int = 512
    hidden_dim: int = 4096
    head_dim: int = 64
    num_heads: int = 8
    num_kv_heads: int = 4
    layer_logit_cap: float = 50.0
    final_logit_cap: float = 30.0
    f_embed: int = 10_000
    tracked: bool = False

    @nn.compact
    def __call__(self, x, pos, mask, aux_ids=()):
        activations = []
        embedder = Embedder(self.vocab_size, self.d_model)
        x = embedder.encode(x)

        for block_id in range(self.num_layers):
            x, actv = TransformerBlock(self.d_model,
                                 self.hidden_dim,
                                 self.head_dim,
                                 self.num_heads,
                                 self.num_kv_heads,
                                 self.layer_logit_cap,
                                 self.f_embed,
                                 self.tracked)(x, pos, mask)
            
            if self.tracked and (not aux_ids or block_id in aux_ids):
                activations.append(actv)
            else:
                activations.append(None)

        x = nn.RMSNorm()(x)

        logits = embedder.decode(x)
        logits = self.final_logit_cap * jnp.tanh(logits / self.final_logit_cap)

        if self.tracked:
            return logits, activations
        return logits