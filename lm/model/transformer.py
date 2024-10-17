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
    tracked: bool = False

    def track(self, module: nn.Module):
        def track_fn(*args, **kwargs):
            val = module(*args, **kwargs)
            self.sow("activations", module.name, val)
            return val
        return track_fn

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

        # MLP Post-norm SAE
        if self.tracked:
            self.sow("activations", "mlp_outs", y)

        x = x + y

        # Residual SAE
        if self.tracked:
            self.sow("activations", "residual_stream", x)

        return x


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

    @nn.compact
    def __call__(self, x, pos, mask):
        embedder = Embedder(self.vocab_size, self.d_model)
        x = embedder.encode(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.d_model,
                                 self.hidden_dim,
                                 self.head_dim,
                                 self.num_heads,
                                 self.num_kv_heads,
                                 self.layer_logit_cap,
                                 self.f_embed)(x, pos, mask)

        x = nn.RMSNorm()(x)

        logits = embedder.decode(x)
        logits = self.final_logit_cap * jnp.tanh(logits / self.final_logit_cap)

        return logits