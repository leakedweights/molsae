import jax.numpy as jnp
import flax.linen as nn

from .attention import GroupedQueryAttention
from .transformer_utils import Embedder, FeedForward

from typing import Optional


class TransformerBlock(nn.Module):
    d_model: int
    hidden_dim: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    logit_cap: float
    f_embed: int
    tracked: bool = False
    modifiers: Optional[tuple[nn.Module]] = None

    @nn.compact
    def __call__(self, x, pos, mask, modifier_args=None):

        if self.modifiers is not None:
            assert len(
                self.modifiers) == 2, "Modifiers must be defined for post-norm MLP and residual stream."
            assert modifier_args is None or len(modifier_args) == 2

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
        mlp_post_norm = nn.RMSNorm()(y)

        mlp_mod_act = ()
        if self.modifiers is not None and self.modifiers[0] is not None:
            mlp_post_norm, mlp_mod_act = self.modifiers[0](
                mlp_post_norm, **modifier_args[0])

        residual = x + mlp_post_norm

        residual_mod_act = ()
        if self.modifiers is not None and self.modifiers[1] is not None:
            residual, residual_mod_act = self.modifiers[1](
                residual, **modifier_args[1])

        if self.tracked:
            extras = (mlp_post_norm, residual), (mlp_mod_act, residual_mod_act)
        else:
            extras = (tuple(), tuple())

        return residual, extras


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
    modifiers: Optional[list[tuple[nn.Module]]] = None

    @nn.compact
    def __call__(self, x, pos, mask, modifier_args=None, aux_ids=()):
        activations = []
        modifier_activations = []
        embedder = Embedder(self.vocab_size, self.d_model)
        x = embedder.encode(x)

        if self.modifiers is not None:
            assert len(
                self.modifiers) == self.num_layers, "Modifier tuples must be defined for every layer. Use `None` for empty modifiers"
            assert modifier_args is None or len(modifier_args) == self.num_layers
            modifiers = self.modifiers
        else:
            modifiers = [(None, None)] * self.num_layers

        if modifier_args is None:
            modifier_args = [({}, {})] * self.num_layers

        for block_id in range(self.num_layers):
            x, (actv, modifier_actv) = TransformerBlock(self.d_model,
                                                        self.hidden_dim,
                                                        self.head_dim,
                                                        self.num_heads,
                                                        self.num_kv_heads,
                                                        self.layer_logit_cap,
                                                        self.f_embed,
                                                        self.tracked,
                                                        modifiers[block_id])(x, pos, mask, modifier_args[block_id])

            if self.tracked and (not aux_ids or block_id in aux_ids):
                activations.append(actv)
                if self.modifiers is not None:
                    modifier_activations.append(modifier_actv)
            else:
                activations.append(None)
                modifier_activations.append(None)

        x = nn.RMSNorm()(x)

        logits = embedder.decode(x)
        logits = self.final_logit_cap * jnp.tanh(logits / self.final_logit_cap)

        if self.tracked:
            if self.modifiers is not None:
                return logits, activations, modifier_activations
            return logits, activations
        return logits
