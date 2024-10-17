import  jax
import jax.numpy as jnp
from functools import partial

from train_utils import setup

@partial(jax.jit, static_argnums=(2,))
def sae_train_step(state, actv, w_recons):

    @jax.jit
    def loss_fn(params):
        enc, dec = state.apply_fn(params, actv)
        recon_loss = jnp.mean((actv - dec) ** 2)
        sparsity = jnp.count_nonzero(enc) / enc.size
        return recon_loss + w_recons * sparsity

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss