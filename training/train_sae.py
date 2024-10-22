import jax
import optax
from jax import random
import jax.numpy as jnp
from flax.training import train_state
import numpy as np

import os
import wandb
from tqdm import trange
from functools import partial

from .train_utils import setup, try_restore_for, create_sharding, save_checkpoint
from lm.model.transformer_utils import causal_mask


def create_sae_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones(1, model.latent_size,))["params"]
    tx = optax.adam(learning_rate)
    print(
        f"Created state with {sum(x.size for x in jax.tree.leaves(params))} parameters.")
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


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


def save_activations(model, params, molecules, config):
    assert model.tracked, "Model must be tracked to save activations!"

    output_dir = config["output_dir"]
    residual_dirs = [
        f"{output_dir}/block_{layer}/residual_stream" for layer in range(model.num_layers)]
    mlp_dirs = [
        f"{output_dir}/block_{layer}/mlp" for layer in range(model.num_layers)]

    for i in range(model.num_layers):
        os.makedirs(residual_dirs[i], exist_ok=True)
        os.makedirs(mlp_dirs[i], exist_ok=True)

    for batch_id, mol_batch in enumerate(molecules):
        seq = mol_batch
        mask = causal_mask(seq, config["pad_token_id"]),
        pos = jnp.arange(0, mol_batch.shape[1])
        _, activations = model.apply({"params": params}, seq, pos, mask)

        for i, layer_act in activations:
            mlp_act, residual_act = layer_act
            np.save(f"{residual_dirs[i]}/{batch_id}.npy", residual_act)
            np.save(f"{mlp_dirs[i]}/{batch_id}.npy", mlp_act)


def train(model, train_ds, config, rng=random.key(0)):
    epochs = config.get("epochs")
    run_id = config.get("run_id", "default-run")
    checkpoint_base = config.get("ckpt_base_dir", "/tmp/checkpoints/")
    checkpoint_dir = f"{checkpoint_base}/{run_id}"

    setup(project_name=config.get("project_name", "MolSAE"),
          run_id=run_id,
          checkpoint_dir=checkpoint_dir,
          resume=config.get("resume", True))

    sharding, mesh = create_sharding()

    state = create_sae_train_state(
        rng, model, learning_rate=config.get("learning_rate"))
    state, start_epoch = try_restore_for(state, checkpoint_dir, mesh)

    for epoch in trange(start_epoch, epochs, desc="Epochs"):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_ds:
            batch = jax.device_put(batch, sharding)

            state, loss = sae_train_step(state, batch)
            loss_value = loss.item()

            epoch_loss += loss_value
            num_batches += 1

        avg_epoch_loss = epoch_loss / \
            num_batches if num_batches > 0 else jnp.inf

        wandb.log({"train_loss": avg_epoch_loss}, step=epoch + 1)

        save_checkpoint(checkpoint_dir, epoch + 1, state)

    wandb.finish()
