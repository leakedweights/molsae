import jax
import optax
import jax.numpy as jnp
from flax.training import train_state

import os
from tqdm.auto import tqdm
import wandb

from sae.model import JSAE, RSAE, step
from data.dataset import create_activation_dataset
from .train_utils import save_checkpoint


def create_sae_train_state(rng, sae, learning_rate):
    params = sae.init(rng, jnp.ones((1, sae.d_model)))['params']
    tx = optax.adam(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=sae.apply,
        params=params,
        tx=tx
    )

    print(
        f"Created state with {sum(x.size for x in jax.tree.leaves(params))} parameters.")

    return state


def jsae_loss(model, params, x, sparsity_coefficient):
    x_reconstructed, pre_activations = model.apply({'params': params}, x)

    reconstruction_error = x - x_reconstructed
    reconstruction_loss = jnp.sum(reconstruction_error**2, axis=-1)

    threshold = jnp.exp(params['log_threshold'])

    l0 = jnp.sum(step(pre_activations, threshold), axis=-1)
    sparsity_loss = sparsity_coefficient * l0
    total_loss = jnp.mean(reconstruction_loss + sparsity_loss)

    avg_reconstruction_loss = jnp.mean(reconstruction_loss)
    avg_sparsity_loss = jnp.mean(sparsity_loss)

    return total_loss, {'reconstruction_loss': avg_reconstruction_loss, 'sparsity_loss': avg_sparsity_loss}


def rsae_loss(model, params, x, sparsity_coefficient):
    x_reconstructed, feature_magnitudes = model.apply({'params': params}, x)

    reconstruction_error = x - x_reconstructed
    reconstruction_loss = jnp.sum(reconstruction_error**2, axis=-1)
    sparsity_loss = jnp.sum(feature_magnitudes, axis=-1)
    scaled_sparsity_loss = sparsity_coefficient * sparsity_loss

    total_loss = jnp.mean(reconstruction_loss + scaled_sparsity_loss)
    avg_reconstruction_loss = jnp.mean(reconstruction_loss)
    avg_sparsity_loss = jnp.mean(sparsity_loss)

    return total_loss, {'reconstruction_loss': avg_reconstruction_loss, 'sparsity_loss': avg_sparsity_loss}


def sae_train_step(saes, states, activations, sae_ids, include_sites, sae_type, sparsity_coefficient):
    new_states = []
    loss_components_list = []

    for layer_id, site_id in sae_ids:
        sae_index = layer_id * len(include_sites) + site_id

        sae = saes[sae_index]
        state = states[sae_index]
        activation = activations[sae_index]

        if sae_type == "relu":
            def loss_fn(params):
                return rsae_loss(sae, params, activation, sparsity_coefficient)
        else:
            def loss_fn(params):
                return jsae_loss(sae, params, activation, sparsity_coefficient)

        (loss_value, aux), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)

        state = state.apply_gradients(grads=grads)
        new_states.append(state)
        loss_components_list.append(aux)

    return new_states, loss_components_list


def train_saes_for(model, config):
    if not model.tracked:
        model.tracked = True
    if config["include_layers"] is None:
        include_layers = list(range(model.num_layers))
    else:
        include_layers = config["include_layers"]
    if config["include_sites"] is None:
        include_sites = ["residual_stream", "mlp"]
    else:
        include_sites = config["include_sites"]

    if config.get("wandb_run_name") is None:
        wandb_run_name = f"sae_{config['sae_mult']}x"
    else:
        wandb_run_name = config["wandb_run_name"]

    if config.get("sae_type") == "relu":
        SAE = RSAE
    else:
        SAE = JSAE

    sae_dim = config["sae_mult"] * model.d_model
    rng = jax.random.PRNGKey(0)

    sae_ids = []
    for layer_id in include_layers:
        for site_id, _ in enumerate(include_sites):
            sae_ids.append((layer_id, site_id))

    saes = []
    states = []
    for layer_id, site_id in sae_ids:
        sae = SAE(d_model=model.d_model, hidden_size=sae_dim)
        sae_rng, rng = jax.random.split(rng)
        state = create_sae_train_state(sae_rng, sae, config["learning_rate"])
        saes.append(sae)
        states.append(state)

    dataset = create_activation_dataset(config["base_dir"],
                                        include_layers,
                                        include_sites,
                                        model.d_model,
                                        config["batch_size"],
                                        config["num_epochs"])

    wandb.init(project=config["wandb_project"], name=wandb_run_name)

    total_steps = (config["num_molecules"] //
                   config["batch_size"]) * config["num_epochs"]
    if config["num_molecules"] % config["batch_size"] != 0:
        total_steps += config["num_epochs"]

    progress_bar = tqdm(total=total_steps)

    step = 0

    for batch in dataset:
        states, loss_components_list = sae_train_step(
            saes=saes,
            states=states,
            activations=batch,
            sae_ids=sae_ids,
            include_sites=include_sites,
            sparsity_coefficient=config["sparsity_coefficient"],
            sae_type=config["sae_type"])

        step += 1

        loss_logs = {}
        for (layer_id, site), loss_components in zip(sae_ids, loss_components_list):
            loss_logs[f"Layer {layer_id+1} {include_sites[site]} Reconstruction Loss"] = loss_components['reconstruction_loss']
            loss_logs[f"Layer {layer_id+1} {include_sites[site]} Sparsity Loss"] = loss_components['sparsity_loss']
            loss_logs[f"Layer {layer_id+1} {include_sites[site]} Total Loss"] = loss_components['reconstruction_loss'] + \
                loss_components['sparsity_loss']

        wandb.log(loss_logs, step=step)

        progress_bar.update(1)

        if step >= total_steps:
            break

    progress_bar.close()
    wandb.finish()

    for sae_id, (layer_id, site_id) in enumerate(sae_ids):
        target_dir = f"{config['checkpoint_dir']}/block{layer_id}/{include_sites[site_id]}"
        os.makedirs(target_dir, exist_ok=True)
        save_checkpoint(target_dir, step, states[sae_id])
