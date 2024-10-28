import os
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np
import orbax.checkpoint as ocp
from collections import defaultdict

from lm.model import Decoder
from lm.model.transformer_utils import causal_mask
from training.train_lm import create_lm_train_state
from training.train_utils import try_restore_for
from .model import RSAE, JSAE
from training.train_sae import create_sae_train_state

def set_modifier_at(layer_id, site_id, lm_params, modifier_params):
    if modifier_params is not None:
        lm_params[f"modifiers_{layer_id}_{site_id}"] = modifier_params
    return lm_params


def restore_modifiers(model, modifier_param_list, lm_params):
    assert len(
        modifier_param_list) == model.num_layers, f"Number of modifier tuples does not match number of layers. Expected {model.num_layers} but got {len(modifier_param_list)}."

    for layer_id in range(model.num_layers):
        assert len(modifier_param_list[layer_id]
                   ) == 2, "Each site should have a modifier."

        lm_params = set_modifier_at(
            layer_id, 0, lm_params, modifier_param_list[layer_id][0])
        lm_params = set_modifier_at(
            layer_id, 1, lm_params, modifier_param_list[layer_id][1])

    return lm_params


def get_merged_params_from_states(full_state, partial_restored_state):
    merged_params = ocp.transform_utils.merge_trees(
        full_state.params, partial_restored_state.params)
    return merged_params


def restore_lm_for_modifiers(lm, modifiers, modifier_params, lm_state_path, mesh=None):
    mod_lm = Decoder(
        hidden_dim=lm.hidden_dim,
        d_model=lm.d_model,
        num_heads=lm.num_heads,
        num_kv_heads=lm.num_kv_heads,
        num_layers=lm.num_layers,
        vocab_size=lm.vocab_size,
        tracked=lm.tracked,
        modifiers=modifiers
    )

    mod_lm_state = create_lm_train_state(mod_lm)
    lm_state = create_lm_train_state(lm)
    lm_state, lm_step = try_restore_for(lm_state, lm_state_path, mesh=mesh)
    print(f"Restored LM at step {lm_step}")
    mod_lm_params = get_merged_params_from_states(mod_lm_state, lm_state)
    mod_lm_params = restore_modifiers(mod_lm, modifier_params, mod_lm_params)

    return mod_lm, mod_lm_params


def get_mod_for_layer(lm, layer_id, sae_config, sites):
    if "mlp" in sites:
        if sae_config["sae_type"] == "relu":
            mlp_sae = RSAE(d_model=lm.d_model,
                           hidden_size=sae_config["sae_mult"] * lm.d_model)
        else:
            mlp_sae = JSAE(d_model=lm.d_model,
                           hidden_size=sae_config["sae_mult"] * lm.d_model)

        mlp_sae_state = create_sae_train_state(mlp_sae)
        mlp_sae_state, mlp_sae_step = try_restore_for(
            mlp_sae_state, f"{sae_config['checkpoint_dir']}/block{layer_id}/mlp")
        print(f"Restored MLP SAE at step {mlp_sae_step}")
        mlp_sae_params = mlp_sae_state.params
    else:
        mlp_sae = None
        mlp_sae_params = None

    if "residual_stream" in sites:
        if sae_config["sae_type"] == "relu":
            residual_sae = RSAE(
                d_model=lm.d_model, hidden_size=sae_config["sae_mult"] * lm.d_model)
        else:
            residual_sae = JSAE(
                d_model=lm.d_model, hidden_size=sae_config["sae_mult"] * lm.d_model)

        residual_sae_state = create_sae_train_state(residual_sae)
        residual_sae_state, residual_sae_step = try_restore_for(
            residual_sae_state, f"{sae_config['checkpoint_dir']}/block{layer_id}/residual_stream")
        print(f"Restored Residual SAE at step {residual_sae_step}")
        residual_sae_params = residual_sae_state.params
    else:
        residual_sae = None
        residual_sae_params = None

    modifiers = (mlp_sae, residual_sae)
    params = (mlp_sae_params, residual_sae_params)

    return modifiers, params


def save_modifier_activations(lm, params, dataset, tokenizer, output_dir, total, sharding=None):
    lm.tracked = True
    os.makedirs(output_dir, exist_ok=True)

    for batch_id, batch in tqdm(enumerate(dataset), total=total):
        if sharding is not None:
            batch = jax.device_put(batch, sharding)

        pos = jnp.arange(batch.shape[-1])
        mask = causal_mask(batch, tokenizer.pad_token_id)
        _, _, activations = lm.apply({"params": params}, batch, pos, mask)

        batch_mlp_activations = {}
        batch_residual_activations = {}

        for layer_id, activation in enumerate(activations):
            mlp_act, residual_act = activation

            if len(mlp_act) != 0:
                batch_mlp_activations[f'mlp_{layer_id}'] = mlp_act
            if len(residual_act) != 0:
                batch_residual_activations[f'residual_{layer_id}'] = residual_act

        if batch_mlp_activations:
            np.savez_compressed(f"{output_dir}/mlp_activations_batch_{batch_id}.npz", **batch_mlp_activations)
        if batch_residual_activations:
            np.savez_compressed(f"{output_dir}/residual_activations_batch_{batch_id}.npz", **batch_residual_activations)


def split_activation_dir_fname(fname):
    model_size, sae_type, sae_mult, sparsity_coefficient = fname.split("-")[1:]
    return model_size, sae_type, int(sae_mult[:-1]), float(sparsity_coefficient)

def extract_batch_number_from_filename(filepath):
    filename = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(filename)[0]
    parts = filename_no_ext.split('_')
    batch_str = parts[-1]
    try:
        batch_number = int(batch_str)
    except ValueError:
        batch_number = -1
    return batch_number

def get_variant_metadata(variant_activations_dir):
    variants = defaultdict(list)
    variant_activations = os.listdir(variant_activations_dir)

    for dir in variant_activations:
        model_size, sae_type, sae_mult, sparsity_coefficient = split_activation_dir_fname(dir)
        files = os.listdir(f"{variant_activations_dir}/{dir}")
        residual_files = [f"{variant_activations_dir}/{dir}/{f}" for f in files if f.endswith(".npz") and f.startswith("residual")]
        mlp_files = [f"{variant_activations_dir}/{dir}/{f}" for f in files if f.endswith(".npz") and f.startswith("mlp")]

        residual_files_sorted = sorted(residual_files, key=extract_batch_number_from_filename)
        mlp_files_sorted = sorted(mlp_files, key=extract_batch_number_from_filename)

        variants[f"model-{model_size}"].append({
            "sae_type": sae_type,
            "sae_mult": sae_mult,
            "sparsity_coefficient": sparsity_coefficient,
            "files": {
                "residual_stream": residual_files_sorted,
                "mlp": mlp_files_sorted
            }
        })

    return variants

def get_activation_stats(activation_files, layer_name, sequence_tensor, tokenizer, activation_threshold=0.01):
    num_neurons = None
    neuron_fire_counts = None
    neuron_token_counts = None
    sequence_activations = []
    total_tokens_processed = 0

    for batch_idx, act_file in enumerate(tqdm(activation_files, desc='Processing Batches')):
        activations = np.load(act_file)[layer_name]
        batch_size, max_seq_length, num_neurons_in_batch = activations.shape

        if neuron_fire_counts is None:
            num_neurons = num_neurons_in_batch
            neuron_fire_counts = np.zeros(num_neurons, dtype=np.int64)
            neuron_token_counts = [defaultdict(int) for _ in range(num_neurons)]

        start_seq_idx = batch_idx * batch_size
        end_seq_idx = start_seq_idx + batch_size
        batch_sequences = sequence_tensor[start_seq_idx:end_seq_idx]

        for seq_idx in range(batch_size):
            sequence = batch_sequences[seq_idx]
            seq_length = len(sequence)
            activations_seq = activations[seq_idx][:seq_length]

            tokens = tokenizer.decode(sequence)

            neuron_activation_sums = activations_seq.sum(axis=0)
            sequence_activations.append(neuron_activation_sums)

            for pos_idx in range(seq_length):
                token = tokens[pos_idx]
                activations_pos = activations_seq[pos_idx]

                firing_neurons = np.where(activations_pos > activation_threshold)[0]

                neuron_fire_counts[firing_neurons] += 1
                total_tokens_processed += 1

                for neuron_idx in firing_neurons:
                    neuron_token_counts[neuron_idx][token] += 1

    return num_neurons, neuron_fire_counts, neuron_token_counts, total_tokens_processed, sequence_activations
