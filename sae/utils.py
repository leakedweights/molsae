import orbax.checkpoint as ocp

from lm.model import Decoder
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
        mlp_sae = None
        residual_sae_state = None

    modifiers = (mlp_sae, residual_sae)
    params = (mlp_sae_params, residual_sae_params)

    return modifiers, params
