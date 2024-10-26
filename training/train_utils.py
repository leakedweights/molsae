import os
import jax.numpy as jnp
import numpy as np
import jax
import wandb
from tqdm import tqdm
from rdkit import Chem
from jax.sharding import NamedSharding, PartitionSpec
import orbax.checkpoint as ocp

from lm.model.transformer_utils import causal_mask


def setup(project_name, run_id, checkpoint_dir, resume):
    run = wandb.init(project=project_name,
                     id=run_id,
                     resume="allow" if resume else "never")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return run


def create_sharding():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec("x",))
    return sharding, mesh


def reshard_state(state, mesh):
    replicate_sharding = NamedSharding(mesh, PartitionSpec())

    def reshard_leaf(x):
        return jax.device_put(x, replicate_sharding)
    return jax.tree.map(reshard_leaf, state)


def save_checkpoint(dir, step, state):
    options = ocp.CheckpointManagerOptions()
    with ocp.CheckpointManager(
        dir,
        options=options,
    ) as mngr:
        mngr.save(step, args=ocp.args.StandardSave(state))


def try_restore_for(item, dir, mesh=None):
    try:
        options = ocp.CheckpointManagerOptions()
        with ocp.CheckpointManager(
            dir,
            options=options,
        ) as mngr:
            step = mngr.latest_step()
            state = mngr.restore(step, args=ocp.args.StandardRestore(item))

            if mesh is not None:
                state = reshard_state(state, mesh)

            return state, step
    except Exception as e:
        print(e)
        return item, 0


def save_activations(model, params, molecule_dataset, output_dir, pad_token_id, sharding):
    assert model.tracked, "Model must be tracked to save activations!"

    output_dir = output_dir
    residual_dirs = [
        f"{output_dir}/block_{layer}/residual_stream" for layer in range(model.num_layers)]
    mlp_dirs = [
        f"{output_dir}/block_{layer}/mlp" for layer in range(model.num_layers)]

    for i in range(model.num_layers):
        os.makedirs(residual_dirs[i], exist_ok=True)
        os.makedirs(mlp_dirs[i], exist_ok=True)

    for batch_id, batch_seq in enumerate(tqdm(molecule_dataset)):
        jax.device_put(batch_seq, sharding)
        mask = causal_mask(batch_seq, pad_token_id).astype(jnp.bool_)

        pos = jnp.arange(0, batch_seq.shape[1])
        _, activations = model.apply({"params": params}, batch_seq, pos, mask)

        for i, layer_act in enumerate(activations):
            mlp_act, residual_act = layer_act
            mlp_act = jnp.reshape(mlp_act, (-1, mlp_act.shape[-1]))
            residual_act = jnp.reshape(
                residual_act, (-1, residual_act.shape[-1]))
            np.save(f"{residual_dirs[i]}/{batch_id}.npy", residual_act)
            np.save(f"{mlp_dirs[i]}/{batch_id}.npy", mlp_act)
