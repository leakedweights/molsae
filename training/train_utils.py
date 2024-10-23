import os
import jax.numpy as jnp
import numpy as np
import jax
import wandb
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

def get_mol_stats(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    try:
        return {
            "qed": Chem.QED.qed(mol),
            **dict(Chem.QED.properties(mol)),
            "mol_weight": Chem.Descriptors.ExactMolWt(mol),
        }
    except:
        return {
            "qed": 0,
        }
    
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
    except:
        return item, 0
    
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