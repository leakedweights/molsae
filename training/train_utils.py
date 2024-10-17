import os
import jax
import wandb
from rdkit import Chem
from jax.sharding import NamedSharding, PartitionSpec
import orbax.checkpoint as ocp

def setup(project_name, run_id, checkpoint_dir, resume):
    wandb.init(project=project_name,
               id=run_id,
               resume="allow" if resume else "never")
    os.makedirs(checkpoint_dir, exist_ok=True)

def create_sharding():
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec("x",))
    return sharding

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

def try_restore_for(item, dir):
    try:
        options = ocp.CheckpointManagerOptions()
        with ocp.CheckpointManager(
            dir,
            options=options,
        ) as mngr:
            step = mngr.latest_step()
            return mngr.restore(step, args=ocp.args.StandardRestore(item)), step
    except:
        return item, 0