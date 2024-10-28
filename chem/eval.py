from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit import RDLogger
from rdkit.Contrib.SA_Score import sascorer

from tqdm.auto import tqdm

RDLogger.DisableLog('rdApp.error')


def clean_molecule(raw_str, tokenizer):
    smiles_str = raw_str.split("<EOM>")[0]
    for key, token in tokenizer.special_tokens.items():
        smiles_str = smiles_str.replace(token, "")
    return smiles_str


def calculate_molecule_properties(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return False, None, None, None, None, None

        qed_value = QED.qed(mol)
        logp_value = Descriptors.MolLogP(mol)
        mol_wt = Descriptors.MolWt(mol)
        mol_len = mol.GetNumAtoms()
        sa_score = sascorer.calculateScore(m)

        return True, qed_value, logp_value, mol_wt, mol_len, sa_score
    except Exception as e:
        return False, None, None, None, None, None


def get_raw_data_stats(lm_output, tokenizer, verbose=False):
    num_valid = 0
    acc_qed = 0.0
    acc_logp = 0.0
    acc_mol_wt = 0.0
    acc_mol_len = 0.0
    acc_sa_score = 0.0
    unique_smiles = set()

    for decoded in tqdm(lm_output):
        smiles_str = clean_molecule(decoded, tokenizer)

        valid, qed, logp, mol_wt, mol_len, sa_score = calculate_molecule_properties(
            smiles_str)
        if valid:
            num_valid += 1

            acc_qed += qed
            acc_logp += logp
            acc_mol_wt += mol_wt
            acc_mol_len += mol_len
            acc_sa_score += sa_score

            unique_smiles.add(smiles_str)
        if verbose:
            print(
                f"Molecule: {smiles_str}, valid: {valid}, QED: {qed}, LogP: {logp}, MolWt: {mol_wt}, MolLen: {mol_len}, SAScore: {sa_score}")

    num_unique = len(unique_smiles)
    unique_percentage = 100 * num_unique / num_valid

    average_qed = acc_qed / num_valid if num_valid else 0
    average_logp = acc_logp / num_valid if num_valid else 0
    average_mol_wt = acc_mol_wt / num_valid if num_valid else 0
    average_mol_len = acc_mol_len / num_valid if num_valid else 0
    average_sa_score = acc_sa_score / num_valid if num_valid else 0

    return {"num_valid": num_valid,
            "percent_valid": 100 * num_valid / len(lm_output),
            "num_unique": num_unique,
            "percent_unique": unique_percentage,
            "avg_qed": average_qed,
            "avg_logp": average_logp,
            "avg_molwt": average_mol_wt,
            "avg_atoms": average_mol_len,
            "avg_sa_score": average_sa_score
            }
