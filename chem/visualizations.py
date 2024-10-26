from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol


def save_molecule_as_svg(mol, filename, highlight_atoms=None, highlight_bonds=None):
    AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    options = drawer.drawOptions()

    options.useBWAtomPalette()
    options.addAtomIndices = True
    options.addBondIndices = True
    options.fixedBondLength = 30
    options.circleAtoms = False
    options.atomLabels = {}

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                        highlightBonds=highlight_bonds)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    with open(filename, 'w') as f:
        f.write(svg)


def visualize_molecule_3d_inline(mol):
    if not mol.GetNumConformers():
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

    pdb_block = Chem.MolToPDBBlock(mol)

    view = py3Dmol.view(width=300, height=300)
    view.addModel(pdb_block, 'pdb')
    view.setStyle({'stick': {}})
    view.setBackgroundColor('white')
    view.zoomTo()
    return view


def get_mol(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise f"Invalid SMILES string: {smiles_str}"

    mol = Chem.AddHs(mol)

    success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if success != 0:
        raise f"Embedding failed for molecule: {smiles_str}"

    AllChem.UFFOptimizeMolecule(mol)

    return mol
