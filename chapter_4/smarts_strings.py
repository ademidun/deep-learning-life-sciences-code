"""
SMARTS strings are used to match a sequence of molecules.
Similar to regular expressions for natural language.
"""
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

def get_smart():

    smiles_list = ["CCCC", "CCOCC", "CCNCC", "CCNCC", "CCSCC"]

    mol_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]

    # now let see which SMILES strings match the SMARTS pattern

    # match 3 adjacent aliphatic carbon atoms
    smarts_query_string = "CCC"
    query = Chem.MolFromSmarts(smarts_query_string)

    match_list = [mol.GetSubstructMatch(query) for mol in mol_list]
    img = MolsToGridImage(mols=mol_list, molsPerRow=4, highlightAtomLists=match_list)
    img.show()

    # natch an aliphatic carbon attached to any atom, attached to another aliphatic carbon
    query = Chem.MolFromSmarts("C[C,O,N]C")
    match_list = [mol.GetSubstructMatch(query) for mol in mol_list]
    img = MolsToGridImage(mols=mol_list, molsPerRow=4, highlightAtomLists=match_list)
    img.show()

if __name__ == '__main__':
    get_smart()
