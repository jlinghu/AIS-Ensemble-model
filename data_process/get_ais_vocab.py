import re
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles
NONE_PHYSICAL_CHARACTERS = (
    '.', ':', '-', '=', '#', '(', ')', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\\', '/', '%')

def encode(smiles, with_atomMap=False):
    """ Transforms given SMILES into Atom-in-SMILES (AiS) tokens. By default, it first canonicalizes the input SMILES.
    In order to get AiS tokens with the same order of SMILES, the input SMILES should be provided with atom map number.

    parameters:
        smiles: str, SMILES
        with_atomMap: if true, it returns AiS with the same order of SMILES.
                      Useful for randomized SMILES, or SMILES augmentation.

    return:
        str, AiS tokens with white space separated.
    """
    smiles_list = smiles.split('.')
    atomInSmiles = []
    for smiles in smiles_list:
        if with_atomMap:
            mol = MolFromSmiles(smiles)
            if mol is None: return
        else:
            tmp = MolFromSmiles(CanonSmiles(smiles))
            if tmp is None: return
            for atom in tmp.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            smiles = MolToSmiles(tmp)
            mol = MolFromSmiles(smiles)
        atomID_sma = {};
        atoms = set()
        for atom in mol.GetAtoms():
            atoms.add(atom.GetSmarts())
            try:
                atomId = atom.GetPropsAsDict()['molAtomMapNumber']
            except:
                atomId = 0
            atom_symbol = atom.GetSymbol()
            chiral_tag = atom.GetChiralTag().name
            charge = atom.GetFormalCharge()
            nHs = atom.GetTotalNumHs()

            if atom.GetIsAromatic():
                atom_symbol = atom_symbol.lower()
            if chiral_tag == 'CHI_TETRAHEDRAL_CCW':
                if nHs:
                    symbol = f"[{atom_symbol}@H]"
                else:
                    symbol = f"[{atom_symbol}@]"
            elif chiral_tag == "CHI_TETRAHEDRAL_CW":
                if nHs:
                    symbol = f"[{atom_symbol}@@H]"
                else:
                    symbol = f"[{atom_symbol}@@]"
            else:
                symbol = atom_symbol
                if nHs:
                    symbol += 'H'
                    if nHs > 1:
                        symbol += '%d' % nHs
            if charge > 0:
                symbol = f"[{symbol}+{charge}]" if charge > 1 else f"[{symbol}+]"
            elif charge < 0:
                symbol = f"[{symbol}{charge}]" if charge < -1 else f"[{symbol}-]"
            ring = 'R' if atom.IsInRing() else '!R'
            neighbs = ''.join(sorted([i.GetSymbol() for i in atom.GetNeighbors()]))
            atomID_sma[atomId] = f'[{symbol};{ring};{neighbs}]'

        ais = []
        for token in smiles_tokenizer(smiles):
            if token in NONE_PHYSICAL_CHARACTERS:
                symbol = token
            else:
                try:
                    atom, atomId = token[1:-1].split(':')
                except:
                    atom, atomId = token, 0
                symbol = atomID_sma[int(atomId)]
            ais.append(symbol)

        atomInSmiles.append(' '.join(ais))

    return ' . '.join(atomInSmiles)

def smiles_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens #' '.join(tokens)

def get_ais(file):
    df = pd.read_csv(file,nrows=10000)
    label = []
    smiles = []
    for row in range(len(df)):
        smiles.append(df['smiles'][row])
        label.append(df['qed'][row])
    all_atomInSMILES = []
    for s in smiles:
        all_atomInSMILES.append(encode(s).split( ))
    vocab = {}
    for i in range(len(all_atomInSMILES)):
        for j in range(len(all_atomInSMILES[i])):
            if all_atomInSMILES[i][j] not in vocab:
                vocab[all_atomInSMILES[i][j]] = len(vocab)

    return all_atomInSMILES,label,vocab
file = '/Users/macbook/Desktop/学习资料/250k_zinc.csv'
a,b,c = get_ais(file)
#
# vocab = pd.DataFrame(c.keys())
# vocab.to_csv('vocab.csv',index=False)

