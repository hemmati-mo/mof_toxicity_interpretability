from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, RDKFingerprint
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch
from torch_geometric.data import Data

# === Initialize fingerprint generators (efficient, outside loop) ===
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
torsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)

# === Feature extraction: Descriptors + Fingerprints ===
def compute_all_features(smiles, idx):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # Compute RDKit descriptors
    descs = []
    for name, func in Descriptors._descList:
        try:
            val = func(mol)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
        except:
            val = 0.0
        descs.append(val)

    # Compute fingerprints
    morgan = morgan_gen.GetFingerprint(mol)
    torsion = torsion_gen.GetFingerprint(mol)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    topological = RDKFingerprint(mol)

    return {
        **{f'desc_{i}': val for i, val in enumerate(descs)},
        **{f'morgan_{i}': int(morgan.GetBit(i)) for i in range(morgan.GetNumBits())},
        **{f'maccs_{i}': int(maccs.GetBit(i)) for i in range(maccs.GetNumBits())},
        **{f'topo_{i}': int(topological.GetBit(i)) for i in range(topological.GetNumBits())},
        **{f'torsion_{i}': int(torsion.GetBit(i)) for i in range(torsion.GetNumBits())}
    }

# === Graph construction: atoms as nodes, bonds as edges ===
def mol_to_graph(mol):
    if mol is None:
        return Data(
            x=torch.empty((0, 14)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 6))
        )

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    x = []
    edge_index = []
    edge_attr = []

    for atom in mol.GetAtoms():
        try:
            g_charge = float(atom.GetProp("_GasteigerCharge"))
        except:
            g_charge = 0.0

        # Valid atom features (14)
        x.append([
            atom.GetAtomicNum(),                          # 0
            atom.GetMass(),                               # 1
            atom.GetTotalDegree(),                        # 2
            atom.GetFormalCharge(),                       # 3
            int(atom.GetHybridization()),                 # 4
            atom.GetImplicitValence(),                    # 5
            atom.GetNumRadicalElectrons(),                # 6
            int(atom.GetIsAromatic()),                    # 7
            int(atom.IsInRingSize(5) or atom.IsInRingSize(6)),  # 8 approx. ring membership
            int(atom.GetChiralTag()),                     # 9
            atom.GetTotalValence(),                       # 10
            atom.GetNumExplicitHs(),                      # 11
            int(atom.GetNoImplicit()),                    # 12
            g_charge                                      # 13
        ])

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feats = [
            bond.GetBondTypeAsDouble(),                   # 0
            int(bond.GetIsAromatic()),                    # 1
            int(bond.IsInRing()),                         # 2
            int(bond.GetStereo()),                        # 3
            int(bond.GetIsConjugated()),                  # 4
            int(bond.GetBondDir())                        # 5
        ]
        edge_index += [[i, j], [j, i]]
        edge_attr += [bond_feats, bond_feats]

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )
