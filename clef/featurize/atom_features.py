from __future__ import annotations

import numpy as np
from rdkit import Chem


ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl")
ELEMENT_INDEX = {symbol: index for index, symbol in enumerate(ELEMENTS)}
HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
ATOM_FEATURE_DIM = 45


def _one_hot(value: int, width: int) -> list[float]:
    block = [0.0] * width
    block[max(0, min(value, width - 1))] = 1.0
    return block


def atom_vector(atom: Chem.Atom) -> np.ndarray:

    element = ELEMENT_INDEX.get(atom.GetSymbol())
    if element is None:
        raise ValueError(f"Unsupported element: {atom.GetSymbol()}")
    hybrid = atom.GetHybridization()
    hybrid_index = HYBRIDIZATIONS.index(hybrid) if hybrid in HYBRIDIZATIONS else 6
    neighbors = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    valence = atom.GetTotalValence()


    attached_hydrogens = sum(
        neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
    )
    values = (
        _one_hot(element, 8)
        + _one_hot(neighbors, 6)
        + _one_hot(formal_charge + 2, 5)
        + _one_hot(hybrid_index, 7)
        + _one_hot(attached_hydrogens, 5)
        + _one_hot(valence, 7)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing()),
           float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)]
        + [float(atom.GetMass()) / 100.0,
           float(atom.GetNumRadicalElectrons()) / 2.0,
           float(atom.GetIsotope()) / 100.0,
           float(attached_hydrogens) / 4.0]
    )
    assert len(values) == ATOM_FEATURE_DIM
    return np.asarray(values, dtype=np.float32)


def graph_arrays(molecule: Chem.Mol, max_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    atom_count = molecule.GetNumAtoms()
    if atom_count > max_atoms:
        raise ValueError(f"Molecule has {atom_count} atoms, limit is {max_atoms}")
    feature = np.zeros((max_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_atoms, dtype=np.float32)
    adjacency = np.zeros((4, max_atoms, max_atoms), dtype=np.float32)
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        feature[i] = atom_vector(atom)
        mask[i] = 1.0
    relation = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    for bond in molecule.GetBonds():
        channel = relation.get(bond.GetBondType())
        if channel is None:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[channel, left, right] = 1.0
        adjacency[channel, right, left] = 1.0
    return adjacency, feature, mask
