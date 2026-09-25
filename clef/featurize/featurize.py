from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Mapping

import numpy as np
from rdkit import Chem

from clef.msutil.binutils import SpectrumBins, create_spectrum_bins

from .atom_features import ELEMENTS, ELEMENT_INDEX, graph_arrays
from .isotopes import formula_exact_mass, isotope_template


FORMULA_BLOCK_WIDTHS = (50, 46, 30, 30, 30, 30, 30, 30)
FORMULA_FEATURE_DIM = sum(FORMULA_BLOCK_WIDTHS)
EVENT_FEATURE_DIM = 45
HYDROGEN_SHIFTS = (-3, -2, -1, 0, 1, 2, 3)


def cumulative_formula_features(counts: np.ndarray) -> np.ndarray:

    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 8:
        raise ValueError("Formula counts must have shape [F,8]")
    if np.any(counts < 0):
        raise ValueError("Formula counts cannot be negative")
    if any(np.any(counts[:, i] >= width) for i, width in enumerate(FORMULA_BLOCK_WIDTHS)):
        raise ValueError("Formula count exceeds cumulative feature capacity")
    result = np.zeros((len(counts), FORMULA_FEATURE_DIM), dtype=np.float32)
    start = 0
    for element, width in enumerate(FORMULA_BLOCK_WIDTHS):
        result[:, start:start + width] = (
            np.arange(width, dtype=np.int64)[None, :] <= counts[:, element, None]
        )
        start += width
    return result


def molecular_counts(molecule: Chem.Mol) -> np.ndarray:
    counts = np.zeros(8, dtype=np.int16)
    for atom in molecule.GetAtoms():
        index = ELEMENT_INDEX.get(atom.GetSymbol())
        if index is None:
            raise ValueError(f"Unsupported element: {atom.GetSymbol()}")
        counts[index] += 1
    return counts


def enumerate_formula_counts(
    precursor_counts: np.ndarray, max_formulas: int = 8192
) -> np.ndarray:
    counts = np.asarray(precursor_counts, dtype=np.int64)
    if counts.shape != (8,) or np.any(counts < 0):
        raise ValueError("Expected eight nonnegative precursor counts")
    total = 1
    for count, width in zip(counts, FORMULA_BLOCK_WIDTHS):
        if count >= width:
            raise ValueError("Precursor exceeds formula encoding capacity")
        total *= int(count) + 1
    if total > max_formulas:
        raise ValueError(f"Molecule has {total} subformulas, limit is {max_formulas}")
    return np.asarray(list(product(*(range(int(c) + 1) for c in counts))), dtype=np.int16)


@dataclass(frozen=True)
class _Event:
    features: np.ndarray
    counts: tuple[int, ...]
    mass: int
    atom_indices: tuple[int, ...]
    cut_count: int
    shift: int
    component: tuple[int, ...]
    molecular_ion: bool = False


def _components_after_cuts(
    atom_count: int, bonds: list[Chem.Bond], removed: tuple[int, ...]
) -> list[tuple[int, ...]]:
    parent = list(range(atom_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    removed_set = set(removed)
    for bond in bonds:
        if bond.GetIdx() not in removed_set:
            join(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    components: dict[int, list[int]] = {}
    for index in range(atom_count):
        components.setdefault(find(index), []).append(index)
    return [tuple(indices) for indices in sorted(components.values(), key=lambda x: x[0])]


def _cut_endpoints(
    cuts: tuple[Chem.Bond, ...], component: tuple[int, ...], slots: int
) -> tuple[int, ...]:
    members = set(component)
    endpoints: list[int] = []
    for bond in cuts:
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (left in members) != (right in members):
            for index in (left, right):
                if index not in endpoints:
                    endpoints.append(index)
                    if len(endpoints) == slots:
                        return tuple(endpoints)
    return tuple(endpoints)


def _event_descriptor(
    *,
    molecule: Chem.Mol,
    precursor_counts: np.ndarray,
    fragment_counts: np.ndarray,
    component: tuple[int, ...],
    cuts: tuple[Chem.Bond, ...],
    shift: int,
    mass: float,
    molecular_ion: bool,
) -> np.ndarray:
    cut_count = len(cuts)
    atom_count = molecule.GetNumAtoms()
    denominator = float(max(atom_count, 1))
    row = np.zeros(EVENT_FEATURE_DIM, dtype=np.float32)


    row[0 if molecular_ion else (1 if shift == 0 else 2)] = 1.0
    row[3] = cut_count / 3.0
    row[4 + shift + 3] = 1.0
    row[11:19] = fragment_counts / denominator
    row[19:27] = (precursor_counts - fragment_counts) / denominator
    row[27] = mass / 511.0
    row[28] = round(mass) / 511.0
    row[29] = len(component) / denominator
    if cuts:
        bond_type = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }
        for bond in cuts:
            channel = bond_type.get(bond.GetBondType())
            if channel is None:
                raise ValueError(f"Unsupported cut bond type: {bond.GetBondType()}")
            row[30 + channel] += 1.0
            row[34] += float(bond.IsInRing())
            row[35] += float(bond.GetIsAromatic())
            row[36] += float(bond.GetIsConjugated())
            for atom_index in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                symbol = molecule.GetAtomWithIdx(atom_index).GetSymbol()
                row[37 + ELEMENT_INDEX[symbol]] += 1.0
        row[30:37] /= cut_count
        row[37:45] /= 2 * cut_count
    return row


def _balanced_selection(events: list[_Event], maximum: int) -> list[_Event]:
    if len(events) <= maximum:
        return events


    molecular = [event for event in events if event.molecular_ion]
    buckets: dict[tuple[int, ...], list[_Event]] = {}
    for event in events:
        if not event.molecular_ion:
            buckets.setdefault(event.counts, []).append(event)
    for bucket in buckets.values():
        bucket.sort(key=lambda e: (e.cut_count, abs(e.shift), e.shift, e.component))
    keys = sorted(buckets, key=lambda key: (sum(key), key))
    selected = molecular[:1]
    depth = 0
    while len(selected) < maximum:
        added = False
        for key in keys:
            bucket = buckets[key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


class MolFeaturizer:


    def __init__(
        self,
        MAX_N: int = 48,
        bin_config: SpectrumBins | Mapping[str, Any] | None = None,
        event_config: Mapping[str, Any] | None = None,
        max_formulas: int = 8192,
        **kwargs: Any,
    ) -> None:
        self.MAX_N = int(MAX_N)
        self.max_formulas = int(max_formulas)
        if self.MAX_N < 1 or self.max_formulas < 1:
            raise ValueError("Capacity limits must be positive")
        if bin_config is None:
            self.bin_config = create_spectrum_bins()
        elif isinstance(bin_config, SpectrumBins):
            self.bin_config = bin_config
        else:
            self.bin_config = create_spectrum_bins(**dict(bin_config))
        config = {
            "enabled": True,
            "max_events": 8192,
            "max_bond_cuts": 3,
            "atom_slots": 4,
            "include_molecular_ion": True,
            "h_shifts": HYDROGEN_SHIFTS,
            "max_bin": 511,
            "elements": ELEMENTS,
            "selection": "balanced",
            "dedupe": "fragment_cut_hshift",
            "max_enumerated_combinations": 250000,
            "min_fragment_atoms": 1,
            "max_fragment_fraction": 0.98,
            "heavy_atom_bond_cuts_only": True,
        }
        config.update(dict(event_config or {}))
        if tuple(config["elements"]) != ELEMENTS or tuple(config["h_shifts"]) != HYDROGEN_SHIFTS:
            raise ValueError("45-dimensional events require the paper's element and H-shift order")
        if int(config["max_events"]) < 1 or int(config["atom_slots"]) < 1:
            raise ValueError("Event and boundary capacities must be positive")
        if int(config["max_bond_cuts"]) > 3:
            raise ValueError("The manuscript permits at most three bond cuts")
        self.event_config = config
        self._extra_options = dict(kwargs)

    def _explicit_hydrogen_molecule(self, molecule: Chem.Mol) -> Chem.Mol:
        if molecule is None:
            raise ValueError("Molecule cannot be None")
        explicit = Chem.AddHs(Chem.Mol(molecule))
        if len(Chem.GetMolFrags(explicit)) != 1:
            raise ValueError("CLEF requires one covalent molecular component")
        if explicit.GetNumAtoms() > self.MAX_N:
            raise ValueError(
                f"Molecule has {explicit.GetNumAtoms()} explicit-H atoms, "
                f"limit is {self.MAX_N}"
            )
        return explicit

    def _enumerate_events(self, molecule: Chem.Mol) -> list[_Event]:
        config = self.event_config
        if not config["enabled"]:
            return []
        precursor = molecular_counts(molecule)
        atom_count = molecule.GetNumAtoms()
        max_bin = min(int(config["max_bin"]), self.bin_config.bin_number)
        bonds = list(molecule.GetBonds())
        heavy_bonds = [
            bond for bond in bonds
            if molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() > 1
            and molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() > 1
        ]

        candidate_bonds = heavy_bonds
        events: list[_Event] = []
        seen: set[tuple[tuple[int, ...], int, int]] = set()
        if config["include_molecular_ion"]:
            mass = formula_exact_mass(precursor)
            nominal = round(mass)
            if 1 <= nominal <= max_bin:
                events.append(_Event(
                    features=_event_descriptor(
                        molecule=molecule, precursor_counts=precursor,
                        fragment_counts=precursor, component=tuple(range(atom_count)),
                        cuts=(), shift=0, mass=mass, molecular_ion=True,
                    ),
                    counts=tuple(map(int, precursor)), mass=nominal,
                    atom_indices=(), cut_count=0, shift=0,
                    component=tuple(range(atom_count)), molecular_ion=True,
                ))

        attempts = 0
        limit = max(int(config["max_enumerated_combinations"]), 0)
        max_cuts = min(3, max(int(config["max_bond_cuts"]), 0))
        for cut_count in range(1, max_cuts + 1):
            for cut_indices in combinations(range(len(candidate_bonds)), cut_count):
                if attempts >= limit:
                    break
                attempts += 1
                cuts = tuple(candidate_bonds[i] for i in cut_indices)
                components = _components_after_cuts(
                    atom_count, bonds, tuple(bond.GetIdx() for bond in cuts)
                )
                if len(components) < 2:
                    continue
                for component in components:
                    if len(component) / atom_count > float(config["max_fragment_fraction"]):
                        continue
                    heavy_count = sum(
                        molecule.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in component
                    )
                    if heavy_count < int(config["min_fragment_atoms"]):
                        continue
                    fragment = np.zeros(8, dtype=np.int16)
                    for atom_index in component:
                        symbol = molecule.GetAtomWithIdx(atom_index).GetSymbol()
                        fragment[ELEMENT_INDEX[symbol]] += 1
                    endpoints = _cut_endpoints(
                        cuts, component, int(config["atom_slots"])
                    )
                    if not endpoints:
                        continue
                    for shift in HYDROGEN_SHIFTS:
                        shifted = fragment.copy()
                        shifted[0] += shift
                        if np.any(shifted < 0) or np.any(shifted > precursor):
                            continue
                        mass = formula_exact_mass(shifted)
                        nominal = round(mass)
                        if not 1 <= nominal <= max_bin:
                            continue
                        key = (component, cut_count, shift)
                        if key in seen:
                            continue
                        seen.add(key)
                        events.append(_Event(
                            features=_event_descriptor(
                                molecule=molecule, precursor_counts=precursor,
                                fragment_counts=shifted, component=component,
                                cuts=cuts, shift=shift, mass=mass,
                                molecular_ion=False,
                            ),
                            counts=tuple(map(int, shifted)), mass=nominal,
                            atom_indices=endpoints, cut_count=cut_count,
                            shift=shift, component=component,
                        ))
            if attempts >= limit:
                break
        return _balanced_selection(events, int(config["max_events"]))

    def _packed_events(
        self, events: list[_Event], *, pad: bool
    ) -> dict[str, np.ndarray]:
        capacity = int(self.event_config["max_events"]) if pad else len(events)
        slots = int(self.event_config["atom_slots"])
        features = np.zeros((capacity, EVENT_FEATURE_DIM), dtype=np.float32)
        counts = np.zeros((capacity, 8), dtype=np.int16)
        mass = np.full(capacity, -1, dtype=np.int64)
        atom_indices = np.full((capacity, slots), -1, dtype=np.int64)
        mask = np.zeros(capacity, dtype=np.float32)
        for i, event in enumerate(events):
            features[i] = event.features
            counts[i] = event.counts
            mass[i] = event.mass
            atom_indices[i, :len(event.atom_indices)] = event.atom_indices
            mask[i] = 1.0
        return {
            "event_features": features,
            "event_counts": counts,
            "event_mass_idx": mass,
            "event_atom_idx": atom_indices,
            "event_mask": mask,
        }

    def _build_events(
        self, molecule: Chem.Mol
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        explicit = self._explicit_hydrogen_molecule(molecule)
        packed = self._packed_events(self._enumerate_events(explicit), pad=True)
        return (
            packed["event_features"], packed["event_mass_idx"],
            packed["event_atom_idx"], packed["event_mask"],
        )

    def __call__(self, molecule: Chem.Mol, **kwargs: Any) -> dict[str, np.ndarray]:
        explicit = self._explicit_hydrogen_molecule(molecule)
        adjacency, atom_features, atom_mask = graph_arrays(explicit, self.MAX_N)
        precursor = molecular_counts(explicit)
        formulas = enumerate_formula_counts(precursor, self.max_formulas)
        formula_mask = np.ones(len(formulas), dtype=np.float32)
        formula_features = cumulative_formula_features(formulas)
        isotope_indices = np.full((len(formulas), 12), -1, dtype=np.int64)
        isotope_intensity = np.zeros((len(formulas), 12), dtype=np.float32)
        for index, counts in enumerate(formulas):
            isotope_indices[index], isotope_intensity[index] = isotope_template(
                counts,
                first_bin_center=self.bin_config.first_bin_center,
                bin_width=self.bin_config.bin_width,
                bin_number=self.bin_config.bin_number,
            )
        packed = self._packed_events(self._enumerate_events(explicit), pad=False)


        packed["event_mass_idx"] = self.bin_config.mass_to_index(
            packed["event_mass_idx"]
        )
        formula_lookup = {tuple(map(int, row)): i for i, row in enumerate(formulas)}
        event_formula_index = np.asarray(
            [formula_lookup[tuple(map(int, row))] for row in packed["event_counts"]],
            dtype=np.int64,
        )
        return {
            "adj": adjacency,
            "vect_feat": atom_features,
            "input_mask": atom_mask,
            "formula_counts": formulas,
            "formula_features": formula_features,
            "formula_mask": formula_mask,
            "isotope_mass_idx": isotope_indices,
            "isotope_intensity": isotope_intensity,
            **packed,
            "event_formula_index": event_formula_index,
        }
