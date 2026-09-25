from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from pyarrow import parquet as pq
from rdkit import Chem
from torch.utils.data import Dataset

from clef.featurize.featurize import MolFeaturizer


_VARIABLE_AXES = frozenset(
    {
        "formula_counts",
        "formula_features",
        "formula_mask",
        "isotope_mass_idx",
        "isotope_intensity",
        "event_features",
        "event_mass_idx",
        "event_atom_idx",
        "event_mask",
        "event_counts",
        "event_formula_index",
    }
)
_PAD_MINUS_ONE = frozenset(
    {"isotope_mass_idx", "event_mass_idx", "event_atom_idx", "event_formula_index"}
)


def _molecule_from_row(row: pd.Series) -> Chem.Mol:


    binary = row.get("rdmol")
    if binary is not None and not (isinstance(binary, float) and np.isnan(binary)):
        try:
            molecule = Chem.Mol(bytes(binary))
            if molecule is not None:
                return molecule
        except (TypeError, ValueError, RuntimeError):
            pass
    smiles = row.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("A dataset row must contain a valid rdmol or smiles value")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    return Chem.AddHs(molecule)


def _candidate_indices(formula_counts: np.ndarray, event_counts: np.ndarray) -> np.ndarray:


    lookup: dict[tuple[int, ...], int] = {}
    for index, counts in enumerate(np.asarray(formula_counts)):
        lookup.setdefault(tuple(int(value) for value in counts), index)
    return np.asarray(
        [lookup.get(tuple(int(value) for value in counts), -1) for counts in event_counts],
        dtype=np.int64,
    )


class _SpectrumTargets:


    def __init__(self, spectrum_bins: Any, pred_config: Mapping[str, Any] | None = None):
        self.spectrum_bins = spectrum_bins
        self.pred_config = dict(pred_config or {})

    def __call__(self, molecule: Chem.Mol, peaks: Any) -> dict[str, np.ndarray]:
        del molecule
        if peaks is None:
            values = np.zeros((0, 2), dtype=np.float64)
        else:
            if len(peaks) == 0:
                values = np.zeros((0, 2), dtype=np.float64)
            else:
                values = np.stack([np.asarray(peak, dtype=np.float64) for peak in peaks])
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("A spectrum must be an array of (m/z, intensity) pairs")
        if values.size and (not np.isfinite(values).all() or (values[:, 1] < 0).any()):
            raise ValueError("Spectrum peaks must be finite with nonnegative intensity")
        dense = self.spectrum_bins.peaks_to_dense(values)
        return {"spect": np.asarray(dense, dtype=np.float32)}


class ParquetDataset(Dataset):


    def __init__(
        self,
        filename: str | Path,
        spectrum_bins: Any,
        featurize_config: Mapping[str, Any] | None = None,
        pred_config: Mapping[str, Any] | None = None,
        filter_config: Mapping[str, Any] | None = None,
        include_targets: bool = True,
    ) -> None:
        self.filename = Path(filename)
        self.filter_config = dict(filter_config or {})
        self.include_targets = bool(include_targets)
        columns = None
        if not self.include_targets:
            columns = [
                name for name in pq.read_schema(self.filename).names if name != "spect"
            ]
        self.df = pd.read_parquet(self.filename, columns=columns).copy()
        if "_source_row_idx" not in self.df:
            self.df["_source_row_idx"] = np.arange(len(self.df), dtype=np.int64)
        self.df = self._filter_rows(self.df).reset_index(drop=True)

        event_dir = self.filter_config.get("event_table_dir")
        self.event_table = self._load_event_table(event_dir) if event_dir else None
        self._event_arrays: dict[str, np.ndarray] | None = None

        options = dict(featurize_config or {})
        options["bin_config"] = spectrum_bins
        if self.event_table is not None:
            event_options = dict(options.get("event_config") or {})
            event_options["enabled"] = False
            options["event_config"] = event_options
        self.featurizer = MolFeaturizer(**options)
        self.pred_featurizer = _SpectrumTargets(spectrum_bins, pred_config)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_event_arrays"] = None
        return state

    def _filter_rows(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        max_atoms = self.filter_config.get("max_n", self.filter_config.get("max_atoms"))
        max_mass = self.filter_config.get("max_mol_wt", self.filter_config.get("max_mass"))
        max_candidates = self.filter_config.get("max_formula_candidates")
        max_atoms = max_atoms if max_atoms is not None and float(max_atoms) > 0 else None
        max_mass = max_mass if max_mass is not None and float(max_mass) > 0 else None
        max_candidates = max_candidates if max_candidates is not None and float(max_candidates) > 0 else None
        if not any(value is not None for value in (max_atoms, max_mass, max_candidates)):
            return dataframe
        keep = np.ones(len(dataframe), dtype=bool)
        if max_atoms is not None and "explicit_atom_count" in dataframe:
            keep &= np.asarray(dataframe["explicit_atom_count"], dtype=float) <= float(max_atoms)
        if max_mass is not None and "mol_wt" in dataframe:
            keep &= np.asarray(dataframe["mol_wt"], dtype=float) <= float(max_mass)
        if max_candidates is not None and "formula_candidate_count" in dataframe:
            keep &= np.asarray(dataframe["formula_candidate_count"], dtype=float) <= float(max_candidates)
        return dataframe.loc[keep]

    @staticmethod
    def _load_event_table(directory: str | Path) -> dict[str, Any]:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(root)
        pointers = np.load(root / "event_ptr.npy", allow_pickle=False)
        source_rows = np.load(root / "source_row_idx.npy", allow_pickle=False)
        if pointers.ndim != 1 or source_rows.ndim != 1 or len(pointers) != len(source_rows) + 1:
            raise ValueError("Event table row pointers and source-row IDs are inconsistent")
        if pointers[0] != 0 or np.any(np.diff(pointers) < 0):
            raise ValueError("Event table row pointers must be nondecreasing from zero")
        source_to_table = {int(source): index for index, source in enumerate(source_rows)}
        if len(source_to_table) != len(source_rows):
            raise ValueError("Event table contains duplicate source-row IDs")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        mass_contract = manifest.get("event_mass_contract", {})
        if mass_contract.get("stored_value") == "zero-based tensor index":
            physical_mass = False
        else:
            physical_mass = not bool(manifest.get("config", {}).get("canonical_mass_index", False))
        return {
            "dir": root,
            "event_ptr": pointers,
            "source_to_table": source_to_table,
            "physical_mass": physical_mass,
        }

    def _table_arrays(self) -> dict[str, np.ndarray]:
        if self.event_table is None:
            raise RuntimeError("No precomputed event table is configured")
        if self._event_arrays is None:
            root = self.event_table["dir"]
            arrays = {
                name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ("event_features", "event_mass_idx", "event_atom_idx")
            }
            count = int(self.event_table["event_ptr"][-1])
            if any(len(array) != count for array in arrays.values()):
                raise ValueError("Event table array lengths differ from its row pointers")
            self._event_arrays = arrays
        return self._event_arrays

    def _events_from_table(
        self, source_row: int, molecule: Chem.Mol, formula_counts: np.ndarray
    ) -> dict[str, np.ndarray]:
        assert self.event_table is not None
        position = self.event_table["source_to_table"].get(source_row)
        if position is None:
            raise KeyError(f"Source row {source_row} is absent from the event table")
        pointers = self.event_table["event_ptr"]
        start, stop = int(pointers[position]), int(pointers[position + 1])
        arrays = self._table_arrays()
        features = np.array(arrays["event_features"][start:stop], dtype=np.float32, copy=True)
        masses = np.array(arrays["event_mass_idx"][start:stop], dtype=np.int64, copy=True)
        atom_ids = np.array(arrays["event_atom_idx"][start:stop], dtype=np.int64, copy=True)
        if self.event_table["physical_mass"]:
            masses -= 1
        if features.ndim != 2 or features.shape[1] != 45:
            raise ValueError("Event table must contain 45-dimensional descriptors")
        if atom_ids.ndim != 2 or atom_ids.shape[1] != 4:
            raise ValueError("Event table must contain four endpoint slots")


        counts = np.rint(features[:, 11:19] * molecule.GetNumAtoms()).astype(np.int64)
        return {
            "event_features": features,
            "event_mass_idx": masses,
            "event_atom_idx": atom_ids,
            "event_mask": np.ones((len(features),), dtype=np.float32),
            "event_counts": counts,
            "event_formula_index": _candidate_indices(formula_counts, counts),
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        row = self.df.iloc[index]
        molecule = _molecule_from_row(row)
        features = self.featurizer(molecule)
        output: dict[str, np.ndarray | int] = {
            key: np.asarray(value) for key, value in features.items()
        }
        if self.event_table is not None:
            output.update(
                self._events_from_table(
                    int(row["_source_row_idx"]),
                    molecule,
                    np.asarray(output["formula_counts"], dtype=np.int64),
                )
            )
        if "event_counts" in output and "event_formula_index" not in output:
            output["event_formula_index"] = _candidate_indices(
                np.asarray(output["formula_counts"]), np.asarray(output["event_counts"])
            )
        if "spect" in self.df:
            output.update(self.pred_featurizer(molecule, row["spect"]))
        output["input_idx"] = int(index)
        return output


def event_dynamic_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:


    if not batch:
        raise ValueError("Cannot collate an empty batch")
    keys = set(batch[0])
    if any(set(item) != keys for item in batch):
        raise ValueError("Examples in a batch have different feature keys")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        arrays = [np.asarray(item[key]) for item in batch]
        if key in _VARIABLE_AXES:
            width = max(1, *(array.shape[0] for array in arrays))
            trailing = arrays[0].shape[1:]
            if any(array.shape[1:] != trailing for array in arrays):
                raise ValueError(f"Mismatched trailing shape for {key}")
            fill = -1 if key in _PAD_MINUS_ONE else 0
            padded = np.full((len(batch), width, *trailing), fill, dtype=arrays[0].dtype)
            for index, array in enumerate(arrays):
                padded[index, : len(array)] = array
            stacked = padded
        else:
            stacked = np.stack(arrays, axis=0)
        result[key] = torch.from_numpy(np.ascontiguousarray(stacked))
    return result


__all__ = ["ParquetDataset", "event_dynamic_collate"]
