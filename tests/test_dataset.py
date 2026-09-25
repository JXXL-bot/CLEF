from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem

from clef.dataset import ParquetDataset, event_dynamic_collate
from clef.featurize.featurize import MolFeaturizer
from clef.msutil.binutils import create_spectrum_bins


def _molecule(smiles: str) -> Chem.Mol:
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_parquet_features_and_dynamic_padding(tmp_path):
    molecules = [_molecule("CO"), _molecule("CCO")]
    path = tmp_path / "structures.parquet"
    pd.DataFrame(
        {
            "rdmol": [molecule.ToBinary() for molecule in molecules],
            "smiles": ["CO", "CCO"],
            "spect": [[[31.0, 1.0], [32.0, 2.0]], [[45.0, 3.0]]],
        }
    ).to_parquet(path, index=False)
    dataset = ParquetDataset(
        path,
        create_spectrum_bins(),
        {"MAX_N": 48, "event_config": {"max_events": 64}},
    )
    batch = event_dynamic_collate([dataset[0], dataset[1]])
    assert tuple(batch["adj"].shape) == (2, 4, 48, 48)
    assert batch["formula_counts"].shape[0] == 2
    assert batch["formula_features"].shape[1] == batch["formula_counts"].shape[1]
    assert batch["event_features"].shape[0] == 2
    assert batch["spect"][0, 30] == 1.0
    assert batch["spect"][0, 31] == 2.0
    assert batch["input_idx"].tolist() == [0, 1]
    structure_only = ParquetDataset(
        path, create_spectrum_bins(), {"MAX_N": 48}, include_targets=False
    )
    assert "spect" not in structure_only.df
    assert "spect" not in structure_only[0]


def test_precomputed_events_follow_source_row_indices(tmp_path):
    molecule = _molecule("CO")
    path = tmp_path / "structures.parquet"
    pd.DataFrame(
        {"rdmol": [molecule.ToBinary()], "smiles": ["CO"]}
    ).to_parquet(path, index=False)
    bins = create_spectrum_bins()
    featurizer = MolFeaturizer(
        MAX_N=48,
        bin_config=bins,
        event_config={"max_events": 64},
    )
    features, masses, atom_ids, mask = featurizer._build_events(molecule)
    count = int(mask.sum())
    table_dir = tmp_path / "events"
    table_dir.mkdir()
    np.save(table_dir / "event_ptr.npy", np.asarray([0, count], dtype=np.int64))
    np.save(table_dir / "source_row_idx.npy", np.asarray([0], dtype=np.int64))
    np.save(table_dir / "event_features.npy", features[:count])
    np.save(table_dir / "event_mass_idx.npy", masses[:count] - 1)
    np.save(table_dir / "event_atom_idx.npy", atom_ids[:count])
    (table_dir / "manifest.json").write_text(
        json.dumps({"event_mass_contract": {"stored_value": "zero-based tensor index"}}),
        encoding="utf-8",
    )
    dataset = ParquetDataset(
        path,
        bins,
        {"MAX_N": 48},
        filter_config={"event_table_dir": table_dir},
    )
    row = dataset[0]
    assert len(row["event_features"]) == count
    assert np.array_equal(row["event_mass_idx"], masses[:count] - 1)
    assert np.all(row["event_formula_index"] >= 0)
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored[0]["event_features"]) == count
