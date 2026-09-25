import csv
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("clef_prepare_dataset", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles", "split", "spectrum_json", "spectrum_kind"])
        writer.writeheader()
        writer.writerows(rows)


def test_build_preserves_declared_splits_and_serializes_explicit_hydrogens(tmp_path):
    source = tmp_path / "input.csv"
    write_csv(source, [
        {"molecule_id": "a", "smiles": "CCO", "split": "train", "spectrum_json": "[[31, 1], [31, 2], [45, 3]]"},
        {"molecule_id": "b", "smiles": "CCCO", "split": "val", "spectrum_json": "[[43, 4]]"},
        {"molecule_id": "c", "smiles": "CCF", "split": "test", "spectrum_json": "[[47, 5]]"},
    ])
    result = prepare.build(type("Args", (), {"input": source, "output_dir": tmp_path / "built", "overwrite": False, "prediction_structures": False})())
    assert result["counts"]["accepted_by_split"] == {"train": 1, "val": 1, "test": 1}
    frame = pq.read_table(tmp_path / "built" / "clef_train.parquet").to_pandas()
    assert [[float(value) for value in peak] for peak in frame.iloc[0]["spect"]] == [[31.0, 3.0], [45.0, 3.0]]
    molecule = Chem.Mol(frame.iloc[0]["rdmol"])
    assert molecule.GetNumAtoms() > Chem.MolFromSmiles("CCO").GetNumAtoms()
    assert frame.iloc[0]["formula"] == "C2H6O"
    assert frame.iloc[0]["mol_id"] == "clef:a"
    assert frame.iloc[0]["eval_id"] == 0


def test_cross_split_connectivity_is_rejected_and_prediction_dummy_is_labeled(tmp_path):
    source = tmp_path / "input.csv"
    write_csv(source, [
        {"molecule_id": "a", "smiles": "CCO", "split": "train", "spectrum_json": "[[31, 1]]"},
        {"molecule_id": "b", "smiles": "CCO", "split": "test", "spectrum_json": "[[31, 1]]"},
        {"molecule_id": "c", "smiles": "CCF", "split": "val", "spectrum_json": ""},
    ])
    result = prepare.build(type("Args", (), {"input": source, "output_dir": tmp_path / "built", "overwrite": False, "prediction_structures": True})())
    assert result["counts"]["accepted"] == 1
    assert result["leakage"]["cross_split_connectivity_overlap"] == {"train__val": 0, "train__test": 0, "val__test": 0}
    row = pq.read_table(tmp_path / "built" / "clef_val.parquet").to_pandas().iloc[0]
    assert [[float(value) for value in peak] for peak in row["spect"]] == [[1.0, 0.0]]
    assert row["spectrum_kind"] == "prediction_dummy"
    assert bool(row["is_observed_spectrum"]) is False
    audit = (tmp_path / "built" / "clef_filter_audit.csv").read_text(encoding="utf-8")
    assert audit.count("cross_split_connectivity_leakage") == 2


def test_noncanonical_split_label_is_rejected(tmp_path):
    source = tmp_path / "input.csv"
    write_csv(source, [
        {"molecule_id": "case-sensitive", "smiles": "CCO", "split": "Train", "spectrum_json": "[[31, 1]]"},
    ])
    result = prepare.build(type("Args", (), {"input": source, "output_dir": tmp_path / "built", "overwrite": False, "prediction_structures": False})())
    assert result["counts"]["accepted"] == 0
    assert result["counts"]["drop_reasons"] == {"split_invalid": 1}


def test_synthetic_example_is_not_labeled_as_an_observed_spectrum(tmp_path):
    source = tmp_path / "input.csv"
    write_csv(source, [
        {"molecule_id": "fixture", "smiles": "CCO", "split": "train", "spectrum_json": "[[31, 1]]", "spectrum_kind": "synthetic_example"},
    ])
    prepare.build(type("Args", (), {"input": source, "output_dir": tmp_path / "built", "overwrite": False, "prediction_structures": False})())
    row = pq.read_table(tmp_path / "built" / "clef_train.parquet").to_pandas().iloc[0]
    assert row["spectrum_kind"] == "synthetic_example"
    assert bool(row["is_observed_spectrum"]) is False
