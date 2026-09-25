from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


ADAPTER_VERSION = "clef_dataset_adapter.v1"
SPLITS = ("train", "val", "test")
ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl"})
MAX_EXPLICIT_ATOMS = 48
MAX_MOL_WT = 400.0
MAX_FORMULA_CANDIDATES = 8192

MZ_MIN = 0.5
MZ_MAX_EXCLUSIVE = 512.5


DATASET_SCHEMA = pa.schema(
    [
        pa.field("mol_id", pa.string()),
        pa.field("molecule_id", pa.string()),
        pa.field("smiles", pa.string()),
        pa.field("split", pa.string()),
        pa.field("formula", pa.string()),
        pa.field("connectivity_key", pa.string()),
        pa.field("inchi_key", pa.string()),
        pa.field("eval_id", pa.int64()),
        pa.field("rdmol", pa.binary()),
        pa.field("spect", pa.list_(pa.list_(pa.float64()))),
        pa.field("spectrum_kind", pa.string()),
        pa.field("is_observed_spectrum", pa.bool_()),
        pa.field("explicit_atom_count", pa.int32()),
        pa.field("mol_wt", pa.float64()),
        pa.field("formula_candidate_count", pa.int64()),
    ]
)


def formula_candidate_count(mol: Chem.Mol) -> int:
    counts: Counter[str] = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    result = 1
    for count in counts.values():
        result *= count + 1
    return result


def read_spectrum(value: str, prediction_structures: bool) -> tuple[list[list[float]], str, int, int]:

    if value is None or not str(value).strip():
        if prediction_structures:
            return [[1.0, 0.0]], "prediction_dummy", 0, 1
        raise ValueError("spectrum_missing")
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("spectrum_json_invalid") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("spectrum_empty")
    sums: dict[float, float] = {}
    for peak in raw:
        if not isinstance(peak, (list, tuple)) or len(peak) != 2:
            raise ValueError("spectrum_peak_not_pair")
        try:
            mz, intensity = float(peak[0]), float(peak[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("spectrum_peak_not_numeric") from exc
        if not math.isfinite(mz) or not math.isfinite(intensity):
            raise ValueError("spectrum_nonfinite")
        if intensity < 0:
            raise ValueError("spectrum_negative_intensity")
        if not (MZ_MIN <= mz < MZ_MAX_EXCLUSIVE):
            raise ValueError("spectrum_mz_out_of_512_range")
        sums[mz] = sums.get(mz, 0.0) + intensity
    peaks = [[mz, sums[mz]] for mz in sorted(sums)]
    if not any(intensity > 0 for _, intensity in peaks):
        raise ValueError("spectrum_total_nonpositive")
    return peaks, "", len(raw), len(peaks)


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_structure(smiles: str) -> tuple[dict[str, Any], str]:
    try:
        source = Chem.MolFromSmiles(smiles, sanitize=True)
        if source is None:
            return {}, "smiles_parse_failed"
        Chem.SanitizeMol(source)
        mol = Chem.AddHs(Chem.Mol(source))
        Chem.SanitizeMol(mol)
        elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
        invalid_elements = sorted(elements.difference(ALLOWED_ELEMENTS))
        if invalid_elements:
            return {}, "element_not_allowed:" + ",".join(invalid_elements)
        atom_count = mol.GetNumAtoms()
        if atom_count > MAX_EXPLICIT_ATOMS:
            return {}, "explicit_atom_count_gt_48"
        mol_wt = float(rdMolDescriptors.CalcExactMolWt(mol))
        if mol_wt > MAX_MOL_WT:
            return {}, "mol_wt_gt_400"
        count = formula_candidate_count(mol)
        if count > MAX_FORMULA_CANDIDATES:
            return {}, "formula_candidate_count_gt_8192"
        source_smiles = Chem.MolToSmiles(source, canonical=True, isomericSmiles=True)
        inchi_key = Chem.MolToInchiKey(source)
        if not inchi_key:
            return {}, "inchi_key_missing"
        return {
            "smiles": source_smiles,
            "rdmol": mol.ToBinary(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "inchi_key": inchi_key,
            "connectivity_key": inchi_key.split("-", 1)[0],
            "explicit_atom_count": int(atom_count),
            "mol_wt": mol_wt,
            "formula_candidate_count": int(count),
        }, ""
    except Exception as exc:
        return {}, "structure_error:" + type(exc).__name__


def audit_row(row: dict[str, Any], *, decision: str, reason: str = "", **extra: Any) -> dict[str, Any]:
    out = {
        "input_row": row["input_row"],
        "molecule_id": row["molecule_id"],
        "split": row["split"],
        "decision": decision,
        "reason": reason,
        "mol_id": row.get("mol_id", ""),
        "connectivity_key": row.get("connectivity_key", ""),
        "formula": row.get("formula", ""),
        "explicit_atom_count": row.get("explicit_atom_count", ""),
        "mol_wt": row.get("mol_wt", ""),
        "formula_candidate_count": row.get("formula_candidate_count", ""),
        "initial_peak_count": row.get("initial_peak_count", ""),
        "final_peak_count": row.get("final_peak_count", ""),
        "duplicate_peak_count": row.get("duplicate_peak_count", ""),
    }
    out.update(extra)
    return out


def parse_input(path: Path, prediction_structures: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    required = {"molecule_id", "smiles", "split", "spectrum_json"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError("input CSV missing columns: " + ", ".join(sorted(missing)))
        for input_row, source_row in enumerate(reader, start=2):
            row = {
                "input_row": input_row,
                "molecule_id": clean_text(source_row.get("molecule_id")),
                "split": clean_text(source_row.get("split")),
            }
            if not row["molecule_id"]:
                audits.append(audit_row(row, decision="drop", reason="molecule_id_missing"))
                continue
            row["mol_id"] = "clef:" + row["molecule_id"]
            if row["split"] not in SPLITS:
                audits.append(audit_row(row, decision="drop", reason="split_invalid"))
                continue
            structure, reason = parse_structure(clean_text(source_row.get("smiles")))
            if reason:
                audits.append(audit_row(row, decision="drop", reason=reason))
                continue
            row.update(structure)
            try:
                peaks, inferred_kind, before, after = read_spectrum(
                    source_row.get("spectrum_json"), prediction_structures
                )
            except ValueError as exc:
                audits.append(audit_row(row, decision="drop", reason=str(exc)))
                continue
            supplied_kind = clean_text(source_row.get("spectrum_kind"))
            if inferred_kind == "prediction_dummy":
                spectrum_kind = inferred_kind
            elif not supplied_kind:
                spectrum_kind = "observed"
            elif supplied_kind in {"observed", "synthetic_example"}:
                spectrum_kind = supplied_kind
            else:
                audits.append(audit_row(row, decision="drop", reason="spectrum_kind_invalid"))
                continue
            row["spect"] = peaks
            row["spectrum_kind"] = spectrum_kind
            row["is_observed_spectrum"] = spectrum_kind == "observed"
            row["initial_peak_count"] = before
            row["final_peak_count"] = after
            row["duplicate_peak_count"] = before - after
            accepted.append(row)
    return accepted, audits


def reject_duplicate_ids(rows: list[dict[str, Any]], audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["molecule_id"] for row in rows)
    kept = []
    for row in rows:
        if counts[row["molecule_id"]] > 1:
            audits.append(audit_row(row, decision="drop", reason="molecule_id_duplicate"))
        else:
            kept.append(row)
    return kept


def reject_cross_split_connectivity(rows: list[dict[str, Any]], audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        locations[row["connectivity_key"]].add(row["split"])
    leaking = {key for key, splits in locations.items() if len(splits) > 1}
    kept = []
    for row in rows:
        if row["connectivity_key"] in leaking:
            audits.append(audit_row(row, decision="drop", reason="cross_split_connectivity_leakage"))
        else:
            kept.append(row)
    return kept


def dataset_record(row: dict[str, Any], eval_id: int) -> dict[str, Any]:
    return {
        "mol_id": row["mol_id"],
        "molecule_id": row["molecule_id"],
        "smiles": row["smiles"],
        "split": row["split"],
        "formula": row["formula"],
        "connectivity_key": row["connectivity_key"],
        "inchi_key": row["inchi_key"],
        "eval_id": eval_id,
        "rdmol": row["rdmol"],
        "spect": row["spect"],
        "spectrum_kind": row["spectrum_kind"],
        "is_observed_spectrum": row["is_observed_spectrum"],
        "explicit_atom_count": row["explicit_atom_count"],
        "mol_wt": row["mol_wt"],
        "formula_candidate_count": row["formula_candidate_count"],
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "input_row", "molecule_id", "split", "decision", "reason", "mol_id",
        "connectivity_key", "formula", "explicit_atom_count", "mol_wt",
        "formula_candidate_count", "initial_peak_count", "final_peak_count",
        "duplicate_peak_count", "eval_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:


    columns = {
        field.name: [record.get(field.name) for record in records]
        for field in DATASET_SCHEMA
    }
    pq.write_table(pa.Table.from_pydict(columns, schema=DATASET_SCHEMA), path, compression="zstd")


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    output_path = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and any(output_path.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is nonempty: {output_path}; use --overwrite to replace it")
    rows, audits = parse_input(input_path, args.prediction_structures)
    rows = reject_duplicate_ids(rows, audits)
    rows = reject_cross_split_connectivity(rows, audits)
    rows.sort(key=lambda item: item["input_row"])
    accepted_audits: list[dict[str, Any]] = []
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for eval_id, row in enumerate(rows):
        record = dataset_record(row, eval_id)
        records_by_split[row["split"]].append(record)
        accepted_audits.append(audit_row(row, decision="accept", eval_id=eval_id))
    all_audits = sorted(audits + accepted_audits, key=lambda item: item["input_row"])
    duplicate_spectra = Counter(
        tuple(tuple(peak) for peak in row["spect"])
        for row in rows if row["is_observed_spectrum"]
    )
    split_connectivity = {
        split: {row["connectivity_key"] for row in values}
        for split, values in records_by_split.items()
    }
    pairs = {
        f"{left}__{right}": len(split_connectivity[left] & split_connectivity[right])
        for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]
    }
    if any(pairs.values()):
        raise AssertionError(f"cross-split connectivity leakage survived filtering: {pairs}")
    temp_parent = output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".clef_dataset_", dir=temp_parent))
    try:
        output_files: list[str] = []
        for split in SPLITS:
            file_name = f"clef_{split}.parquet"
            write_parquet(staging / file_name, records_by_split[split])
            output_files.append(file_name)
        split_rows = []
        for split in SPLITS:
            for record in records_by_split[split]:
                split_rows.append({
                    "split": split, "eval_id": record["eval_id"], "mol_id": record["mol_id"],
                    "molecule_id": record["molecule_id"], "connectivity_key": record["connectivity_key"],
                })
        write_csv(staging / "clef_split_ids.csv", split_rows)
        write_csv(staging / "clef_filter_audit.csv", all_audits)
        output_files.extend(["clef_split_ids.csv", "clef_filter_audit.csv"])
        distributions = {
            split: {
                "rows": len(records_by_split[split]),
                "formula": dict(sorted(Counter(row["formula"] for row in records_by_split[split]).items())),
                "explicit_atom_count": dict(sorted(Counter(str(row["explicit_atom_count"]) for row in records_by_split[split]).items())),
            }
            for split in SPLITS
        }
        manifest = {
            "schema": "clef.dataset_manifest.v1",
            "adapter_version": ADAPTER_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {"path": str(input_path)},
            "contract": {
                "declared_splits_preserved": True,
                "allowed_splits": list(SPLITS),
                "explicit_hydrogens": True,
                "max_explicit_atoms": MAX_EXPLICIT_ATOMS,
                "max_mol_wt": MAX_MOL_WT,
                "allowed_elements": sorted(ALLOWED_ELEMENTS),
                "formula_candidate_count": "product(element_count + 1)",
                "max_formula_candidate_count": MAX_FORMULA_CANDIDATES,
                "observed_mz_range": {"min_inclusive": MZ_MIN, "max_exclusive": MZ_MAX_EXCLUSIVE},
                "duplicate_peak_handling": "sum exact equal raw m/z values; no m/z rounding or binning",
                "prediction_structures": bool(args.prediction_structures),
            },
            "counts": {
                "accepted": len(rows), "dropped": len(audits),
                "accepted_by_split": {split: len(values) for split, values in records_by_split.items()},
                "drop_reasons": dict(sorted(Counter(row["reason"] for row in audits).items())),
                "observed_duplicate_spectrum_rows": sum(count for count in duplicate_spectra.values() if count > 1),
                "observed_duplicate_spectrum_groups": sum(1 for count in duplicate_spectra.values() if count > 1),
            },
            "distributions": distributions,
            "leakage": {"cross_split_connectivity_overlap": pairs, "status": "pass"},
            "outputs": output_files,
            "resume_strategy": "Rebuild all splits and dependent event tables if the source CSV or filtering code changes; eval_id is regenerated.",
            "downstream_invalidation": [
                "compatibility audit", "train and validation Parquet row coordinates", "event tables",
                "candidate manifests", "retrieval inputs", "prediction identity sidecars", "all metrics joined by eval_id",
            ],
        }
        manifest_path = staging / "clef_dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output_path.exists():
            shutil.rmtree(output_path)
        os.replace(staging, output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare molecular spectra for CLEF.')
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 CSV with molecule_id, smiles, split, spectrum_json, and optional spectrum_kind")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-structures", action="store_true", help="allow blank spectrum_json and label [[1, 0]] as a non-observed dummy")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory atomically after a successful rebuild")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({"accepted": manifest["counts"]["accepted"], "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
