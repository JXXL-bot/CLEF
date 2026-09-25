from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl"})
N_BINS = 512
MAX_MASS = 400.0
MAX_ATOMS = 48
MAX_SUBFORMULAS = 8192
NEAR_SPECTRUM_THRESHOLD = 0.995


def present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip() and value.strip().lower() not in {"nan", "none", "null"})
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return False
    except (TypeError, ValueError):
        pass
    return True


def text_value(value: object) -> str:
    return str(value).strip() if present(value) else ""


def parse_msp_metadata(path: Path) -> dict[str, dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "").replace("_", "")
            current.setdefault(key, value.strip())
    if current:
        records.append(current)
    by_id: dict[str, dict[str, str]] = {}
    for record in records:
        record_id = record.get("id", "")
        if record_id:
            by_id.setdefault(record_id, record)
    return by_id


def standardized_molecule(row: pd.Series) -> Chem.Mol:
    molecule = None
    if present(row.get("rdmol")):
        try:
            molecule = Chem.Mol(bytes(row["rdmol"]))
        except (TypeError, ValueError, RuntimeError):
            molecule = None
    if molecule is None:
        smiles = text_value(row.get("smiles"))
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError("invalid structure")
    Chem.SanitizeMol(molecule)
    heavy = Chem.RemoveHs(Chem.Mol(molecule))
    Chem.SanitizeMol(heavy)
    if len(Chem.GetMolFrags(heavy)) != 1:
        raise ValueError("multiple covalent components")
    explicit = Chem.AddHs(heavy)
    Chem.SanitizeMol(explicit)
    if any(atom.GetNumImplicitHs() for atom in explicit.GetAtoms()):
        raise ValueError("implicit hydrogen after standardization")
    return explicit


def formula_candidate_count(molecule: Chem.Mol) -> int:
    counts = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    result = 1
    for count in counts.values():
        result *= count + 1
    return result


def connectivity_key(row: pd.Series, molecule: Chem.Mol) -> str:
    supplied = text_value(row.get("inchi_key")).upper()
    if supplied.startswith("INCHIKEY="):
        supplied = supplied.split("=", 1)[1]
    key = supplied or Chem.MolToInchiKey(Chem.RemoveHs(Chem.Mol(molecule)))
    if not key:
        raise ValueError("missing InChIKey")
    return key.split("-", 1)[0]


def dense_spectrum(value: object) -> np.ndarray:
    result = np.zeros(N_BINS, dtype=np.float32)
    if value is None:
        return result
    for peak in value:
        try:
            mz, intensity = float(peak[0]), float(peak[1])
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(mz) or not np.isfinite(intensity):
            continue
        index = round(mz - 1.0)
        if 0 <= index < N_BINS:
            result[index] += max(intensity, 0.0)
    return result


def tic_normalize(spectrum: np.ndarray) -> np.ndarray:
    total = float(spectrum.sum())
    return spectrum / total if total > 0 else spectrum.copy()


def sqrt_cosine_vector(spectrum: np.ndarray) -> np.ndarray:
    transformed = np.sqrt(np.maximum(spectrum, 0))
    norm = float(np.linalg.norm(transformed))
    return transformed / norm if norm > 0 else transformed


def spectrum_key(spectrum: np.ndarray) -> bytes:
    return np.round(tic_normalize(spectrum), decimals=7).astype("<f4").tobytes()


def sparse_spectrum(spectrum: np.ndarray) -> list[list[float]]:
    maximum = float(spectrum.max())
    scaled = spectrum / maximum * 999.0
    indices = np.flatnonzero(scaled > 0)
    return [[float(index + 1), float(scaled[index])] for index in indices]


def compatible_records(source: pd.DataFrame, msp_by_id: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, np.ndarray, list[bytes]]:
    required = {"mol_id", "spect"}
    if missing := required.difference(source.columns):
        raise ValueError("SWGDRUG table missing columns: " + ", ".join(sorted(missing)))
    if "rdmol" not in source and "smiles" not in source:
        raise ValueError("SWGDRUG table requires rdmol or smiles")
    grouped: dict[str, list[tuple[int, pd.Series, Chem.Mol, np.ndarray, float, int]]] = defaultdict(list)
    for index, row in source.iterrows():
        try:
            molecule = standardized_molecule(row)
            if {atom.GetSymbol() for atom in molecule.GetAtoms()}.difference(ALLOWED_ELEMENTS):
                continue
            if molecule.GetNumAtoms() > MAX_ATOMS:
                continue
            exact_mass = float(Descriptors.ExactMolWt(molecule))
            if exact_mass > MAX_MASS:
                continue
            subformulas = formula_candidate_count(molecule)
            if subformulas > MAX_SUBFORMULAS:
                continue
            mol_id = text_value(row.get("mol_id"))
            molecule_id = text_value(row.get("molecule_id"))
            if molecule_id and mol_id.startswith("clef:") and mol_id[5:] != molecule_id:
                raise ValueError("inconsistent molecule identifiers")
            candidate_ids = [mol_id, molecule_id]
            if mol_id.startswith("clef:"):
                candidate_ids.append(mol_id[5:])
            msp = next((msp_by_id[value] for value in candidate_ids if value in msp_by_id), None)
            if msp is None:
                continue
            connectivity = connectivity_key(row, molecule)
            msp_key = text_value(msp.get("inchikey")).upper().split("-", 1)[0]
            if msp_key and msp_key != connectivity:
                continue
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            msp_formula = text_value(msp.get("formula"))
            if msp_formula and msp_formula != formula:
                continue
            msp_mass = text_value(msp.get("exactmass"))
            if msp_mass and abs(float(msp_mass) - exact_mass) > 0.01:
                continue
            spectrum = dense_spectrum(row["spect"])
            if not np.any(spectrum > 0):
                continue
            grouped[connectivity].append((index, row, molecule, spectrum, exact_mass, subformulas))
        except (TypeError, ValueError, RuntimeError):
            continue
    if not grouped:
        raise ValueError("no compatible SWGDRUG records")
    records: list[dict[str, object]] = []
    cosine_vectors: list[np.ndarray] = []
    exact_keys: list[bytes] = []
    for eval_id, connectivity in enumerate(sorted(grouped)):
        group = grouped[connectivity]
        _, row, molecule, _, exact_mass, subformulas = min(group, key=lambda item: item[0])
        consensus = np.stack([tic_normalize(item[3]) for item in group]).mean(axis=0).astype(np.float32)
        inchi_key = text_value(row.get("inchi_key")) or Chem.MolToInchiKey(Chem.RemoveHs(Chem.Mol(molecule)))
        records.append({
            "eval_id": eval_id,
            "mol_id": text_value(row.get("mol_id")),
            "molecule_id": text_value(row.get("molecule_id")) or text_value(row.get("mol_id")).removeprefix("clef:"),
            "smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)), canonical=True, isomericSmiles=True),
            "rdmol": molecule.ToBinary(),
            "spect": sparse_spectrum(consensus),
            "spectrum_kind": "observed",
            "is_observed_spectrum": True,
            "inchi_key": inchi_key,
            "connectivity_key": connectivity,
            "formula": rdMolDescriptors.CalcMolFormula(molecule),
            "explicit_atom_count": molecule.GetNumAtoms(),
            "mol_wt": exact_mass,
            "formula_candidate_count": subformulas,
        })
        cosine_vectors.append(sqrt_cosine_vector(consensus))
        exact_keys.append(spectrum_key(consensus))
    return pd.DataFrame(records), np.stack(cosine_vectors).astype(np.float32), exact_keys


def split_file(root: Path, split: str) -> Path:
    for name in (f"clef_{split}.parquet", f"mona_{split}.parquet", f"{split}.parquet"):
        path = root / name
        if path.is_file():
            return path
    manifest_path = root / "split_manifest.parquet"
    if manifest_path.is_file():
        manifest = pd.read_parquet(manifest_path)
        if {"split", "split_parquet"}.issubset(manifest.columns):
            paths = manifest.loc[manifest["split"].eq(split), "split_parquet"].drop_duplicates().tolist()
            if len(paths) == 1:
                path = root / str(paths[0])
                if path.is_file():
                    return path
    raise FileNotFoundError(f"{split} Parquet not found under {root}")


def reference_spectra(root: Path) -> tuple[set[str], set[bytes], np.ndarray]:
    frames = [pd.read_parquet(split_file(root, split)) for split in ("train", "val")]
    train_val = pd.concat(frames, ignore_index=True)
    if "spect" not in train_val:
        raise ValueError("MoNA train/val Parquet requires spect")
    connectivities: set[str] = set()
    keys: set[bytes] = set()
    vectors: list[np.ndarray] = []
    for _, row in train_val.iterrows():
        molecule = standardized_molecule(row)
        connectivities.add(connectivity_key(row, molecule))
        spectrum = dense_spectrum(row["spect"])
        if not np.any(spectrum > 0):
            raise ValueError("MoNA train/val contains an empty spectrum")
        keys.add(spectrum_key(spectrum))
        vectors.append(sqrt_cosine_vector(spectrum))
    return connectivities, keys, np.stack(vectors).astype(np.float32)


def maximum_cosine(queries: np.ndarray, references: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    result = np.zeros(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        result[start:end] = np.max(queries[start:end] @ references.T, axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SWGDRUG EI spectra for MoNA external evaluation")
    parser.add_argument("--swgdrug", required=True, type=Path)
    parser.add_argument("--swgdrug-msp", required=True, type=Path)
    parser.add_argument("--mona-split-root", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = pd.read_parquet(args.swgdrug)
    compatible, vectors, exact_keys = compatible_records(source, parse_msp_metadata(args.swgdrug_msp))
    cohorts: list[tuple[str, pd.DataFrame]] = []
    for index, root in enumerate(args.mona_split_root, start=1):
        connectivities, reference_keys, references = reference_spectra(root)
        near = maximum_cosine(vectors, references) >= NEAR_SPECTRUM_THRESHOLD
        keep = (
            ~compatible["connectivity_key"].isin(connectivities).to_numpy()
            & ~np.array([key in reference_keys for key in exact_keys], dtype=bool)
            & ~near
        )
        cohorts.append((f"swgdrug_external_s{index}.parquet", compatible.loc[keep].copy()))
    args.output_dir.mkdir(parents=True)
    compatible.to_parquet(args.output_dir / "swgdrug_full_compatible.parquet", index=False)
    print(f"compatible: {len(compatible)}")
    for name, cohort in cohorts:
        cohort.to_parquet(args.output_dir / name, index=False)
        print(f"{name}: {len(cohort)}")


if __name__ == "__main__":
    main()
