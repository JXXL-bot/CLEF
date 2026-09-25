from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolHash
from rdkit.Chem.Scaffolds import MurckoScaffold


SPLITS = ("train", "val", "test")
OUTPUT_FIELDS = ("molecule_id", "smiles", "split", "spectrum_json", "connectivity_key")


def spectrum_vector(value: str) -> np.ndarray:
    peaks = json.loads(value)
    if not isinstance(peaks, list) or not peaks:
        raise ValueError("invalid spectrum")
    spectrum = np.zeros(512, dtype=np.float64)
    for peak in peaks:
        if not isinstance(peak, list) or len(peak) != 2:
            raise ValueError("invalid peak")
        mz, intensity = float(peak[0]), float(peak[1])
        if not np.isfinite(mz) or not np.isfinite(intensity) or intensity < 0:
            raise ValueError("invalid peak value")
        index = int(mz) - 1
        if mz != index + 1 or not 0 <= index < 512:
            raise ValueError("invalid m/z")
        spectrum[index] += intensity
    total = float(spectrum.sum())
    if total <= 0:
        raise ValueError("empty spectrum")
    return spectrum / total


def sparse_spectrum(spectrum: np.ndarray) -> str:
    peaks = [[float(index + 1), float(spectrum[index])] for index in np.flatnonzero(spectrum > 1.0e-8)]
    if not peaks:
        index = int(np.argmax(spectrum))
        peaks = [[float(index + 1), float(spectrum[index])]]
    return json.dumps(peaks, separators=(",", ":"))


def consensus_rows(input_csv: Path) -> list[dict[str, str]]:
    grouped: dict[str, list[tuple[dict[str, str], np.ndarray]]] = defaultdict(list)
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"molecule_id", "smiles", "inchi_key", "spectrum_json"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("MoNA source CSV is missing required columns")
        for row in reader:
            molecule = Chem.MolFromSmiles(row["smiles"])
            if molecule is None:
                raise ValueError("invalid SMILES in MoNA source")
            key = Chem.MolToInchiKey(molecule)
            if not key or key != row["inchi_key"]:
                raise ValueError("SMILES and InChIKey disagree")
            grouped[key].append((row, spectrum_vector(row["spectrum_json"])))
    output: list[dict[str, str]] = []
    for identity in sorted(grouped):
        records = grouped[identity]
        if len(records) == 1:
            row = dict(records[0][0])
            row["spectrum_json"] = sparse_spectrum(records[0][1])
            output.append(row)
            continue
        vectors = np.stack([np.sqrt(spectrum) for _, spectrum in records])
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1.0e-12)
        similarities = np.clip(vectors @ vectors.T, 0.0, 1.0)
        medoid = int(np.argmax(similarities.mean(axis=1)))
        keep = np.ones(len(records), dtype=bool) if len(records) <= 2 else similarities[medoid] >= 0.80
        keep[medoid] = True
        centroid = vectors[keep].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1.0e-12)
        spectrum = np.square(centroid)
        spectrum /= max(float(spectrum.sum()), 1.0e-12)
        row = dict(records[medoid][0])
        row["spectrum_json"] = sparse_spectrum(spectrum)
        output.append(row)
    if not output:
        raise ValueError("MoNA source CSV contains no records")
    return output


def scaffold_key(molecule: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms():
        return "murcko:" + Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    return "acyclic:" + rdMolHash.MolHash(molecule, rdMolHash.HashFunction.AnonymousGraph)


def feature_labels(molecule: Chem.Mol) -> dict[str, str]:
    explicit = Chem.AddHs(Chem.Mol(molecule))
    mass = Descriptors.ExactMolWt(explicit)
    heavy_atoms = molecule.GetNumHeavyAtoms()
    element_family = "+".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()}))
    mass_edges = (0, 100, 150, 200, 250, 300, 350, 400, float("inf"))
    heavy_edges = (0, 8, 12, 16, 20, 24, 32, 48, float("inf"))
    mass_bin = next(index for index, upper in enumerate(mass_edges[1:]) if mass < upper)
    heavy_bin = next(index for index, upper in enumerate(heavy_edges[1:]) if heavy_atoms < upper)
    return {"mass": str(mass_bin), "heavy": str(heavy_bin), "elements": element_family}


def assign_groups(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    features: dict[str, dict[str, str]] = {}
    for row in rows:
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            raise ValueError("invalid consensus SMILES")
        key = scaffold_key(molecule)
        groups[key].append(row)
        features[row["molecule_id"]] = feature_labels(molecule)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    target_size = {split: max(len(rows) * fraction, 1.0) for split, fraction in fractions.items()}
    categories = ("mass", "heavy", "elements")
    weights = {"mass": 0.4, "heavy": 0.35, "elements": 0.25}
    totals = {category: Counter(features[row["molecule_id"]][category] for row in rows) for category in categories}
    current_size: Counter[str] = Counter()
    current_features = {split: {category: Counter() for category in categories} for split in SPLITS}
    assignment: dict[str, str] = {}
    for key in order:
        group = groups[key]
        group_features = {
            category: Counter(features[row["molecule_id"]][category] for row in group)
            for category in categories
        }
        scores: dict[str, float] = {}
        for split in SPLITS:
            size_fill = (current_size[split] + 0.5 * len(group)) / target_size[split]
            feature_fill = 0.0
            for category in categories:
                weighted = 0.0
                for label, count in group_features[category].items():
                    target = max(totals[category][label] * fractions[split], 1.0)
                    weighted += count * (current_features[split][category][label] + 0.5 * count) / target
                feature_fill += weights[category] * weighted / len(group)
            overshoot = max(0.0, (current_size[split] + len(group) - target_size[split]) / target_size[split])
            scores[split] = size_fill + feature_fill + 20.0 * overshoot * overshoot
        chosen = min(SPLITS, key=lambda split: (scores[split], SPLITS.index(split)))
        assignment[key] = chosen
        current_size[chosen] += len(group)
        for category in categories:
            current_features[chosen][category].update(group_features[category])
    return assignment


def split_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    assignments = assign_groups(rows, seed)
    output: list[dict[str, str]] = []
    seen_identity: dict[str, str] = {}
    seen_scaffold: dict[str, str] = {}
    for row in rows:
        molecule = Chem.MolFromSmiles(row["smiles"])
        key = scaffold_key(molecule)
        split = assignments[key]
        connectivity = row["inchi_key"].split("-", 1)[0]
        if connectivity in seen_identity and seen_identity[connectivity] != split:
            raise ValueError("connectivity crossed split boundary")
        if key in seen_scaffold and seen_scaffold[key] != split:
            raise ValueError("scaffold crossed split boundary")
        seen_identity[connectivity] = split
        seen_scaffold[key] = split
        output.append({
            "molecule_id": row["molecule_id"],
            "smiles": row["smiles"],
            "split": split,
            "spectrum_json": row["spectrum_json"],
            "connectivity_key": connectivity,
        })
    return output


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staged_name, path)
    finally:
        Path(staged_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build three MoNA scaffold partitions for CLEF.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    targets = [output_dir / f"mona_s{index}.csv" for index in (1, 2, 3)]
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError("MoNA partition output already exists")
    rows = consensus_rows(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed, target in zip((1234, 2345, 3456), targets):
        write_output(target, split_rows(rows, seed))


if __name__ == "__main__":
    main()
