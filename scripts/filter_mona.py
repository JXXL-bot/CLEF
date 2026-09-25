from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


ALLOWED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17})
DERIVATIZATION = re.compile(
    r"\bderivati[sz](?:ation|ed|e)\b|\b(?:MSTFA|BSTFA|MTBSTFA|TBDMS|TMS|trimethylsilyl|silylat(?:ed|ion))\b|\bchemically\s+modified\b.{0,80}\bbefore\s+GC(?:-MS)?\b",
    re.IGNORECASE,
)
MIXTURE = re.compile(
    r"\bmixtures?\b|\bco[- ]?elut(?:ion|ing|ed)\b|\bmultiple\s+(?:compounds?|derivatives?|species)\b|\bmore\s+than\s+one\s+(?:compound|derivative|species)\b",
    re.IGNORECASE,
)
ENERGY = re.compile(
    r"\b(?:ionization potential|electron energy|ionization energy)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:eV)?",
    re.IGNORECASE,
)
LICENSE = re.compile(r"^(?:CC0|CC\s+BY(?:-NC)?(?:-SA)?)\b", re.IGNORECASE)
ADDUCT = re.compile(
    r"\bionization\s*=\s*esi\b|\[\s*m\s*\+\s*(?:h|na)\s*\]\s*\+",
    re.IGNORECASE,
)
ISOTOPES = {
    1: ((0, 0.999885), (1, 0.000115)),
    6: ((0, 0.9893), (1, 0.0107)),
    7: ((0, 0.99636), (1, 0.00364)),
    8: ((0, 0.99757), (1, 0.00038), (2, 0.00205)),
    9: ((0, 1.0),),
    15: ((0, 1.0),),
    16: ((0, 0.9499), (1, 0.0075), (2, 0.0425), (4, 0.0001)),
    17: ((0, 0.7576), (2, 0.2424)),
}
FIELDS = (
    "molecule_id",
    "smiles",
    "inchi_key",
    "connectivity_key",
    "formula",
    "spectrum_json",
)


def msp_records(path: Path) -> Iterator[tuple[dict[str, str], list[tuple[float, float]], int]]:
    headers: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    reading_peaks = False

    def finish():
        nonlocal headers, peaks, reading_peaks
        if not headers and not peaks:
            return None
        result = headers, peaks, len(peaks)
        headers = {}
        peaks = []
        reading_peaks = False
        return result

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                result = finish()
                if result is not None:
                    yield result
                continue
            if not reading_peaks and ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
                if key.strip().lower() == "num peaks":
                    reading_peaks = True
                continue
            if reading_peaks:
                for token in line.split(";"):
                    values = token.strip().split()
                    if len(values) < 2:
                        continue
                    try:
                        peaks.append((float(values[0]), float(values[1])))
                    except ValueError:
                        peaks.append((math.nan, math.nan))
    result = finish()
    if result is not None:
        yield result


def comment_fields(text: str) -> dict[str, str]:
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = re.findall(r'"([^"]*)"', text)
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            fields.setdefault(" ".join(key.lower().split()), value.strip())
    return fields


def eligible_metadata(headers: dict[str, str], fields: dict[str, str]) -> bool:
    spectrum_type = headers.get("spectrum_type", "").strip().upper()
    ion_mode = headers.get("ion_mode", "").strip().upper()
    instrument = headers.get("instrument_type", "").strip()
    comments = headers.get("comments", "")
    if spectrum_type and spectrum_type != "MS1":
        return False
    if ion_mode and ion_mode != "P":
        return False
    if re.search(r"(?:^|[- /])(?:CI|APCI|FI)(?:$|[- /])", instrument, re.IGNORECASE):
        return False
    if not re.search(r"(?:^EI-B$|GC-EI)", instrument, re.IGNORECASE):
        return False
    if ADDUCT.search(comments):
        return False
    match = ENERGY.search(comments)
    if match is not None and abs(float(match.group(1)) - 70.0) > 0.51:
        return False
    text = " ".join((headers.get("name", ""), comments))
    if any(key.startswith("derivatization ") for key in fields):
        return False
    if DERIVATIZATION.search(text) or MIXTURE.search(text):
        return False
    if not LICENSE.match(fields.get("license", "").strip()):
        return False
    return True


def parse_structure(headers: dict[str, str], fields: dict[str, str]):
    smiles = fields.get("smiles") or fields.get("computed smiles") or ""
    inchi = fields.get("inchi") or headers.get("inchi", "")
    if inchi.startswith("InChI=InChI="):
        inchi = inchi[len("InChI="):]
    smiles_mol = Chem.MolFromSmiles(smiles) if smiles else None
    inchi_mol = Chem.MolFromInchi(inchi) if inchi else None
    candidates = [mol for mol in (smiles_mol, inchi_mol) if mol is not None]
    if not candidates:
        raise ValueError("structure_missing")
    keys = {Chem.MolToInchiKey(mol) for mol in candidates}
    if len(keys) != 1 or not next(iter(keys)):
        raise ValueError("structure_identity_conflict")
    inchi_key = next(iter(keys))
    if headers.get("inchikey", "").strip() not in {"", inchi_key}:
        raise ValueError("header_identity_conflict")
    heavy = Chem.RemoveHs(Chem.Mol(candidates[0]))
    Chem.SanitizeMol(heavy)
    if len(Chem.GetMolFrags(heavy)) != 1:
        raise ValueError("multiple_components")
    if sum(atom.GetFormalCharge() for atom in heavy.GetAtoms()) != 0:
        raise ValueError("non_neutral")
    if any(atom.GetIsotope() or atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS for atom in heavy.GetAtoms()):
        raise ValueError("unsupported_structure")
    molecule = Chem.AddHs(heavy)
    Chem.SanitizeMol(molecule)
    if molecule.GetNumAtoms() > 48 or Descriptors.ExactMolWt(molecule) > 400:
        raise ValueError("structure_limit")
    counts = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    formula_count = math.prod(count + 1 for count in counts.values())
    if formula_count > 8192:
        raise ValueError("formula_limit")
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    if headers.get("formula", "").strip() != formula:
        raise ValueError("formula_mismatch")
    return molecule, Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True), inchi_key, formula


def normalize_spectrum(peaks: list[tuple[float, float]]) -> np.ndarray:
    dense = np.zeros(512, dtype=np.float64)
    raw_total = 0.0
    outside_total = 0.0
    for mz, intensity in peaks:
        if not math.isfinite(mz) or not math.isfinite(intensity) or mz <= 0 or intensity <= 0:
            raise ValueError("invalid_peak")
        raw_total += intensity
        index = int(math.floor(mz - 0.5))
        if 0 <= index < 512:
            dense[index] += intensity
        else:
            outside_total += intensity
    if raw_total <= 0 or outside_total / raw_total > 0.01:
        raise ValueError("outside_mz_grid")
    total = float(dense.sum())
    if total <= 0:
        raise ValueError("empty_spectrum")
    dense /= total
    return dense


def isotope_envelope_limit(molecule: Chem.Mol) -> int:
    distribution = np.asarray([1.0], dtype=np.float64)
    table = Chem.GetPeriodicTable()
    parent_mz = 0.0
    for atom in molecule.GetAtoms():
        parent_mz += table.GetMostCommonIsotopeMass(atom.GetSymbol())
        isotopes = ISOTOPES[atom.GetAtomicNum()]
        atom_distribution = np.zeros(max(shift for shift, _ in isotopes) + 1)
        for shift, abundance in isotopes:
            atom_distribution[shift] = abundance
        distribution = np.convolve(distribution, atom_distribution)
    positions = np.flatnonzero(distribution >= distribution.max() * 1.0e-3)
    return int(round(parent_mz)) + int(positions[-1] if len(positions) else 0)


def clean_records(path: Path) -> list[dict[str, str]]:
    candidates: list[tuple[dict[str, str], tuple[float, ...], str]] = []
    seen_accessions: set[str] = set()
    for headers, peaks, parsed_count in msp_records(path):
        accession = headers.get("db#", "").strip()
        if not accession or accession in seen_accessions:
            continue
        seen_accessions.add(accession)
        fields = comment_fields(headers.get("comments", ""))
        if fields.get("accession", accession) != accession:
            continue
        try:
            if int(headers.get("num peaks", "")) != parsed_count:
                continue
            if not eligible_metadata(headers, fields):
                continue
            molecule, smiles, inchi_key, formula = parse_structure(headers, fields)
            spectrum = normalize_spectrum(peaks)
            if float(spectrum[min(max(isotope_envelope_limit(molecule), 0), 512):].sum()) > 0.01:
                continue
        except (ValueError, OverflowError, KeyError):
            continue
        sparse = [[float(index + 1), float(spectrum[index])] for index in np.flatnonzero(spectrum > 0)]
        row = {
            "molecule_id": accession,
            "smiles": smiles,
            "inchi_key": inchi_key,
            "connectivity_key": inchi_key.split("-", 1)[0],
            "formula": formula,
            "spectrum_json": json.dumps(sparse, separators=(",", ":")),
        }
        candidates.append((row, tuple(np.round(spectrum, 12).tolist()), inchi_key))
    by_spectrum: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, (_, key, _) in enumerate(candidates):
        by_spectrum[key].append(index)
    rejected: set[int] = set()
    for indices in by_spectrum.values():
        identities = {candidates[index][2] for index in indices}
        if len(identities) > 1:
            rejected.update(indices)
        else:
            rejected.update(indices[1:])
    return [row for index, (row, _, _) in enumerate(candidates) if index not in rejected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter MoNA MSP records for CLEF.")
    parser.add_argument("--input-msp", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.input_msp.expanduser().resolve()
    target = args.output_csv.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)
    rows = clean_records(source)
    if not rows:
        raise ValueError("no eligible MoNA records")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staged_name, target)
    finally:
        Path(staged_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
