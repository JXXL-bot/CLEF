from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from clef.featurize.featurize import MolFeaturizer
from clef.msutil import binutils


_FEATURIZER: MolFeaturizer | None = None
_ROW_TIMEOUT_SECONDS: int = 0
_MASS_BIN_FEATURE_INDEX: int | None = None
_MASS_BIN_FEATURE_SCALE: float = 1.0


class RowTimeoutError(TimeoutError):
    pass


def timeout_handler(signum, frame):
    raise RowTimeoutError("event enumeration timed out")


def init_worker(config: Dict[str, Any]) -> None:
    global _FEATURIZER
    global _ROW_TIMEOUT_SECONDS
    global _MASS_BIN_FEATURE_INDEX
    global _MASS_BIN_FEATURE_SCALE
    _ROW_TIMEOUT_SECONDS = int(config.get("row_timeout_seconds", 0))


    _MASS_BIN_FEATURE_INDEX = (
        5 + len(config["h_shifts"]) + 2 * len(config["elements"])
    )
    _MASS_BIN_FEATURE_SCALE = 511.0
    spect_bin = binutils.create_spectrum_bins(
        first_bin_center=1.0,
        bin_width=1.0,
        bin_number=int(config["bin_number"]),
    )
    _FEATURIZER = MolFeaturizer(
        MAX_N=int(config["max_n"]),
        bin_config=spect_bin,
        event_config={
            "enabled": True,
            "max_events": int(config["max_events"]),
            "max_bond_cuts": int(config["max_bond_cuts"]),
            "atom_slots": int(config["atom_slots"]),
            "include_molecular_ion": bool(config["include_molecular_ion"]),
            "h_shifts": [int(x) for x in config["h_shifts"]],
            "max_bin": int(config["max_bin"]),
            "elements": list(config["elements"]),
            "max_enumerated_combinations": int(config["max_enumerated_combinations"]),
            "min_fragment_atoms": int(config["min_fragment_atoms"]),
            "max_fragment_fraction": float(config["max_fragment_fraction"]),
        },
    )


def mol_from_row(row: Dict[str, Any]) -> Chem.Mol:
    if "rdmol" in row and row["rdmol"] is not None:
        mol = Chem.Mol(row["rdmol"])
        if mol is not None:
            return mol
    smiles = str(row.get("smiles", "") or "").strip()
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
    raise ValueError("could not parse molecule")


def build_payload(task: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    source_row_idx, record = task
    if _FEATURIZER is None:
        raise RuntimeError("worker featurizer was not initialized")
    old_handler = None
    try:
        if _ROW_TIMEOUT_SECONDS > 0:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, float(_ROW_TIMEOUT_SECONDS))
        mol = mol_from_row(record)
        features, mass_idx, atom_idx, mask = _FEATURIZER._build_events(mol)
        n_events = int(mask.sum())
        features = features[:n_events].astype(np.float32, copy=True)
        mass_idx = mass_idx[:n_events].astype(np.int64, copy=True)
        if n_events:

            if np.any(mass_idx < 1):
                raise ValueError("physical event m/z must be positive")
            mass_idx -= 1
            if _MASS_BIN_FEATURE_INDEX is None:
                raise RuntimeError("mass-bin feature index was not initialized")
            features[:, _MASS_BIN_FEATURE_INDEX] -= (
                1.0 / _MASS_BIN_FEATURE_SCALE
            )
        return {
            "source_row_idx": int(source_row_idx),
            "features": features,
            "mass_idx": mass_idx,
            "atom_idx": atom_idx[:n_events].astype(np.int64, copy=False),
            "n_events": int(n_events),
            "failed": False,
        }
    except Exception:
        return {
            "source_row_idx": int(source_row_idx),
            "features": np.zeros((0, 1), dtype=np.float32),
            "mass_idx": np.zeros((0,), dtype=np.int64),
            "atom_idx": np.zeros((0, 1), dtype=np.int64),
            "n_events": 0,
            "failed": True,
        }
    finally:
        if _ROW_TIMEOUT_SECONDS > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def count_payload(task: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    payload = build_payload(task)
    return {
        "source_row_idx": int(payload["source_row_idx"]),
        "n_events": int(payload["n_events"]),
        "failed": bool(payload["failed"]),
    }


def filter_df(df: pd.DataFrame, max_n: int, max_mass: float) -> pd.DataFrame:
    keep = []
    for _, row in df.iterrows():
        try:
            mol = mol_from_row(row)
            ok = True
            if max_n and mol.GetNumAtoms() > int(max_n):
                ok = False
            if max_mass and rdMolDescriptors.CalcExactMolWt(mol) > float(max_mass):
                ok = False
            keep.append(ok)
        except Exception:
            keep.append(False)
    return df[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def input_file_identity(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def source_row_list(tasks: List[Tuple[int, Dict[str, Any]]]) -> List[int]:
    return [int(source_row_idx) for source_row_idx, _ in tasks]


def normalize_row_stats(row_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_row_idx": int(row["source_row_idx"]),
            "n_events": int(row["n_events"]),
            "failed": bool(row["failed"]),
        }
        for row in row_stats
    ]


def write_count_cache(
    path: Path,
    args: argparse.Namespace,
    config: Dict[str, Any],
    input_identity: Dict[str, int],
        original_rows: int,
        tasks: List[Tuple[int, Dict[str, Any]]],
        row_stats: List[Dict[str, Any]]) -> None:
    atomic_write_json(path, {
        "version": 1,
        "input_parquet": str(args.input_parquet),
        "input_identity": input_identity,
        "original_rows": int(original_rows),
        "max_rows": int(args.max_rows),
        "config": config,
        "source_row_idx": source_row_list(tasks),
        "row_stats": normalize_row_stats(row_stats),
    })


def load_count_cache(
    path: Path,
    args: argparse.Namespace,
    config: Dict[str, Any],
    input_identity: Dict[str, int],
        original_rows: int,
        tasks: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[clef-event-table] ignore unreadable count cache {path}: {exc}", flush=True)
        return []
    expected_sources = source_row_list(tasks)
    if (
            int(cache.get("version", -1)) != 1
            or cache.get("input_parquet") != str(args.input_parquet)
            or cache.get("input_identity") != input_identity
            or int(cache.get("original_rows", -1)) != int(original_rows)
            or int(cache.get("max_rows", -1)) != int(args.max_rows)
            or cache.get("config") != config
            or cache.get("source_row_idx") != expected_sources):
        print(f"[clef-event-table] ignore stale count cache {path}", flush=True)
        return []
    row_stats = normalize_row_stats(list(cache.get("row_stats", [])))
    if len(row_stats) > len(tasks):
        print(f"[clef-event-table] ignore oversized count cache {path}", flush=True)
        return []
    for i, row in enumerate(row_stats):
        if int(row["source_row_idx"]) != int(tasks[i][0]):
            print(f"[clef-event-table] ignore misaligned count cache {path}", flush=True)
            return []
    return row_stats


def open_or_create_memmap(path: Path, dtype: np.dtype, shape: Tuple[int, ...]) -> np.memmap:
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if tuple(array.shape) != tuple(shape) or np.dtype(array.dtype) != np.dtype(dtype):
            raise RuntimeError(
                f"existing memmap shape/dtype mismatch for {path}: "
                f"got shape={array.shape} dtype={array.dtype}, expected shape={shape} dtype={np.dtype(dtype)}"
            )
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def write_progress(
    path: Path,
    args: argparse.Namespace,
    config: Dict[str, Any],
    input_identity: Dict[str, int],
        total_events: int,
        event_dim: int,
        atom_slots: int,
        rows_total: int,
        rows_written: int) -> None:
    atomic_write_json(path, {
        "version": 1,
        "input_parquet": str(args.input_parquet),
        "input_identity": input_identity,
        "config": config,
        "total_events": int(total_events),
        "event_dim": int(event_dim),
        "atom_slots": int(atom_slots),
        "rows_total": int(rows_total),
        "rows_written": int(rows_written),
    })


def load_write_progress(
    path: Path,
    args: argparse.Namespace,
    config: Dict[str, Any],
    input_identity: Dict[str, int],
        total_events: int,
        event_dim: int,
        atom_slots: int,
        rows_total: int) -> int:
    if not path.exists():
        return 0
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[clef-event-table] ignore unreadable write progress {path}: {exc}", flush=True)
        return 0
    if (
            int(progress.get("version", -1)) != 1
            or progress.get("input_parquet") != str(args.input_parquet)
            or progress.get("input_identity") != input_identity
            or progress.get("config") != config
            or int(progress.get("total_events", -1)) != int(total_events)
            or int(progress.get("event_dim", -1)) != int(event_dim)
            or int(progress.get("atom_slots", -1)) != int(atom_slots)
            or int(progress.get("rows_total", -1)) != int(rows_total)):
        print(f"[clef-event-table] ignore stale write progress {path}", flush=True)
        return 0
    return max(0, min(int(progress.get("rows_written", 0)), int(rows_total)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Precompute CLEF graph-cut events.')
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-events", type=int, default=8192)
    parser.add_argument("--max-bond-cuts", type=int, default=3)
    parser.add_argument("--atom-slots", type=int, default=4)
    parser.add_argument("--h-shifts", default="-3,-2,-1,0,1,2,3")
    parser.add_argument("--elements", default="H,C,N,O,F,P,S,Cl")
    parser.add_argument("--max-enumerated-combinations", type=int, default=250000)
    parser.add_argument("--min-fragment-atoms", type=int, default=1)
    parser.add_argument("--max-fragment-fraction", type=float, default=0.98)
    parser.add_argument("--bin-number", type=int, default=512)
    parser.add_argument("--max-bin", type=int, default=511)
    parser.add_argument("--max-n", type=int, default=48)
    parser.add_argument("--filter-max-n", type=int, default=48)
    parser.add_argument("--filter-max-mass", type=float, default=400.0)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test cap after filtering; 0 means all rows.")
    parser.add_argument("--row-timeout-seconds", type=int, default=0, help="Skip rows whose event enumeration exceeds this many seconds; 0 disables timeout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_identity = input_file_identity(Path(args.input_parquet))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "max_events": int(args.max_events),
        "max_bond_cuts": int(args.max_bond_cuts),
        "atom_slots": int(args.atom_slots),
        "h_shifts": [int(x) for x in str(args.h_shifts).split(",") if str(x).strip()],
        "elements": [x.strip() for x in str(args.elements).split(",") if x.strip()],
        "bin_number": int(args.bin_number),
        "max_bin": int(args.max_bin),
        "max_n": int(args.max_n),
        "include_molecular_ion": True,
        "max_enumerated_combinations": int(args.max_enumerated_combinations),
        "min_fragment_atoms": int(args.min_fragment_atoms),
        "max_fragment_fraction": float(args.max_fragment_fraction),
        "row_timeout_seconds": int(args.row_timeout_seconds),
    }

    df = pd.read_parquet(args.input_parquet)
    df["_source_row_idx"] = np.arange(len(df), dtype=np.int64)
    original_rows = int(len(df))
    df = filter_df(df, max_n=int(args.filter_max_n), max_mass=float(args.filter_max_mass))
    if int(args.max_rows) > 0:
        df = df.head(int(args.max_rows)).reset_index(drop=True)
    tasks = [
        (int(row["_source_row_idx"]), {k: row[k] for k in row.index if k != "_source_row_idx"})
        for _, row in df.iterrows()
    ]

    checkpoint_interval = max(int(args.checkpoint_interval), 1)
    count_cache_path = output_dir / "row_stats.json"
    write_progress_path = output_dir / "write_progress.json"

    row_stats: List[Dict[str, Any]] = load_count_cache(
        count_cache_path,
        args=args,
        config=config,
        input_identity=input_identity,
        original_rows=original_rows,
        tasks=tasks,
    )
    counts = Counter()
    for row in row_stats:
        counts["failed"] += int(bool(row["failed"]))
        counts["events"] += int(row["n_events"])
    if row_stats:
        print(
            f"[clef-event-table] pass=1 resume counted={len(row_stats)}/{len(tasks)} "
            f"events={int(counts['events'])} failed={int(counts['failed'])}",
            flush=True,
        )
    print(
        f"[clef-event-table] pass=1 count rows={len(tasks)} workers={int(args.workers)} chunksize={int(args.chunksize)}",
        flush=True,
    )
    if len(row_stats) < len(tasks):
        remaining_tasks = tasks[len(row_stats):]
        with mp.Pool(processes=int(args.workers), initializer=init_worker, initargs=(config,)) as pool:
            iterator = pool.imap(count_payload, remaining_tasks, chunksize=max(int(args.chunksize), 1))
            for i, payload in enumerate(iterator, start=len(row_stats) + 1):
                row_stats.append(payload)
                counts["failed"] += int(bool(payload["failed"]))
                counts["events"] += int(payload["n_events"])
                if i % checkpoint_interval == 0 or i == len(tasks):
                    write_count_cache(
                        count_cache_path,
                        args=args,
                        config=config,
                        input_identity=input_identity,
                        original_rows=original_rows,
                        tasks=tasks,
                        row_stats=row_stats,
                    )
                    print(
                        f"[clef-event-table] pass=1 counted={i}/{len(tasks)} "
                        f"events={int(counts['events'])} failed={int(counts['failed'])} "
                        f"checkpoint={count_cache_path}",
                        flush=True,
                    )
    else:
        print(f"[clef-event-table] pass=1 complete from cache {count_cache_path}", flush=True)
    write_count_cache(
        count_cache_path,
        args=args,
        config=config,
        input_identity=input_identity,
        original_rows=original_rows,
        tasks=tasks,
        row_stats=row_stats,
    )

    event_dim = 14 + len(config["h_shifts"]) + 3 * len(config["elements"])
    atom_slots = int(config["atom_slots"])
    total_events = int(sum(int(row["n_events"]) for row in row_stats))
    source_row_idx = np.asarray([int(row["source_row_idx"]) for row in row_stats], dtype=np.int64)
    event_ptr = np.zeros((len(row_stats) + 1,), dtype=np.int64)
    for i, row in enumerate(row_stats):
        event_ptr[i + 1] = event_ptr[i] + int(row["n_events"])

    np.save(output_dir / "event_ptr.npy", event_ptr)
    np.save(output_dir / "source_row_idx.npy", source_row_idx)

    event_features = open_or_create_memmap(
        output_dir / "event_features.npy",
        dtype=np.float32,
        shape=(total_events, event_dim),
    )
    event_mass_idx = open_or_create_memmap(
        output_dir / "event_mass_idx.npy",
        dtype=np.int64,
        shape=(total_events,),
    )
    event_atom_idx = open_or_create_memmap(
        output_dir / "event_atom_idx.npy",
        dtype=np.int64,
        shape=(total_events, atom_slots),
    )
    write_start = load_write_progress(
        write_progress_path,
        args=args,
        config=config,
        input_identity=input_identity,
        total_events=total_events,
        event_dim=event_dim,
        atom_slots=atom_slots,
        rows_total=len(tasks),
    )
    if write_start:
        print(
            f"[clef-event-table] pass=2 resume written={write_start}/{len(tasks)} "
            f"progress={write_progress_path}",
            flush=True,
        )
    print(
        f"[clef-event-table] pass=2 write rows={len(tasks)} total_events={total_events} event_dim={event_dim}",
        flush=True,
    )
    if write_start < len(tasks):
        remaining_tasks = tasks[write_start:]
        with mp.Pool(processes=int(args.workers), initializer=init_worker, initargs=(config,)) as pool:
            iterator = pool.imap(build_payload, remaining_tasks, chunksize=max(int(args.chunksize), 1))
            for offset, row in enumerate(iterator):
                i = write_start + offset
                expected_n = int(row_stats[i]["n_events"])
                if int(row["n_events"]) != expected_n:
                    raise RuntimeError(
                        f"event count changed for source row {row['source_row_idx']}: "
                        f"pass1={expected_n} pass2={row['n_events']}"
                    )
                if int(row["source_row_idx"]) != int(row_stats[i]["source_row_idx"]):
                    raise RuntimeError(
                        f"source row changed at table row {i}: "
                        f"pass1={row_stats[i]['source_row_idx']} pass2={row['source_row_idx']}"
                    )
                start = int(event_ptr[i])
                end = int(event_ptr[i + 1])
                if end > start:
                    event_features[start:end] = row["features"]
                    event_mass_idx[start:end] = row["mass_idx"]
                    event_atom_idx[start:end] = row["atom_idx"]
                rows_written = i + 1
                if rows_written % checkpoint_interval == 0 or rows_written == len(tasks):
                    event_features.flush()
                    event_mass_idx.flush()
                    event_atom_idx.flush()
                    write_progress(
                        write_progress_path,
                        args=args,
                        config=config,
                        input_identity=input_identity,
                        total_events=total_events,
                        event_dim=event_dim,
                        atom_slots=atom_slots,
                        rows_total=len(tasks),
                        rows_written=rows_written,
                    )
                    print(
                        f"[clef-event-table] pass=2 written={rows_written}/{len(tasks)} "
                        f"checkpoint={write_progress_path}",
                        flush=True,
                    )
    else:
        print(f"[clef-event-table] pass=2 complete from progress {write_progress_path}", flush=True)
    event_features.flush()
    event_mass_idx.flush()
    event_atom_idx.flush()
    if input_file_identity(Path(args.input_parquet)) != input_identity:
        raise RuntimeError("input Parquet changed while the event table was being built")
    write_progress(
        write_progress_path,
        args=args,
        config=config,
        input_identity=input_identity,
        total_events=total_events,
        event_dim=event_dim,
        atom_slots=atom_slots,
        rows_total=len(tasks),
        rows_written=len(tasks),
    )

    observed = np.diff(event_ptr)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_parquet": str(args.input_parquet),
        "output_dir": str(output_dir),
        "original_rows": original_rows,
        "filtered_rows": int(len(row_stats)),
        "max_rows": int(args.max_rows),
        "total_events": total_events,
        "event_feature_dim": event_dim,
        "atom_slots": atom_slots,
        "max_events": int(args.max_events),
        "max_observed_events": int(observed.max()) if observed.size else 0,
        "mean_observed_events": float(observed.mean()) if observed.size else 0.0,
        "failed_rows": int(counts["failed"]),
        "workers": int(args.workers),
        "chunksize": int(args.chunksize),
        "config": config,
        "event_mass_contract": {
            "stored_value": "zero-based tensor index",
            "physical_mz": "stored_value + 1",
            "first_bin_center": 1.0,
            "bin_width": 1.0,
            "mass_bin_feature_index": (
                5 + len(config["h_shifts"]) + 2 * len(config["elements"])
            ),
        },
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
