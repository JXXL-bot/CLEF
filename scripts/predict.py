from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run CLEF spectrum prediction.')
    parser.add_argument("--dataset", required=True, help="Prepared molecular Parquet file")
    parser.add_argument("--event-table", help="Optional precomputed event-table directory")
    parser.add_argument("--checkpoint", required=True, help="CLEF model state dictionary")
    parser.add_argument("--meta", help="Optional model and featurizer metadata (JSON or pickle)")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--eval-id-column", default="")
    parser.add_argument("--event-source-row-column", default="")
    parser.add_argument("--target-free", "--omit-targets", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=50)
    return parser.parse_args()


def load_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".json":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError("Model metadata must be a dictionary")
    return metadata


def load_model(path: Path, metadata: dict[str, Any], device: torch.device, bins: int):
    from clef.model.spectrum_model import CLEFSpectrumModel

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:

        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a CLEF model state dictionary")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model state must be a dictionary")
    options = checkpoint.get("model_kwargs", metadata.get("model_kwargs", {}))
    if not isinstance(options, dict):
        raise TypeError("model_kwargs must be a dictionary")
    options = dict(options)
    options.setdefault("spectrum_bins", bins)
    model = CLEFSpectrumModel(**options)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_rows < 0:
        raise ValueError("batch-size must be positive; num-workers and max-rows cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {args.device}")

    sys.path.insert(0, str(Path(args.project_root).resolve()))
    from clef.dataset import ParquetDataset, event_dynamic_collate
    from clef.msutil.binutils import create_spectrum_bins

    metadata = load_metadata(args.meta)
    bin_options = metadata.get("spectrum_bin_config") or {}
    bins = create_spectrum_bins(**bin_options)
    dataset = ParquetDataset(
        args.dataset,
        bins,
        metadata.get("featurize_config") or {},
        metadata.get("pred_config") or {},
        filter_config={"event_table_dir": args.event_table} if args.event_table else {},
        include_targets=not args.target_free,
    )
    if args.event_source_row_column:
        column = args.event_source_row_column
        if column not in dataset.df:
            raise KeyError(f"Missing event source row column: {column}")
        source_rows = np.asarray(dataset.df[column], dtype=np.int64)
        if len(source_rows) != len(set(source_rows.tolist())):
            raise ValueError("Event source row IDs must be unique")
        dataset.df["_source_row_idx"] = source_rows
    row_count = min(len(dataset), args.max_rows) if args.max_rows else len(dataset)
    if row_count == 0:
        raise ValueError("No input rows are available")
    frame = dataset.df.iloc[:row_count]
    if args.eval_id_column:
        if args.eval_id_column not in frame:
            raise KeyError(f"Missing evaluation ID column: {args.eval_id_column}")
        eval_ids = np.asarray(frame[args.eval_id_column], dtype=np.int64)
    else:
        eval_ids = np.asarray(frame["_source_row_idx"], dtype=np.int64)
    if len(np.unique(eval_ids)) != row_count:
        raise ValueError("Evaluation IDs must be unique")
    if not args.target_free and "spect" not in frame:
        raise ValueError("Dataset has no measured spectra; use --target-free")

    model = load_model(Path(args.checkpoint), metadata, device, bins.bin_number)
    source = Subset(dataset, range(row_count)) if row_count < len(dataset) else dataset
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": event_dynamic_collate,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_options["prefetch_factor"] = max(1, args.prefetch_factor)
    loader = DataLoader(source, **loader_options)
    batches = loader
    if args.progress:
        from tqdm import tqdm

        batches = tqdm(loader, desc="Predicting spectra")

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            inputs = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
                if key not in {"spect", "input_idx"}
            }
            output = model(**inputs)
            if not isinstance(output, dict) or "spect" not in output:
                raise TypeError("CLEF model must return a spectrum under 'spect'")
            prediction = output["spect"].detach().to("cpu", dtype=torch.float32).numpy()
            if prediction.ndim != 2 or prediction.shape[1] != bins.bin_number:
                raise ValueError(f"Unexpected predicted spectrum shape: {prediction.shape}")
            predictions.append(prediction)
            indices.append(batch["input_idx"].to("cpu", dtype=torch.int64).numpy())
            if not args.target_free:
                targets.append(batch["spect"].to("cpu", dtype=torch.float32).numpy())
            if not args.progress and args.log_every_batches > 0:
                if batch_number % args.log_every_batches == 0 or batch_number == len(loader):
                    print(f"Predicted {sum(map(len, indices))}/{row_count} spectra", flush=True)

    input_indices = np.concatenate(indices)
    if not np.array_equal(input_indices, np.arange(row_count, dtype=np.int64)):
        raise ValueError("Prediction rows do not match dataset order")
    predicted = np.concatenate(predictions)
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Predicted intensities must be finite and nonnegative")
    arrays = {"eval_ids": eval_ids, "pred_spect": predicted}
    if not args.target_free:
        target = np.concatenate(targets)
        if target.shape != predicted.shape or not np.isfinite(target).all():
            raise ValueError("Measured spectrum shape or values are invalid")
        arrays["true_spect"] = target

    output_npz = Path(args.output_npz)
    output_report = Path(args.output_report)
    if output_npz.exists() or output_report.exists():
        raise FileExistsError("Prediction outputs already exist")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "event_table": str(Path(args.event_table).resolve()) if args.event_table else None,
        "output_npz": str(output_npz.resolve()),
        "target_free": bool(args.target_free),
        "spectra": row_count,
        "bins": int(predicted.shape[1]),
        "seconds": round(time.monotonic() - started, 3),
    }
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
