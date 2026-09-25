from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

from clef.dataset import ParquetDataset, event_dynamic_collate
from clef.model.spectrum_model import CLEFSpectrumModel
from clef.msutil.binutils import create_spectrum_bins
from clef.train import fit


def test_train_checkpoint_and_target_free_prediction(tmp_path):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    dataset_path = tmp_path / "molecule.parquet"
    pd.DataFrame({
        "rdmol": [molecule.ToBinary()],
        "smiles": ["CO"],
        "spect": [[[31.0, 1.0], [32.0, 0.5]]],
    }).to_parquet(dataset_path, index=False)
    dataset = ParquetDataset(dataset_path, create_spectrum_bins())
    loader = DataLoader(dataset, batch_size=1, collate_fn=event_dynamic_collate)
    model = CLEFSpectrumModel(
        graph_width=32, graph_layers=2, event_width=32, latent_count=4,
        dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "model.pt"
    fit(model, loader, loader, optimizer, epochs=1, device="cpu",
        checkpoint_path=checkpoint)
    assert checkpoint.is_file()

    predictions = tmp_path / "prediction.npz"
    report = tmp_path / "report.json"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(repo_root / "scripts" / "predict.py"),
        "--dataset", str(dataset_path),
        "--checkpoint", str(checkpoint),
        "--output-npz", str(predictions),
        "--output-report", str(report),
        "--target-free",
    ], check=True, cwd=repo_root)
    with np.load(predictions, allow_pickle=False) as output:
        assert output["eval_ids"].tolist() == [0]
        assert output["pred_spect"].shape == (1, 512)
        assert np.isfinite(output["pred_spect"]).all()
        assert "true_spect" not in output
