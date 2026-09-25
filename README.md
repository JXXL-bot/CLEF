# CLEF: Cleavage Events Guide Electron Ionization Mass Spectrum Prediction

## Requirements

Python 3.9 and PyTorch 1.13.1.

```bash
python -m pip install torch==1.13.1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Datasets

- [MoNA](https://mona.fiehnlab.ucdavis.edu/)
- [SWGDRUG Mass Spectral Library](https://www.swgdrug.org/ms.htm)

NIST23 is available for purchase through NIST distributors.

## Quick Start

1. Provide a CSV with `molecule_id`, `smiles`, `split`, and `spectrum_json`.
2. Prepare the data:

   ```bash
   python scripts/prepare_dataset.py --input YOUR_DATA.csv --output-dir data
   ```

3. Check the installation:

   ```bash
   python -m pytest -q tests
   ```
