# Training Family A and B Models

This describes how to train the next-day wildfire spread CNN (Family A: no SAR, Family B: + Sentinel-1 SAR) using data produced by the pipeline.

## Branch

Training code lives on the **model-training** branch (created from `cleanup/pipeline-reorg`). Data stays wherever your pipeline writes it (e.g. `processed/<fire_name>/`); it is not committed to git.

## Test locally (no data required)

Before pushing to the VM, run the smoke test to confirm imports, dataset layout, and one training step.

**Python version:** PyTorch does not support Python 3.13 yet. Use **Python 3.11 or 3.12** for the training env (your main pipeline venv can stay on 3.13). Example:

```bash
# Option A: separate venv for training (recommended if you use 3.13 for the rest of the project)
python3.12 -m venv .venv-train
source .venv-train/bin/activate   # or: .venv-train\Scripts\activate on Windows
pip install -r requirements.txt -r requirements-training.txt
python -m src.training.smoke_test

# Option B: if your default python is already 3.11 or 3.12
pip install -r requirements.txt -r requirements-training.txt
python -m src.training.smoke_test
```

This creates a tiny fake fire run (same zarr layout as the pipeline), builds the dataset for Family A and B, runs a forward pass, and runs one training step. If it prints **Smoke test PASSED.** you're good to push and run on the VM.

## Setup

1. **Install pipeline deps** (if not already):
   ```bash
   pip install -r requirements.txt
   ```

2. **Install training deps**:
   ```bash
   pip install -r requirements-training.txt
   ```
   This adds PyTorch and scikit-learn.

## Data

Each **fire run** is a directory containing (as produced by `data_pipeline.py`):

- `fire_masks.zarr` – daily fire mask (time, y, x)
- `weather.zarr` – GridMET vars (time, y, x)
- `satellite.zarr` – SAR backscatter (time, y, x, band) – can be placeholder
- `static.zarr` – elevation, slope, aspect (y, x)

Put all such directories under a single root, e.g. `processed/`, and optionally list them in a text file (one path per line, relative to that root).

## Running training

- **Family A** (no SAR; 11 channels: fire + weather + topo):
  ```bash
  python -m src.training.train_cnn --data-root processed --family A --epochs 20 --batch-size 8
  ```

- **Family B** (+ Sentinel-1 SAR; 13 channels):
  ```bash
  python -m src.training.train_cnn --data-root processed --family B --epochs 20 --batch-size 8
  ```

If you have a list of fire dirs (e.g. after generating several runs):

```bash
python -m src.training.train_cnn --data-root processed --fires-file configs/fires_for_training.txt --family A --epochs 30
```

Fires are split **by event** into train/val/test (default 80/10/10). Outputs go to `experiments/<run_name>/` (best checkpoint, config, test metrics).

## What the code does

- **Dataset** (`src/training/dataset.py`): For each (fire, day), loads inputs at day `t` and label = burned at day `t+1`. Family A uses fire mask + 7 weather + 3 static; Family B adds 2 SAR bands. Optionally crops random patches of `--patch-size` (default 128) so training fits in memory.
- **Model** (`src/models/cnn_spread.py`): Small U-Net-style CNN with configurable input channels (11 or 13).
- **Train script** (`src/training/train_cnn.py`): Train/val loop, saves best model by validation recall, then evaluates on held-out test fires and writes precision/recall to `test_metrics.json`.

## Getting it working

1. Run the **data pipeline** for at least one fire so you have one full run dir under `processed/`.
2. (Optional) Edit `configs/fires_for_training.txt` and add one line per fire dir (e.g. `camp_fire_2018`).
3. Install training deps and run:
   ```bash
   python -m src.training.train_cnn --data-root processed --family A --epochs 5 --batch-size 4
   ```
4. If you have multiple fires, add them to the list and re-run; the script will split them into train/val/test by fire event.

## Running on the VM (data lives there)

1. **Push the branch** from your laptop (after smoke test passes):
   ```bash
   git push -u origin model-training
   ```

2. **On the VM**: clone or pull, then switch to the branch:
   ```bash
   cd /path/to/wildfire-spread-nisar
   git fetch origin
   git checkout model-training
   ```

3. **Install deps** on the VM (if not already):
   ```bash
   pip install -r requirements.txt -r requirements-training.txt
   ```

4. **Point training at your data**: use the directory where the pipeline wrote the fire runs (e.g. `processed/` or a full path). If all runs are immediate subdirs of that path, you don't need a fires file:
   ```bash
   python -m src.training.train_cnn --data-root /path/to/processed --family A --epochs 20 --batch-size 8
   ```
   If you use a fires list (one path per line, relative to `--data-root`):
   ```bash
   python -m src.training.train_cnn --data-root /path/to/processed --fires-file configs/fires_for_training.txt --family A --epochs 20
   ```

5. **Optional**: run the smoke test on the VM once to confirm the environment:
   ```bash
   python -m src.training.smoke_test
   ```
