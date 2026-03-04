## Wildfire Spread with NISAR and Deep Learning

This repository contains an end‑to‑end pipeline for building wildfire event datasets and training convolutional neural networks to predict next‑day wildfire spread. It was developed around early NISAR and Sentinel‑1 SAR data, with a focus on California but support for global fires.

### High‑level components

- **Data pipeline (`src/pipeline/`)**  
  - `data_pipeline.py`: main end‑to‑end pipeline that, for each fire event and date range, builds aligned Zarr datasets:
    - `fire_masks.zarr` (time, y, x)
    - `weather.zarr` (time, y, x, channels)
    - `satellite.zarr` (time, y, x, band)
    - `static.zarr` (y, x, layers)
  - NISAR discovery and validation utilities (`discover_nisar_fires.py`, `summarize_runs.py`, `validate_run.py`, `scan_nisar_coverage.py`) for finding candidate fires, checking data quality, and producing summary CSV/JSON files.
  - Batch tools (`batch_process.py`, `rerun_failed.py`) to run the pipeline across manifests of many fires.

- **Training (`src/training/`)**  
  - `dataset.py`: dataset and sampling logic for patch‑based training on the processed Zarr stores, including options for oversampling rare fire pixels and stratified splits.
  - `train_cnn.py`: U‑Net–style CNN training script with support for different model families (A/B/C), NISAR global data, and diagnostics such as AUC‑PR.
  - Plotting utilities (`plot_seed_summary.py`, `plot_family_training_curves_for_poster.py`, `plot_family_comparison_for_poster.py`, `plot_example_prediction.py`) for inspecting training curves, family comparisons, and example predictions.

- **Configuration and docs**
  - `configs/`: manifests, coverage scans, and training lists for NISAR and baseline experiments.
  - `docs/NISAR_DATA.md`: notes on how NISAR data are discovered, validated, and used.
  - `docs/TRAINING.md`: additional details on training configurations and options.

- **Poster assets (`poster_assets/`)**  
  - Final PNGs and CSVs used to generate a Synopsys‑style poster: dataset characteristics, family A/B/C training curves, family comparison bar chart, and example prediction maps.

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-training.txt
```

3. Configure external services and secrets in `.env` (not committed), for example:

- `EARTHDATA_TOKEN` (ASF downloads and related services)
- `OPENTOPOGRAPHY_API_KEY` (terrain data)
- `EARTHENGINE_PROJECT` (fire mask data)
- `FIRMS_MAP_KEY` (NASA FIRMS fire detections, if used)

### Typical workflows

- **Run the data pipeline for a set of fires**

  1. Prepare or update a manifest CSV in `configs/` (see existing examples).
  2. Run the batch processor:

  ```bash
  python -m src.pipeline.batch_process \
    --manifest-csv configs/nisar_global_fires.csv \
    --output-root processed/nisar_global
  ```

- **Train a model family (A/B/C)**

  1. Use the appropriate training list in `configs/` (e.g., `fires_for_training.txt` or a NISAR‑specific list).
  2. Train a CNN:

  ```bash
  python -m src.training.train_cnn \
    --data-root processed/nisar_global \
    --fires-file configs/fires_nisar_training.txt \
    --family C \
    --out-dir experiments
  ```

- **Generate analysis and poster figures**

  - Dataset characteristics:

  ```bash
  python -m src.pipeline.plot_dataset_characteristics_for_poster \
    --summaries processed/overnight_01_summary.csv \
               processed/overnight_02_summary.csv \
               processed/overnight_03_summary.csv \
               processed/overnight_04_summary.csv \
               processed/overnight_05_summary.csv \
               configs/nisar_global_summary.csv \
    --output poster_assets/dataset_characteristics.png
  ```

  - Family comparison bar chart:

  ```bash
  python -m src.training.plot_family_comparison_for_poster \
    --csv poster_assets/seed_comparison_ABC.csv \
    --output poster_assets/family_ABC_comparison.png
  ```

### Notes

- Large, derived artifacts (data under `processed/`, experiment runs in `experiments/`, and archived baseline bundles under `poster_assets/baseline_ab_*`) are ignored by git and can be regenerated from the pipeline and training scripts.
- For a deeper dive into the data pipeline or training setup, start with `docs/NISAR_DATA.md` and `docs/TRAINING.md`. 
