#!/usr/bin/env python3
"""
Hyperparameter search for wildfire spread models using Optuna.

This script is designed for single-GPU (e.g., T4) runs. It runs multiple
short training trials by invoking `train_cnn.py` and evaluates each trial
using the selection_score from test_metrics.json (primarily val AUC-PR).

Example (Family A):
  python -m src.training.hparam_optuna \\
    --family A \\
    --data-root processed \\
    --fires-file configs/fires_for_training.txt \\
    --trials 20

Example (Family C / NISAR):
  python -m src.training.hparam_optuna \\
    --family B \\
    --data-root processed/nisar_global \\
    --fires-file configs/fires_nisar_training.txt \\
    --trials 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]


def _run_trial(
    trial_number: int,
    family: str,
    data_root: Path,
    fires_file: Path | None,
    max_epochs: int,
) -> float:
    """Launch a single training trial as a subprocess and return its score."""
    run_name = f"hparam_{family}_trial{trial_number}"
    out_dir = ROOT / "experiments"

    # Hyperparameters to tune
    lr = optuna.trial.Trial.current().suggest_float("lr", 1e-4, 3e-3, log=True)  # type: ignore[attr-defined]
    batch_size = optuna.trial.Trial.current().suggest_categorical("batch_size", [4, 8, 16])  # type: ignore[attr-defined]
    loss_type = optuna.trial.Trial.current().suggest_categorical("loss", ["bce", "focal"])  # type: ignore[attr-defined]

    fire_oversample_frac = 0.0
    if family.upper() == "B":
        fire_oversample_frac = optuna.trial.Trial.current().suggest_float("fire_oversample_frac", 0.0, 0.5)  # type: ignore[attr-defined]

    args = [
        sys.executable,
        "-m",
        "src.training.train_cnn",
        "--data-root",
        str(data_root),
        "--family",
        family,
        "--epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        f"{lr:.6g}",
        "--run-name",
        run_name,
        "--out-dir",
        str(out_dir),
        "--loss",
        loss_type,
    ]

    if fires_file is not None:
        args.extend(["--fires-file", str(fires_file)])

    if fire_oversample_frac > 0:
        args.extend(["--fire-oversample-frac", f"{fire_oversample_frac:.3f}"])

    if loss_type == "bce":
        # Pos weight search for BCE
        pos_weight = optuna.trial.Trial.current().suggest_float("pos_weight", 5.0, 300.0, log=True)  # type: ignore[attr-defined]
        args.extend(["--pos-weight", f"{pos_weight:.3f}"])
    else:
        # Focal parameters
        gamma = optuna.trial.Trial.current().suggest_float("focal_gamma", 1.0, 3.0)  # type: ignore[attr-defined]
        alpha = optuna.trial.Trial.current().suggest_float("focal_alpha", 0.3, 0.9)  # type: ignore[attr-defined]
        args.extend(["--focal-gamma", f"{gamma:.3f}", "--focal-alpha", f"{alpha:.3f}"])

    print(f"\n[Trial {trial_number}] Running: {' '.join(args)}")
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[Trial {trial_number}] training failed with return code {proc.returncode}")
        # Penalize failed trials
        return -1e6

    metrics_path = out_dir / run_name / "test_metrics.json"
    if not metrics_path.exists():
        print(f"[Trial {trial_number}] Missing metrics file: {metrics_path}")
        return -1e6

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Prefer selection_score (based on val AUC-PR); fall back to test AUC-PR
    score = metrics.get("selection_score")
    if score is None or (isinstance(score, float) and (score != score)):  # NaN check
        score = metrics.get("auc_pr", 0.0)

    print(f"[Trial {trial_number}] score={score}")
    return float(score)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for wildfire CNN")
    parser.add_argument("--family", type=str, choices=["A", "B"], required=True, help="Model family to tune")
    parser.add_argument("--data-root", type=Path, required=True, help="Root of processed fire directories")
    parser.add_argument("--fires-file", type=Path, default=None, help="Optional fires file to restrict runs")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=8, help="Epochs per trial (keep small)")
    args = parser.parse_args()

    family = args.family.upper()
    data_root = args.data_root
    fires_file = args.fires_file if args.fires_file and args.fires_file.exists() else None

    def objective(trial: optuna.trial.Trial) -> float:
        return _run_trial(trial.number, family, data_root, fires_file, args.max_epochs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\nBest trial:")
    print(study.best_trial.number, study.best_trial.value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

