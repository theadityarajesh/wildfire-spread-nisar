#!/usr/bin/env python3
"""
Create one consolidated summary figure across multiple seed runs.

Example:
  python -m src.training.plot_seed_summary \
    --runs familyA_opt_seed42 familyA_opt_seed123 familyA_opt_seed999 \
    --label "Family A" \
    --output experiments/familyA_3seed_summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_history(run_dir: Path) -> list[dict]:
    p = run_dir / "history.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing history file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_test_metrics(run_dir: Path) -> dict:
    p = run_dir / "test_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing test metrics file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float_array(vals: list[float]) -> np.ndarray:
    out = np.array(vals, dtype=np.float64)
    return out


def _nanmeanstd(arr2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(arr2d, axis=0), np.nanstd(arr2d, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate 3+ seed runs into one summary figure")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names under experiments/ (e.g., familyA_opt_seed42 familyA_opt_seed123)",
    )
    parser.add_argument("--label", required=True, help="Figure label/title prefix (e.g., Family A)")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory root")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    exp_root = Path(args.experiments_dir)
    run_dirs = [exp_root / r for r in args.runs]

    histories = []
    metrics = []
    run_names = []
    for run_dir in run_dirs:
        histories.append(_load_history(run_dir))
        metrics.append(_load_test_metrics(run_dir))
        run_names.append(run_dir.name)

    # Align epochs via intersection so runs with differing lengths still work.
    epoch_sets = [set(int(row["epoch"]) for row in h) for h in histories]
    common_epochs = sorted(set.intersection(*epoch_sets))
    if not common_epochs:
        raise RuntimeError("No common epochs across runs; cannot aggregate.")
    epochs = np.array(common_epochs, dtype=np.int32)

    def stacked_key(key: str) -> np.ndarray:
        rows = []
        for h in histories:
            by_epoch = {int(row["epoch"]): float(row.get(key, np.nan)) for row in h}
            rows.append([by_epoch.get(int(e), np.nan) for e in epochs])
        return np.array(rows, dtype=np.float64)

    train_loss = stacked_key("train_loss")
    val_loss = stacked_key("val_loss")
    val_precision = stacked_key("val_precision")
    val_recall = stacked_key("val_recall")
    val_auc_pr = stacked_key("val_auc_pr")

    tr_loss_mu, tr_loss_sd = _nanmeanstd(train_loss)
    va_loss_mu, va_loss_sd = _nanmeanstd(val_loss)
    va_prec_mu, va_prec_sd = _nanmeanstd(val_precision)
    va_rec_mu, va_rec_sd = _nanmeanstd(val_recall)
    va_auc_mu, va_auc_sd = _nanmeanstd(val_auc_pr)

    test_precision = _to_float_array([float(m.get("precision", np.nan)) for m in metrics])
    test_recall = _to_float_array([float(m.get("recall", np.nan)) for m in metrics])
    test_auc = _to_float_array([float(m.get("auc_pr", np.nan)) for m in metrics])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: loss mean +/- std
    ax = axes[0]
    ax.plot(epochs, tr_loss_mu, label="Train loss (mean)", linewidth=2)
    ax.fill_between(epochs, tr_loss_mu - tr_loss_sd, tr_loss_mu + tr_loss_sd, alpha=0.2)
    ax.plot(epochs, va_loss_mu, label="Val loss (mean)", linewidth=2)
    ax.fill_between(epochs, va_loss_mu - va_loss_sd, va_loss_mu + va_loss_sd, alpha=0.2)
    ax.set_title("Loss (mean ± std across seeds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: validation metrics mean +/- std
    ax = axes[1]
    ax.plot(epochs, va_prec_mu, label="Val precision", linewidth=2)
    ax.fill_between(epochs, va_prec_mu - va_prec_sd, va_prec_mu + va_prec_sd, alpha=0.2)
    ax.plot(epochs, va_rec_mu, label="Val recall", linewidth=2)
    ax.fill_between(epochs, va_rec_mu - va_rec_sd, va_rec_mu + va_rec_sd, alpha=0.2)
    ax.plot(epochs, va_auc_mu, label="Val AUC-PR", linewidth=2)
    ax.fill_between(epochs, va_auc_mu - va_auc_sd, va_auc_mu + va_auc_sd, alpha=0.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Validation Metrics (mean ± std)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: test metrics per seed + mean line
    ax = axes[2]
    x = np.arange(len(run_names))
    ax.scatter(x, test_precision, label="Test precision", s=45)
    ax.scatter(x, test_recall, label="Test recall", s=45)
    ax.scatter(x, test_auc, label="Test AUC-PR", s=45)
    ax.axhline(np.nanmean(test_precision), linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(np.nanmean(test_recall), linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(np.nanmean(test_auc), linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in run_names], fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test Metrics by Seed")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(f"{args.label}: 3-seed Summary", fontsize=14)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Also save machine-readable summary.
    summary = {
        "label": args.label,
        "runs": run_names,
        "test_precision_mean": float(np.nanmean(test_precision)),
        "test_precision_std": float(np.nanstd(test_precision)),
        "test_recall_mean": float(np.nanmean(test_recall)),
        "test_recall_std": float(np.nanstd(test_recall)),
        "test_auc_pr_mean": float(np.nanmean(test_auc)),
        "test_auc_pr_std": float(np.nanstd(test_auc)),
        "output_png": str(out_path),
    }
    with (out_path.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary figure: {out_path}")
    print(f"Wrote summary stats:   {out_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

