#!/usr/bin/env python3
"""
Poster-ready training curve summary across multiple seeds.

This is a slimmer, more poster-focused version of plot_seed_summary:
- focuses on validation metrics vs. epoch
- aggregates mean ± std across seeds

Example:
  python -m src.training.plot_family_training_curves_for_poster \
    --runs familyC_nisar_hq_seed42 familyC_nisar_hq_seed123 familyC_nisar_hq_seed999 \
    --label "Family C (NISAR-like SAR)" \
    --output poster_assets/familyC_training_curves_poster.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

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


def _nanmeanstd(arr2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(arr2d, axis=0), np.nanstd(arr2d, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Poster-ready training curves (val metrics vs epoch, aggregated across seeds)."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names under experiments/ (e.g., familyC_nisar_hq_seed42 ...)",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label for the figure (e.g., 'Family C (NISAR-like SAR)').",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Experiments directory root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path (e.g., poster_assets/familyC_training_curves_poster.png).",
    )
    args = parser.parse_args()

    exp_root = Path(args.experiments_dir)
    run_dirs: List[Path] = [exp_root / r for r in args.runs]

    histories = []
    for run_dir in run_dirs:
        histories.append(_load_history(run_dir))

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

    val_loss = stacked_key("val_loss")
    val_precision = stacked_key("val_precision")
    val_recall = stacked_key("val_recall")
    val_auc_pr = stacked_key("val_auc_pr")

    va_loss_mu, va_loss_sd = _nanmeanstd(val_loss)
    va_prec_mu, va_prec_sd = _nanmeanstd(val_precision)
    va_rec_mu, va_rec_sd = _nanmeanstd(val_recall)
    va_auc_mu, va_auc_sd = _nanmeanstd(val_auc_pr)

    # Slightly larger canvas for poster usage.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Panel 1: validation loss and AUC-PR
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(
        epochs,
        va_loss_mu,
        label="Val loss (mean)",
        color="#1f77b4",
        linewidth=2,
    )
    ax.fill_between(
        epochs,
        va_loss_mu - va_loss_sd,
        va_loss_mu + va_loss_sd,
        color="#1f77b4",
        alpha=0.15,
    )
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("BCE loss", color="#1f77b4", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=12)

    ax2.plot(
        epochs,
        va_auc_mu,
        label="Val AUC-PR (mean)",
        color="#2ca02c",
        linewidth=2,
    )
    ax2.fill_between(
        epochs,
        va_auc_mu - va_auc_sd,
        va_auc_mu + va_auc_sd,
        color="#2ca02c",
        alpha=0.15,
    )
    ax2.set_ylabel("AUC-PR", color="#2ca02c", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=12)
    ax.set_title("Validation loss and AUC-PR", fontsize=16)
    ax.grid(True, alpha=0.3)

    # Panel 2: validation precision and recall
    ax = axes[1]
    ax.plot(
        epochs,
        va_prec_mu,
        label="Val precision@0.5",
        color="#ff7f0e",
        linewidth=2,
    )
    ax.fill_between(
        epochs,
        va_prec_mu - va_prec_sd,
        va_prec_mu + va_prec_sd,
        color="#ff7f0e",
        alpha=0.15,
    )
    ax.plot(
        epochs,
        va_rec_mu,
        label="Val recall@0.5",
        color="#1f77b4",
        linewidth=2,
    )
    ax.fill_between(
        epochs,
        va_rec_mu - va_rec_sd,
        va_rec_mu + va_rec_sd,
        color="#1f77b4",
        alpha=0.15,
    )
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Validation precision and recall", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="lower right", fontsize=12)

    fig.suptitle(
        f"{args.label}: training dynamics (mean ± std across seeds)",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote poster training-curve figure: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

