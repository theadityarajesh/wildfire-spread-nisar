#!/usr/bin/env python3
"""
Poster-ready comparison of Families A, B, C.

Reads seed-wise metrics from:
  poster_assets/seed_comparison_ABC.csv

and writes a grouped bar chart with mean ± std for each family/metric to:
  poster_assets/family_ABC_comparison.png  (by default)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_seed_metrics(csv_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    families = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: Dict[str, List[Dict[str, float]]] = {}
        for row in reader:
            fam = row.get("family", "").strip()
            if not fam:
                continue
            rows.setdefault(fam, []).append(row)

    for fam, fam_rows in rows.items():
        prec = [float(r["precision"]) for r in fam_rows]
        rec = [float(r["recall"]) for r in fam_rows]
        auc = [float(r["auc_pr"]) for r in fam_rows]
        families[fam] = {
            "precision": np.array(prec, dtype=np.float64),
            "recall": np.array(rec, dtype=np.float64),
            "auc_pr": np.array(auc, dtype=np.float64),
        }
    return families


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create poster-ready Family A/B/C comparison bar chart."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("poster_assets/seed_comparison_ABC.csv"),
        help="CSV with columns: family,precision,recall,auc_pr,...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poster_assets/family_ABC_comparison.png"),
        help="Output PNG path for comparison figure.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Seed comparison CSV not found: {args.csv}")

    fam_stats = _load_seed_metrics(args.csv)
    if not fam_stats:
        raise RuntimeError(f"No family entries found in {args.csv}")

    families = sorted(fam_stats.keys())
    metrics = ["auc_pr", "precision", "recall"]
    metric_labels = {
        "auc_pr": "Test AUC-PR",
        "precision": "Test precision@0.5",
        "recall": "Test recall@0.5",
    }

    n_fam = len(families)
    n_met = len(metrics)

    x = np.arange(n_met, dtype=np.float64)
    width = 0.25 if n_fam == 3 else 0.7 / max(n_fam, 1)
    offsets = (np.arange(n_fam) - (n_fam - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    colors = {
        "A": "#1f77b4",
        "B": "#ff7f0e",
        "C": "#2ca02c",
    }

    for i, fam in enumerate(families):
        means = []
        stds = []
        for m in metrics:
            arr = fam_stats[fam][m]
            mu, sd = _mean_std(arr)
            means.append(mu)
            stds.append(sd)
        means_arr = np.array(means, dtype=np.float64)
        stds_arr = np.array(stds, dtype=np.float64)

        bars = ax.bar(
            x + offsets[i],
            means_arr,
            width=width,
            yerr=stds_arr,
            label=f"Family {fam}",
            color=colors.get(fam, None),
            alpha=0.9,
            capsize=4,
        )
        # Add numeric labels above bars (poster-friendly).
        for bar, val in zip(bars, means_arr):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Families A/B/C: Test performance (mean ± std across seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote family comparison figure: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

