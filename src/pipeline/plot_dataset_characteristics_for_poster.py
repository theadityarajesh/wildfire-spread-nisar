#!/usr/bin/env python3
"""
Poster-ready dataset characteristics visualization.

Aggregates one or more *_summary.csv files (e.g., overnight_01_summary.csv)
and produces a compact multi-panel figure summarizing:
- number of fires by status (OK/WARN/FAIL)
- distribution of fire_active_fraction
- distribution of satellite_nonzero_fraction
- scatter of fire_active_fraction vs satellite_nonzero_fraction

Example:
  python -m src.pipeline.plot_dataset_characteristics_for_poster \
    --summaries processed/overnight_01_summary.csv \
               processed/overnight_02_summary.csv \
               processed/overnight_03_summary.csv \
               processed/overnight_04_summary.csv \
               processed/overnight_05_summary.csv \
    --output poster_assets/dataset_characteristics.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _try_float(val: str) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_summaries(paths: List[Path]) -> Tuple[List[dict], List[Path]]:
    rows: List[dict] = []
    used: List[Path] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        used.append(p)
    return rows, used


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create poster-ready dataset characteristics figure from *_summary.csv files."
    )
    parser.add_argument(
        "--summaries",
        nargs="+",
        type=Path,
        default=[
            Path("processed/overnight_01_summary.csv"),
            Path("processed/overnight_02_summary.csv"),
            Path("processed/overnight_03_summary.csv"),
            Path("processed/overnight_04_summary.csv"),
            Path("processed/overnight_05_summary.csv"),
        ],
        help="One or more summary CSVs with columns like run_dir,status,fire_active_fraction,satellite_nonzero_fraction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poster_assets/dataset_characteristics.png"),
        help="Output PNG path for the dataset characteristics figure.",
    )
    args = parser.parse_args()

    rows, used_paths = _load_summaries(args.summaries)
    if not rows:
        raise RuntimeError(
            f"No rows loaded from summaries. Checked: {[str(p) for p in args.summaries]}"
        )

    statuses: Dict[str, int] = {}
    fire_fracs: List[float] = []
    sar_fracs: List[float] = []

    for row in rows:
        status = (row.get("status") or "").strip()
        if not status:
            status = "UNKNOWN"
        statuses[status] = statuses.get(status, 0) + 1

        fa = _try_float(row.get("fire_active_fraction", ""))
        if fa is not None and np.isfinite(fa):
            fire_fracs.append(fa)

        sa = _try_float(row.get("satellite_nonzero_fraction", ""))
        if sa is not None and np.isfinite(sa):
            sar_fracs.append(sa)

    fire_arr = np.array(fire_fracs, dtype=np.float64) if fire_fracs else np.array([])
    sar_arr = np.array(sar_fracs, dtype=np.float64) if sar_fracs else np.array([])

    # Slightly larger canvas and fonts for poster use.
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))

    # Panel 1: counts by status
    ax = axes[0, 0]
    status_keys = sorted(statuses.keys())
    counts = [statuses[k] for k in status_keys]
    bars = ax.bar(status_keys, counts, color="#1f77b4", alpha=0.9)
    ax.set_ylabel("# fires", fontsize=13)
    ax.set_title("Pipeline status", fontsize=15)

    if counts:
        y_max = max(counts)
        # Add a bit of headroom so count labels sit clearly above bars.
        ax.set_ylim(0, y_max * 1.15)
        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                c + y_max * 0.03,
                str(c),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Panel 2: fire_active_fraction histogram
    ax = axes[0, 1]
    if fire_arr.size > 0:
        ax.hist(fire_arr, bins=10, color="#ff7f0e", alpha=0.9, edgecolor="black")
        ax.set_xlabel("Fire-active fraction", fontsize=13)
        ax.set_ylabel("# fires", fontsize=13)
        ax.set_title("Burned fraction per fire", fontsize=15)
    else:
        ax.text(
            0.5,
            0.5,
            "No fire_active_fraction data",
            ha="center",
            va="center",
            fontsize=11,
        )
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Panel 3: satellite_nonzero_fraction histogram
    ax = axes[1, 0]
    if sar_arr.size > 0:
        ax.hist(sar_arr, bins=10, color="#2ca02c", alpha=0.9, edgecolor="black")
        ax.set_xlabel("SAR non-zero fraction", fontsize=13)
        ax.set_ylabel("# fires", fontsize=13)
        ax.set_title("SAR coverage per fire", fontsize=15)
    else:
        ax.text(
            0.5,
            0.5,
            "No satellite_nonzero_fraction data",
            ha="center",
            va="center",
            fontsize=11,
        )
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Panel 4: scatter fire_active_fraction vs satellite_nonzero_fraction
    ax = axes[1, 1]
    if fire_arr.size > 0 and sar_arr.size > 0 and fire_arr.size == sar_arr.size:
        ax.scatter(
            fire_arr,
            sar_arr,
            s=24,
            alpha=0.7,
            edgecolors="none",
            color="#9467bd",
        )
        ax.set_xlabel("Fire-active fraction", fontsize=13)
        ax.set_ylabel("SAR non-zero fraction", fontsize=13)
        ax.set_title("Burned vs SAR coverage", fontsize=15)
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient paired data",
            ha="center",
            va="center",
            fontsize=11,
        )
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Slightly larger title, with reduced whitespace above the panels.
    fig.suptitle("Dataset characteristics", fontsize=19, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote dataset characteristics figure: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

