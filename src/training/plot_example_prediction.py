#!/usr/bin/env python3
"""
Create a poster-friendly 3-panel figure showing:

- Input fire mask at day t
- Model-predicted probability of burn at day t+1
- Observed burn mask at day t+1

Example:
  python -m src.training.plot_example_prediction \
    --checkpoint experiments/familyB_opt_gap1_seed42/best.pt \
    --fire-dir processed/overnight_05/camp_fire_2018_o5w1 \
    --family B \
    --output poster_assets/example_prediction_camp_fire_2018_o5w1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import sys

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.cnn_spread import build_spread_cnn
from src.training.dataset import WildfireSpreadDataset, FIRE_MASK_VAR


def _pick_day_idx_auto(fire_dir: Path) -> int:
    """
    Choose a day index with substantial next-day burn area.

    Heuristic: among all t where we have t+1, pick t whose label(t+1)
    has the largest number of burning pixels.
    """
    fire_ds = xr.open_zarr(fire_dir / "fire_masks.zarr")
    fire_da = fire_ds[FIRE_MASK_VAR]
    if "time" not in fire_da.dims or fire_da.sizes["time"] < 2:
        fire_ds.close()
        return 0

    n_time = int(fire_da.sizes["time"])
    scores = []
    for t in range(n_time - 1):
        label = fire_da.isel(time=t + 1).values
        scores.append(float(np.nansum(label > 0)))

    fire_ds.close()
    scores_arr = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores_arr).any() or scores_arr.max() <= 0:
        return 0
    return int(np.nanargmax(scores_arr))


def _compute_precision_recall(
    prob: np.ndarray,
    label: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Binary precision/recall at a fixed probability threshold."""
    prob_flat = prob.reshape(-1).astype(np.float32)
    label_flat = (label.reshape(-1) >= 0.5).astype(np.uint8)
    pred_bin = (prob_flat >= threshold).astype(np.uint8)

    tp = int(np.sum((pred_bin == 1) & (label_flat == 1)))
    fp = int(np.sum((pred_bin == 1) & (label_flat == 0)))
    fn = int(np.sum((pred_bin == 0) & (label_flat == 1)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall)


def _make_figure(
    fire_t: np.ndarray,
    prob_t1: np.ndarray,
    label_t1: np.ndarray,
    precision: float,
    recall: float,
    out_path: Path,
    title_prefix: Optional[str] = None,
) -> None:
    """Render a 3-panel prediction figure (vertically stacked for poster use)."""
    fig, axes = plt.subplots(3, 1, figsize=(4.5, 9.0))

    ax = axes[0]
    im0 = ax.imshow(fire_t, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.set_title("Fire mask (day t)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im1 = ax.imshow(prob_t1, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.contour(
        prob_t1,
        levels=[0.5],
        colors="white",
        linewidths=0.7,
    )
    ax.set_title("Predicted burn probability (t+1)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im2 = ax.imshow(label_t1, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.set_title("Observed burn mask (t+1)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    if title_prefix:
        fig.suptitle(
            f"{title_prefix}\nPrecision@0.5={precision:.3f}, Recall@0.5={recall:.3f}",
            fontsize=11,
            ha="center",
            y=0.955,
        )
    else:
        fig.suptitle(
            f"Next-day spread prediction\nPrecision@0.5={precision:.3f}, Recall@0.5={recall:.3f}",
            fontsize=11,
            ha="center",
            y=0.955,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot one example next-day spread prediction for poster use."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (best.pt).",
    )
    parser.add_argument(
        "--fire-dir",
        type=Path,
        required=True,
        help="Processed fire directory (with fire_masks.zarr, weather.zarr, satellite.zarr, static.zarr).",
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=["A", "B"],
        default="B",
        help="Model family: A (no SAR) or B (+ SAR).",
    )
    parser.add_argument(
        "--sar-max-gap-days",
        type=int,
        default=1,
        help="Max day gap for nearest SAR match (for Family B inputs).",
    )
    parser.add_argument(
        "--day-idx",
        type=int,
        default=None,
        help="Fire day index t to visualize (0-based). If omitted, pick t with the largest burn area at t+1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the figure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Optional device override (e.g., "cuda:0"); default picks CUDA if available.',
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default=None,
        help="Optional title prefix for the figure (e.g., fire name).",
    )
    args = parser.parse_args()

    fire_dir = args.fire_dir
    if not fire_dir.exists():
        raise FileNotFoundError(f"Fire directory not found: {fire_dir}")
    if not (fire_dir / "fire_masks.zarr").exists():
        raise FileNotFoundError(f"Missing fire_masks.zarr in {fire_dir}")

    if args.day_idx is None:
        day_idx = _pick_day_idx_auto(fire_dir)
    else:
        day_idx = int(args.day_idx)

    ds = WildfireSpreadDataset(
        fire_dirs=[fire_dir],
        family=args.family,
        patch_size=None,
        sar_max_gap_days=args.sar_max_gap_days,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No (t, t+1) samples found in {fire_dir}")

    if day_idx < 0 or day_idx >= len(ds):
        raise ValueError(f"day_idx {day_idx} out of range [0, {len(ds) - 1}]")

    x, y = ds[day_idx]
    x_np = x.numpy()
    y_np = y.numpy()[0]

    fire_t = x_np[0]

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_spread_cnn(family=args.family).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x_batch = x.unsqueeze(0).to(device)
        logits = model(x_batch)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    precision, recall = _compute_precision_recall(prob, y_np, threshold=0.5)

    _make_figure(
        fire_t=fire_t,
        prob_t1=prob,
        label_t1=y_np,
        precision=precision,
        recall=recall,
        out_path=args.output,
        title_prefix=args.title_prefix,
    )

    print(f"Wrote example prediction figure to: {args.output}")
    print(f"Day index t={day_idx} (label at t+1 used for target).")
    print(f"Precision@0.5={precision:.3f}, Recall@0.5={recall:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

