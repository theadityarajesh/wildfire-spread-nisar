#!/usr/bin/env python3
"""
Train Family A (no SAR) or Family B (+ Sentinel-1 SAR) CNN for next-day wildfire spread.

Usage:
  python -m src.training.train_cnn --data-root processed --fires-file configs/fires_for_training.txt --family A --epochs 20
  python -m src.training.train_cnn --data-root processed --fires-file configs/fires_for_training.txt --family B --epochs 20 --batch-size 8

Splits fires into train/val/test by event (e.g. 80/10/10). Saves best checkpoint and metrics to experiments/<run_name>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

# Add project root for imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.cnn_spread import build_spread_cnn
from src.training.dataset import WildfireSpreadDataset, train_val_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _jsonable_args(args: argparse.Namespace) -> dict:
    """Convert argparse values to JSON-serializable types."""
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _flatten_binary(pred: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Flatten prediction probabilities and binary targets to 1D NumPy arrays."""
    pred_np = pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
    target_np = (target.detach().cpu().numpy().reshape(-1) >= 0.5).astype(np.uint8)
    return pred_np, target_np


def _precision_recall_at_threshold(
    pred_np: np.ndarray, target_np: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    pred_bin = (pred_np >= threshold).astype(np.uint8)
    tp = int(np.sum((pred_bin == 1) & (target_np == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_np == 0)))
    fn = int(np.sum((pred_bin == 0) & (target_np == 1)))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall)


def _compute_auc_pr(pred_np: np.ndarray, target_np: np.ndarray) -> float:
    """Average precision (AUC-PR); NaN when one class is missing."""
    n_pos = int(target_np.sum())
    n_total = int(target_np.size)
    if n_total == 0 or n_pos == 0 or n_pos == n_total:
        return float("nan")
    return float(average_precision_score(target_np, pred_np))


def _metric_str(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "nan"


def _selection_score(metrics: dict) -> float:
    """Model-selection score: prefer val AUC-PR when defined, else val recall."""
    auc_pr = float(metrics.get("auc_pr", float("nan")))
    if np.isfinite(auc_pr):
        return auc_pr
    return float(metrics.get("recall", 0.0))


def save_training_curves(history: list[dict], out_path: Path) -> None:
    """Save poster-friendly training curves PNG."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_prec = [h["train_precision"] for h in history]
    val_prec = [h["val_precision"] for h in history]
    train_rec = [h["train_recall"] for h in history]
    val_rec = [h["val_recall"] for h in history]
    train_auc = [h["train_auc_pr"] for h in history]
    val_auc = [h["val_auc_pr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss, label="Val loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_prec, label="Train precision@0.5")
    axes[1].plot(epochs, train_rec, label="Train recall@0.5")
    axes[1].plot(epochs, val_prec, label="Val precision@0.5")
    axes[1].plot(epochs, val_rec, label="Val recall@0.5")
    axes[1].set_title("Precision/Recall vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, train_auc, label="Train AUC-PR")
    axes[2].plot(epochs, val_auc, label="Val AUC-PR")
    axes[2].set_title("AUC-PR vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Average precision")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pr_curve(pred_np: np.ndarray, target_np: np.ndarray, out_path: Path, title: str) -> float:
    """Save precision-recall curve PNG and return AUC-PR."""
    auc_pr = _compute_auc_pr(pred_np, target_np)
    fig, ax = plt.subplots(figsize=(6, 5))

    if np.isfinite(auc_pr):
        precision, recall, _ = precision_recall_curve(target_np, pred_np)
        ax.plot(recall, precision, linewidth=2, label=f"PR curve (AUC-PR={auc_pr:.3f})")
        p05, r05 = _precision_recall_at_threshold(pred_np, target_np, threshold=0.5)
        ax.scatter([r05], [p05], s=40, label="Threshold=0.5")
    else:
        ax.text(
            0.5,
            0.5,
            "AUC-PR undefined\n(one class only in evaluation set)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc_pr


def _load_fire_dirs(data_root: Path, fires_file: Path) -> list[Path]:
    """Load list of fire run directories from a file or directory scan."""
    data_root = Path(data_root)
    if fires_file and fires_file.exists():
        dirs = []
        for line in fires_file.read_text().strip().splitlines():
            line = line.strip().split("#")[0]
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = data_root / p
            if p.is_dir():
                dirs.append(p)
            else:
                logger.warning("Fire dir not found: %s", p)
        return dirs
    # Fallback: all immediate subdirs of data_root that contain fire_masks.zarr
    dirs = [d for d in data_root.iterdir() if d.is_dir() and (d / "fire_masks.zarr").exists()]
    return sorted(dirs)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute thresholded precision/recall and AUC-PR."""
    pred_np, target_np = _flatten_binary(pred, target)
    prec, rec = _precision_recall_at_threshold(pred_np, target_np, threshold=threshold)
    auc_pr = _compute_auc_pr(pred_np, target_np)
    return {
        "precision": prec,
        "recall": rec,
        "auc_pr": auc_pr,
        "target_positive_fraction": float(target_np.mean()) if target_np.size else float("nan"),
        "pred_positive_fraction": float((pred_np >= threshold).mean()) if pred_np.size else float("nan"),
    }


def estimate_pos_weight(dataset: WildfireSpreadDataset, max_samples: int = 64) -> float:
    """
    Estimate BCE positive-class weight from a subset of training samples.

    pos_weight ~= negatives / positives. Returns 1.0 if estimate is unstable.
    """
    n = len(dataset)
    if n == 0:
        return 1.0
    sample_n = max(1, min(max_samples, n))
    if sample_n == n:
        indices = np.arange(n)
    else:
        indices = np.linspace(0, n - 1, num=sample_n).round().astype(int)

    pos = 0.0
    total = 0.0
    for idx in indices:
        _, y = dataset[int(idx)]
        y_np = y.numpy()
        pos += float(np.sum(y_np >= 0.5))
        total += float(y_np.size)

    neg = max(total - pos, 0.0)
    if pos <= 0.0:
        return 1.0
    # Clamp to avoid extreme instability; allow high weight for very rare positives (e.g. NISAR global).
    return float(np.clip(neg / pos, 1.0, 500.0))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    n = 0
    all_pred, all_target = [], []
    skipped_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            skipped_batches += 1
            logger.warning("Skipping non-finite training batch loss: %s", loss.detach().cpu().item())
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            all_pred.append(pred)
            all_target.append(y)
    if n == 0 or not all_pred:
        avg_loss = float("nan")
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "auc_pr": float("nan"),
            "target_positive_fraction": float("nan"),
            "pred_positive_fraction": float("nan"),
        }
    else:
        avg_loss = total_loss / n
        pred_cat = torch.cat(all_pred, dim=0)
        target_cat = torch.cat(all_target, dim=0)
        metrics = compute_metrics(pred_cat, target_cat)
    if skipped_batches:
        logger.warning("Skipped %d non-finite training batches this epoch", skipped_batches)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_raw: bool = False,
) -> tuple[float, dict] | tuple[float, dict, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred, all_target = [], []
    skipped_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = torch.sigmoid(logits)
        all_pred.append(pred)
        all_target.append(y)
    if n == 0 or not all_pred:
        avg_loss = float("nan")
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "auc_pr": float("nan"),
            "target_positive_fraction": float("nan"),
            "pred_positive_fraction": float("nan"),
        }
        pred_cat = torch.empty(0)
        target_cat = torch.empty(0)
    else:
        avg_loss = total_loss / n
        pred_cat = torch.cat(all_pred, dim=0)
        target_cat = torch.cat(all_target, dim=0)
        metrics = compute_metrics(pred_cat, target_cat)
    if skipped_batches:
        logger.warning("Skipped %d non-finite eval batches", skipped_batches)
    if return_raw:
        return avg_loss, metrics, pred_cat, target_cat
    return avg_loss, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train wildfire spread CNN (Family A or B)")
    parser.add_argument("--data-root", type=str, default="processed", help="Root directory containing fire run dirs")
    parser.add_argument("--fires-file", type=Path, default=None, help="Text file listing fire dir paths (relative to data-root) or one path per line")
    parser.add_argument("--family", type=str, choices=["A", "B"], default="A", help="A = no SAR, B = + Sentinel-1 SAR")
    parser.add_argument("--patch-size", type=int, default=128, help="Spatial patch size (default 128)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--sar-max-gap-days", type=int, default=2, help="Max day gap for nearest SAR match in Family B")
    parser.add_argument(
        "--fire-oversample-frac",
        type=float,
        default=0.0,
        help="Fraction of training patches to center on a fire pixel when present (0=off, 0.5=half). Use for very rare positives (e.g. NISAR global).",
    )
    parser.add_argument("--no-pos-weight", action="store_true", help="Disable class-weighted BCE loss")
    parser.add_argument("--pos-weight", type=float, default=None, help="Manual BCE positive-class weight (default: auto-estimate)")
    parser.add_argument("--pos-weight-samples", type=int, default=64, help="Samples used to auto-estimate pos_weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments"), help="Base dir for run outputs")
    parser.add_argument("--run-name", type=str, default=None, help="Run name (default: family_A_YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    fire_dirs = _load_fire_dirs(data_root, args.fires_file)
    if not fire_dirs:
        logger.error("No fire directories found. Use --data-root and optionally --fires-file.")
        return 1

    train_dirs, val_dirs, test_dirs = train_val_test_split(
        fire_dirs, args.train_frac, args.val_frac, args.test_frac, args.seed
    )
    logger.info("Fires: train=%d val=%d test=%d", len(train_dirs), len(val_dirs), len(test_dirs))

    train_ds = WildfireSpreadDataset(
        train_dirs,
        family=args.family,
        patch_size=args.patch_size,
        sar_max_gap_days=args.sar_max_gap_days,
        fire_oversample_frac=args.fire_oversample_frac,
    )
    val_ds = WildfireSpreadDataset(
        val_dirs,
        family=args.family,
        patch_size=args.patch_size,
        sar_max_gap_days=args.sar_max_gap_days,
    )
    test_ds = WildfireSpreadDataset(
        test_dirs,
        family=args.family,
        patch_size=args.patch_size,
        sar_max_gap_days=args.sar_max_gap_days,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_spread_cnn(family=args.family).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.no_pos_weight:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Loss: unweighted BCEWithLogitsLoss")
    else:
        if args.pos_weight is None:
            pos_weight_value = estimate_pos_weight(train_ds, max_samples=args.pos_weight_samples)
            logger.info("Loss: auto-estimated pos_weight=%.3f from %d samples", pos_weight_value, args.pos_weight_samples)
        else:
            pos_weight_value = float(args.pos_weight)
            logger.info("Loss: using manual pos_weight=%.3f", pos_weight_value)
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    run_name = args.run_name or f"family_{args.family}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(_jsonable_args(args), f, indent=2)

    history: list[dict] = []
    best_val_score = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_auc_pr": train_metrics["auc_pr"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_auc_pr": val_metrics["auc_pr"],
                "train_target_positive_fraction": train_metrics["target_positive_fraction"],
                "train_pred_positive_fraction": train_metrics["pred_positive_fraction"],
                "val_target_positive_fraction": val_metrics["target_positive_fraction"],
                "val_pred_positive_fraction": val_metrics["pred_positive_fraction"],
            }
        )
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  "
            "train_prec=%.3f train_rec=%.3f train_aucpr=%s  "
            "val_prec=%.3f val_rec=%.3f val_aucpr=%s  "
            "val_pos=%.4f val_pred_pos=%.4f",
            epoch,
            train_loss,
            val_loss,
            train_metrics["precision"],
            train_metrics["recall"],
            _metric_str(float(train_metrics["auc_pr"])),
            val_metrics["precision"],
            val_metrics["recall"],
            _metric_str(float(val_metrics["auc_pr"])),
            float(val_metrics["target_positive_fraction"]),
            float(val_metrics["pred_positive_fraction"]),
        )
        current_score = _selection_score(val_metrics)
        if current_score > best_val_score:
            best_val_score = current_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                },
                out_dir / "best.pt",
            )
            logger.info("  -> saved best.pt (selection_score=%s)", _metric_str(best_val_score))

    # Final test set evaluation
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_metrics, test_pred, test_target = evaluate(
        model, test_loader, criterion, device, return_raw=True
    )
    logger.info(
        "Test: loss=%.4f  precision=%.3f  recall=%.3f  auc_pr=%s",
        test_loss,
        test_metrics["precision"],
        test_metrics["recall"],
        _metric_str(float(test_metrics["auc_pr"])),
    )

    # Save plots for poster/report use.
    save_training_curves(history, out_dir / "training_curves.png")
    test_pred_np, test_target_np = _flatten_binary(test_pred, test_target)
    test_auc_pr_plot = save_pr_curve(
        test_pred_np,
        test_target_np,
        out_dir / "test_pr_curve.png",
        title=f"Test Precision-Recall Curve ({args.family})",
    )

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    test_metrics = {
        **test_metrics,
        "loss": test_loss,
        "auc_pr_curve_plot": test_auc_pr_plot,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "selection_score": float(ckpt.get("best_val_score", float("nan"))),
        "artifacts": {
            "training_curves_png": str(out_dir / "training_curves.png"),
            "test_pr_curve_png": str(out_dir / "test_pr_curve.png"),
            "history_json": str(history_path),
        },
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
