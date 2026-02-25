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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root for imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.cnn_spread import build_spread_cnn, FAMILY_A_CHANNELS, FAMILY_B_CHANNELS
from src.training.dataset import WildfireSpreadDataset, train_val_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _jsonable_args(args: argparse.Namespace) -> dict:
    """Convert argparse values to JSON-serializable types."""
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


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
    """Compute precision, recall, and approximate AUC-PR (binary at threshold)."""
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return {"precision": prec, "recall": rec}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    n = 0
    all_pred, all_target = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            all_pred.append(pred)
            all_target.append(y)
    avg_loss = total_loss / max(n, 1)
    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = compute_metrics(pred_cat, target_cat)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred, all_target = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = torch.sigmoid(logits)
        all_pred.append(pred)
        all_target.append(y)
    avg_loss = total_loss / max(n, 1)
    pred_cat = torch.cat(all_pred, dim=0)
    target_cat = torch.cat(all_target, dim=0)
    metrics = compute_metrics(pred_cat, target_cat)
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

    train_ds = WildfireSpreadDataset(train_dirs, family=args.family, patch_size=args.patch_size)
    val_ds = WildfireSpreadDataset(val_dirs, family=args.family, patch_size=args.patch_size)
    test_ds = WildfireSpreadDataset(test_dirs, family=args.family, patch_size=args.patch_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_spread_cnn(family=args.family).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = args.run_name or f"family_{args.family}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(_jsonable_args(args), f, indent=2)

    best_val_recall = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  val_precision=%.3f  val_recall=%.3f",
            epoch, train_loss, val_loss, val_metrics["precision"], val_metrics["recall"],
        )
        if val_metrics["recall"] > best_val_recall:
            best_val_recall = val_metrics["recall"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, out_dir / "best.pt")
            logger.info("  -> saved best.pt (recall=%.3f)", best_val_recall)

    # Final test set evaluation
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, device)
    logger.info("Test: loss=%.4f  precision=%.3f  recall=%.3f", test_loss, test_metrics["precision"], test_metrics["recall"])
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
