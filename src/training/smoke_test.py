#!/usr/bin/env python3
"""
Smoke test for training pipeline: runs without real data.

Creates a tiny fake fire run (same zarr layout as data_pipeline.py), then:
- Builds WildfireSpreadDataset (Family A and B)
- Runs one batch through the CNN
- Optionally runs 1 training step

Run locally before pushing to VM:
  python -m src.training.smoke_test

Requires: pip install -r requirements.txt -r requirements-training.txt
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

# Project root
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def make_fake_fire_run(out_dir: Path, n_time: int = 4, h: int = 32, w: int = 32) -> None:
    """Write minimal zarrs matching data_pipeline.py layout."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [f"2024-07-{20 + i:02d}" for i in range(n_time)]

    # fire_masks.zarr: fire_mask (time, y, x)
    fire = xr.DataArray(
        np.random.randint(0, 2, (n_time, h, w), dtype=np.uint8),
        dims=["time", "y", "x"],
        coords={"time": times, "y": np.arange(h), "x": np.arange(w)},
    )
    fire.to_dataset(name="fire_mask").to_zarr(out_dir / "fire_masks.zarr", mode="w", consolidated=True)

    # weather.zarr: pr, tmmn, tmmx, vs, th, rmax, rmin (time, y, x)
    weather_vars = ["pr", "tmmn", "tmmx", "vs", "th", "rmax", "rmin"]
    data_vars = {}
    for v in weather_vars:
        data_vars[v] = (["time", "y", "x"], np.random.rand(n_time, h, w).astype(np.float32))
    weather = xr.Dataset(
        data_vars,
        coords={"time": times, "y": np.arange(h), "x": np.arange(w)},
    )
    weather.to_zarr(out_dir / "weather.zarr", mode="w", consolidated=True)

    # satellite.zarr: sar_backscatter (time, y, x, band)
    sar = xr.DataArray(
        np.random.rand(n_time, h, w, 2).astype(np.float32) * 0.1,
        dims=["time", "y", "x", "band"],
        coords={"time": times, "y": np.arange(h), "x": np.arange(w), "band": ["VV", "VH"]},
    )
    sar.to_dataset(name="sar_backscatter").to_zarr(out_dir / "satellite.zarr", mode="w", consolidated=True)

    # static.zarr: elevation, slope, aspect (y, x)
    static = xr.Dataset(
        {
            "elevation": (["y", "x"], np.random.rand(h, w).astype(np.float32) * 1000),
            "slope": (["y", "x"], np.random.rand(h, w).astype(np.float32) * 30),
            "aspect": (["y", "x"], np.random.rand(h, w).astype(np.float32) * 360),
        },
        coords={"y": np.arange(h), "x": np.arange(w)},
    )
    static.to_zarr(out_dir / "static.zarr", mode="w", consolidated=True)


def main() -> int:
    print("Smoke test: creating fake fire run...")
    tmp = Path(tempfile.mkdtemp(prefix="wildfire_smoke_"))
    try:
        make_fake_fire_run(tmp, n_time=4, h=32, w=32)
        fire_dirs = [tmp]

        from src.training.dataset import WildfireSpreadDataset
        from src.models.cnn_spread import (
            build_spread_cnn,
            FAMILY_A_CHANNELS,
            FAMILY_B_CHANNELS,
        )
        import torch

        # Dataset
        for fam in ("A", "B"):
            ds = WildfireSpreadDataset(fire_dirs, family=fam, patch_size=16)
            assert len(ds) >= 1, f"Family {fam}: expected at least 1 sample"
            x, y = ds[0]
            assert x.dim() == 3 and y.dim() == 3
            c = FAMILY_A_CHANNELS if fam == "A" else FAMILY_B_CHANNELS
            assert x.shape[0] == c, f"Family {fam}: expected {c} channels, got {x.shape[0]}"
            print(f"  Dataset Family {fam}: {len(ds)} samples, sample shape x={x.shape} y={y.shape}")

        # Model forward
        for fam in ("A", "B"):
            model = build_spread_cnn(family=fam)
            c = FAMILY_A_CHANNELS if fam == "A" else FAMILY_B_CHANNELS
            x = torch.randn(2, c, 16, 16)
            out = model(x)
            assert out.shape == (2, 1, 16, 16), f"Family {fam}: expected (2,1,16,16), got {out.shape}"
            print(f"  Model Family {fam}: forward OK")

        # One training step
        from torch.utils.data import DataLoader
        train_ds = WildfireSpreadDataset(fire_dirs, family="A", patch_size=16)
        loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        model = build_spread_cnn(family="A")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = next(iter(loader))
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        print("  One training step: OK")

        print("Smoke test PASSED.")
        return 0
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
