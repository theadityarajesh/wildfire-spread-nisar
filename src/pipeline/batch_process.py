#!/usr/bin/env python3
"""
Batch-process fires from a manifest CSV through the data pipeline.

Manifest CSV format (header required):
    name,start,end,west,south,east,north

Usage:
    python src/pipeline/batch_process.py \
        --manifest-csv configs/nisar_global_fires.csv \
        --output-root processed/nisar_global
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process fires from a manifest CSV"
    )
    parser.add_argument(
        "--manifest-csv",
        required=True,
        help="Path to manifest CSV (name,start,end,west,south,east,north)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory for processed output (each fire gets a subdirectory)",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python executable (default: python3)",
    )
    parser.add_argument(
        "--max-fires",
        type=int,
        default=0,
        help="Max fires to process (0 = all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip fires whose output directory already has zarr files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.max_fires > 0:
        rows = rows[: args.max_fires]

    if not rows:
        print("No fires in manifest.")
        return

    output_root = Path(args.output_root)
    repo_root = Path(__file__).resolve().parents[2]
    results = []

    print(f"Processing {len(rows)} fires from {manifest_path}")
    print(f"Output root: {output_root}")
    print("=" * 60)

    for i, row in enumerate(rows, start=1):
        name = row["name"]
        out_dir = output_root / name

        if args.skip_existing:
            required = {"fire_masks.zarr", "weather.zarr", "satellite.zarr", "static.zarr"}
            present = (
                {p.name for p in out_dir.iterdir() if p.is_dir()}
                if out_dir.exists()
                else set()
            )
            if required.issubset(present):
                print(f"[{i}/{len(rows)}] SKIP {name} (has all 4 required zarrs)")
                results.append({"name": name, "status": "skipped", "seconds": 0})
                continue

        cmd = [
            args.python_bin,
            "src/pipeline/data_pipeline.py",
            "--fire-name", name,
            "--start-date", row["start"],
            "--end-date", row["end"],
            "--bounds", row["west"], row["south"], row["east"], row["north"],
            "--output-dir", str(out_dir),
        ]

        if args.dry_run:
            print(f"[{i}/{len(rows)}] DRY-RUN: {' '.join(cmd)}")
            results.append({"name": name, "status": "dry-run", "seconds": 0})
            continue

        print(f"[{i}/{len(rows)}] Processing {name}...")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=repo_root)
        elapsed = round(time.time() - t0, 1)
        status = "ok" if proc.returncode == 0 else f"fail(rc={proc.returncode})"
        results.append({"name": name, "status": status, "seconds": elapsed})
        print(f"[{i}/{len(rows)}] {name}: {status} ({elapsed}s)")

    # Write results summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_root / f"batch_results_{ts}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBatch results: {summary_path}")

    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"].startswith("fail"))
    skip = sum(1 for r in results if r["status"] == "skipped")
    print(f"Summary: {ok} ok, {fail} failed, {skip} skipped out of {len(results)}")


if __name__ == "__main__":
    main()
