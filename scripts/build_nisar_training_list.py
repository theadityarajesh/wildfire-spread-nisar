#!/usr/bin/env python3
"""
Build a Model C (NISAR) training list from the summary CSV so only valid runs are used.

Valid = status is PASS or WARN (not FAIL), and optionally:
  - fire_active_fraction > 0 (has some fire signal)
  - satellite_nonzero_fraction >= min_sat (has real SAR; default 0.2 excludes placeholder-only runs)

Usage:
  python scripts/build_nisar_training_list.py \\
    --summary configs/nisar_global_summary.csv \\
    --output configs/fires_nisar_training.txt \\
    --data-root-prefix processed/nisar_global \\
    --min-sat 0.2 \\
    --require-fire
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NISAR training list from summary CSV")
    parser.add_argument("--summary", required=True, help="Path to summary CSV from summarize_runs.py")
    parser.add_argument("--output", required=True, help="Output text file (one fire name per line)")
    parser.add_argument(
        "--data-root-prefix",
        default="processed/nisar_global",
        help="Prefix to strip from run_dir to get fire name (default: processed/nisar_global)",
    )
    parser.add_argument(
        "--min-sat",
        type=float,
        default=0.2,
        help="Minimum satellite_nonzero_fraction to include (0 = allow placeholder SAR; 0.2 = real SAR only)",
    )
    parser.add_argument(
        "--require-fire",
        action="store_true",
        help="Exclude runs with fire_active_fraction = 0",
    )
    parser.add_argument(
        "--include-warn-only",
        action="store_true",
        help="Include only WARN (exclude PASS if you want to be strict)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: summary file not found: {summary_path}")
        return 1

    prefix = args.data_root_prefix.rstrip("/") + "/"
    valid = []
    skipped = []

    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("Error: empty or invalid CSV")
            return 1
        for row in reader:
            run_dir = row.get("run_dir", "").strip()
            status = (row.get("status", "") or "").strip().upper()
            if status == "FAIL":
                skipped.append((run_dir, "FAIL"))
                continue
            if args.include_warn_only and status != "WARN":
                skipped.append((run_dir, f"status={status} (not WARN)"))
                continue

            try:
                fire_frac = float(row.get("fire_active_fraction") or 0)
            except (TypeError, ValueError):
                fire_frac = 0.0
            try:
                sat_frac = float(row.get("satellite_nonzero_fraction") or 0)
            except (TypeError, ValueError):
                sat_frac = 0.0

            if args.require_fire and fire_frac <= 0:
                skipped.append((run_dir, f"fire_active_fraction=0"))
                continue
            if sat_frac < args.min_sat:
                skipped.append((run_dir, f"satellite_nonzero_fraction={sat_frac:.3f} < {args.min_sat}"))
                continue

            name = run_dir.replace(prefix, "").split("/")[0] if prefix in run_dir else Path(run_dir).name
            valid.append((name, fire_frac, sat_frac))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for name, _, _ in valid:
            f.write(name + "\n")

    print("Included (valid NISAR runs):")
    for name, fire_frac, sat_frac in valid:
        print(f"  {name}  fire_frac={fire_frac:.2e}  sat_frac={sat_frac:.3f}")
    print(f"\nSkipped {len(skipped)} run(s):")
    for run_dir, reason in skipped:
        short = run_dir.replace(prefix, "").split("/")[0] if prefix in run_dir else run_dir
        print(f"  {short}: {reason}")
    print(f"\nWrote {len(valid)} fire(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
