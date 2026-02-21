#!/usr/bin/env python3
"""Batch-validate wildfire runs and print a cloud-ready summary."""

import argparse
import csv
import json
import sys
from pathlib import Path

from validate_run import evaluate_run


def _discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    # Include all immediate subdirectories so incomplete runs are surfaced as FAIL.
    runs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return runs


def _status_from_codes(error_count: int, warning_count: int) -> str:
    if error_count > 0:
        return "FAIL"
    if warning_count > 0:
        return "WARN"
    return "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize many wildfire pipeline output runs")
    parser.add_argument(
        "--root-dir",
        default="processed",
        help="Directory containing per-fire run subdirectories (default: processed)",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific run directory to validate (repeatable). If set, root scan is skipped.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional CSV path for machine-readable summary",
    )
    parser.add_argument(
        "--output-json",
        help="Optional JSON path for machine-readable summary",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit nonzero if any run has warnings",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(p) for p in args.run_dir]
    else:
        run_dirs = _discover_runs(Path(args.root_dir))

    if not run_dirs:
        print("No run directories found.")
        sys.exit(1)

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        result, code = evaluate_run(run_dir)
        errors = len(result.errors)
        warnings = len(result.warnings)
        status = _status_from_codes(errors, warnings if code == 0 else warnings)
        fire_days = result.metrics.get("fire_timesteps", "")
        sat_days = result.metrics.get("satellite_timesteps", "")
        rows.append(
            {
                "run_dir": str(run_dir),
                "status": status,
                "errors": errors,
                "warnings": warnings,
                "fire_timesteps": fire_days,
                "satellite_timesteps": sat_days,
                "fire_active_fraction": result.metrics.get("fire_active_fraction", ""),
                "satellite_nonzero_fraction": result.metrics.get("satellite_nonzero_fraction", ""),
                "notes": " | ".join(result.warnings[:3] + result.errors[:3]),
            }
        )

    # Print compact terminal summary
    print("=" * 100)
    print("Run Summary")
    print("=" * 100)
    print(
        f"{'status':<8} {'run_dir':<42} {'err':>3} {'warn':>4} {'fire_t':>6} {'sat_t':>6} {'fire%':>8} {'sat%':>8}"
    )
    for row in rows:
        fire_frac = row["fire_active_fraction"]
        sat_frac = row["satellite_nonzero_fraction"]
        fire_txt = f"{float(fire_frac):.3f}" if fire_frac != "" else "-"
        sat_txt = f"{float(sat_frac):.3f}" if sat_frac != "" else "-"
        print(
            f"{row['status']:<8} {str(row['run_dir'])[:42]:<42} "
            f"{int(row['errors']):>3} {int(row['warnings']):>4} "
            f"{str(row['fire_timesteps']):>6} {str(row['satellite_timesteps']):>6} "
            f"{fire_txt:>8} {sat_txt:>8}"
        )
    print("-" * 100)

    total = len(rows)
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_warn = sum(1 for r in rows if r["status"] == "WARN")
    n_pass = total - n_fail - n_warn
    print(f"Totals: {total} runs | PASS={n_pass} WARN={n_warn} FAIL={n_fail}")
    print("=" * 100)

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {csv_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote JSON: {json_path}")

    if n_fail > 0:
        sys.exit(1)
    if args.fail_on_warnings and n_warn > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
