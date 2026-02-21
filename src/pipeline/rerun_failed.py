#!/usr/bin/env python3
"""Rerun failed wildfire pipeline runs from a summary JSON report."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun failed incidents listed in summarize_runs JSON output"
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to summary JSON produced by summarize_runs.py",
    )
    parser.add_argument(
        "--include-warn",
        action="store_true",
        help="Also rerun WARN entries (default reruns only FAIL entries)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of reruns (0 = no cap)",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python executable to use (default: python3)",
    )
    parser.add_argument(
        "--manifest-csv",
        help="Optional manifest CSV (name,start,end,west,south,east,north) for runs missing run_config.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rerun commands without executing",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f"Summary JSON not found: {summary_path}")

    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Invalid summary format: expected a JSON list")

    wanted_status = {"FAIL", "WARN"} if args.include_warn else {"FAIL"}
    selected = [r for r in rows if str(r.get("status", "")).upper() in wanted_status]
    if args.max_runs > 0:
        selected = selected[: args.max_runs]

    if not selected:
        print("No matching runs to rerun.")
        return

    repo_root = Path(__file__).resolve().parents[2]
    manifest_by_name = {}
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        if not manifest_path.exists():
            raise SystemExit(f"Manifest CSV not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("name")
                if name:
                    manifest_by_name[name] = row
    out_rows = []

    print(f"Selected {len(selected)} run(s) for rerun")
    for i, row in enumerate(selected, start=1):
        run_dir = Path(str(row.get("run_dir", "")))
        run_cfg = run_dir / "run_config.json"
        if run_cfg.exists():
            cfg = json.loads(run_cfg.read_text(encoding="utf-8"))
        else:
            fire_name = run_dir.name
            m = manifest_by_name.get(fire_name)
            if not m:
                out_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "status_before": row.get("status", ""),
                        "return_code": 2,
                        "seconds": 0.0,
                        "note": "missing run_config.json and no manifest match",
                    }
                )
                print(
                    f"[{i}/{len(selected)}] SKIP {run_dir} "
                    "(missing run_config.json and no manifest match)"
                )
                continue
            cfg = {
                "fire_name": fire_name,
                "start_date": m["start"],
                "end_date": m["end"],
                "bounds": [
                    float(m["west"]),
                    float(m["south"]),
                    float(m["east"]),
                    float(m["north"]),
                ],
                "output_dir": str(run_dir),
            }
        cmd = [
            args.python_bin,
            "src/pipeline/data_pipeline.py",
            "--fire-name",
            str(cfg["fire_name"]),
            "--start-date",
            str(cfg["start_date"]),
            "--end-date",
            str(cfg["end_date"]),
            "--bounds",
            str(cfg["bounds"][0]),
            str(cfg["bounds"][1]),
            str(cfg["bounds"][2]),
            str(cfg["bounds"][3]),
            "--output-dir",
            str(cfg["output_dir"]),
        ]

        if args.dry_run:
            out_rows.append(
                {
                    "run_dir": str(run_dir),
                    "status_before": row.get("status", ""),
                    "return_code": 0,
                    "seconds": 0.0,
                    "note": "dry-run",
                }
            )
            print(f"[{i}/{len(selected)}] DRY-RUN {' '.join(cmd)}")
            continue

        print(f"[{i}/{len(selected)}] RUN {run_dir}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=repo_root)
        dt = round(time.time() - t0, 1)
        out_rows.append(
            {
                "run_dir": str(run_dir),
                "status_before": row.get("status", ""),
                "return_code": int(proc.returncode),
                "seconds": dt,
                "note": "",
            }
        )
        print(f"[{i}/{len(selected)}] DONE rc={proc.returncode} time={dt}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = summary_path.parent / f"rerun_results_{ts}.json"
    out_json.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"Wrote rerun results: {out_json}")


if __name__ == "__main__":
    main()
