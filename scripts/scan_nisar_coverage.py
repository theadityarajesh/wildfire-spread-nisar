#!/usr/bin/env python3
"""
Scan regions (from a CSV) for NISAR SAR coverage via ASF.

Reads a manifest CSV with columns: name, start, end, west, south, east, north
(optional: note). For each region, queries ASF for NISAR GSLC and GCOV scenes
in that bbox and date range, and writes an output CSV with scene_count_gslc,
scene_count_gcov, and total. Use this to find which fire regions actually have
NISAR data before running the full pipeline.

Usage:
  # Use the fallback manifest as input (same format)
  python scripts/scan_nisar_coverage.py \\
    --input configs/nisar_global_fires.csv \\
    --output configs/nisar_coverage_scan.csv

  # Or any CSV with: name, start, end, west, south, east, north
  python scripts/scan_nisar_coverage.py --input regions.csv --output coverage.csv

Requires: asf_search, EARTHDATA_TOKEN in .env or environment.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root for asf_search (used in pipeline)
_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


def search_nisar_count_asf(
    west: float, south: float, east: float, north: float,
    start_date: str, end_date: str,
    dataset_name: str, product_name: str,
) -> int:
    """Return number of NISAR scenes from ASF for the given bbox and date range."""
    try:
        import asf_search as asf
    except ImportError:
        print("asf_search not installed. pip install asf_search", file=sys.stderr)
        return -1

    token = os.getenv("EARTHDATA_TOKEN", "").strip()
    if not token:
        print("EARTHDATA_TOKEN not set. Set it in .env or environment.", file=sys.stderr)
        return -1

    wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
    try:
        results = asf.search(
            dataset=dataset_name,
            processingLevel=product_name,
            intersectsWith=wkt,
            start=start_date,
            end=end_date,
        )
        return len(results)
    except Exception as e:
        print(f"  ASF search failed ({dataset_name} {product_name}): {e}", file=sys.stderr)
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan regions for NISAR coverage via ASF"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with columns: name, start, end, west, south, east, north (optional: note)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path with added columns: scene_count_gslc, scene_count_gcov, scene_count_total",
    )
    parser.add_argument(
        "--no-gcov",
        action="store_true",
        help="Skip GCOV search (only query GSLC)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    rows = []
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    if not rows:
        print("No rows in input CSV.", file=sys.stderr)
        return 1

    # Ensure we have required columns
    required = {"name", "start", "end", "west", "south", "east", "north"}
    first = rows[0]
    missing = required - set(first.keys())
    if missing:
        print(f"Input CSV missing columns: {missing}. Need: name, start, end, west, south, east, north", file=sys.stderr)
        return 1

    try:
        import asf_search as asf
    except ImportError:
        print("asf_search not installed. pip install asf_search", file=sys.stderr)
        return 1

    out_fieldnames = list(first.keys()) + ["scene_count_gslc", "scene_count_gcov", "scene_count_total"]

    for i, row in enumerate(rows):
        name = row["name"]
        start = row["start"].strip()[:10]
        end = row["end"].strip()[:10]
        west = float(row["west"])
        south = float(row["south"])
        east = float(row["east"])
        north = float(row["north"])

        print(f"[{i+1}/{len(rows)}] {name} ({start}..{end}) ...", flush=True)

        n_gslc = search_nisar_count_asf(
            west, south, east, north, start, end,
            asf.DATASET.NISAR, asf.PRODUCT_TYPE.GSLC,
        )
        row["scene_count_gslc"] = str(n_gslc) if n_gslc >= 0 else ""

        n_gcov = 0
        if not args.no_gcov:
            n_gcov = search_nisar_count_asf(
                west, south, east, north, start, end,
                asf.DATASET.NISAR, asf.PRODUCT_TYPE.GCOV,
            )
        row["scene_count_gcov"] = str(n_gcov) if n_gcov >= 0 else ""
        total = (n_gslc if n_gslc >= 0 else 0) + (n_gcov if n_gcov >= 0 else 0)
        row["scene_count_total"] = str(total) if (n_gslc >= 0 or n_gcov >= 0) else ""

        print(f"    GSLC={row['scene_count_gslc']} GCOV={row['scene_count_gcov']} total={row['scene_count_total']}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    with_coverage = sum(1 for r in rows if int(r.get("scene_count_total", 0) or 0) > 0)
    print(f"Regions with NISAR coverage: {with_coverage}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
