#!/usr/bin/env python3
"""Compute bounds for GeoTIFFs in a directory and output unioned bbox.

Examples:
  python3 src/pipeline/get_bounds.py --input_dir /path/to/fire_tiffs
  python3 src/pipeline/get_bounds.py --input_dir /path/to/fire_tiffs --crs EPSG:4326 --out combined_bounds.json --format geojson

This writes a GeoJSON bbox polygon by default when `--format geojson` is used.
"""
import argparse
import json
import os
from glob import glob
from typing import List, Tuple

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception as e:
    raise RuntimeError("rasterio is required: pip install rasterio") from e


def list_tiffs(dirpath: str) -> List[str]:
    patterns = ["*.tif", "*.tiff"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(dirpath, p)))
    files = [
        f
        for f in sorted(files)
        if os.path.isfile(f) and not os.path.basename(f).startswith(".")
    ]
    return files


def get_file_bounds(
    fp: str, dst_crs: str = "EPSG:4326"
) -> Tuple[Tuple[float, float, float, float], str]:
    """Return (minx, miny, maxx, maxy) in `dst_crs` and the source CRS string."""
    with rasterio.open(fp) as src:
        b = src.bounds  # left, bottom, right, top
        src_crs = src.crs
        if src_crs is None:
            # assume already in dst_crs
            tb = (b.left, b.bottom, b.right, b.top)
        else:
            tb = transform_bounds(
                src_crs, dst_crs, b.left, b.bottom, b.right, b.top, densify_pts=21
            )
        return (tb[0], tb[1], tb[2], tb[3]), (
            src_crs.to_string() if src_crs else "None"
        )


def union_bounds(bounds_list: List[Tuple[float, float, float, float]]):
    if not bounds_list:
        return None
    minxs = [b[0] for b in bounds_list]
    minys = [b[1] for b in bounds_list]
    maxxs = [b[2] for b in bounds_list]
    maxys = [b[3] for b in bounds_list]
    return (min(minxs), min(minys), max(maxxs), max(maxys))


def bbox_to_geojson(bbox: Tuple[float, float, float, float], crs: str = "EPSG:4326"):
    minx, miny, maxx, maxy = bbox
    coords = [
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny],
    ]
    return {
        "type": "Feature",
        "properties": {"crs": crs},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def main():
    p = argparse.ArgumentParser(
        description="Compute combined bounds for TIFF files in a directory"
    )
    p.add_argument("--input_dir", required=True, help="Directory with GeoTIFF files")
    p.add_argument(
        "--crs",
        default="EPSG:4326",
        help="Destination CRS for bounds (default EPSG:4326)",
    )
    p.add_argument(
        "--out", default=None, help="Write output to file (json or geojson) if provided"
    )
    p.add_argument(
        "--format",
        choices=["text", "json", "geojson"],
        default="text",
        help="Output format",
    )
    p.add_argument("--per-file", action="store_true", help="Also print per-file bounds")
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    files = list_tiffs(args.input_dir)
    if not files:
        raise SystemExit(f"No TIFF files found in {args.input_dir}")

    per_bounds = []
    for f in files:
        try:
            b, src_crs = get_file_bounds(f, dst_crs=args.crs)
            per_bounds.append((f, b, src_crs))
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    bboxes = [b for (_, b, _) in per_bounds]
    union = union_bounds(bboxes)

    # Print
    if args.per_file:
        print("Per-file bounds (in {}):".format(args.crs))
        for f, b, src in per_bounds:
            print(f"{os.path.basename(f)} -> {b}  (src_crs={src})")

    if union is None:
        raise SystemExit("No valid bounds computed")

    if args.format == "text":
        print("Combined bounds (minx, miny, maxx, maxy) in {}:".format(args.crs))
        print(union)
    elif args.format == "json":
        obj = {"crs": args.crs, "bbox": list(union)}
        s = json.dumps(obj, indent=2)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(s)
            print("Wrote JSON to", args.out)
        else:
            print(s)
    elif args.format == "geojson":
        gj = bbox_to_geojson(union, crs=args.crs)
        s = json.dumps(gj, indent=2)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(s)
            print("Wrote GeoJSON to", args.out)
        else:
            print(s)


if __name__ == "__main__":
    main()
