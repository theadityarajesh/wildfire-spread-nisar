#!/usr/bin/env python3
"""
Discover global fires that overlap with available NISAR SAR coverage.

Steps:
1. Search ASF for all NISAR GSLC and GCOV scenes.
2. For each scene footprint, query NASA FIRMS for active fire detections.
3. Cluster fire detections into individual fire events.
4. Output a manifest CSV ready for the data pipeline.

If FIRMS returns no detections (e.g. missing MAP_KEY), a fallback manifest of
known fire-prone regions is written so you can run the pipeline and filter
by NISAR coverage afterward (see scripts/scan_nisar_coverage.py).

Usage (on VM with .venv-train activated):
    python src/pipeline/discover_nisar_fires.py --output-csv configs/nisar_global_fires.csv

Requires: asf_search, requests
Auth: EARTHDATA_TOKEN in env / .env (for ASF).
      For FIRMS fire-overlap: request a free MAP_KEY at
      https://firms.modaps.eosdis.nasa.gov/api/map_key and set FIRMS_MAP_KEY in .env.
"""

import argparse
import csv
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def search_nisar_scenes():
    """Search ASF for all available NISAR SAR products (GSLC and GCOV)."""
    try:
        import asf_search as asf
    except ImportError:
        logger.error("asf_search not installed. pip install asf_search")
        return []

    scenes = []
    seen = set()  # avoid duplicates when same scene appears as GSLC and GCOV

    datasets_products = [
        (asf.DATASET.NISAR, asf.PRODUCT_TYPE.GSLC),
        (asf.DATASET.NISAR, asf.PRODUCT_TYPE.GCOV),
    ]

    for ds, prod in datasets_products:
        logger.info(f"Searching ASF: dataset={ds}, product={prod}")
        try:
            results = asf.search(
                dataset=ds,
                processingLevel=prod,
                start="2025-01-01",
                end=datetime.utcnow().strftime("%Y-%m-%d"),
            )
            logger.info(f"  Found {len(results)} scenes")
            for r in results:
                props = r.properties
                scene_name = props.get("sceneName", "")
                if scene_name in seen:
                    continue
                seen.add(scene_name)
                geom = r.geometry
                scenes.append({
                    "scene_name": scene_name,
                    "start_date": props.get("startTime", ""),
                    "end_date": props.get("stopTime", ""),
                    "geometry": geom,
                    "dataset": str(ds),
                    "product": str(prod),
                })
        except Exception as e:
            logger.warning(f"  Search failed: {e}")

    return scenes


def _bbox_from_geometry(geom: dict) -> Optional[tuple]:
    """Extract (west, south, east, north) from a GeoJSON geometry."""
    coords = geom.get("coordinates", [])
    if not coords:
        return None
    if geom["type"] == "Polygon":
        ring = coords[0]
    elif geom["type"] == "MultiPolygon":
        ring = coords[0][0]
    else:
        return None
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def query_firms_fires(bbox: tuple, start_date: str, end_date: str):
    """
    Query NASA FIRMS for active fire detections within a bounding box.

    FIRMS area API requires a dedicated MAP_KEY (not Earthdata token).
    Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key
    and set FIRMS_MAP_KEY in .env. If unset, EARTHDATA_TOKEN is tried
    (may return empty for area/csv).
    """
    import requests

    west, south, east, north = bbox
    map_key = os.getenv("FIRMS_MAP_KEY", "").strip() or os.getenv("EARTHDATA_TOKEN", "").strip()

    fires = []

    try:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{map_key}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/1/"
            f"{start_date}"
        )
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            lines = resp.text.strip().split("\n")
            if len(lines) > 1:
                reader = csv.DictReader(lines)
                for row in reader:
                    try:
                        fires.append({
                            "latitude": float(row.get("latitude", 0)),
                            "longitude": float(row.get("longitude", 0)),
                            "acq_date": row.get("acq_date", ""),
                            "confidence": row.get("confidence", ""),
                            "frp": float(row.get("frp", 0)),
                        })
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        logger.warning(f"FIRMS query failed: {e}")

    if not fires:
        try:
            url = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/"
            resp = requests.get(
                f"{url}{map_key}/VIIRS_SNPP_NRT/0/1",
                timeout=60,
            )
            logger.info(f"  FIRMS fallback status: {resp.status_code}")
        except Exception:
            pass

    return fires


def _parse_acq_date(s: str) -> Optional[str]:
    """Return YYYY-MM-DD if parseable, else None."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()[:10]
    if len(s) != 10:
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        return None


def cluster_fires(detections: list, bbox: tuple, days_buffer: int = 3):
    """
    Cluster fire detections into fire events.

    Simple approach: group by date proximity and spatial proximity.
    Returns a list of fire event dicts with bounding boxes.
    Skips detections with missing or invalid acq_date.
    """
    if not detections:
        return []

    by_date = {}
    for d in detections:
        date = _parse_acq_date(d.get("acq_date", ""))
        if date is None:
            continue
        by_date.setdefault(date, []).append(d)

    dates_sorted = sorted(by_date.keys())
    if not dates_sorted:
        return []

    events = []
    current_dates = [dates_sorted[0]]
    for i in range(1, len(dates_sorted)):
        prev = datetime.strptime(dates_sorted[i - 1], "%Y-%m-%d")
        curr = datetime.strptime(dates_sorted[i], "%Y-%m-%d")
        if (curr - prev).days <= days_buffer:
            current_dates.append(dates_sorted[i])
        else:
            events.append(current_dates)
            current_dates = [dates_sorted[i]]
    events.append(current_dates)

    fire_events = []
    for event_dates in events:
        pts = []
        for d in event_dates:
            pts.extend(by_date[d])
        if not pts:
            continue
        lats = [p["latitude"] for p in pts]
        lons = [p["longitude"] for p in pts]
        fire_west = min(lons) - 0.1
        fire_south = min(lats) - 0.1
        fire_east = max(lons) + 0.1
        fire_north = max(lats) + 0.1
        start = min(event_dates)
        end = max(event_dates)
        start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1)
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=2)
        fire_events.append({
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "west": round(fire_west, 4),
            "south": round(fire_south, 4),
            "east": round(fire_east, 4),
            "north": round(fire_north, 4),
            "n_detections": len(pts),
            "n_days": len(event_dates),
        })

    return fire_events


def main():
    parser = argparse.ArgumentParser(
        description="Discover global fires with NISAR SAR coverage"
    )
    parser.add_argument(
        "--output-csv",
        default="configs/nisar_global_fires.csv",
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=5,
        help="Minimum fire detections to include (default: 5)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Max NISAR scenes to check with FIRMS (0 = all). Use e.g. 500 to test quickly.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NISAR Fire Discovery")
    logger.info("=" * 60)

    # Step 1: Find all NISAR scenes
    logger.info("\n[1/3] Searching for NISAR scenes on ASF...")
    scenes = search_nisar_scenes()

    if not scenes:
        logger.warning("No NISAR scenes found. Trying CMR fallback...")
        scenes = _cmr_fallback_scenes()

    if not scenes:
        logger.error("No NISAR scenes found anywhere. Nothing to cross-reference.")
        return 1

    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
        logger.info(f"Limited to first {len(scenes)} scenes (--max-scenes={args.max_scenes})")
    logger.info(f"\nFound {len(scenes)} NISAR scenes total")

    # Step 2: Check for fires in each scene footprint
    logger.info("\n[2/3] Querying FIRMS for fires in NISAR footprints...")
    all_fire_events = []

    for i, scene in enumerate(scenes):
        bbox = _bbox_from_geometry(scene["geometry"])
        if bbox is None:
            continue
        logger.info(
            f"  Scene {i+1}/{len(scenes)}: {scene['scene_name'][:40]}... "
            f"bbox=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})"
        )

        start = scene.get("start_date", "")[:10]
        end = scene.get("end_date", "")[:10]
        if not start or not end:
            continue

        fires = query_firms_fires(bbox, start, end)
        logger.info(f"    {len(fires)} fire detections")

        if fires:
            events = cluster_fires(fires, bbox)
            for ev in events:
                ev["nisar_scene"] = scene["scene_name"]
                ev["nisar_dataset"] = scene["dataset"]
            all_fire_events.extend(events)

    # Filter by minimum detections
    all_fire_events = [
        e for e in all_fire_events if e["n_detections"] >= args.min_detections
    ]

    logger.info(f"\n[3/3] Found {len(all_fire_events)} fire events with NISAR coverage")

    if not all_fire_events:
        logger.warning(
            "No fires found overlapping NISAR scenes. "
            "This is expected given NISAR's limited early coverage."
        )
        logger.info(
            "Alternative: manually create a manifest with known post-Oct-2025 fires "
            "in fire-prone regions and let the pipeline search for NISAR data."
        )
        _write_fallback_manifest(args.output_csv)
        return 0

    # Write manifest CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name", "start", "end", "west", "south", "east", "north",
        "n_detections", "n_days", "nisar_scene",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, ev in enumerate(all_fire_events):
            writer.writerow({
                "name": f"nisar_fire_{i+1:03d}",
                "start": ev["start_date"],
                "end": ev["end_date"],
                "west": ev["west"],
                "south": ev["south"],
                "east": ev["east"],
                "north": ev["north"],
                "n_detections": ev["n_detections"],
                "n_days": ev["n_days"],
                "nisar_scene": ev.get("nisar_scene", ""),
            })

    logger.info(f"Wrote manifest: {out_path}")
    logger.info(f"Next: process these fires with the data pipeline:")
    logger.info(f"  python src/pipeline/batch_process.py --manifest-csv {out_path} --output-root processed/nisar_global")
    return 0


def _cmr_fallback_scenes():
    """Query CMR directly for NISAR scene locations when asf_search fails."""
    try:
        import requests
    except ImportError:
        return []

    scenes = []
    collections = ["NISAR_L2_GSLC_BETA_V1", "NISAR_L2_GCOV_BETA_V1", "NISAR_L1_RSLC_BETA_V1"]

    for short_name in collections:
        try:
            resp = requests.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params={
                    "short_name": short_name,
                    "page_size": 100,
                    "temporal": "2025-01-01T00:00:00Z,",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            logger.info(f"  CMR {short_name}: {len(entries)} granules")
            for entry in entries:
                boxes = entry.get("boxes", [])
                if boxes:
                    parts = boxes[0].split()
                    if len(parts) == 4:
                        s, w, n, e = [float(x) for x in parts]
                        scenes.append({
                            "scene_name": entry.get("title", short_name),
                            "start_date": entry.get("time_start", ""),
                            "end_date": entry.get("time_end", ""),
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[w, s], [e, s], [e, n], [w, n], [w, s]]
                                ],
                            },
                            "dataset": short_name,
                            "product": "CMR",
                        })
        except Exception as e:
            logger.warning(f"  CMR query for {short_name} failed: {e}")

    return scenes


def _write_fallback_manifest(output_csv: str):
    """
    Write a manifest of known recent large fires in fire-prone regions
    where NISAR may have coverage. These are post-Oct 2025 fires
    that the pipeline can attempt to process.
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    known_fires = [
        {
            "name": "australia_nsw_dec2025",
            "start": "2025-12-01",
            "end": "2025-12-15",
            "west": 149.5, "south": -34.0, "east": 150.5, "north": -33.0,
            "note": "NSW Australia fire season",
        },
        {
            "name": "greece_attica_oct2025",
            "start": "2025-10-15",
            "end": "2025-10-25",
            "west": 23.5, "south": 37.8, "east": 24.2, "north": 38.3,
            "note": "Mediterranean fire season",
        },
        {
            "name": "chile_valparaiso_jan2026",
            "start": "2026-01-10",
            "end": "2026-01-20",
            "west": -71.8, "south": -33.5, "east": -71.0, "north": -32.8,
            "note": "South American fire season",
        },
        {
            "name": "portugal_algarve_nov2025",
            "start": "2025-11-01",
            "end": "2025-11-10",
            "west": -8.5, "south": 37.0, "east": -7.8, "north": 37.5,
            "note": "Iberian Peninsula late fire",
        },
        {
            "name": "indonesia_kalimantan_nov2025",
            "start": "2025-11-05",
            "end": "2025-11-20",
            "west": 116.0, "south": -2.5, "east": 117.5, "north": -1.0,
            "note": "Indonesian peat fire season",
        },
    ]

    fieldnames = ["name", "start", "end", "west", "south", "east", "north", "note"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fire in known_fires:
            writer.writerow(fire)

    logger.info(f"Wrote fallback manifest with {len(known_fires)} candidate fires: {out_path}")
    logger.info(
        "These are candidate fires in fire-prone regions post-NISAR launch. "
        "The pipeline will search ASF for any available NISAR data in these areas."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
