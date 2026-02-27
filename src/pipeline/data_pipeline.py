#!/usr/bin/env python3
"""
===============================================================================
WILDFIRE DATA PROCESSING PIPELINE
===============================================================================

This script downloads and processes all data needed for wildfire prediction:
- Fire masks from Google Earth Engine (VIIRS)
- Weather data from GridMET
- SAR satellite data from ASF (NISAR or Sentinel-1)
- Topography from OpenTopography

Each section is clearly separated and modular for easy debugging.

Author: Aditya Rajesh
Date: January 2026
===============================================================================
"""

import os
import re
import json
import shutil
import sys
import zipfile
import argparse
import logging
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds

# Check for optional libraries
LIBRARIES_AVAILABLE = {
    "ee": False,
    "geemap": False,
    "pygridmet": False,
    "asf_search": False,
    "bmi_topography": False,
}

try:
    import ee

    LIBRARIES_AVAILABLE["ee"] = True
except ImportError:
    pass

try:
    import geemap

    LIBRARIES_AVAILABLE["geemap"] = True
except ImportError:
    pass

try:
    import pygridmet

    LIBRARIES_AVAILABLE["pygridmet"] = True
except ImportError:
    pass

try:
    import asf_search as asf

    LIBRARIES_AVAILABLE["asf_search"] = True
except ImportError:
    pass

try:
    import requests

    LIBRARIES_AVAILABLE["requests"] = True
except ImportError:
    pass

# Optional: load credentials from .env (copy .env.example to .env and fill in)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


def _auto_utm_epsg(lon: float, lat: float) -> str:
    """Return the EPSG code for the UTM zone that contains (lon, lat)."""
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone_number:02d}"
    return f"EPSG:327{zone_number:02d}"


class Config:
    """Configuration constants for the pipeline."""

    # Default spatial reference system (UTM Zone 10N for California).
    # When TARGET_CRS is "auto", the pipeline will pick the correct UTM zone
    # based on the centroid of the fire bounding box.
    TARGET_CRS = "auto"

    # Grid resolution in meters
    TARGET_RESOLUTION = 500

    # Buffer around fire in meters (to capture spread areas)
    BUFFER_DISTANCE = 5000

    # NISAR availability date (when it started producing science data)
    NISAR_START_DATE = datetime(2025, 10, 1)

    # Convert SAR backscatter/intensity to dB for model-friendly scaling.
    SAR_TO_DB = True
    SAR_DB_EPS = 1e-8
    # Aggregate multiple same-day SAR acquisitions into one daily slice.
    # Options: "mean", "median", "max", "none"
    SAR_DAILY_AGGREGATION = "mean"
    # Retry policy for transient GridMET service failures.
    WEATHER_MAX_RETRIES = 3
    WEATHER_RETRY_BASE_SECONDS = 5.0
    WEATHER_RETRY_JITTER_SECONDS = 2.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def check_library(library_name: str, purpose: str) -> bool:
    """
    Check if a library is available and log appropriate message.

    Args:
        library_name: Name of the library (e.g., 'ee', 'pygridmet')
        purpose: What the library is used for

    Returns:
        True if available, False otherwise
    """
    if LIBRARIES_AVAILABLE.get(library_name, False):
        logger.info(f"✓ {library_name} available for {purpose}")
        return True
    else:
        logger.warning(f"✗ {library_name} not available for {purpose}")
        logger.warning(f"  Install with: pip install {library_name}")
        return False


def create_date_list(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Create a list of dates between start and end (inclusive).

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of datetime objects
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def resize_nearest_2d(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize 2D array to target shape using nearest-neighbor indexing."""
    src_h, src_w = arr.shape
    dst_h, dst_w = target_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return arr
    y_idx = np.linspace(0, src_h - 1, dst_h).round().astype(int)
    x_idx = np.linspace(0, src_w - 1, dst_w).round().astype(int)
    return arr[np.ix_(y_idx, x_idx)]


def calculate_grid_parameters(
    bounds_wgs84: Tuple[float, float, float, float],
    target_crs: str,
    resolution: int,
    buffer: int,
) -> Tuple[Tuple[float, float, float, float], Tuple[int, int], object]:
    """
    Calculate grid parameters for processing.

    Args:
        bounds_wgs84: (west, south, east, north) in WGS84
        target_crs: Target coordinate reference system
        resolution: Grid resolution in meters
        buffer: Buffer distance in meters

    Returns:
        Tuple of (buffered_bounds, grid_shape, grid_transform)
    """
    west, south, east, north = bounds_wgs84

    # Transform bounds to target CRS
    bounds_target = transform_bounds("EPSG:4326", target_crs, west, south, east, north)

    # Apply buffer
    minx, miny, maxx, maxy = bounds_target
    buffered_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

    # Calculate grid dimensions
    width = int((buffered_bounds[2] - buffered_bounds[0]) / resolution)
    height = int((buffered_bounds[3] - buffered_bounds[1]) / resolution)

    grid_shape = (height, width)
    grid_transform = from_bounds(*buffered_bounds, width, height)

    return buffered_bounds, grid_shape, grid_transform


# ============================================================================
# FIRE MASK PROCESSING (Google Earth Engine)
# ============================================================================


def download_fire_masks_gee(
    bounds_wgs84: Tuple[float, float, float, float],
    date_list: List[datetime],
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download daily fire masks from Google Earth Engine VIIRS data.

    This function uses Google Earth Engine to download VIIRS active fire
    detections for each day in the date range.

    Args:
        bounds_wgs84: Bounding box in WGS84 (west, south, east, north)
        date_list: List of dates to process
        grid_shape: Target grid shape (height, width)
        grid_transform: Rasterio transform for target grid
        target_crs: Target coordinate reference system
        output_path: Path to save fire_masks.zarr

    Returns:
        True if successful, False otherwise

    """
    logger.info("Processing fire masks from Google Earth Engine...")

    # Check if libraries are available
    if not check_library("ee", "Google Earth Engine"):
        logger.error("Cannot process fire masks without Earth Engine")
        return False

    if not check_library("geemap", "direct GEE downloads"):
        logger.warning("geemap not available - using placeholder")
        logger.warning("Install geemap for actual downloads: pip install geemap")

    try:
        # Initialize Earth Engine. Newer setups may require explicit project.
        ee_project = os.getenv("EARTHENGINE_PROJECT") or os.getenv("EE_PROJECT")
        if ee_project:
            ee.Initialize(project=ee_project)
            logger.info(f"✓ Earth Engine initialized (project={ee_project})")
        else:
            ee.Initialize()
            logger.info("✓ Earth Engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        logger.error("Run: earthengine authenticate")
        logger.error(
            "If you see 'no project found', set one with: earthengine set_project YOUR_GCP_PROJECT_ID"
        )
        logger.error(
            "Or export EARTHENGINE_PROJECT=YOUR_GCP_PROJECT_ID before running the pipeline"
        )
        return False

    # Create region of interest
    west, south, east, north = bounds_wgs84
    roi = ee.Geometry.Rectangle([west, south, east, north])

    # Load VIIRS fire dataset
    # Dataset: FIRMS (Fire Information for Resource Management System)
    viirs = ee.ImageCollection("FIRMS").filterBounds(roi)

    fire_masks = []
    placeholder_days = 0

    for date in date_list:
        date_str = date.strftime("%Y-%m-%d")
        next_date_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"  Processing {date_str}...")

        # Filter to single day
        daily_fires = viirs.filterDate(date_str, next_date_str)

        # Create binary mask (fire detected = 1, no fire = 0)
        fire_mask = daily_fires.select("T21").mosaic().gt(0).unmask(0)

        if LIBRARIES_AVAILABLE["geemap"]:
            import geemap

            fire_array = geemap.ee_to_numpy(fire_mask, region=roi, scale=500)
            # ee_to_numpy returns (height, width) or (height, width, bands); ensure 2D
            if fire_array.ndim > 2:
                fire_array = np.squeeze(fire_array)
            if fire_array.ndim != 2:
                logger.warning(
                    f"  Unexpected shape {fire_array.shape} for {date_str}, using placeholder"
                )
                fire_array = np.zeros(grid_shape, dtype=np.uint8)
                placeholder_days += 1
            elif fire_array.shape != grid_shape:
                logger.warning(
                    f"  GEE array shape {fire_array.shape} != grid {grid_shape} for {date_str}; resizing to target grid"
                )
                fire_array = resize_nearest_2d(fire_array.astype(np.uint8), grid_shape)
            fire_masks.append(fire_array)
        else:
            fire_masks.append(np.zeros(grid_shape, dtype=np.uint8))
            placeholder_days += 1

    if not fire_masks:
        logger.error("No fire mask data collected")
        return False

    # Stack into xarray DataArray
    fire_da = xr.DataArray(
        np.stack(fire_masks),
        dims=["time", "y", "x"],
        coords={
            "time": [d.strftime("%Y-%m-%d") for d in date_list],
            "y": np.arange(grid_shape[0]),
            "x": np.arange(grid_shape[1]),
        },
    )

    # Add spatial reference
    fire_da.rio.write_crs(target_crs, inplace=True)
    fire_da.rio.write_transform(grid_transform, inplace=True)

    # Save to zarr
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fire_da.to_dataset(name="fire_mask").to_zarr(
        output_path, mode="w", consolidated=True
    )

    logger.info(f"✓ Fire masks saved to {output_path}")
    logger.info(f"  Shape: {fire_da.shape} (time, y, x)")
    logger.info(
        f"  Source: GEE FIRMS/VIIRS (placeholder days: {placeholder_days}/{len(date_list)})"
    )

    return True


# ============================================================================
# WEATHER DATA PROCESSING (GridMET)
# ============================================================================


def download_weather_gridmet(
    bounds_wgs84: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download weather data from GridMET.

    GridMET provides daily meteorological data at 4km resolution for the
    contiguous United States.

    Args:
        bounds_wgs84: Bounding box in WGS84
        start_date: Start date
        end_date: End date
        grid_shape: Target grid shape
        grid_transform: Rasterio transform
        target_crs: Target CRS
        output_path: Path to save weather.zarr

    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading weather data from GridMET...")

    if not check_library("pygridmet", "weather data"):
        return False

    # Create bounding box geometry
    from shapely.geometry import box

    west, south, east, north = bounds_wgs84
    geometry = box(west, south, east, north)

    # Variables to download
    variables = [
        "pr",  # Precipitation (mm)
        "tmmn",  # Minimum temperature (K)
        "tmmx",  # Maximum temperature (K)
        "vs",  # Wind speed (m/s)
        "th",  # Wind direction (degrees)
        "rmax",  # Maximum relative humidity (%)
        "rmin",  # Minimum relative humidity (%)
    ]

    try:
        # Format dates for API
        dates = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        logger.info(f"  Requesting {len(variables)} variables...")
        logger.info(f"  Date range: {dates[0]} to {dates[1]}")

        # Download data with retry/backoff for transient service-side failures.
        weather_data = None
        last_error = None
        for attempt in range(1, Config.WEATHER_MAX_RETRIES + 1):
            try:
                weather_data = pygridmet.get_bygeom(
                    geometry,
                    dates,
                    variables=variables,
                    snow=False,  # Don't need snow parameters
                )
                break
            except Exception as e:
                last_error = e
                if attempt >= Config.WEATHER_MAX_RETRIES:
                    raise
                sleep_s = (
                    Config.WEATHER_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    + random.uniform(0.0, Config.WEATHER_RETRY_JITTER_SECONDS)
                )
                logger.warning(
                    f"  GridMET attempt {attempt}/{Config.WEATHER_MAX_RETRIES} failed: {e}"
                )
                logger.warning(f"  Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)

        if weather_data is None:
            raise RuntimeError(f"GridMET download failed: {last_error}")

        logger.info(f"  Downloaded {len(weather_data.data_vars)} variables")

        # PyGridMET returns xarray with dims (time, lat, lon). Set spatial dims for rioxarray.
        if "lat" in weather_data.dims and "lon" in weather_data.dims:
            weather_data = weather_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        weather_data = weather_data.rio.write_crs("EPSG:4326")

        # Reproject to target CRS and resample to target grid
        weather_data = weather_data.rio.reproject(
            target_crs,
            shape=grid_shape,
            transform=grid_transform,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # Save to zarr
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weather_data.to_zarr(output_path, mode="w", consolidated=True)

        logger.info(f"✓ Weather data saved to {output_path}")
        logger.info(f"  Variables: {list(weather_data.data_vars)}")
        logger.info(f"  Shape: {dict(weather_data.dims)}")

        return True

    except Exception as e:
        logger.error(f"Failed to download weather data: {e}")
        logger.exception("Full traceback:")
        return False


def _is_conus(bounds_wgs84: Tuple[float, float, float, float]) -> bool:
    """Check if a WGS84 bounding box lies within the contiguous US."""
    west, south, east, north = bounds_wgs84
    return west >= -125 and east <= -66 and south >= 24 and north <= 50


def download_weather_openmeteo(
    bounds_wgs84: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download weather data from Open-Meteo (global, free, no API key).

    Queries the Open-Meteo Historical Weather API at the bounding-box centroid
    and broadcasts the daily timeseries uniformly across the target grid.
    This is acceptable because weather variables change slowly over the spatial
    extent of a single fire (~10-50 km), while the model mainly needs the
    temporal signal.

    Returns True if successful, False otherwise.
    """
    logger.info("Downloading weather data from Open-Meteo (global fallback)...")

    if not check_library("requests", "HTTP requests"):
        return False

    import requests as _requests

    west, south, east, north = bounds_wgs84
    lat = (south + north) / 2
    lon = (west + east) / 2

    daily_vars = [
        "precipitation_sum",
        "temperature_2m_min",
        "temperature_2m_max",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
    ]
    hourly_vars = ["relative_humidity_2m"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join(daily_vars),
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
    }

    try:
        resp = _requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        hourly = data.get("hourly", {})

        times = [
            np.datetime64(d) for d in daily.get("time", [])
        ]
        n_times = len(times)
        if n_times == 0:
            logger.error("Open-Meteo returned no daily data")
            return False

        height, width = grid_shape

        def _broadcast(values: list) -> np.ndarray:
            arr = np.array(values, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            return np.broadcast_to(arr[:, None, None], (n_times, height, width)).copy()

        pr = _broadcast(daily.get("precipitation_sum", [0.0] * n_times))

        tmmn_c = daily.get("temperature_2m_min", [0.0] * n_times)
        tmmn = _broadcast([t + 273.15 for t in tmmn_c])

        tmmx_c = daily.get("temperature_2m_max", [0.0] * n_times)
        tmmx = _broadcast([t + 273.15 for t in tmmx_c])

        vs = _broadcast(daily.get("wind_speed_10m_max", [0.0] * n_times))
        th = _broadcast(daily.get("wind_direction_10m_dominant", [0.0] * n_times))

        # Compute daily rmax / rmin from hourly relative_humidity_2m.
        hourly_rh = hourly.get("relative_humidity_2m", [])
        hourly_times = hourly.get("time", [])
        if hourly_rh and hourly_times:
            rh_by_day: dict[str, list[float]] = {}
            for t_str, rh_val in zip(hourly_times, hourly_rh):
                day = t_str[:10]
                if rh_val is not None:
                    rh_by_day.setdefault(day, []).append(float(rh_val))
            rmax_list = []
            rmin_list = []
            for d in daily.get("time", []):
                vals = rh_by_day.get(d, [50.0])
                rmax_list.append(max(vals))
                rmin_list.append(min(vals))
        else:
            rmax_list = [50.0] * n_times
            rmin_list = [30.0] * n_times
        rmax = _broadcast(rmax_list)
        rmin = _broadcast(rmin_list)

        coords = {
            "time": np.array(times),
            "y": np.arange(height),
            "x": np.arange(width),
        }

        ds = xr.Dataset(
            {
                "pr": (["time", "y", "x"], pr),
                "tmmn": (["time", "y", "x"], tmmn),
                "tmmx": (["time", "y", "x"], tmmx),
                "vs": (["time", "y", "x"], vs),
                "th": (["time", "y", "x"], th),
                "rmax": (["time", "y", "x"], rmax),
                "rmin": (["time", "y", "x"], rmin),
            },
            coords=coords,
        )
        ds = ds.rio.write_crs(target_crs)
        ds = ds.rio.write_transform(grid_transform)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(output_path, mode="w", consolidated=True)

        logger.info(f"✓ Weather data saved to {output_path} (Open-Meteo)")
        logger.info(f"  Variables: {list(ds.data_vars)}")
        logger.info(f"  Shape: {dict(ds.dims)}")
        return True

    except Exception as e:
        logger.error(f"Failed to download weather from Open-Meteo: {e}")
        logger.exception("Full traceback:")
        return False


def download_weather(
    bounds_wgs84: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download weather data, choosing the best available source.

    Uses GridMET for fires within the contiguous US (higher resolution),
    falls back to Open-Meteo for fires anywhere else in the world.
    """
    if _is_conus(bounds_wgs84) and LIBRARIES_AVAILABLE.get("pygridmet", False):
        logger.info("Fire is within CONUS - using GridMET")
        ok = download_weather_gridmet(
            bounds_wgs84, start_date, end_date,
            grid_shape, grid_transform, target_crs, output_path,
        )
        if ok:
            return True
        logger.warning("GridMET failed, falling back to Open-Meteo...")

    return download_weather_openmeteo(
        bounds_wgs84, start_date, end_date,
        grid_shape, grid_transform, target_crs, output_path,
    )


# ============================================================================
# SATELLITE DATA PROCESSING (ASF - NISAR/Sentinel-1)
# ============================================================================


def should_use_nisar(start_date: datetime) -> bool:
    """
    Determine if NISAR data should be used based on date.

    NISAR became operational in November 2025. For earlier dates,
    use Sentinel-1 instead.

    Args:
        start_date: Start date of fire event

    Returns:
        True if NISAR should be used, False for Sentinel-1
    """
    return start_date >= Config.NISAR_START_DATE


def _extract_scene_date(name: str, fallback: datetime) -> str:
    """Extract YYYY-MM-DD from SAR filename; fallback to provided date."""
    # Match YYYY-MM-DD
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    # Match YYYYMMDDT...
    m = re.search(r"(20\d{2})(\d{2})(\d{2})T\d{6}", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return fallback.strftime("%Y-%m-%d")


def _find_raster_files(temp_dir: Path) -> List[Path]:
    """Find raster files directly or inside downloaded zip archives."""
    raster_files = list(temp_dir.rglob("*.tif")) + list(temp_dir.rglob("*.tiff"))
    if raster_files:
        return raster_files

    zip_files = list(temp_dir.rglob("*.zip"))
    if not zip_files:
        return []

    extract_dir = temp_dir / "unzipped"
    extract_dir.mkdir(exist_ok=True)
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(extract_dir / zf.stem)
        except zipfile.BadZipFile:
            logger.warning(f"  Could not unzip {zf.name}; skipping")

    return list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff"))


def download_sar_data_asf(
    bounds_wgs84: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download SAR satellite data from Alaska Satellite Facility (ASF).

    This function downloads ANALYSIS-READY RTC (Radiometric Terrain Corrected)
    products, which are already processed and ready to use.

    NISAR is used for fires from November 2025 onwards.
    Sentinel-1 is used for earlier fires.

    Authentication: set EARTHDATA_TOKEN in environment (NASA Earthdata bearer token).

    Args:
        bounds_wgs84: Bounding box in WGS84
        start_date: Start date
        end_date: End date
        grid_shape: Target grid shape
        grid_transform: Rasterio transform
        target_crs: Target CRS
        output_path: Path to save satellite.zarr

    Returns:
        True if successful, False otherwise

    NOTE: This downloads RTC products which are ALREADY PROCESSED.
    No additional SAR processing (calibration, terrain correction) needed!
    """
    logger.info("Downloading SAR satellite data from ASF...")

    if not check_library("asf_search", "SAR data"):
        return False

    token = os.getenv("EARTHDATA_TOKEN")
    if not token or not token.strip():
        logger.error("EARTHDATA_TOKEN not set. Set it in .env or environment.")
        logger.error("Generate a token at: https://urs.earthdata.nasa.gov/profile")
        return False

    session = asf.ASFSession().auth_with_token(token.strip())

    # Create WKT polygon from bounds (ASF requires closed polygon: first vertex = last)
    west, south, east, north = bounds_wgs84
    wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"

    temp_dir = output_path.parent / "temp_sar"

    try:
        # Inclusive day handling for ASF end date.
        exact_start = start_date
        exact_end = end_date + timedelta(days=1)
        widened_start = start_date - timedelta(days=6)
        widened_end = end_date + timedelta(days=7)

        # Search strategy tuned for training utility:
        # 1) Prefer Sentinel-1 GRD for broad historical coverage and raster compatibility
        # 2) Widen time window if exact daily range has no acquisition
        # 3) Fallback to SLC, then OPERA RTC (often HDF5 containers)
        if should_use_nisar(start_date):
            logger.info("  Using NISAR-era search strategy")
            query_plan = [
                (asf.DATASET.NISAR, asf.PRODUCT_TYPE.GSLC, exact_start, exact_end),
                (asf.DATASET.NISAR, asf.PRODUCT_TYPE.GSLC, widened_start, widened_end),
            ]
        else:
            logger.info("  Using Sentinel-era search strategy with fallbacks")
            query_plan = [
                (asf.DATASET.SENTINEL1, asf.PRODUCT_TYPE.GRD_HD, exact_start, exact_end),
                (asf.DATASET.SENTINEL1, asf.PRODUCT_TYPE.GRD_HD, widened_start, widened_end),
                (asf.DATASET.SENTINEL1, asf.PRODUCT_TYPE.SLC, widened_start, widened_end),
                (asf.DATASET.OPERA_S1, asf.PRODUCT_TYPE.RTC, exact_start, exact_end),
                (asf.DATASET.OPERA_S1, asf.PRODUCT_TYPE.RTC, widened_start, widened_end),
            ]

        results = None
        dataset = None
        product_type = None
        chosen_window = None
        for ds_name, prod_name, q_start, q_end in query_plan:
            logger.info(
                f"  Searching ASF: dataset={ds_name}, product={prod_name}, window={q_start.date()}..{(q_end - timedelta(days=1)).date()}"
            )
            candidate = asf.search(
                dataset=ds_name,
                processingLevel=prod_name,
                intersectsWith=wkt,
                start=q_start.strftime("%Y-%m-%d"),
                end=q_end.strftime("%Y-%m-%d"),
            )
            logger.info(f"    Found {len(candidate)} scenes")
            if len(candidate) > 0:
                results = candidate
                dataset = ds_name
                product_type = prod_name
                chosen_window = (q_start, q_end)
                break

        if results is None:
            results = []

        if len(results) == 0:
            logger.warning("  No SAR data found for this region/time")
            logger.info(
                "  Source: placeholder (all search fallbacks returned 0 scenes)"
            )
            logger.info("  Creating placeholder satellite.zarr...")
            create_placeholder_sar(
                grid_shape, grid_transform, target_crs, output_path, start_date
            )
            return True

        logger.info(
            f"  Using SAR source dataset={dataset}, product={product_type}, scenes={len(results)}"
        )
        if chosen_window is not None and (
            chosen_window[0] != exact_start or chosen_window[1] != exact_end
        ):
            logger.info(
                f"  Note: used widened time window {chosen_window[0].date()}..{(chosen_window[1] - timedelta(days=1)).date()} for scene availability"
            )

        # Session already created above (token or username/password)

        # Download to temporary directory
        temp_dir.mkdir(exist_ok=True)

        logger.info(f"  Downloading {len(results)} scenes...")
        logger.info(f"  This may take a while (scenes are ~500MB each)")

        # Download files. Use serial mode for stability on local macOS/Python
        # environments where ASF multiprocessing can terminate unexpectedly.
        results.download(path=str(temp_dir), session=session, processes=1)

        logger.info("  Download complete, processing SAR rasters...")

        # ================================================================
        # Process downloaded RTC GeoTIFFs
        # RTC products are ALREADY PROCESSED - just need to:
        # 1. Open the GeoTIFFs
        # 2. Reproject to target grid
        # 3. Stack into zarr
        # ================================================================

        sar_by_time = {}
        detected_bands = set()

        # Find downloaded GeoTIFFs directly or inside zip archives.
        rtc_files = sorted(_find_raster_files(temp_dir))

        if not rtc_files:
            logger.warning("  No GeoTIFF files found in download payload")
            create_placeholder_sar(
                grid_shape, grid_transform, target_crs, output_path, start_date
            )
            return True

        for rtc_file in rtc_files:
            logger.info(f"    Processing {rtc_file.name}...")

            # Open RTC GeoTIFF
            sar_data = rioxarray.open_rasterio(rtc_file, masked=True)

            # Reproject to target grid
            sar_data = sar_data.rio.reproject(
                target_crs,
                shape=grid_shape,
                transform=grid_transform,
                resampling=rasterio.enums.Resampling.bilinear,
            )

            # Parse acquisition key + polarization for proper (time, band) stacking.
            name_low = rtc_file.name.lower()
            band_match = re.search(r"[-_](vv|vh|hh|hv)[-_]", name_low)
            band_label = band_match.group(1).upper() if band_match else "B0"
            detected_bands.add(band_label)

            ts_match = re.search(r"(20\d{6}t\d{6})", name_low)
            if ts_match:
                acquisition_key = ts_match.group(1)
                date_str = f"{acquisition_key[0:4]}-{acquisition_key[4:6]}-{acquisition_key[6:8]}"
            else:
                date_str = _extract_scene_date(rtc_file.name, start_date)
                acquisition_key = date_str

            # For single-band files, extract first band.
            arr = sar_data.values
            if arr.ndim == 3:  # (band, y, x)
                arr = arr[0]
            arr = arr.astype(np.float32)
            if Config.SAR_TO_DB:
                # Sentinel/RTC rasters are typically positive-valued. Convert to dB.
                arr = np.where(arr > 0, 10.0 * np.log10(arr + Config.SAR_DB_EPS), np.nan)

            if acquisition_key not in sar_by_time:
                sar_by_time[acquisition_key] = {"date": date_str, "bands": {}}
            sar_by_time[acquisition_key]["bands"][band_label] = arr

        if not sar_by_time:
            logger.warning("  No raster data could be parsed from SAR files")
            create_placeholder_sar(
                grid_shape, grid_transform, target_crs, output_path, start_date
            )
            return True

        band_names = sorted(detected_bands) if detected_bands else ["B0"]

        agg_method = str(Config.SAR_DAILY_AGGREGATION).lower().strip()
        if agg_method not in {"none", "mean", "median", "max"}:
            logger.warning(
                f"  Unknown SAR_DAILY_AGGREGATION='{Config.SAR_DAILY_AGGREGATION}', defaulting to 'mean'"
            )
            agg_method = "mean"

        if agg_method != "none":
            by_date = {}
            for entry in sar_by_time.values():
                date_str = entry["date"]
                if date_str not in by_date:
                    by_date[date_str] = {"date": date_str, "bands": {}}
                for b_name, arr in entry["bands"].items():
                    by_date[date_str]["bands"].setdefault(b_name, []).append(arr)

            aggregated = {}
            for date_str, date_entry in by_date.items():
                aggregated[date_str] = {"date": date_str, "bands": {}}
                for b_name, arr_list in date_entry["bands"].items():
                    if len(arr_list) == 1:
                        aggregated_arr = arr_list[0]
                    else:
                        stack = np.stack(arr_list, axis=0)
                        if agg_method == "mean":
                            aggregated_arr = np.nanmean(stack, axis=0)
                        elif agg_method == "median":
                            aggregated_arr = np.nanmedian(stack, axis=0)
                        else:  # "max"
                            aggregated_arr = np.nanmax(stack, axis=0)
                    aggregated[date_str]["bands"][b_name] = aggregated_arr.astype(
                        np.float32
                    )

            logger.info(
                f"  Applied SAR daily aggregation ({agg_method}): "
                f"{len(sar_by_time)} acquisitions -> {len(aggregated)} daily timesteps"
            )
            sar_by_time = aggregated

        time_keys = sorted(sar_by_time.keys())
        sar_stacked = np.zeros(
            (len(time_keys), grid_shape[0], grid_shape[1], len(band_names)),
            dtype=np.float32,
        )
        dates = []

        for t_idx, key in enumerate(time_keys):
            entry = sar_by_time[key]
            dates.append(entry["date"])
            for b_idx, b_name in enumerate(band_names):
                if b_name in entry["bands"]:
                    sar_stacked[t_idx, :, :, b_idx] = entry["bands"][b_name]

        sar_da = xr.DataArray(
            sar_stacked,
            dims=["time", "y", "x", "band"],
            coords={
                "time": dates,
                "y": np.arange(grid_shape[0]),
                "x": np.arange(grid_shape[1]),
                "band": band_names,
            },
        )
        sar_da.attrs["units"] = "dB" if Config.SAR_TO_DB else "linear"
        sar_da.rio.write_crs(target_crs, inplace=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sar_da.to_dataset(name="sar_backscatter").to_zarr(
            output_path, mode="w", consolidated=True
        )

        logger.info(f"✓ SAR data saved to {output_path}")
        logger.info(f"  Shape: {sar_da.shape} (time, y, x, band)")
        logger.info(
            f"  Source: ASF real data (dataset={dataset}, product={product_type}, scenes={len(results)})"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to process SAR data: {e}")
        logger.exception("Full traceback:")
        return False

    finally:
        # Clean up downloaded SAR temp files
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("  Cleaned up temp_sar directory")
            except OSError as e:
                logger.warning(f"  Could not remove temp_sar: {e}")


def create_placeholder_sar(
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
    date: datetime,
):
    """Create placeholder SAR data when no real data is available."""
    logger.info("  Creating placeholder SAR data...")

    placeholder = xr.DataArray(
        np.zeros((1, grid_shape[0], grid_shape[1], 2)),
        dims=["time", "y", "x", "band"],
        coords={
            "time": [date.strftime("%Y-%m-%d")],
            "y": np.arange(grid_shape[0]),
            "x": np.arange(grid_shape[1]),
            "band": ["VV", "VH"],
        },
    )
    placeholder.attrs["units"] = "dB" if Config.SAR_TO_DB else "linear"
    placeholder.rio.write_crs(target_crs, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder.to_dataset(name="sar_backscatter").to_zarr(
        output_path, mode="w", consolidated=True
    )


# ============================================================================
# STATIC TOPOGRAPHY PROCESSING (OpenTopography)
# ============================================================================


def download_topography(
    bounds_wgs84: Tuple[float, float, float, float],
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
    api_key: Optional[str] = None,
) -> bool:
    """
    Download topography data from OpenTopography.

    Downloads DEM (Digital Elevation Model) and calculates slope and aspect.

    Args:
        bounds_wgs84: Bounding box in WGS84
        grid_shape: Target grid shape
        grid_transform: Rasterio transform
        target_crs: Target CRS
        output_path: Path to save static.zarr
        api_key: OpenTopography API key

    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading topography from OpenTopography...")

    # Check API key
    api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")

    if not api_key:
        logger.error("OpenTopography API key not provided")
        logger.error("Get free key at: https://portal.opentopography.org/requestService")
        logger.error("Set OPENTOPOGRAPHY_API_KEY env var or pass as argument")
        return False

    west, south, east, north = bounds_wgs84

    try:
        # Use HTTP request to OpenTopography API
        if not check_library("requests", "HTTP requests"):
            return False

        import requests

        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",  # 30m SRTM DEM
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }

        logger.info("  Requesting DEM from OpenTopography...")
        response = requests.get(url, params=params, stream=True, timeout=300)
        response.raise_for_status()

        # Save to temporary file
        temp_dem = output_path.parent / "temp_dem.tif"
        temp_dem.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_dem, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("  DEM downloaded, processing...")

        # Open DEM
        dem = rioxarray.open_rasterio(temp_dem, masked=True)

        # Reproject to target grid
        dem = dem.rio.reproject(
            target_crs,
            shape=grid_shape,
            transform=grid_transform,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # Get elevation values
        elevation = dem.values[0]  # Remove band dimension

        # Calculate slope and aspect
        logger.info("  Calculating slope and aspect...")

        # Calculate gradients (rise over run)
        dy, dx = np.gradient(elevation, Config.TARGET_RESOLUTION)

        # Calculate slope (in degrees)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Calculate aspect (in degrees, 0=North, clockwise)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = (aspect + 360) % 360

        # Create dataset
        static_ds = xr.Dataset(
            {
                "elevation": (["y", "x"], elevation),
                "slope": (["y", "x"], slope),
                "aspect": (["y", "x"], aspect),
            },
            coords={"y": np.arange(grid_shape[0]), "x": np.arange(grid_shape[1])},
        )

        # Add spatial reference
        static_ds.rio.write_crs(target_crs, inplace=True)
        static_ds.rio.write_transform(grid_transform, inplace=True)

        # Save to zarr
        output_path.parent.mkdir(parents=True, exist_ok=True)
        static_ds.to_zarr(output_path, mode="w", consolidated=True)

        logger.info(f"✓ Static topography saved to {output_path}")
        logger.info(f"  Layers: elevation, slope, aspect")
        logger.info(f"  Shape: {static_ds.dims}")

        # Clean up temp file
        if temp_dem.exists():
            temp_dem.unlink()

        return True

    except Exception as e:
        logger.error(f"Failed to process topography: {e}")
        logger.exception("Full traceback:")
        return False


# ============================================================================
# MAIN PROCESSING CLASS
# ============================================================================


class WildfireDataProcessor:
    """
    Main coordinator for wildfire data processing.

    This class orchestrates the entire pipeline, calling individual
    processing functions in sequence.
    """

    def __init__(
        self,
        fire_name: str,
        start_date: str,
        end_date: str,
        bounds: Tuple[float, float, float, float],
        output_dir: str,
        opentopo_api_key: Optional[str] = None,
    ):
        """
        Initialize the processor.

        Args:
            fire_name: Name of fire event
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bounds: (west, south, east, north) in WGS84
            output_dir: Directory to save processed data
            opentopo_api_key: OpenTopography API key (or set OPENTOPOGRAPHY_API_KEY)
        """
        self.fire_name = fire_name
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.bounds_wgs84 = bounds
        self.output_dir = Path(output_dir)
        self.opentopo_api_key = opentopo_api_key

        # Resolve CRS: "auto" -> pick UTM zone from bounding-box centroid.
        if Config.TARGET_CRS.lower() == "auto":
            centroid_lon = (bounds[0] + bounds[2]) / 2
            centroid_lat = (bounds[1] + bounds[3]) / 2
            self.target_crs = _auto_utm_epsg(centroid_lon, centroid_lat)
            logger.info(f"Auto-detected CRS: {self.target_crs}")
        else:
            self.target_crs = Config.TARGET_CRS

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Persist run configuration for reproducibility and rerun tooling.
        try:
            run_config = {
                "fire_name": fire_name,
                "start_date": start_date,
                "end_date": end_date,
                "bounds": list(bounds),
                "output_dir": str(self.output_dir),
                "target_crs": self.target_crs,
                "target_resolution": Config.TARGET_RESOLUTION,
            }
            with open(self.output_dir / "run_config.json", "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=2)
        except OSError as e:
            logger.warning(f"Could not write run_config.json: {e}")

        # Calculate grid parameters
        self.bounds_target, self.grid_shape, self.grid_transform = (
            calculate_grid_parameters(
                bounds,
                self.target_crs,
                Config.TARGET_RESOLUTION,
                Config.BUFFER_DISTANCE,
            )
        )

        # Create date list
        self.date_list = create_date_list(self.start_date, self.end_date)

        logger.info("=" * 80)
        logger.info(f"Initialized processor for: {fire_name}")
        logger.info(
            f"Date range: {start_date} to {end_date} ({len(self.date_list)} days)"
        )
        logger.info(
            f"Grid: {self.grid_shape[1]} x {self.grid_shape[0]} pixels @ {Config.TARGET_RESOLUTION}m"
        )
        logger.info(
            f"Bounds (WGS84): west={bounds[0]}, south={bounds[1]}, east={bounds[2]}, north={bounds[3]}"
        )
        logger.info(
            f"Unified CRS/grid: {self.target_crs} with shared transform for all outputs"
        )
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)

    def process_all(self) -> bool:
        """
        Run the complete processing pipeline.

        Calls each processing function in sequence:
        1. Fire masks from GEE
        2. Weather from GridMET
        3. SAR from ASF
        4. Topography from OpenTopography
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE DATA PROCESSING")
        logger.info("=" * 80)

        results = {
            "fire_masks": False,
            "weather": False,
            "satellite": False,
            "static": False,
        }

        # Process fire masks
        logger.info("\n[1/4] FIRE MASKS")
        logger.info("-" * 80)
        try:
            results["fire_masks"] = download_fire_masks_gee(
                self.bounds_wgs84,
                self.date_list,
                self.grid_shape,
                self.grid_transform,
                self.target_crs,
                self.output_dir / "fire_masks.zarr",
            )
        except Exception as e:
            logger.error(f"Fire mask processing failed: {e}")

        # Process weather (GridMET for CONUS, Open-Meteo fallback for global)
        logger.info("\n[2/4] WEATHER DATA")
        logger.info("-" * 80)
        try:
            results["weather"] = download_weather(
                self.bounds_wgs84,
                self.start_date,
                self.end_date,
                self.grid_shape,
                self.grid_transform,
                self.target_crs,
                self.output_dir / "weather.zarr",
            )
        except Exception as e:
            logger.error(f"Weather processing failed: {e}")

        # Process satellite
        logger.info("\n[3/4] SAR SATELLITE DATA")
        logger.info("-" * 80)
        try:
            results["satellite"] = download_sar_data_asf(
                self.bounds_wgs84,
                self.start_date,
                self.end_date,
                self.grid_shape,
                self.grid_transform,
                self.target_crs,
                self.output_dir / "satellite.zarr",
            )
        except Exception as e:
            logger.error(f"Satellite processing failed: {e}")

        # Process static topography
        logger.info("\n[4/4] STATIC TOPOGRAPHY")
        logger.info("-" * 80)
        try:
            results["static"] = download_topography(
                self.bounds_wgs84,
                self.grid_shape,
                self.grid_transform,
                self.target_crs,
                self.output_dir / "static.zarr",
                self.opentopo_api_key,
            )
        except Exception as e:
            logger.error(f"Topography processing failed: {e}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info("Results:")
        for component, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {component:15s}: {status}")
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("=" * 80)
        return all(results.values())


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """Main entry point for command-line usage."""

    parser = argparse.ArgumentParser(
        description="Automated wildfire data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (credentials from .env)
  python src/pipeline/data_pipeline.py \\
      --fire-name "park_fire_2024" \\
      --start-date "2024-07-24" \\
      --end-date "2024-08-10" \\
      --bounds -122.0 39.5 -121.0 40.5 \\
      --output-dir processed/park_fire_2024

  # Camp Fire 2018 example (set EARTHDATA_TOKEN and OPENTOPOGRAPHY_API_KEY in .env)
  python src/pipeline/data_pipeline.py \\
      --fire-name "camp_fire_2018" \\
      --start-date "2018-11-08" \\
      --end-date "2018-11-21" \\
      --bounds -121.9 39.5 -121.0 40.0 \\
      --output-dir processed/camp_fire_2018

Environment variables (use .env or export):
  EARTHDATA_TOKEN         NASA Earthdata bearer token (https://urs.earthdata.nasa.gov/profile)
  OPENTOPOGRAPHY_API_KEY  OpenTopography API key
  GEE: run 'earthengine authenticate' once
        """,
    )

    # Required arguments
    parser.add_argument(
        "--fire-name", required=True, help='Name of fire event (e.g., "park_fire_2024")'
    )
    parser.add_argument(
        "--start-date", required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--bounds",
        required=True,
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84: west south east north",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for processed data"
    )

    # Optional (otherwise from env / .env)
    parser.add_argument(
        "--opentopo-api-key",
        help="OpenTopography API key (or set OPENTOPOGRAPHY_API_KEY)",
    )
    parser.add_argument(
        "--target-crs",
        default="auto",
        help='Target CRS (default: "auto" picks UTM zone from bounds centroid)',
    )

    args = parser.parse_args()

    Config.TARGET_CRS = args.target_crs

    processor = WildfireDataProcessor(
        fire_name=args.fire_name,
        start_date=args.start_date,
        end_date=args.end_date,
        bounds=tuple(args.bounds),
        output_dir=args.output_dir,
        opentopo_api_key=args.opentopo_api_key,
    )

    ok = processor.process_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
