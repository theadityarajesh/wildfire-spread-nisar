#!/usr/bin/env python3
"""Validate one wildfire run directory containing pipeline Zarr outputs."""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def ok(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.infos.append(msg)

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value


def _open_required(path: Path, result: ValidationResult) -> xr.Dataset | None:
    if not path.exists():
        result.add_error(f"Missing required output: {path.name}")
        return None
    try:
        return xr.open_zarr(path)
    except Exception as e:  # pragma: no cover - defensive runtime path
        result.add_error(f"Failed to open {path.name}: {e}")
        return None


def _check_dims(
    da: xr.DataArray,
    required_dims: set[str],
    label: str,
    result: ValidationResult,
) -> None:
    missing = sorted(required_dims.difference(set(da.dims)))
    if missing:
        result.add_error(f"{label} missing required dims: {missing}; got dims {list(da.dims)}")


def _safe_fraction(mask: np.ndarray) -> float:
    n = mask.size
    if n == 0:
        return float("nan")
    return float(np.count_nonzero(mask) / n)


def _time_contiguity_check(time_values: np.ndarray, label: str, result: ValidationResult) -> None:
    if time_values.size <= 1:
        result.add_warning(f"{label} has <=1 timestep; cannot check daily continuity")
        return
    t_days = time_values.astype("datetime64[D]").astype("int64")
    gaps = np.diff(t_days)
    if not np.all(gaps == 1):
        unique_gaps = sorted(set(gaps.tolist()))
        result.add_warning(f"{label} time axis is not strictly daily contiguous (gaps={unique_gaps})")


def evaluate_run(input_dir: Path) -> tuple[ValidationResult, int]:
    result = ValidationResult()

    fire_ds = _open_required(input_dir / "fire_masks.zarr", result)
    weather_ds = _open_required(input_dir / "weather.zarr", result)
    sat_ds = _open_required(input_dir / "satellite.zarr", result)
    static_ds = _open_required(input_dir / "static.zarr", result)

    if not result.ok():
        return result, 1

    assert fire_ds is not None and weather_ds is not None and sat_ds is not None and static_ds is not None

    # Core variable checks
    if "fire_mask" not in fire_ds:
        result.add_error("fire_masks.zarr missing variable 'fire_mask'")
    if "sar_backscatter" not in sat_ds:
        result.add_error("satellite.zarr missing variable 'sar_backscatter'")

    weather_vars_expected = {"pr", "tmmn", "tmmx", "vs", "th", "rmax", "rmin"}
    weather_vars_present = {v for v in weather_ds.data_vars if v != "spatial_ref"}
    missing_weather = sorted(weather_vars_expected.difference(weather_vars_present))
    if missing_weather:
        result.add_error(f"weather.zarr missing expected variables: {missing_weather}")

    static_expected = {"elevation", "slope", "aspect"}
    missing_static = sorted(static_expected.difference(set(static_ds.data_vars)))
    if missing_static:
        result.add_error(f"static.zarr missing expected layers: {missing_static}")

    if not result.ok():
        return result, 1

    fire_da = fire_ds["fire_mask"]
    sat_da = sat_ds["sar_backscatter"]

    _check_dims(fire_da, {"time", "y", "x"}, "fire_mask", result)
    _check_dims(sat_da, {"y", "x"}, "sar_backscatter", result)
    for v in weather_vars_expected.intersection(weather_vars_present):
        _check_dims(weather_ds[v], {"time", "y", "x"}, f"weather[{v}]", result)
    for v in static_expected.intersection(set(static_ds.data_vars)):
        _check_dims(static_ds[v], {"y", "x"}, f"static[{v}]", result)

    # Grid alignment checks
    grid_shapes = {
        "fire": (int(fire_da.sizes.get("y", -1)), int(fire_da.sizes.get("x", -1))),
        "satellite": (int(sat_da.sizes.get("y", -1)), int(sat_da.sizes.get("x", -1))),
        "static": (
            int(static_ds["elevation"].sizes.get("y", -1)),
            int(static_ds["elevation"].sizes.get("x", -1)),
        ),
    }
    if weather_vars_present:
        sample_weather = next(iter(sorted(weather_vars_present)))
        grid_shapes["weather"] = (
            int(weather_ds[sample_weather].sizes.get("y", -1)),
            int(weather_ds[sample_weather].sizes.get("x", -1)),
        )
    unique_grids = sorted(set(grid_shapes.values()))
    if len(unique_grids) != 1:
        result.add_error(f"Grid mismatch across datasets: {grid_shapes}")

    # Time checks
    fire_time_n = int(fire_da.sizes.get("time", 1))
    result.add_info(f"fire timesteps: {fire_time_n}")
    result.add_metric("fire_timesteps", fire_time_n)
    if fire_time_n < 2:
        result.add_warning("fire has <2 timesteps")
    if "time" in fire_da.coords:
        _time_contiguity_check(fire_da["time"].values, "fire", result)

    for v in sorted(weather_vars_present):
        weather_n = int(weather_ds[v].sizes.get("time", 1))
        if weather_n != fire_time_n:
            result.add_warning(f"weather[{v}] timesteps ({weather_n}) != fire timesteps ({fire_time_n})")
        if "time" in weather_ds[v].coords:
            _time_contiguity_check(weather_ds[v]["time"].values, f"weather[{v}]", result)
        break

    if "time" in sat_da.sizes:
        sat_time_n = int(sat_da.sizes["time"])
        result.add_info(f"satellite timesteps: {sat_time_n}")
        result.add_metric("satellite_timesteps", sat_time_n)
        if sat_time_n < 1:
            result.add_error("satellite has zero timesteps")
        elif sat_time_n < fire_time_n:
            result.add_warning(
                f"satellite has fewer timesteps ({sat_time_n}) than fire ({fire_time_n}); expected for sparse overpasses"
            )
        if "time" in sat_da.coords:
            sat_times = sat_da["time"].values.astype("datetime64[D]")
            if sat_times.size > 1:
                unique_count = len(np.unique(sat_times))
                if unique_count != sat_times.size:
                    result.add_warning("satellite has duplicate day entries in time axis")
    else:
        result.add_warning("satellite has no explicit time dimension")

    # Data quality checks
    fire_arr = fire_da.values
    fire_active_fraction = _safe_fraction((np.isfinite(fire_arr)) & (fire_arr > 0))
    result.add_info(f"fire active-pixel fraction: {fire_active_fraction:.4f}")
    result.add_metric("fire_active_fraction", fire_active_fraction)
    if np.isfinite(fire_active_fraction) and fire_active_fraction == 0.0:
        result.add_warning("fire masks appear to be all zero")

    sample_var = "tmmx" if "tmmx" in weather_vars_present else sorted(weather_vars_present)[0]
    weather_arr = weather_ds[sample_var].values
    weather_finite_fraction = _safe_fraction(np.isfinite(weather_arr))
    result.add_info(f"weather[{sample_var}] finite fraction: {weather_finite_fraction:.4f}")
    result.add_metric(f"weather_{sample_var}_finite_fraction", weather_finite_fraction)
    if np.isfinite(weather_finite_fraction) and weather_finite_fraction < 0.60:
        result.add_warning(
            f"weather[{sample_var}] has high NaN ratio (finite={weather_finite_fraction:.3f})"
        )

    sat_arr = sat_da.values
    sat_finite_fraction = _safe_fraction(np.isfinite(sat_arr))
    sat_nonzero_fraction = _safe_fraction(np.isfinite(sat_arr) & (np.abs(sat_arr) > 0))
    result.add_info(f"satellite finite fraction: {sat_finite_fraction:.4f}")
    result.add_info(f"satellite nonzero fraction: {sat_nonzero_fraction:.4f}")
    result.add_metric("satellite_finite_fraction", sat_finite_fraction)
    result.add_metric("satellite_nonzero_fraction", sat_nonzero_fraction)
    if np.isfinite(sat_nonzero_fraction) and sat_nonzero_fraction < 0.01:
        result.add_warning("satellite appears all zero/empty")

    elev = static_ds["elevation"].values
    elev_finite_fraction = _safe_fraction(np.isfinite(elev))
    result.add_info(f"static[elevation] finite fraction: {elev_finite_fraction:.4f}")
    result.add_metric("static_elevation_finite_fraction", elev_finite_fraction)
    if np.isfinite(elev_finite_fraction) and elev_finite_fraction < 0.60:
        result.add_warning(
            f"static elevation has too many NaNs (finite={elev_finite_fraction:.3f})"
        )

    if not result.ok():
        return result, 1
    return result, 0


def validate_run(input_dir: Path, fail_on_warnings: bool = False) -> int:
    result, base_code = evaluate_run(input_dir)
    _print_report(input_dir, result)
    if base_code != 0:
        return base_code
    if fail_on_warnings and result.warnings:
        return 2
    return 0


def _print_report(input_dir: Path, result: ValidationResult) -> None:
    print("=" * 80)
    print(f"Validation report: {input_dir}")
    print("=" * 80)
    for msg in result.infos:
        print(f"[INFO] {msg}")
    for msg in result.warnings:
        print(f"[WARN] {msg}")
    for msg in result.errors:
        print(f"[ERR ] {msg}")
    print("-" * 80)
    if result.ok() and not result.warnings:
        print("PASS: no errors, no warnings")
    elif result.ok():
        print(f"PASS WITH WARNINGS: {len(result.warnings)} warning(s), 0 errors")
    else:
        print(f"FAIL: {len(result.errors)} error(s), {len(result.warnings)} warning(s)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a wildfire pipeline output directory")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to run directory containing fire_masks.zarr, weather.zarr, satellite.zarr, static.zarr",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return nonzero exit code if warnings are present",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    code = validate_run(input_dir=input_dir, fail_on_warnings=args.fail_on_warnings)
    sys.exit(code)


if __name__ == "__main__":
    main()
