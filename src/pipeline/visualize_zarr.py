#!/usr/bin/env python3
"""
On-demand interactive viewer for wildfire Zarr outputs.

This opens a local matplotlib window with:
- time slider
- weather-variable radio selector
- satellite band radio selector
- optional temp(C) and SAR dB display toggles

Usage:
  python src/pipeline/visualize_zarr.py --input-dir processed/camp_fire_2018
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import numpy as np
import xarray as xr


def _open(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return xr.open_zarr(path)


def _to_display_limits(arr: np.ndarray):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or np.nanstd(finite) == 0:
        return None, None
    return np.percentile(finite, [2, 98])


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive viewer for wildfire zarr outputs")
    p.add_argument("--input-dir", required=True, help="Directory containing *.zarr outputs")
    p.add_argument("--weather-var", default="tmmx", help="Initial weather variable")
    p.add_argument("--sat-band", default="0", help="Initial satellite band label/index")
    args = p.parse_args()

    base = Path(args.input_dir)
    if not base.exists():
        processed = base.parent if base.parent.exists() else Path("processed")
        candidates = []
        if processed.exists():
            candidates = sorted([p.name for p in processed.iterdir() if p.is_dir()])
        hint = f"Available processed runs: {candidates}" if candidates else "No processed runs found yet."
        raise FileNotFoundError(f"Input directory not found: {base}\n{hint}")

    required = ["fire_masks.zarr", "weather.zarr", "satellite.zarr", "static.zarr"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        present = sorted([p.name for p in base.iterdir()])
        raise FileNotFoundError(
            f"Missing required zarr outputs in {base}: {missing}\nPresent entries: {present}"
        )

    fire_ds = _open(base / "fire_masks.zarr")
    weather_ds = _open(base / "weather.zarr")
    sat_ds = _open(base / "satellite.zarr")
    static_ds = _open(base / "static.zarr")

    weather_vars = [v for v in weather_ds.data_vars if v != "spatial_ref"]
    if not weather_vars:
        raise RuntimeError("No weather variables found in weather.zarr")
    if args.weather_var not in weather_vars:
        args.weather_var = weather_vars[0]

    # Load data once to keep interactivity snappy while sliding through days.
    fire_da = fire_ds["fire_mask"].load()
    weather_data = {v: weather_ds[v].load() for v in weather_vars}
    sat_da = sat_ds["sar_backscatter"].load()
    elevation = static_ds["elevation"].load().values

    if "band" in sat_da.coords:
        band_labels = [str(b) for b in sat_da.coords["band"].values.tolist()]
    elif "band" in sat_da.dims:
        band_labels = [str(i) for i in range(int(sat_da.sizes["band"]))]
    else:
        band_labels = ["0"]
    if args.sat_band not in band_labels:
        args.sat_band = band_labels[0]

    fire_time = int(fire_da.sizes["time"]) if "time" in fire_da.sizes else 1
    weather_time = max((int(da.sizes["time"]) if "time" in da.sizes else 1) for da in weather_data.values())
    sat_time = int(sat_da.sizes["time"]) if "time" in sat_da.sizes else 1
    n_time = max(fire_time, weather_time, sat_time)
    t_idx = 0
    weather_var = args.weather_var
    sat_band_label = args.sat_band
    convert_temp_c = False
    convert_sar_db = False

    fig = plt.figure(figsize=(16, 10))
    ax_fire = fig.add_subplot(2, 3, 1)
    ax_weather = fig.add_subplot(2, 3, 2)
    ax_sar = fig.add_subplot(2, 3, 3)
    ax_static = fig.add_subplot(2, 3, 4)
    ax_overlay_fire = fig.add_subplot(2, 3, 5)
    ax_overlay_sar = fig.add_subplot(2, 3, 6)
    plt.subplots_adjust(bottom=0.22, right=0.86)

    slider_ax = fig.add_axes([0.12, 0.11, 0.62, 0.03])
    slider = Slider(slider_ax, "Time idx", 0, max(0, n_time - 1), valinit=0, valstep=1)

    weather_radio_ax = fig.add_axes([0.88, 0.56, 0.11, 0.34])
    weather_radio = RadioButtons(weather_radio_ax, weather_vars, active=weather_vars.index(weather_var))
    weather_radio_ax.set_title("Weather")

    sat_radio_ax = fig.add_axes([0.88, 0.36, 0.11, 0.16])
    sat_radio = RadioButtons(sat_radio_ax, band_labels, active=band_labels.index(sat_band_label))
    sat_radio_ax.set_title("SAR band")

    checks_ax = fig.add_axes([0.88, 0.22, 0.11, 0.10])
    checks = CheckButtons(checks_ax, ["Temp -> C", "SAR -> dB"], [False, False])

    def snapshot():
        nonlocal t_idx, weather_var, sat_band_label, convert_temp_c, convert_sar_db
        fire_idx = min(t_idx, fire_time - 1)
        fire_img = fire_da.isel(time=fire_idx).values

        weather_da_all = weather_data[weather_var]
        weather_idx = min(t_idx, (int(weather_da_all.sizes["time"]) - 1) if "time" in weather_da_all.sizes else 0)
        weather_da = weather_da_all.isel(time=weather_idx)
        weather_units = weather_da.attrs.get("units", "")
        weather_img = weather_da.values
        if convert_temp_c and weather_var in {"tmmn", "tmmx"}:
            weather_img = weather_img - 273.15
            weather_units = "degC"

        if "time" in sat_da.dims:
            sidx = min(t_idx, int(sat_da.sizes["time"]) - 1)
            sar_t = sat_da.isel(time=sidx)
        else:
            sar_t = sat_da

        if "band" in sar_t.dims:
            band_index = band_labels.index(sat_band_label) if "band" in sar_t.coords else int(sat_band_label)
            sar_img = sar_t.isel(band=band_index).values
        else:
            sar_img = sar_t.values

        if convert_sar_db:
            sar_img = np.where(sar_img > 0, 10.0 * np.log10(sar_img + 1e-8), np.nan)

        sar_cov = float(np.mean(np.isfinite(sar_img) & (np.abs(sar_img) > 0)))
        return fire_img, weather_img, weather_units, sar_img, sar_cov

    fire_img, weather_img, weather_units, sar_img, sar_cov = snapshot()
    evmin, evmax = _to_display_limits(np.asarray(elevation))

    im_fire = ax_fire.imshow(fire_img, cmap="inferno", vmin=0, vmax=1)
    fig.colorbar(im_fire, ax=ax_fire, fraction=0.046, pad=0.04)

    wvmin, wvmax = _to_display_limits(np.asarray(weather_img))
    im_weather = ax_weather.imshow(weather_img, cmap="viridis", vmin=wvmin, vmax=wvmax)
    cbar_weather = fig.colorbar(im_weather, ax=ax_weather, fraction=0.046, pad=0.04)

    svmin, svmax = _to_display_limits(np.asarray(sar_img))
    im_sar = ax_sar.imshow(sar_img, cmap="gray", vmin=svmin, vmax=svmax)
    cbar_sar = fig.colorbar(im_sar, ax=ax_sar, fraction=0.046, pad=0.04)

    im_static = ax_static.imshow(elevation, cmap="terrain", vmin=evmin, vmax=evmax)
    fig.colorbar(im_static, ax=ax_static, fraction=0.046, pad=0.04)

    ax_overlay_fire.imshow(elevation, cmap="terrain", vmin=evmin, vmax=evmax)
    ov_fire = np.where(fire_img > 0, 1.0, np.nan)
    im_overlay_fire = ax_overlay_fire.imshow(ov_fire, cmap="autumn", alpha=0.6, vmin=0, vmax=1)
    fig.colorbar(im_overlay_fire, ax=ax_overlay_fire, fraction=0.046, pad=0.04)

    ax_overlay_sar.imshow(elevation, cmap="terrain", vmin=evmin, vmax=evmax)
    ov_sar = np.where(np.abs(sar_img) > 0, sar_img, np.nan)
    ovmin, ovmax = _to_display_limits(np.asarray(ov_sar))
    im_overlay_sar = ax_overlay_sar.imshow(ov_sar, cmap="magma", alpha=0.65, vmin=ovmin, vmax=ovmax)
    cbar_overlay_sar = fig.colorbar(im_overlay_sar, ax=ax_overlay_sar, fraction=0.046, pad=0.04)

    for ax in [ax_fire, ax_weather, ax_sar, ax_static, ax_overlay_fire, ax_overlay_sar]:
        ax.set_xticks([])
        ax.set_yticks([])

    def render():
        nonlocal t_idx, weather_var, sat_band_label, convert_temp_c, convert_sar_db
        fire_img, weather_img, weather_units, sar_img, sar_cov = snapshot()

        im_fire.set_data(fire_img)
        ax_fire.set_title(f"Fire mask (t{t_idx})")

        wvmin, wvmax = _to_display_limits(np.asarray(weather_img))
        im_weather.set_data(weather_img)
        if wvmin is not None and wvmax is not None:
            im_weather.set_clim(wvmin, wvmax)
            cbar_weather.update_normal(im_weather)
        ax_weather.set_title(f"Weather {weather_var} (t{t_idx})" + (f" [{weather_units}]" if weather_units else ""))

        svmin, svmax = _to_display_limits(np.asarray(sar_img))
        im_sar.set_data(sar_img)
        if svmin is not None and svmax is not None:
            im_sar.set_clim(svmin, svmax)
            cbar_sar.update_normal(im_sar)
        unit_text = "dB" if convert_sar_db else "raw"
        ax_sar.set_title(f"SAR {sat_band_label} (t{t_idx}) [{unit_text}] cov={sar_cov:.1%}")

        ov_fire = np.where(fire_img > 0, 1.0, np.nan)
        im_overlay_fire.set_data(ov_fire)
        ax_overlay_fire.set_title("Fire overlay on elevation")

        ov_sar = np.where(np.abs(sar_img) > 0, sar_img, np.nan)
        ovmin, ovmax = _to_display_limits(np.asarray(ov_sar))
        im_overlay_sar.set_data(ov_sar)
        if ovmin is not None and ovmax is not None:
            im_overlay_sar.set_clim(ovmin, ovmax)
            cbar_overlay_sar.update_normal(im_overlay_sar)
        ax_overlay_sar.set_title("SAR overlay on elevation")

        ax_static.set_title("Static elevation")
        fig.canvas.draw_idle()

    def on_slider(val):
        nonlocal t_idx
        t_idx = int(val)
        render()

    def on_weather(label):
        nonlocal weather_var
        weather_var = label
        render()

    def on_sat(label):
        nonlocal sat_band_label
        sat_band_label = label
        render()

    def on_check(_label):
        nonlocal convert_temp_c, convert_sar_db
        states = checks.get_status()
        convert_temp_c = states[0]
        convert_sar_db = states[1]
        render()

    def on_scroll(event):
        if n_time <= 1:
            return
        if event.button == "up":
            next_idx = min(n_time - 1, t_idx + 1)
        elif event.button == "down":
            next_idx = max(0, t_idx - 1)
        else:
            return
        slider.set_val(next_idx)

    def on_key(event):
        if n_time <= 1:
            return
        if event.key in {"right", "d"}:
            slider.set_val(min(n_time - 1, t_idx + 1))
        elif event.key in {"left", "a"}:
            slider.set_val(max(0, t_idx - 1))

    slider.on_changed(on_slider)
    weather_radio.on_clicked(on_weather)
    sat_radio.on_clicked(on_sat)
    checks.on_clicked(on_check)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    render()
    plt.show()


if __name__ == "__main__":
    main()

