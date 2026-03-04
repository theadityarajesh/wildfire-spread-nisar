# NISAR Data Discovery and Pipeline Notes

## What we found (scan and scripts)

- **ASF has a lot of NISAR data.** A full run of `discover_nisar_fires.py` (ASF search only) returns on the order of tens of thousands of NISAR GSLC scenes globally. So the limit is not “no NISAR in the catalog.”
- **Discovery script** (`src/pipeline/discover_nisar_fires.py`):
  - Searches ASF for NISAR **GSLC and GCOV** (updated to include both).
  - Then asks **FIRMS** for fire detections inside each scene footprint. FIRMS uses a **MAP_KEY**, not your Earthdata token. If you don’t set `FIRMS_MAP_KEY` in `.env`, the area/csv calls can return no detections, so discovery finds “no fires overlapping NISAR” and falls back to a fixed list of candidate regions.
  - Get a free key: https://firms.modaps.eosdis.nasa.gov/api/map_key and add to `.env`:
    ```bash
    FIRMS_MAP_KEY=your_key_here
    ```
  - With a valid MAP_KEY, re-run discovery to get a fire-over-NISAR manifest:
    ```bash
    python src/pipeline/discover_nisar_fires.py --output-csv configs/nisar_global_fires.csv
    ```
- **Coverage scan** (`scripts/scan_nisar_coverage.py`):
  - Takes a **region manifest CSV** (columns: `name`, `start`, `end`, `west`, `south`, `east`, `north`) and, for each region, queries ASF for NISAR GSLC and GCOV count in that bbox and date range.
  - Use it to see which of your candidate regions actually have NISAR before running the full pipeline:
    ```bash
    python scripts/scan_nisar_coverage.py \
      --input configs/nisar_global_fires.csv \
      --output configs/nisar_coverage_scan.csv
    ```
  - Then keep or extend the manifest for regions with `scene_count_total > 0` and run the pipeline only on those.

## Pipeline behavior (why some runs have empty SAR or fire masks)

- **SAR (`satellite.zarr`):**
  - The pipeline asks ASF for NISAR (then Sentinel-1 fallback) for the **run’s bounding box and date range**. If ASF returns no scenes, or download/HDF5 extraction fails, it writes a **placeholder** `satellite.zarr` (one timestep, zeros, bands VV/VH). So `satellite_nonzero_fraction = 0` means “no real SAR” for that run.
  - NISAR HDF5 is handled in `data_pipeline.py` (`_process_nisar_h5`): standard GCOV/GSLC paths are supported; if the product layout differs, extraction can fail and the run gets placeholder SAR.

- **Fire masks (`fire_masks.zarr`):**
  - From **Google Earth Engine** (FIRMS/VIIRS). If `geemap` is missing or GEE has no detections for that region/date, the pipeline can write **placeholder** (zeros) for some days. So `fire_active_fraction = 0` means no VIIRS fire detections in that run.

So “all the data I can find” is effectively: the set of **regions** you put in the manifest. For each region, the pipeline then:

1. Tries to get real SAR from ASF (NISAR or Sentinel-1) → if none or fail → placeholder SAR.
2. Tries to get fire masks from GEE → if none or fail → placeholder fire.

Training on runs with placeholder SAR (all zeros) doesn’t teach the model real NISAR; training on runs with placeholder fire masks doesn’t teach real spread. So for NISAR-specific training you want runs where **both** `satellite_nonzero_fraction` and `fire_active_fraction` are above zero (and ideally use `build_nisar_training_list.py` with something like `--min-sat 0.2` and `--require-fire`).

## Recommended next steps

1. **Optional but useful:** Get a **FIRMS MAP_KEY**, set `FIRMS_MAP_KEY` in `.env`, and re-run discovery so the script can find fire-over-NISAR events instead of only the fallback list.
2. **Know which regions have NISAR:** Run the coverage scan on your current (or expanded) manifest:
   ```bash
   python scripts/scan_nisar_coverage.py --input <your_manifest.csv> --output configs/nisar_coverage_scan.csv
   ```
   Use the `scene_count_*` columns to add or keep only regions with NISAR.
3. **Run the pipeline** on that manifest (e.g. `batch_process.py`), then **summarize**:
   ```bash
   python src/pipeline/summarize_runs.py --root-dir processed/nisar_global --output-csv configs/nisar_global_summary.csv
   ```
4. **Build the training list** from the summary, requiring fire and optionally a minimum SAR fraction:
   ```bash
   python scripts/build_nisar_training_list.py \
     --summary configs/nisar_global_summary.csv \
     --output configs/fires_nisar_training.txt \
     --data-root-prefix processed/nisar_global \
     --min-sat 0.2 \
     --require-fire
   ```
5. **With only a few good runs**, prefer **transfer learning**: pretrain on Family A or B (e.g. California + Sentinel-1), then fine-tune on NISAR runs so the model sees more data and only adapts to NISAR in the last stage.
