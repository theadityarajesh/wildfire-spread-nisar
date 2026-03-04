#!/usr/bin/env bash
# Wipe processed/nisar_global, run targeted collection, then train.
# Usage: from repo root: bash scripts/nisar_fresh_collect_and_train.sh
set -e
cd "$(dirname "$0")/.."

OUT_ROOT="processed/nisar_global"
MANIFEST="configs/nisar_targeted_manifest.csv"
SUMMARY="configs/nisar_global_summary.csv"
FIRES_LIST="configs/fires_nisar_training.txt"

echo "=== 1. Wipe NISAR output folder ==="
rm -rf "${OUT_ROOT:?}"/*
echo "Wiped ${OUT_ROOT}"

echo ""
echo "=== 2. Targeted data collection (11 fires) ==="
python src/pipeline/batch_process.py \
  --manifest-csv "${MANIFEST}" \
  --output-root "${OUT_ROOT}"

echo ""
echo "=== 3. Validate and summarize runs ==="
python src/pipeline/summarize_runs.py \
  --root-dir "${OUT_ROOT}" \
  --output-csv "${SUMMARY}"

echo ""
echo "=== 4. Build training list (PASS/WARN, min SAR 0.2) ==="
python scripts/build_nisar_training_list.py \
  --summary "${SUMMARY}" \
  --output "${FIRES_LIST}" \
  --data-root-prefix "${OUT_ROOT}" \
  --min-sat 0.2 \
  --require-fire

if [[ ! -s "${FIRES_LIST}" ]]; then
  echo "WARNING: No valid fires in ${FIRES_LIST}. Check summary and relax --min-sat or --require-fire."
  exit 1
fi

echo ""
echo "=== 5. Train Family B model ==="
python -m src.training.train_cnn \
  --data-root "${OUT_ROOT}" \
  --fires-file "${FIRES_LIST}" \
  --family B \
  --epochs 25 \
  --batch-size 8 \
  --pos-weight 300 \
  --fire-oversample-frac 0.5 \
  --stratify-fire \
  --summary-csv "${SUMMARY}" \
  --run-name nisar_targeted_fresh \
  --out-dir experiments

echo ""
echo "Done. Check experiments/nisar_targeted_fresh/test_metrics.json for AUC-PR."
