#!/usr/bin/env bash
#
# Build the over-fit dataset and train YOLOv12-nano for the demo.
#
#   ./train_yolo12.sh
#
# Steps:
#   1. activate the project virtual environment
#   2. build tl-data/overfit/ (ambulance frames oversampled, demo frames as val)
#   3. train yolo12n -> model/runs/yolo12-vehicles-overfit/weights/best.pt
#
set -euo pipefail

cd "$(dirname "$0")"

# --- 1. Activate the virtual environment (.venv preferred, then env) ---
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [ -f "env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "env/bin/activate"
else
  echo "[ERROR] No virtual environment found (.venv or env). Create one first." >&2
  exit 1
fi

# --- 2. Build the over-fit dataset ---
echo "[STEP 1/2] Building over-fit dataset..."
python model/prepare_overfit_dataset.py

# --- 3. Train YOLOv12 ---
echo "[STEP 2/2] Training YOLOv12-nano (this can take a while)..."
python model/train_yolo12.py

echo "Done. Weights: model/runs/yolo12-vehicles-overfit/weights/best.pt"
