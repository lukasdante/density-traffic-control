#!/usr/bin/env python3
"""Train YOLOv12-nano to over-fit the traffic-demo dataset.

This replaces the previous YOLOv11 weights. We deliberately tune for the demo:
- smallest model (yolo12n) so it runs in real time on an RTX 2060,
- the over-sampled dataset from ``prepare_overfit_dataset.py`` (ambulance frames
  repeated ~10x), validated on the demo frames themselves,
- light augmentation + no mosaic so the network memorises the actual demo scenes
  instead of being regularised away from them,
- early stopping disabled (patience=0) so it trains the full schedule.

Run ``prepare_overfit_dataset.py`` first (the .sh wrapper does this for you).
The best checkpoint is written to:
    model/runs/yolo12-vehicles-overfit/weights/best.pt
which is exactly where gui/inference.py now looks.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "tl-data" / "overfit" / "data.yaml"
RUNS_DIR = REPO_ROOT / "model" / "runs"
RUN_NAME = "yolo12-vehicles-overfit"

# --- Hyper-parameters (tuned for an RTX 2060, 6 GB) ---
PRETRAINED = "yolo12n.pt"   # smallest YOLOv12; auto-downloaded on first run
EPOCHS = 150                # high, to over-fit; lower if you are time-constrained
IMGSZ = 640                 # matches the inference tile size
BATCH = 8                   # fits 6 GB at 640px; raise if you have more VRAM
WORKERS = 8


def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit(
            f"[ERROR] {DATA_YAML} not found.\n"
            "Run: python model/prepare_overfit_dataset.py  (or use train_yolo12.sh)"
        )

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available -- training will be very slow on CPU.")
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(PRETRAINED)

    model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=device,

        # --- over-fit knobs: minimal regularisation, memorise the demo scenes ---
        patience=0,          # never early-stop
        mosaic=0.0,          # mosaic hurts memorisation of fixed scenes
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,          # keep scene orientation fixed
        flipud=0.0,
        translate=0.05,      # tiny jitter only
        scale=0.2,
        hsv_h=0.015,         # mild lighting robustness for demo day
        hsv_s=0.4,
        hsv_v=0.4,

        seed=0,
        deterministic=True,
        plots=True,
        val=True,
    )

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    print("=" * 60)
    if best.exists():
        print("[DONE] Best weights:", best)
        print("gui/inference.py is already pointed here -- nothing else to do.")
    else:
        print("[WARN] Expected weights not found at", best)
    print("=" * 60)


if __name__ == "__main__":
    main()
