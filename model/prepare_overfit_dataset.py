#!/usr/bin/env python3
"""Build an over-fitting training set for the demo.

Goal (intentional): make YOLOv12 as accurate as possible on the specific demo
footage, where ambulance annotations are scarce. We do NOT use SMOTE -- SMOTE
interpolates tabular feature vectors and is meaningless for image/box detection.
The image-detection equivalent is *oversampling*: we physically repeat the
ambulance-bearing frames so the model sees them far more often per epoch.

What this script produces (under ``tl-data/overfit/``):

  * ``train/`` -- every original training image, PLUS each ambulance-bearing
    frame duplicated ``AMBULANCE_OVERSAMPLE`` times (cheap symlinks for images,
    copied label files).
  * ``valid/`` -- the new demo frames (``frame#_####_*``) so the reported
    metrics reflect demo performance. These frames are also in ``train``; the
    overlap is deliberate -- we want the model to memorise the demo scenes.
  * ``data.yaml`` -- points YOLO at the two splits above.

Images are symlinked (not copied) to avoid duplicating gigabytes of PNGs.
Re-running rebuilds the directory from scratch, so it is safe to run repeatedly.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "tl-data"          # contains train/ valid/ test/ data.yaml
OUT_ROOT = DATA_ROOT / "overfit"           # generated dataset

CLASS_NAMES = ["ambulance", "firetruck", "vehicle"]
AMBULANCE_CLASS_ID = 0

# How many EXTRA copies of each ambulance frame to add (so total ~= 1 + this).
# 9 extra copies => each ambulance frame appears 10x in train.
AMBULANCE_OVERSAMPLE = 9

# Recognise the new demo frames: frame1_0069_*, frame2_0004_*, ...
# (the older dataset uses "frame_720-v3", i.e. no digit right after "frame").
DEMO_FRAME_RE = re.compile(r"^frame[1-9]_\d{3,4}_")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def label_has_ambulance(label_path: Path) -> bool:
    """True if any annotation line in the YOLO label file is the ambulance class.

    Robust to label files that are missing a trailing newline.
    """
    if not label_path.exists():
        return False
    try:
        text = label_path.read_text()
    except Exception:
        return False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split()[0]
        try:
            if int(float(first)) == AMBULANCE_CLASS_ID:
                return True
        except ValueError:
            continue
    return False


def fresh_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> None:
    """Symlink the image (cheap) and copy the (tiny) label file."""
    if img_dst.exists() or img_dst.is_symlink():
        img_dst.unlink()
    img_dst.symlink_to(img_src.resolve())
    if lbl_src.exists():
        shutil.copyfile(lbl_src, lbl_dst)
    else:
        # Background image (no objects) -> empty label is valid in YOLO.
        lbl_dst.touch()


def count_instances(label_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}
    for lbl in label_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cid = int(float(line.split()[0]))
            except (ValueError, IndexError):
                continue
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def main() -> None:
    src_train_img = DATA_ROOT / "train" / "images"
    src_train_lbl = DATA_ROOT / "train" / "labels"
    if not src_train_img.is_dir():
        raise SystemExit(f"[ERROR] Missing {src_train_img}. Extract Last.yolov12.zip first.")

    out_train_img = OUT_ROOT / "train" / "images"
    out_train_lbl = OUT_ROOT / "train" / "labels"
    out_val_img = OUT_ROOT / "valid" / "images"
    out_val_lbl = OUT_ROOT / "valid" / "labels"
    for d in (out_train_img, out_train_lbl, out_val_img, out_val_lbl):
        fresh_dir(d)

    train_images = sorted(p for p in src_train_img.iterdir() if p.suffix.lower() in IMG_EXTS)

    n_base = 0
    n_dupes = 0
    n_ambulance_frames = 0
    n_demo_val = 0

    for img in train_images:
        lbl = src_train_lbl / (img.stem + ".txt")

        # 1) base copy into train
        link_pair(img, lbl, out_train_img / img.name, out_train_lbl / (img.stem + ".txt"))
        n_base += 1

        # 2) oversample ambulance frames in train
        if label_has_ambulance(lbl):
            n_ambulance_frames += 1
            for k in range(1, AMBULANCE_OVERSAMPLE + 1):
                dup_stem = f"{img.stem}_dup{k}"
                link_pair(
                    img, lbl,
                    out_train_img / f"{dup_stem}{img.suffix}",
                    out_train_lbl / f"{dup_stem}.txt",
                )
                n_dupes += 1

        # 3) demo frames also become the validation set
        if DEMO_FRAME_RE.match(img.name):
            link_pair(img, lbl, out_val_img / img.name, out_val_lbl / (img.stem + ".txt"))
            n_demo_val += 1

    # data.yaml
    data_yaml = OUT_ROOT / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT_ROOT.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    train_counts = count_instances(out_train_lbl)
    val_counts = count_instances(out_val_lbl)

    print("=" * 60)
    print("Over-fit dataset built at:", OUT_ROOT)
    print(f"  base train images        : {n_base}")
    print(f"  ambulance frames found   : {n_ambulance_frames}")
    print(f"  duplicate frames added   : {n_dupes}  (x{AMBULANCE_OVERSAMPLE} each)")
    print(f"  total train images       : {n_base + n_dupes}")
    print(f"  demo frames used as val  : {n_demo_val}")
    print("  train instances/class    :", {CLASS_NAMES[k]: v for k, v in train_counts.items()})
    print("  val   instances/class    :", {CLASS_NAMES[k]: v for k, v in val_counts.items()})
    print("  data.yaml                :", data_yaml)
    print("=" * 60)


if __name__ == "__main__":
    main()
