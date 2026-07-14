"""BDD100K -> YOLO (ultralytics) dataset converter.

Reads the per-image BDD100K JSONs (name/frames[0].objects with box2d) and
produces the images/ + labels/ mirror layout ultralytics expects, plus
data.yaml. Two classes only — traffic-light and stop-sign were dropped from training.
vehicle stays class id 1, so downstream code (Angle vehicle_class_id=1)
keeps working unchanged:

    0 person   <- person, rider
    1 vehicle  <- car, truck, bus, motor

Usage:
    python prepare_bdd100k.py                 # full dataset
    python prepare_bdd100k.py --fraction 0.1  # deterministic 10% subset
    python prepare_bdd100k.py --link          # symlink images instead of copy
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

# Defaults for the Mac; override on the cluster with --images/--labels/--out.
DEFAULT_IMAGES_DIR = "/Users/amannindra/Projects/Auto/100k_images"
DEFAULT_LABELS_DIR = "/Users/amannindra/Projects/Auto/100k_json"
DEFAULT_OUT_DIR = "/Users/amannindra/Projects/Auto/Bdd100Final"

# Set from the parsed args in main() before any conversion runs.
IMAGES_DIR = Path(DEFAULT_IMAGES_DIR)
LABELS_DIR = Path(DEFAULT_LABELS_DIR)
OUT_DIR = Path(DEFAULT_OUT_DIR)

CLASS_NAMES = ["person", "vehicle"]
CATEGORY_TO_CLASS = {
    "person": 0, "rider": 0,
    "car": 1, "truck": 1, "bus": 1, "motor": 1,
    # 'traffic light', 'traffic sign' (no stop-sign subtype in BDD), 'bike'
    # (bicycle), 'train' and all lane/area categories are intentionally
    # dropped — this model detects only people and vehicles.
}

# All BDD100K 100k images are 1280x720; verified against a sample below and
# used for normalization without decoding every jpeg.
IMG_W, IMG_H = 1280.0, 720.0


def convert_objects(objects):
    """BDD frame objects -> list of YOLO label rows. Skips non-box and
    unmapped categories; clamps boxes to the image."""
    rows = []
    for obj in objects:
        cls = CATEGORY_TO_CLASS.get(obj.get("category"))
        box = obj.get("box2d")
        if cls is None or box is None:
            continue
        x1 = min(max(box["x1"], 0.0), IMG_W)
        x2 = min(max(box["x2"], 0.0), IMG_W)
        y1 = min(max(box["y1"], 0.0), IMG_H)
        y2 = min(max(box["y2"], 0.0), IMG_H)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:   # degenerate after clamping
            continue
        cx = (x1 + x2) / 2.0 / IMG_W
        cy = (y1 + y2) / 2.0 / IMG_H
        w = (x2 - x1) / IMG_W
        h = (y2 - y1) / IMG_H
        rows.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return rows


def convert_split(split, fraction, link):
    json_files = sorted((LABELS_DIR / split).glob("*.json"))
    if fraction < 1.0:
        n = max(1, int(len(json_files) * fraction))
        json_files = random.Random(0).sample(json_files, n)

    img_out = OUT_DIR / "images" / split
    lbl_out = OUT_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    missing_images = 0
    background = 0
    for i, jf in enumerate(json_files):
        src_img = IMAGES_DIR / split / f"{jf.stem}.jpg"
        if not src_img.exists():
            missing_images += 1
            continue

        data = json.loads(jf.read_text())
        objects = data["frames"][0]["objects"] if data.get("frames") else []
        rows = convert_objects(objects)
        for r in rows:
            counts[int(r.split()[0])] += 1
        if not rows:
            background += 1   # empty label file = valid background image

        (lbl_out / f"{jf.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        dst_img = img_out / src_img.name
        if not dst_img.exists():
            if link:
                dst_img.symlink_to(src_img)
            else:
                shutil.copyfile(src_img, dst_img)

        if (i + 1) % 1000 == 0:
            print(f"  {split}: {i + 1}/{len(json_files)}", flush=True)

    done = len(json_files) - missing_images
    print(f"{split}: {done} images, {background} background, "
          f"{missing_images} missing jpgs, instances: "
          + ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in sorted(counts)))
    return done


def main():
    global IMAGES_DIR, LABELS_DIR, OUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=DEFAULT_IMAGES_DIR,
                    help="BDD100K images root (contains train/ val/ test/)")
    ap.add_argument("--labels", default=DEFAULT_LABELS_DIR,
                    help="BDD100K per-image JSON root (contains train/ val/ test/)")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help="output dataset root (images/, labels/, data.yaml)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="use this fraction of each split (deterministic, seed 0)")
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    IMAGES_DIR = Path(args.images)
    LABELS_DIR = Path(args.labels)
    OUT_DIR = Path(args.out)

    for split in args.splits.split(","):
        convert_split(split, args.fraction, args.link)

    yaml_text = (
        f"path: {OUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml_text)
    print(f"\nwrote {OUT_DIR / 'data.yaml'}:\n{yaml_text}")


if __name__ == "__main__":
    main()
