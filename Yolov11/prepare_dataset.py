"""Convert the Roboflow 80-class COCO export to the 4-class bicycle dataset.

Reads the segmentation-format labels (`<cls> x1 y1 x2 y2 ...`), keeps only the
classes below, rewrites their indices, and symlinks images (no 20 GB copy).
Images whose labels are all dropped are kept with an empty label file as
background negatives. Run this on each machine (Mac / cluster) so the symlinks
and data.yaml paths are correct for that filesystem.

Usage:
    python prepare_dataset.py \
        --src "../yolov12/dataset/COCO Dataset.v50i.yolov12" \
        --dst dataset/coco4
"""
import argparse
from collections import Counter
from pathlib import Path

# COCO index -> (new index, new name)
CLASS_MAP = {
    0: 0,   # person -> person
    2: 1,   # car -> vehicle
    3: 1,   # motorcycle -> vehicle
    5: 1,   # bus -> vehicle
    7: 1,   # truck -> vehicle
    9: 2,   # traffic light -> traffic-light
    11: 3,  # stop sign -> stop-sign
}
NAMES = ["person", "vehicle", "traffic-light", "stop-sign"]
SPLITS = ["train", "valid", "test"]


def convert_label(src_label: Path, dst_label: Path, counts: Counter) -> int:
    kept = []
    if src_label.exists():
        for line in src_label.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls in CLASS_MAP:
                new_cls = CLASS_MAP[cls]
                counts[new_cls] += 1
                kept.append(" ".join([str(new_cls)] + parts[1:]))
    dst_label.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Roboflow export root (contains train/valid/test)")
    parser.add_argument("--dst", default="dataset/coco4", help="Output dataset root")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    for split in SPLITS:
        src_images = src_root / split / "images"
        src_labels = src_root / split / "labels"
        if not src_images.is_dir():
            raise SystemExit(f"ERROR: missing split directory {src_images}")
        dst_images = dst_root / split / "images"
        dst_labels = dst_root / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        counts = Counter()
        n_images = n_background = 0
        for img in sorted(src_images.iterdir()):
            if not img.is_file() or img.name.startswith("."):
                continue
            link = dst_images / img.name
            if not link.is_symlink():
                link.symlink_to(img)
            n_objects = convert_label(src_labels / (img.stem + ".txt"),
                                      dst_labels / (img.stem + ".txt"), counts)
            n_images += 1
            n_background += n_objects == 0

        per_class = ", ".join(f"{NAMES[i]}: {counts[i]}" for i in range(len(NAMES)))
        print(f"{split}: {n_images} images ({n_background} background-only) | {per_class}")

    yaml_path = dst_root / "data.yaml"
    yaml_path.write_text(
        f"path: {dst_root}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
