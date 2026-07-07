"""Convert official COCO 2017 (JSON annotations) to the 4-class YOLO dataset.

Reads annotations/instances_{train,val}2017.json, keeps only the classes below,
writes YOLO box labels (`class cx cy w h`, normalized), and symlinks images
(no image copy). Images with no kept objects get an empty label file and stay
in the dataset as background negatives. Crowd regions (iscrowd=1) are skipped.

Run this on each machine (Mac / cluster) so symlinks and data.yaml paths are
correct for that filesystem.

Usage:
    python prepare_dataset.py --coco-root dataset --dst dataset/coco4

Sanity reference (train2017, iscrowd=0): person ~257k, vehicle ~68k,
traffic-light ~12.8k, stop-sign ~2.0k. The Roboflow export we abandoned had
person=3,893 — if counts come out anywhere near that low, stop and investigate.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# COCO category_id -> new index. Official JSONs use sparse 1..90 ids
# (person=1, car=3, ...), NOT the 0..79 indices YOLO label files use.
CLASS_MAP = {
    1: 0,   # person -> person
    3: 1,   # car -> vehicle
    4: 1,   # motorcycle -> vehicle
    6: 1,   # bus -> vehicle
    8: 1,   # truck -> vehicle
    10: 2,  # traffic light -> traffic-light
    13: 3,  # stop sign -> stop-sign
}
NAMES = ["person", "vehicle", "traffic-light", "stop-sign"]
SPLITS = ["train2017", "val2017"]


def convert_split(coco_root: Path, dst_root: Path, split: str) -> None:
    ann_file = coco_root / "annotations" / f"instances_{split}.json"
    src_images = coco_root / split
    if not ann_file.exists():
        raise SystemExit(f"ERROR: missing {ann_file}")
    if not src_images.is_dir():
        raise SystemExit(f"ERROR: missing image directory {src_images}")

    print(f"[{split}] loading {ann_file.name} ...")
    coco = json.loads(ann_file.read_text())

    # image_id -> YOLO label lines
    labels = defaultdict(list)
    counts = Counter()
    img_info = {im["id"]: im for im in coco["images"]}
    for ann in coco["annotations"]:
        cls = CLASS_MAP.get(ann["category_id"])
        if cls is None or ann.get("iscrowd", 0):
            continue
        im = img_info[ann["image_id"]]
        iw, ih = im["width"], im["height"]
        x, y, w, h = ann["bbox"]  # absolute pixels, top-left + size
        # Clip to the image and drop anything degenerate.
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, iw), min(y + h, ih)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
        bw, bh = (x2 - x1) / iw, (y2 - y1) / ih
        labels[ann["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        counts[cls] += 1

    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    n_images = n_background = n_missing = 0
    for image_id, im in img_info.items():
        src = src_images / im["file_name"]
        if not src.exists():  # tolerate a partially extracted image dir, but count it
            n_missing += 1
            continue
        link = dst_images / im["file_name"]
        if not link.is_symlink():
            link.symlink_to(src.resolve())
        lines = labels.get(image_id, [])
        (dst_labels / (Path(im["file_name"]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_images += 1
        n_background += not lines

    per_class = ", ".join(f"{NAMES[i]}: {counts[i]}" for i in range(len(NAMES)))
    print(f"[{split}] {n_images} images ({n_background} background-only, "
          f"{n_missing} listed-but-missing) | {per_class}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", default="dataset",
                        help="dir containing annotations/, train2017/, val2017/")
    parser.add_argument("--dst", default="dataset/coco4", help="output dataset root")
    args = parser.parse_args()

    coco_root = Path(args.coco_root).resolve()
    dst_root = Path(args.dst).resolve()
    for split in SPLITS:
        convert_split(coco_root, dst_root, split)

    yaml_path = dst_root / "data.yaml"
    yaml_path.write_text(
        f"path: {dst_root}\n"
        "train: images/train2017\n"
        "val: images/val2017\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
