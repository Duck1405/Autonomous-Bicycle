import argparse
import json
import shutil
import tarfile
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare small HybridNets SageMaker smoke-test assets")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("/home/aman/Projects/Auto/Autonomous-Bicycle/100k_images"),
        help="Source image root with train/ and val/ directories.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("/home/aman/Projects/Auto/HybridNets/data/nonzip"),
        help="Source label root containing bdd_seg_gt/ and bdd_lane_gt/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/aman/Projects/Auto/HybridNets/data/smoke_subset"),
        help="Destination directory for the small smoke-test dataset.",
    )
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--val-count", type=int, default=16)
    return parser.parse_args()


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def copy_split(
    split: str,
    limit: int,
    images_root: Path,
    road_root: Path,
    lane_root: Path,
    output_root: Path,
) -> list[str]:
    source_images = images_root / split
    source_road = road_root / split
    source_lane = lane_root / split

    dest_images = output_root / "images" / split
    dest_road = output_root / "nonzip" / "bdd_seg_gt" / split
    dest_lane = output_root / "nonzip" / "bdd_lane_gt" / split

    for directory in (dest_images, dest_road, dest_lane):
        directory.mkdir(parents=True, exist_ok=True)

    selected_stems: list[str] = []
    for lane_mask_path in sorted(source_lane.glob("*.png")):
        if len(selected_stems) >= limit:
            break
        stem = lane_mask_path.stem
        image_path = find_image_for_stem(source_images, stem)
        road_mask_path = source_road / f"{stem}.png"
        if image_path is None or not road_mask_path.exists():
            continue

        shutil.copy2(image_path, dest_images / image_path.name)
        shutil.copy2(road_mask_path, dest_road / road_mask_path.name)
        shutil.copy2(lane_mask_path, dest_lane / lane_mask_path.name)
        selected_stems.append(stem)

    if len(selected_stems) != limit:
        raise RuntimeError(f"Requested {limit} {split} samples, but only found {len(selected_stems)} matching records.")
    return selected_stems


def build_tar(source_dir: Path, tar_path: Path) -> None:
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source_dir, arcname=source_dir.name)


def main() -> None:
    args = parse_args()
    images_root = args.images_root.expanduser().resolve()
    labels_root = args.labels_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    road_root = labels_root / "bdd_seg_gt"
    lane_root = labels_root / "bdd_lane_gt"

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_stems = copy_split("train", args.train_count, images_root, road_root, lane_root, output_root)
    val_stems = copy_split("val", args.val_count, images_root, road_root, lane_root, output_root)

    tar_path = output_root / "hybridNets_data_smoke.tar"
    build_tar(output_root / "nonzip", tar_path)

    manifest = {
        "images_root": str((output_root / "images").resolve()),
        "bundle_tar": str(tar_path.resolve()),
        "train_count": len(train_stems),
        "val_count": len(val_stems),
        "train_stems": train_stems,
        "val_stems": val_stems,
        "suggested_s3_images_prefix": "s3://autonomous-bike/data/hybridnets_smoke/images/",
        "suggested_s3_bundle_object": "s3://autonomous-bike/data/hybridnets_smoke/hybridNets_data_smoke.tar",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print("\nUpload commands:")
    print(f"aws s3 sync {output_root / 'images'} s3://autonomous-bike/data/hybridnets_smoke/images/ --no-progress --only-show-errors")
    print(f"aws s3 cp {tar_path} s3://autonomous-bike/data/hybridnets_smoke/hybridNets_data_smoke.tar --no-progress --only-show-errors")


if __name__ == "__main__":
    main()
