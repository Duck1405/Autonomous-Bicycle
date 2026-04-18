"""Convert all .heic/.HEIC images in a folder to .jpg.

Usage:
    python heic_to_jpg.py [--input test_images] [--output test_images_jpg]
                          [--quality 95] [--delete-original]

Dependencies:
    pip install pillow pillow-heif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ExifTags

try:
    from pillow_heif import register_heif_opener
except ImportError:
    sys.stderr.write(
        "Missing dependency 'pillow-heif'. Install it with:\n"
        "    pip install pillow-heif\n"
    )
    sys.exit(1)

register_heif_opener()


def convert_file(src: Path, dst: Path, quality: int) -> None:
    with Image.open(src) as img:
        img = _apply_exif_orientation(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"quality": quality, "optimize": True}
        exif = img.info.get("exif")
        if exif:
            save_kwargs["exif"] = exif
        img.save(dst, format="JPEG", **save_kwargs)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Rotate image according to EXIF orientation so JPEG appears upright."""
    try:
        exif = img.getexif()
        if not exif:
            return img
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None:
            return img
        orientation = exif.get(orientation_key)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert HEIC images to JPG.")
    parser.add_argument(
        "--input", "-i", type=Path, default=Path("test_images"),
        help="Folder containing .heic/.HEIC files (default: test_images)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output folder for .jpg files (default: same as input)",
    )
    parser.add_argument(
        "--quality", "-q", type=int, default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    parser.add_argument(
        "--delete-original", action="store_true",
        help="Delete the .heic file after successful conversion",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .jpg files",
    )
    args = parser.parse_args()

    in_dir: Path = args.input
    out_dir: Path = args.output or in_dir

    if not in_dir.is_dir():
        sys.stderr.write(f"Input folder not found: {in_dir}\n")
        return 1

    heic_files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".heic"
    )
    if not heic_files:
        print(f"No .heic files found in {in_dir}")
        return 0

    print(f"Found {len(heic_files)} HEIC file(s) in {in_dir}")
    converted = skipped = failed = 0

    for src in heic_files:
        dst = out_dir / (src.stem + ".jpg")
        if dst.exists() and not args.overwrite:
            print(f"  [skip] {src.name} -> {dst.name} (already exists)")
            skipped += 1
            continue
        try:
            convert_file(src, dst, args.quality)
            print(f"  [ok]   {src.name} -> {dst.name}")
            converted += 1
            if args.delete_original:
                src.unlink()
        except Exception as exc:
            print(f"  [fail] {src.name}: {exc}")
            failed += 1

    print(
        f"\nDone. Converted: {converted}, Skipped: {skipped}, Failed: {failed}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
