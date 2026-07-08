import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
    
    
def get_files(segment_folder):
    return [path for path in segment_folder.iterdir() if path.is_file()]
def get_directories(cipo_path):
    return [path for path in cipo_path.iterdir() if path.is_dir()]
def remove_first_parent(path: Path) -> Path:
    return Path(*path.parts[1:])
def check_dir(check):
    if check.is_dir():
        print(f"Directory Found: {check}")
        return True
    else:
        print(f"Directory Not Found: {check}")
        return False
def check_file(check):
    if check.is_file():
        print(f"File Found: {check}")
    else:
        print("File Not Found")
    


def split_dataset(coco_root, dst_root):
    print("split_dataset")
    print(coco_root)
    print(dst_root)
    annotation = coco_root / "annotations"
    train2017 = coco_root / "train2017"
    val2017 = coco_root / "val2017"
    if check_dir(annotation) == False:
        sys.exit()
    if check_dir(train2017) == False:
        sys.exit()
    if check_dir(val2017) == False:
        sys.exit()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="dataset",help="dir containing annotations/, train2017/, val2017/")
    parser.add_argument("--dst", default="dataset/coco4", help="output dataset root")
    args = parser.parse_args()
    coco_root = Path(args.coco_root).resolve()
    print(f"coco_root: {coco_root}")
    dst_root = Path(args.dst).resolve()
    print(f"dst_root: {dst_root}")
    split_dataset(coco_root, dst_root)

if __name__ == "__main__":
    main()