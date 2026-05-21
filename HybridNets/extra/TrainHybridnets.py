import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = REPO_ROOT / "projects"
DEFAULT_PROJECT_TEMPLATE = "bdd100k"
DEFAULT_PROJECT_NAME = "bdd100k_sagemaker"

CHANNEL_ALIASES = {
    "images": ("SM_CHANNEL_IMAGES",),
    "annotations": (
        "SM_CHANNEL_ANNOTATIONS",
        "SM_CHANNEL_DET_ANNOT",
        "SM_CHANNEL_DET_ANNOTATIONS",
    ),
    "drivable_masks": (
        "SM_CHANNEL_DRIVABLE_MASKS",
        "SM_CHANNEL_DRIVABLE",
        "SM_CHANNEL_DA_SEG_ANNOT",
        "SM_CHANNEL_DA_SEG",
        "SM_CHANNEL_DA_SEG_MASKS",
    ),
    "lane_masks": (
        "SM_CHANNEL_LANE_MASKS",
        "SM_CHANNEL_LANE",
        "SM_CHANNEL_LL_SEG_ANNOT",
        "SM_CHANNEL_LL_SEG",
        "SM_CHANNEL_LL_SEG_MASKS",
    ),
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def default_num_gpus():
    env_value = first_env("SM_NUM_GPUS")
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        "Train HybridNets on SageMaker BDD100K channels"
    )
    parser.add_argument(
        "--project-template",
        default=DEFAULT_PROJECT_TEMPLATE,
        help="Base YAML template from projects/<name>.yml",
    )
    parser.add_argument(
        "--project-name",
        default=DEFAULT_PROJECT_NAME,
        help="Generated project name written under projects/<name>.yml",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="val")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--compound-coef", type=int, default=3)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--freeze-backbone", type=parse_bool, default=False)
    parser.add_argument("--freeze-det", type=parse_bool, default=False)
    parser.add_argument("--freeze-seg", type=parse_bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--es-min-delta", type=float, default=0.0)
    parser.add_argument("--es-patience", type=int, default=0)
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--debug", type=parse_bool, default=False)
    parser.add_argument("--cal-map", type=parse_bool, default=True)
    parser.add_argument("--verbose", type=parse_bool, default=True)
    parser.add_argument("--plots", type=parse_bool, default=True)
    parser.add_argument("--num-gpus", type=int, default=default_num_gpus())
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--amp", type=parse_bool, default=False)
    parser.add_argument(
        "--images-root",
        default=first_env(*CHANNEL_ALIASES["images"]),
        help="Root channel directory containing train/ and val/ images",
    )
    parser.add_argument(
        "--annotations-root",
        default=first_env(*CHANNEL_ALIASES["annotations"]),
        help="Root channel directory containing train/ and val/ detection JSON",
    )
    parser.add_argument(
        "--drivable-masks-root",
        default=first_env(*CHANNEL_ALIASES["drivable_masks"]),
        help="Root channel directory containing train/ and val/ drivable PNG masks",
    )
    parser.add_argument(
        "--lane-masks-root",
        default=first_env(*CHANNEL_ALIASES["lane_masks"]),
        help="Root channel directory containing train/ and val/ lane PNG masks",
    )
    parser.add_argument(
        "--saved-path",
        default=os.environ.get("SM_MODEL_DIR", str(REPO_ROOT / "checkpoints")),
        help="Checkpoint/model output root",
    )
    parser.add_argument(
        "--log-path",
        default=os.environ.get(
            "SM_OUTPUT_DATA_DIR", str(REPO_ROOT / "checkpoints")
        ),
        help="TensorBoard/log output root",
    )
    return parser


def require_dir(path_str, description):
    if not path_str:
        raise ValueError(f"Missing required {description}")
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{description} path does not exist: {path}")
    return path.resolve()


def require_split_dirs(root, split, description):
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{description} split directory does not exist: {split_dir}"
        )
    return split_dir


def count_top_level_files(path):
    try:
        return sum(1 for child in path.iterdir() if child.is_file())
    except FileNotFoundError:
        return 0


def load_template(project_template):
    template_path = PROJECTS_DIR / f"{project_template}.yml"
    if not template_path.exists():
        raise FileNotFoundError(f"Project template not found: {template_path}")
    with template_path.open("r", encoding="utf-8") as handle:
        return template_path, yaml.safe_load(handle)


def write_project_config(args):
    template_path, config = load_template(args.project_template)
    dataset = config.setdefault("dataset", {})
    model = config.setdefault("model", {})
    dataset["dataroot"] = str(require_dir(args.images_root, "images root"))
    dataset["labelroot"] = str(
        require_dir(args.annotations_root, "annotations root")
    )
    dataset["segroot"] = [
        str(require_dir(args.drivable_masks_root, "drivable masks root")),
        str(require_dir(args.lane_masks_root, "lane masks root")),
    ]
    dataset["train_set"] = args.train_split
    dataset["test_set"] = args.test_split
    model["image_size"] = [args.image_width, args.image_height]

    for root_label, root_path in (
        ("images", Path(dataset["dataroot"])),
        ("annotations", Path(dataset["labelroot"])),
        ("drivable_masks", Path(dataset["segroot"][0])),
        ("lane_masks", Path(dataset["segroot"][1])),
    ):
        train_dir = require_split_dirs(root_path, args.train_split, root_label)
        test_dir = require_split_dirs(root_path, args.test_split, root_label)
        print(
            f"{root_label}: train={train_dir} ({count_top_level_files(train_dir)} files), "
            f"val={test_dir} ({count_top_level_files(test_dir)} files)"
        )

    project_path = PROJECTS_DIR / f"{args.project_name}.yml"
    with project_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    output_metadata = {
        "template": str(template_path),
        "generated_project": str(project_path),
        "dataset": dataset,
        "model": model,
    }

    model_dir = Path(args.saved_path)
    log_dir = Path(args.log_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_path, model_dir / project_path.name)
    with (log_dir / "hybridnets_sagemaker_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output_metadata, handle, indent=2)

    return project_path


def build_train_command(args):
    cmd = [
        sys.executable,
        "train.py",
        "-p",
        args.project_name,
        "-c",
        str(args.compound_coef),
        "-n",
        str(args.num_workers),
        "-b",
        str(args.batch_size),
        "--freeze_backbone",
        str(args.freeze_backbone),
        "--freeze_det",
        str(args.freeze_det),
        "--freeze_seg",
        str(args.freeze_seg),
        "--lr",
        str(args.lr),
        "--optim",
        args.optim,
        "--num_epochs",
        str(args.num_epochs),
        "--val_interval",
        str(args.val_interval),
        "--save_interval",
        str(args.save_interval),
        "--es_min_delta",
        str(args.es_min_delta),
        "--es_patience",
        str(args.es_patience),
        "--log_path",
        args.log_path,
        "--saved_path",
        args.saved_path,
        "--debug",
        str(args.debug),
        "--cal_map",
        str(args.cal_map),
        "--verbose",
        str(args.verbose),
        "--plots",
        str(args.plots),
        "--num_gpus",
        str(args.num_gpus),
        "--conf_thres",
        str(args.conf_thres),
        "--iou_thres",
        str(args.iou_thres),
        "--amp",
        str(args.amp),
    ]
    if args.backbone:
        cmd.extend(["-bb", args.backbone])
    if args.load_weights:
        cmd.extend(["-w", args.load_weights])
    return cmd


def main():
    parser = build_parser()
    args = parser.parse_args()

    project_path = write_project_config(args)
    print(f"Generated SageMaker project config: {project_path}")

    train_cmd = build_train_command(args)
    print("Launching HybridNets training command:")
    print(" ".join(train_cmd))

    completed = subprocess.run(train_cmd, cwd=REPO_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
