from __future__ import annotations

import argparse
from typing import Any

from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import Compute, InputData, StoppingCondition


DEFAULT_IMAGE_S3_URI = "s3://autonomous-bike/100k_images/"
DEFAULT_ANNOTATION_S3_URI = "s3://autonomous-bike/100k/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker V3 training job for train.py using a custom Docker image."
    )
    parser.add_argument("--role", required=True, help="IAM role ARN used by the SageMaker training job.")
    parser.add_argument("--training-image", required=True, help="ECR image URI for the custom training container.")
    parser.add_argument("--job-name", default=None, help="Optional explicit training job name.")
    parser.add_argument("--instance-type", default="ml.g6e.2xlarge", help="SageMaker training instance type.")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of SageMaker training instances.")
    parser.add_argument("--volume-size-gb", type=int, default=100, help="Attached EBS volume size in GB.")
    parser.add_argument("--max-runtime-seconds", type=int, default=86400, help="Maximum job runtime in seconds.")
    parser.add_argument("--images-s3-uri", default=DEFAULT_IMAGE_S3_URI, help="S3 prefix containing image split folders.")
    parser.add_argument(
        "--annotations-s3-uri",
        default=DEFAULT_ANNOTATION_S3_URI,
        help="S3 prefix containing annotation split folders.",
    )
    parser.add_argument(
        "--input-mode",
        default="FastFile",
        choices=["File", "FastFile"],
        help="SageMaker channel input mode.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--backbone", default="resnet101", choices=["resnet50", "resnet101"])
    parser.add_argument("--base-lr", type=float, default=0.007)
    parser.add_argument("--max-iters", type=int, default=30000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-power", type=float, default=0.9)
    parser.add_argument("--train-hflip-prob", type=float, default=0.5)
    parser.add_argument("--save-file", default="final_model.pth")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="val")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-sync-bn", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--disable-auto-discover-classes", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--category-map-s3-uri", default=None, help="Optional S3 URI to a category_map.json object.")
    return parser


def build_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    hyperparameters: dict[str, Any] = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "test-batch-size": args.test_batch_size,
        "num-workers": args.num_workers,
        "image-height": args.image_height,
        "image-width": args.image_width,
        "backbone": args.backbone,
        "base-lr": args.base_lr,
        "max-iters": args.max_iters,
        "weight-decay": args.weight_decay,
        "momentum": args.momentum,
        "lr-power": args.lr_power,
        "train-hflip-prob": args.train_hflip_prob,
        "save-file": args.save_file,
        "train-split": args.train_split,
        "test-split": args.test_split,
        "amp": str(not args.disable_amp).lower(),
        "sync-bn": str(not args.disable_sync_bn).lower(),
        "progress": str(not args.disable_progress).lower(),
        "no-auto-discover-classes": str(args.disable_auto_discover_classes).lower(),
    }
    if args.pretrained_backbone:
        hyperparameters["pretrained-backbone"] = "true"
    if args.max_train_samples is not None:
        hyperparameters["max-train-samples"] = args.max_train_samples
    if args.max_test_samples is not None:
        hyperparameters["max-test-samples"] = args.max_test_samples
    if args.category_map_s3_uri:
        hyperparameters["category-map-path"] = "/opt/ml/input/data/category_map/category_map.json"
    return hyperparameters


def main() -> None:
    args = build_parser().parse_args()

    trainer = ModelTrainer(
        base_job_name=args.job_name or "autonomous-bicycle-train",
        training_image=args.training_image,
        role=args.role,
        training_input_mode=args.input_mode,
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            volume_size_in_gb=args.volume_size_gb,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=args.max_runtime_seconds,
        ),
        hyperparameters=build_hyperparameters(args),
    )

    input_data_config = [
        InputData(
            channel_name="images",
            data_source=args.images_s3_uri,
        ),
        InputData(
            channel_name="annotations",
            data_source=args.annotations_s3_uri,
        ),
    ]
    if args.category_map_s3_uri:
        input_data_config.append(
            InputData(
                channel_name="category_map",
                data_source=args.category_map_s3_uri,
            )
        )

    training_job = trainer.train(
        input_data_config=input_data_config,
    )

    print(f"Started training job: {training_job.name}")


if __name__ == "__main__":
    main()
