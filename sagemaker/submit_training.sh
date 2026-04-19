#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./sagemaker/submit_training.sh \
    --role arn:aws:iam::123456789012:role/MySageMakerRole \
    --training-image 123456789012.dkr.ecr.us-west-2.amazonaws.com/autonomous-bicycle-train:latest \
    [--instance-type ml.g6e.2xlarge] \
    [--instance-count 1] \
    [--images-s3-uri s3://autonomous-bike/100k_images/] \
    [--annotations-s3-uri s3://autonomous-bike/100k/] \
    [--job-name autonomous-bicycle-train]

Submits a SageMaker V3 ModelTrainer job using the custom training image.
Additional arguments are forwarded to sagemaker/model_trainer_v3.py.
EOF
}

ROLE=""
TRAINING_IMAGE=""
INSTANCE_TYPE="ml.g6e.2xlarge"
INSTANCE_COUNT="1"
IMAGES_S3_URI="s3://autonomous-bike/100k_images/"
ANNOTATIONS_S3_URI="s3://autonomous-bike/100k/"
JOB_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)
      ROLE="${2:-}"
      shift 2
      ;;
    --training-image)
      TRAINING_IMAGE="${2:-}"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="${2:-}"
      shift 2
      ;;
    --instance-count)
      INSTANCE_COUNT="${2:-}"
      shift 2
      ;;
    --images-s3-uri)
      IMAGES_S3_URI="${2:-}"
      shift 2
      ;;
    --annotations-s3-uri)
      ANNOTATIONS_S3_URI="${2:-}"
      shift 2
      ;;
    --job-name)
      JOB_NAME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$ROLE" || -z "$TRAINING_IMAGE" ]]; then
  echo "--role and --training-image are required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CMD=(
  python
  "${REPO_ROOT}/sagemaker/model_trainer_v3.py"
  --role "${ROLE}"
  --training-image "${TRAINING_IMAGE}"
  --instance-type "${INSTANCE_TYPE}"
  --instance-count "${INSTANCE_COUNT}"
  --images-s3-uri "${IMAGES_S3_URI}"
  --annotations-s3-uri "${ANNOTATIONS_S3_URI}"
)

if [[ -n "$JOB_NAME" ]]; then
  CMD+=(--job-name "${JOB_NAME}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Submitting SageMaker training job:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
