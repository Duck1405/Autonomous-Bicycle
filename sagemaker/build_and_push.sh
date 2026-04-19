#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./sagemaker/build_and_push.sh \
    --account-id 123456789012 \
    --region us-west-2 \
    [--repository autonomous-bicycle-train] \
    [--tag latest]

Builds the SageMaker training image locally, creates the ECR repository if needed,
logs Docker into ECR, tags the image, and pushes it.
EOF
}

ACCOUNT_ID=""
REGION=""
REPOSITORY="autonomous-bicycle-train"
TAG="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account-id)
      ACCOUNT_ID="${2:-}"
      shift 2
      ;;
    --region)
      REGION="${2:-}"
      shift 2
      ;;
    --repository)
      REPOSITORY="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$ACCOUNT_ID" || -z "$REGION" ]]; then
  echo "--account-id and --region are required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_IMAGE="${REPOSITORY}:${TAG}"
ECR_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}:${TAG}"

echo "Ensuring ECR repository exists: ${REPOSITORY}"
aws ecr describe-repositories \
  --repository-names "${REPOSITORY}" \
  --region "${REGION}" >/dev/null 2>&1 || \
aws ecr create-repository \
  --repository-name "${REPOSITORY}" \
  --region "${REGION}" >/dev/null

echo "Logging Docker into ECR: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building local image: ${LOCAL_IMAGE}"
docker build \
  -f "${REPO_ROOT}/sagemaker/train.Dockerfile" \
  -t "${LOCAL_IMAGE}" \
  "${REPO_ROOT}"

echo "Tagging image: ${ECR_IMAGE}"
docker tag "${LOCAL_IMAGE}" "${ECR_IMAGE}"

echo "Pushing image: ${ECR_IMAGE}"
docker push "${ECR_IMAGE}"

echo
echo "Pushed image URI:"
echo "${ECR_IMAGE}"
