# SageMaker V3 training for `train.py`

This folder contains a custom SageMaker training path for the top-level `train.py` script.

## What is included

- `train.Dockerfile`: custom GPU training image using Python 3.10 and PyTorch 2.8
- `requirements.train.txt`: minimal Python dependencies required by `train.py`
- `train_entrypoint.py`: maps SageMaker channels and runtime paths into `train.py` arguments
- `model_trainer_v3.py`: launches a SageMaker Python SDK V3 `ModelTrainer` job with your image URI

## Expected S3 layout

The training script expects each root to contain split folders:

- `s3://autonomous-bike/100k_images/train/*.jpg`
- `s3://autonomous-bike/100k_images/val/*.jpg`
- `s3://autonomous-bike/100k/train/*.json`
- `s3://autonomous-bike/100k/val/*.json`

These are mounted by SageMaker as separate channels:

- `images` -> `/opt/ml/input/data/images`
- `annotations` -> `/opt/ml/input/data/annotations`

## Build and push the image

Replace the placeholders with your AWS account and region.

Helper script:

```bash
chmod +x sagemaker/build_and_push.sh

./sagemaker/build_and_push.sh \
  --account-id <account> \
  --region <region> \
  --repository autonomous-bicycle-train \
  --tag latest
```

Manual commands:

```bash
aws ecr create-repository --repository-name autonomous-bicycle-train

aws ecr get-login-password --region <region> | \
docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

docker build -f sagemaker/train.Dockerfile \
  -t autonomous-bicycle-train:latest \
  .

docker tag autonomous-bicycle-train:latest \
  <account>.dkr.ecr.<region>.amazonaws.com/autonomous-bicycle-train:latest

docker push <account>.dkr.ecr.<region>.amazonaws.com/autonomous-bicycle-train:latest
```

## Launch training with SageMaker V3

Install the SageMaker SDK in the environment where you submit jobs:

```bash
pip install sagemaker
```

Then launch the job:

Helper script:

```bash
chmod +x sagemaker/submit_training.sh

./sagemaker/submit_training.sh \
  --role arn:aws:iam::<account>:role/<sagemaker-execution-role> \
  --training-image <account>.dkr.ecr.<region>.amazonaws.com/autonomous-bicycle-train:latest \
  --instance-type ml.g6e.2xlarge \
  --instance-count 1 \
  --epochs 5 \
  --batch-size 4 \
  --test-batch-size 4 \
  --num-workers 4
```

Direct Python command:

```bash
python sagemaker/model_trainer_v3.py \
  --role arn:aws:iam::<account>:role/<sagemaker-execution-role> \
  --training-image <account>.dkr.ecr.<region>.amazonaws.com/autonomous-bicycle-train:latest \
  --instance-type ml.g6e.2xlarge \
  --instance-count 1 \
  --images-s3-uri s3://autonomous-bike/100k_images/ \
  --annotations-s3-uri s3://autonomous-bike/100k/ \
  --epochs 5 \
  --batch-size 4 \
  --test-batch-size 4 \
  --num-workers 4
```

## Notes

- `train.py` supports multi-GPU on a single instance via local DDP. If you use a multi-GPU instance, make sure `--batch-size` and `--test-batch-size` are divisible by the visible GPU count.
- The training script writes checkpoints to `SM_MODEL_DIR` and metrics/reports to `SM_OUTPUT_DATA_DIR`, which SageMaker captures automatically.
- This setup uses a custom training image. It does not rely on SageMaker script mode.
- Recommended first run: `ml.g6e.2xlarge`, `instance_count=1`, `image_height=360`, `image_width=640`, `batch_size=4`, `test_batch_size=4`, `num_workers=4`. This keeps you on a single GPU with 48 GiB of GPU memory, which clears the `>=32 GB` requirement without forcing multi-GPU DDP. Move to a larger single instance before attempting multi-instance training because `train.py` is written for single-node DDP, not multi-node orchestration.
