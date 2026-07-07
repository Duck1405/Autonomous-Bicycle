#!/bin/bash
#SBATCH --job-name=yolo11n
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda3
source ~/.bashrc
source activate yolo11

export PYTHONUNBUFFERED=1

# 4-class COCO from prepare_dataset.py (run it once on the cluster first).
DATA_YAML=/home/anindra/data/ObjectDetection/yolo11Dataset/coco4/data.yaml

# GPU preflight: fail fast (job -> FAILED) if this node can't give us CUDA,
# e.g. the broken MPS daemon (CUDA error 805) that killed jobs 170777/170778.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
echo "=== GPU preflight on $(hostname) ==="
nvidia-smi || exit 1
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot initialize CUDA'; print('CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

# --device 0,1 -> ultralytics DDP across the two requested L40S GPUs
python train.py --size n --data "$DATA_YAML" --device 0,1 --epoch 150
