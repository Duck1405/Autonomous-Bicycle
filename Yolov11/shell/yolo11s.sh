#!/bin/bash
#SBATCH --job-name=yolo11s
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 250 epochs of yolo11s ≈ 42-50 h on one L40S. If the job dies at the
# walltime anyway, resubmit continuing from the checkpoint:
#   python train.py --size s --data "$DATA_YAML" --resume runs/yolo11s_coco4/weights/last.pt

module load anaconda3
source ~/.bashrc
source activate yolo

export PYTHONUNBUFFERED=1

# 4-class COCO from prepare_dataset.py (run it once on the cluster first).
# Do NOT train on yolo11Dataset/data.yaml — that's the Roboflow 80-class yaml
# whose train split has zero label files (killed jobs 174789-174791)

# DATA_YAML=/home/anindra/data/Autonomous-Bicycle/Yolov11/dataset/data.yaml
DATA_YAML=/home/anindra/data/ObjectDetection/Bdd100k/Bdd100kFinal/data.yaml

# GPU preflight: fail fast (job -> FAILED) if this node can't give us CUDA,
# e.g. the broken MPS daemon (CUDA error 805) that killed jobs 170777/170778.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
echo "=== GPU preflight on $(hostname) ==="
nvidia-smi || exit 1
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot initialize CUDA'; print('CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

python train.py --size s --data "$DATA_YAML" --device "0" --workers 20 --epochs 250