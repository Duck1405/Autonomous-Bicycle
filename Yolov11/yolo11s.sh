#!/bin/bash
#SBATCH --job-name=yolo11s
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda3
source ~/.bashrc
source activate yolo11

export PYTHONUNBUFFERED=1

# GPU preflight: fail fast (job -> FAILED) if this node can't give us CUDA,
# e.g. the broken MPS daemon (CUDA error 805) that killed jobs 170777/170778.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
echo "=== GPU preflight on $(hostname) ==="
nvidia-smi || exit 1
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot initialize CUDA'; print('CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

python train.py --size s
