#!/bin/bash
#SBATCH --job-name=lanenet_l40s
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda3
source ~/.bashrc
source activate /home/anindra/data/conda/envs/LaneNet310

torchrun --standalone --nproc_per_node=2 \
    /home/anindra/data/Autonomous-Bicycle/LaneNet/lanenet-lane-detection-pytorch/train.py \
    --dataset /home/anindra/data/TUSimple/train_set/training \
    --epochs 25 \
    --bs 12 \
    --num_workers 16 \
    --lr 0.0001
    --save ./log


python train_hnet.py   --tusimple_root /home/anindra/data/TUSimple/train_set   --epochs 50 --bs 32 --lr 5e-5 --num_workers 32 --poly_order 3 --save .
/log_hnet

