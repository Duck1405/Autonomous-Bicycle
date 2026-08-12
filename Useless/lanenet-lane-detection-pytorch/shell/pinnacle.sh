#!/bin/bash
#SBATCH --job-name=lanenet_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


module load anaconda3

conda init

conda activate LaneNet310

python /home/anindra/data/Autonomous-Bicycle/LaneNet/lanenet-lane-detection-pytorch/train.py \
    --dataset /home/anindra/data/TUSimple/train_set/training \
    --epochs 25 \
    --bs 12 \
    --save ./log