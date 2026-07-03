#!/bin/bash
#SBATCH --job-name=LaneATTresnet34
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda3
source ~/.bashrc
source activate /home/anindra/data/conda/envs/LaneNet310

python main.py train --exp_name LaneATTresnet34Aug --cfg cfgs/laneatt_culane_resnet34.yml

