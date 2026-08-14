#!/bin/bash

source /home/mlc/miniconda3/etc/profile.d/conda.sh
conda activate LaneNet310

python trt_video_benchmark.py --laneatt-on False --video Videos2/IMG_2635.mp4 --no-render &
python trt_video_benchmark.py --yolo-on False --video Videos2/IMG_2635.mp4 --no-render &

wait


