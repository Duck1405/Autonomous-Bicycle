import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment
import sys
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument("mode", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Explicit checkpoint epoch for evaluation/video inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to pickle file")
    parser.add_argument("--view", choices=["all", "mistakes"], help="Show predictions")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")
    args = parser.parse_args()
    # if args.cfg is None and args.mode == "train":
    #     raise Exception("If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
    # if args.resume and args.mode == "test":
    #     raise Exception("args.resume is set on `test` mode: can't resume testing")
    # if args.epoch is not None and args.mode == 'train':
    #     raise Exception("The `epoch` parameter should not be set when training")
    # if args.view is not None and args.mode != "test":
    #     raise Exception('Visualization is only available during evaluation')

    return args


'''

python main.py train --exp_name Testing --cfg /Users/amannindra/Projects/Auto/Autonomous-Bicycle/LaneATT/cfgs/laneatt_culane_resnet18_laptop.yml
python main.py train --exp_name LaneATTresnet34Aug2Test --cfg /Users/amannindra/Projects/Auto/Autonomous-Bicycle/LaneATT/cfgs/laneatt_culane_resnet34_test.yml --epoch 1 --cpu 28 --view all
python main.py train --exp_name LaneATTresnet34Final --cfg /Users/amannindra/Projects/Auto/Autonomous-Bicycle/LaneATT/cfgs/laneatt_culane_resnet34_new.yml --view all

'''
def main():
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
        
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')

        
    if device.type != "cuda" and not args.cpu:
        # Exit non-zero so SLURM reports FAILED — a bare sys.exit() exits 0 and the
        # scheduler marks a job that never trained as COMPLETED.
        sys.exit(f"ERROR: no usable GPU (torch.cuda.is_available() is False), device would be '{device}'. "
                 "Check nvidia-smi / the node's MPS daemon, or pass --cpu to run on CPU deliberately.")
    
        
    runner = Runner(cfg, exp, device, view=args.view, resume=args.resume, deterministic=args.deterministic)

    if args.mode == 'train':
        try:
            runner.train()
        except KeyboardInterrupt:
           logging.info('Training interrupted.')
    if args.mode == 'test':
        runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)
    
    conf_threshold = 0.5
    nms_thres = 50
    nms_topk = 4
    match_tolerance = 0.05
    keep_threshold = 0.3
    
    VIDEO_DIR = Path("/home/anindra/data/Autonomous-Bicycle/LaneATT/video_input")
    OUTPUT_DIR = Path(exp.results_dirpath) / "video_inference"
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    path_video = VIDEO_DIR
    output_folder = OUTPUT_DIR
    # After training, always select by validation F1 rather than --epoch.
    if args.mode == 'train' and exp.get_best_validation_epoch() is None:
        last_epoch = exp.get_last_checkpoint_epoch()
        if last_epoch < 1:
            raise RuntimeError('No saved checkpoint is available for video inference')
        runner.eval(last_epoch, on_val=True)
    runner.get_video_inference(conf_threshold = conf_threshold,  nms_thres = nms_thres, nms_topk = nms_topk, path_video = path_video, output_folder = output_folder, epoch=args.epoch if args.mode == 'test' else None)

if __name__ == '__main__':
    main()
