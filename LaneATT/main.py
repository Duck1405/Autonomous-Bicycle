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
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
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
    runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)
    
    conf_threshold = 0.5
    nms_thres = 50
    nms_topk = 4
    
    PROJECT_ROOT = Path(__file__).resolve().parent   # main.py lives at LaneATT/
    VIDEO_DIR = PROJECT_ROOT / "video_input"          # folder; runner picks 1/2/3.mp4
    OUTPUT_DIR = PROJECT_ROOT / f"video_output_{cfg.get_model_name()}"
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    path_video = VIDEO_DIR
    output_folder = OUTPUT_DIR
    runner.get_video_inference(conf_threshold = conf_threshold,  nms_thres = nms_thres, nms_topk = nms_topk, path_video = path_video, output_folder = output_folder)
    


if __name__ == '__main__':
    main()
