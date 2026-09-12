"""Run an explicitly selected LaneATT checkpoint through the shared video pipeline."""
import argparse
from pathlib import Path
import time
from lib.config import Config
from lib.video import VideoInference
import torch


def video_inference(MODELS, files, frame_limit=1000, output_root=Path('video_output_4'),
                    yolo_conf=0.2, nms_thres=50, nms_topk=4,
                    conf_threshold=0.5, keep_threshold=0.3,
                    yolo_iou=0.15, match_tolerance=0.05, device=None):
    

    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    files = [Path(files)] if isinstance(files, (str, Path)) else [Path(f) for f in files]
    if not files:
        raise ValueError('Provide at least one input video')
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    for config_path, path_model, path_yolo in MODELS:
        for path in (config_path, path_model, path_yolo):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        cfg = Config(config_path)
        checkpoint = Path(path_model)
        output_folder = Path(output_root) / checkpoint.parent.parent.name / checkpoint.stem
        print(f'Checkpoint: {checkpoint} -> {output_folder}')
        # Each tuple uses its own architecture, checkpoint and YOLO weights.
        video = VideoInference(
            model_archiecture=cfg.get_model(), model_path=str(checkpoint),
            frame_limit=frame_limit, output_folder=output_folder, device=device,
            view=False, yolo_path=str(path_yolo), yolo_conf=yolo_conf,
            yolo_iou=yolo_iou, nms_thres=nms_thres, nms_topk=nms_topk,
            conf_threshold=conf_threshold, keep_threshold=keep_threshold,
            match_tolerance=match_tolerance)
        start = time.perf_counter()
        for file in files:
            video.set_video_path(str(file))
            video.video_eval()
        print(f'{checkpoint}: {time.perf_counter() - start:.1f} s')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('experiments/LaneATTresnet34Aug2'))
    parser.add_argument('--epoch', required=True, type=int, help='Exact checkpoint epoch to load')
    parser.add_argument('--config', type=Path, help='Defaults to the experiment config.yaml')
    parser.add_argument('--videos', nargs='+', type=Path, required=True)
    parser.add_argument('--yolo', type=Path, default=Path('onnxmodels/YolloS/yolo11s_coco4.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('video_inference'))
    parser.add_argument('--frame-limit', type=int, default=99999)
    parser.add_argument('--yolo-conf', type=float, default=0.7)
    parser.add_argument('--yolo-iou', type=float, default=0.15)
    parser.add_argument('--nms-thres', type=float, default=50)
    parser.add_argument('--nms-topk', type=int, default=8)
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    parser.add_argument('--keep-threshold', type=float, default=0.3)
    parser.add_argument('--match-tolerance', type=float, default=0.05)
    parser.add_argument('--device', help='For example cpu or cuda:0; default auto-selects')
    args = parser.parse_args(argv)
    if args.epoch < 1:
        parser.error('--epoch must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.experiment / 'models' / f'model_{args.epoch:04d}.pt'
    config = args.config or args.experiment / 'config.yaml'
    video_inference(
        [(config, checkpoint, args.yolo)], args.videos,
        frame_limit=args.frame_limit, output_root=args.output_dir,
        yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou,
        nms_thres=args.nms_thres, nms_topk=args.nms_topk,
        conf_threshold=args.conf_threshold, keep_threshold=args.keep_threshold,
        match_tolerance=args.match_tolerance, device=args.device)


if __name__ == '__main__':
    main()
