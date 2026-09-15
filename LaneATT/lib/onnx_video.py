"""ONNX Runtime backends for the shared VideoInference video pipeline.

LaneATT expects raw (1, N, 77) proposals. YOLO expects the project's four-class
export with embedded NMS, (1, N, 6); raw YOLO heads need a different decoder.
CUDA remains the default; --provider cpu enables explicit local verification.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import onnxruntime as ort

from .video import DecodedYoloDrawing, VideoInference
from jetson_tools.postprocess import LaneHysteresis, laneatt_decode, yolo_decode
from jetson_tools.preprocess import pre_laneatt, pre_yolo_meta

DEFAULT_MODELS = {
    'laneatt': 'LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx',
    'yolo': 'LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx',
}


def parse_args(label, argv=None):
    parser = argparse.ArgumentParser(
        description=f'{label} ONNX inference with shared video.py visualization.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', type=Path, default=Path('Videos2/IMG_6893_30fps.mp4'))
    parser.add_argument('--onnx', type=Path, default=Path(DEFAULT_MODELS[label]))
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--start-frame', type=int, default=0, help='zero-based starting frame')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--render', type=Path,
                        default=Path(f'LaneATT/jetson_video/{label}_onnx.mp4'),
                        help='side-by-side original and annotated output video')
    parser.add_argument('--no-render', action='store_true', help='time inference without decoding or saving video')
    parser.add_argument('--codec', default='mp4v')
    parser.add_argument('--json', type=Path, default=Path('Benchmark'), help='benchmark JSON directory')
    parser.add_argument('--provider', choices=['cuda', 'cpu'], default='cuda',
                        help='explicit ONNX Runtime execution provider; no silent CUDA-to-CPU fallback')
    if label == 'laneatt':
        parser.add_argument('--no-hysteresis', action='store_true',
                            help='skip temporal acceptance before shared ego-lane selection')
    else:
        parser.add_argument('--conf', type=float, default=0.35,
                            help='additional confidence filter after embedded NMS')
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.start_frame < 0 or args.warmup < 0:
        parser.error('--frames must be positive; --start-frame and --warmup must be non-negative')
    if label == 'yolo' and not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    return args


def run(label, args):
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    provider = 'CUDAExecutionProvider' if args.provider == 'cuda' else 'CPUExecutionProvider'
    if provider not in ort.get_available_providers():
        raise RuntimeError(f'{provider} unavailable; install a compatible ONNX Runtime or explicitly use --provider cpu')
    providers = [provider] if args.provider == 'cpu' else [provider, 'CPUExecutionProvider']
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    active = session.get_providers()
    if provider not in active:
        raise RuntimeError(f'Requested {provider}, but session activated {active}')
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError('Expected a single-input, single-output ONNX model')
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    expected_input = [1, 3, 360, 640] if label == 'laneatt' else [1, 3, 640, 640]
    if inp.type != 'tensor(float)' or len(inp.shape) != 4 or any(
            isinstance(dim, int) and dim != expected for dim, expected in zip(inp.shape, expected_input)):
        raise ValueError(f'Expected float32 input {expected_input}, got {inp.shape} {inp.type}')
    width = 77 if label == 'laneatt' else 6
    if len(out.shape) != 3 or isinstance(out.shape[-1], int) and out.shape[-1] != width:
        raise ValueError(f'Expected {label} output (1, N, {width}), got {out.shape}; '
                         'YOLO rendering requires the export with embedded NMS')
    print(f'onnxruntime {ort.__version__}, providers {active}')
    print(f'{label}: {args.onnx}\n    in {inp.name} {inp.shape}\n    out {out.name} {out.shape}')
    pipeline = VideoInference(video_path=str(args.video), frame_limit=args.frames,
                              model_path=args.onnx, device=f'ONNX Runtime/{provider}', initialize_models=False)
    if label == 'yolo':
        pipeline.yolo = DecodedYoloDrawing()
    hysteresis = None
    if label == 'laneatt' and not args.no_hysteresis:
        hysteresis = LaneHysteresis(conf_threshold=pipeline.conf_threshold,
                                    keep_threshold=pipeline.keep_threshold,
                                    match_tolerance=pipeline.match_tolerance)
    render = not args.no_render
    t_pre = t_eng = t_render = 0.0

    def preprocess(frame):
        if label == 'laneatt':
            return pre_laneatt(frame), None
        arr, r, dx, dy = pre_yolo_meta(frame)
        return arr, (r, dx, dy)

    def warmup(first):
        arr, _ = preprocess(first)
        for _ in range(args.warmup):
            session.run([out.name], {inp.name: arr})

    def process(frame):
        nonlocal t_pre, t_eng, t_render
        t0 = time.perf_counter()
        arr, meta = preprocess(frame)
        t_pre += time.perf_counter() - t0
        t0 = time.perf_counter()
        raw = session.run([out.name], {inp.name: arr})[0]
        t_eng += time.perf_counter() - t0
        if not render:
            return None
        t0 = time.perf_counter()
        evaluation, boxes = [], []
        if label == 'laneatt':
            lanes = laneatt_decode(raw, conf_threshold=pipeline.keep_threshold,
                                   nms_thres=pipeline.nms_thres, nms_topk=pipeline.nms_topk)
            if hysteresis is not None:
                lanes = hysteresis(lanes)
            evaluation = [SimpleNamespace(points=lane['points'], metadata={'conf': lane['conf']})
                          for lane in lanes]
        else:
            boxes = yolo_decode(raw, *meta, conf_threshold=args.conf)
        annotated = pipeline.get_frame(frame, evaluation=evaluation, yolo_results=boxes)
        t_render += time.perf_counter() - t0
        return annotated

    stats = pipeline.video_eval(start_frame=args.start_frame, output_path=args.render,
                                codec=args.codec, frame_processor=process, warmup=warmup, render=render)
    done = stats['frames']
    elapsed = stats['elapsed_seconds']
    def ms(seconds):
        return 1000 * seconds / done
    def fps(seconds):
        return done / seconds if seconds > 0 else 0.0
    result = {
        'frames': done, 'video': str(args.video), 'runtime': 'onnxruntime',
        'onnxruntime': ort.__version__, 'providers': active, 'model': str(args.onnx),
        'start_frame': args.start_frame, 'warmup': args.warmup, 'render': stats['output'],
        'read_ms': ms(stats['read_seconds']),
        'models': {label: {'preprocess_ms': ms(t_pre), 'engine_ms': ms(t_eng)}},
        'engines_only_ms': ms(t_eng), 'engines_only_fps': fps(t_eng),
        'render_ms': ms(t_render + stats['write_seconds']),
        'pipeline_ms': ms(elapsed), 'pipeline_fps': fps(elapsed),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        folder = args.json.parent if args.json.is_file() else args.json
        folder.mkdir(parents=True, exist_ok=True)
        n = 1
        while (folder / f'Benchmark_{n}.json').exists():
            n += 1
        path = folder / f'Benchmark_{n}.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Wrote {path}')
    return result
