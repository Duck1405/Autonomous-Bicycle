"""Strict checkpoint audit and reproducible video visualization; no training."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def tensor(frame, size):
    rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
    return (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]


def predict(model, frame, size):
    start = time.perf_counter()
    logits = model(tensor(frame, size))
    if not torch.isfinite(logits).all():
        raise RuntimeError('Nonfinite logits')
    labels = F.interpolate(logits, size=(size[1], size[0]), mode='bilinear', align_corners=False).argmax(1)[0].numpy().astype(np.uint8)
    return labels, {'seconds': time.perf_counter() - start,
                    'logit_range': [float(logits.min()), float(logits.max())],
                    'road_fraction': float((labels == 0).mean()),
                    'classes': [CLASSES[i] for i in np.unique(labels)]}


def panel(frame, labels, title):
    img = cv2.resize(frame, (640, 360))
    if labels is not None:
        mask = cv2.resize(labels, (640, 360), interpolation=cv2.INTER_NEAREST) == 0
        img[mask] = (img[mask] * .55 + np.array([0, 255, 0]) * .45).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (640, 32), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_image(path, img):
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f'Cannot write {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=str(ROOT.parents[1] / 'LaneATT/video_input/IMG_5105.mp4'))
    parser.add_argument('--frames', type=int, default=180, help='Consecutive source frames; 0 means entire video')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/checkpoint_validation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok or fps <= 0:
        raise RuntimeError('Cannot decode source video')
    report = {'video': args.video, 'source_fps': fps, 'source_frames': count,
              'source_size': list(first.shape[:2]), 'device': 'cpu',
              'preprocessing': 'RGB / 255, ImageNet mean/std',
              'label_mapping': 'Standard Cityscapes train IDs; names not embedded in checkpoints',
              'checkpoints': {}, 'clip': {}, 'samples': []}
    models = {}
    with torch.inference_mode():
        for slim in [True, False]:
            arch = 'DDRNet_23_slim' if slim else 'DDRNet_23'
            filename = 'DDRNet23s_imagenet.pth' if slim else 'DDRNet23_imagenet.pth'
            mod = module(ROOT.parent / 'classification' / f'{arch}.py', f'classification_{arch}')
            m = mod.get_model().eval()
            state = torch.load(ROOT / 'models' / filename, map_location='cpu', weights_only=True)
            m.load_state_dict(state, strict=True)
            result = m(tensor(first, (224, 224)))
            assert result.shape == (1, 1000) and torch.isfinite(result).all()
            report['checkpoints'][filename] = {'architecture': f'classification/{arch}.py', 'strict_load': True, 'video_frame_output_shape': list(result.shape), 'purpose': 'ImageNet classification; no road mask'}
            filename = 'best_val_smaller.pth' if slim else 'best_val.pth'
            mod = module(ROOT / f'{arch}.py', f'segmentation_{arch}')
            m = mod.DualResNet(mod.BasicBlock, [2]*4, planes=32 if slim else 64, spp_planes=128, head_planes=64 if slim else 128, augment=False).eval()
            state = torch.load(ROOT / 'models' / filename, map_location='cpu', weights_only=True)
            # Only explicitly identified training-only parameters are excluded.
            removed = [k for k in state if k.startswith('model.seghead_extra.') or k.startswith('loss.')]
            clean = {k.removeprefix('model.'): v for k, v in state.items() if k not in removed}
            m.load_state_dict(clean, strict=True)
            models[arch] = m
            report['checkpoints'][filename] = {'architecture': f'segmentation/{arch}.py', 'strict_load': True, 'excluded_training_keys': removed, 'purpose': '19-class semantic segmentation'}
            print(filename, 'strict load passed', flush=True)
        dest = args.output / 'road_comparison.mp4'
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 360))
        if not writer.isOpened():
            raise RuntimeError('Cannot open output writer')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        timings = {k: [] for k in models}
        n = 0
        try:
            while args.frames == 0 or n < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                panels = [panel(frame, None, f'Original | {n / fps:.2f}s')]
                for name, m in models.items():
                    labels, stats = predict(m, frame, (1024, 512))
                    timings[name].append(stats['seconds'])
                    panels.append(panel(frame, labels, name + ' | road in green'))
                joined = np.concatenate(panels, axis=1)
                writer.write(joined)
                if n in [0, 90, 179]:
                    save_image(args.output / f'clip_frame_{n:04d}.jpg', joined)
                n += 1
                if n % 30 == 0:
                    print('Rendered frames', n, flush=True)
        finally:
            writer.release()
        report['clip'] = {'frames': n, 'inference_size': [512, 1024], 'duration_seconds': n/fps, 'mean_inference_and_postprocess_seconds': {k: float(np.mean(v)) for k, v in timings.items()}}
        for index in np.linspace(0, count-1, 9).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {index}')
            panels = [panel(frame, None, f'Original | {index/fps:.2f}s')]
            sample = {'frame': int(index), 'seconds': index/fps, 'inference_size': [1024, 2048]}
            for name, m in models.items():
                labels, stats = predict(m, frame, (2048, 1024))
                sample[name] = stats
                panels.append(panel(frame, labels, name + ' | road in green'))
                save_image(args.output / f'labels_{name}_{index:05d}.png', labels)
            save_image(args.output / f'sample_{index:05d}.jpg', np.concatenate(panels, axis=1))
            report['samples'].append(sample)
            print('Full-resolution sample', index, flush=True)
    cap.release()
    check = cv2.VideoCapture(str(dest))
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        assert frame.shape[:2] == (360, 1920)
        decoded += 1
    check.release()
    assert decoded == n and n > 0, (decoded, n)
    report['output_video_decoded_frames'] = decoded
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print('Verified', decoded, 'output frames;', dest, flush=True)


if __name__ == '__main__':
    main()
