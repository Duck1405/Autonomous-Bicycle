"""BDD100K per-image JSON adapter for LaneATT.

Use root=<parent containing 100k_images and 100k_json>, or supply
root=<100k_images> and annotations_root=<100k_json>. Evaluation uses
CULane-style polyline IoU, NOT the official BDD100K segmentation metric.
"""
import json
import logging
from pathlib import Path

import numpy as np

from .lane_dataset_loader import LaneDatasetLoader
from lib.lane_attributes import encode_attributes


class BDD100K(LaneDatasetLoader):
    def __init__(self, max_lanes=None, split='train', root=None,
                 annotations_root=None, official_metric=True, multilabel=False):
        if root is None:
            raise ValueError('Please specify the BDD100K root directory')
        if split not in ('train', 'val', 'test'):
            raise ValueError('BDD100K split must be train, val, or test')
        self.multilabel = multilabel
        self.split = split
        root = Path(root)
        self.root = root / '100k_images' if (root / '100k_images').is_dir() else root
        self.annotations_root = (Path(annotations_root) if annotations_root is not None
                                 else self.root.parent / '100k_json')
        self.img_w, self.img_h = 1280, 720
        self.official_metric = official_metric  # raster vs continuous CULane IoU
        self.logger = logging.getLogger(__name__)
        self.annotations = []
        self.load_annotations()
        if max_lanes is not None and max_lanes < self.max_lanes:
            raise ValueError(f'max_lanes={max_lanes} would discard labels; need {self.max_lanes}')
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    @staticmethod
    def _sample_polyline(vertices):
        """Legacy BDD codes describe outgoing segments: CCC + endpoint is cubic."""
        points = []
        i = 0
        while i < len(vertices) - 1:
            if vertices[i][2] == 'C':
                if i + 3 >= len(vertices) or any(v[2] != 'C' for v in vertices[i:i+3]):
                    raise ValueError('Malformed BDD100K cubic polyline')
                p = np.asarray([v[:2] for v in vertices[i:i+4]], dtype=float)
                t = np.linspace(0, 1, 33)[:, None]
                samples = ((1-t)**3*p[0] + 3*(1-t)**2*t*p[1]
                           + 3*(1-t)*t**2*p[2] + t**3*p[3])
                i += 3
            elif vertices[i][2] == 'L':
                p = np.asarray([v[:2] for v in vertices[i:i+2]], dtype=float)
                t = np.linspace(0, 1, max(2, int(np.linalg.norm(p[1]-p[0])/5)+1))[:, None]
                samples = (1-t)*p[0] + t*p[1]
                i += 1
            else:
                raise ValueError(f'Unsupported BDD polyline code: {vertices[i][2]}')
            points.extend(map(tuple, samples))
        return points

    def load_annotation(self, annotation_path, with_attributes=False):
        with open(annotation_path) as handle:
            data = json.load(handle)
        frames = data.get('frames', [])
        if len(frames) != 1 or 'objects' not in frames[0]:
            raise ValueError(f'Expected one labeled frame in {annotation_path}')
        lanes = []
        lane_attributes = []
        for obj in frames[0]['objects']:
            if not obj.get('category', '').startswith('lane/'):
                continue
            if self.multilabel and obj['category'] == 'lane/crosswalk':
                continue
            if obj.get('attributes', {}).get('direction') == 'vertical':
                continue  # cross-road markings are not longitudinal lane boundaries
            vertices = obj.get('poly2d', [])
            points = self._sample_polyline(vertices)
            # LaneATT requires a single x for each y, strictly ordered by y.
            by_y = {}
            for x, y in points:
                if np.isfinite(x) and np.isfinite(y) and 0 <= x < self.img_w and 0 <= y < self.img_h:
                    by_y.setdefault(float(y), float(x))
            lane = [(x, y) for y, x in sorted(by_y.items())]
            if len(lane) >= 2 and lane[-1][1] - lane[0][1] >= 1:
                lanes.append(lane)
                lane_attributes.append(encode_attributes(obj['category'], obj.get('attributes', {})))
        return (lanes, lane_attributes) if with_attributes else lanes

    def load_annotations(self):
        image_dir = self.root / self.split
        label_dir = self.annotations_root / self.split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f'Missing split directories: {image_dir}, {label_dir}')
        self.annotations = []
        self.max_lanes = 1
        for path in sorted(image_dir.glob('*.jpg')):
            label = label_dir / (path.stem + '.json')
            if not label.is_file():
                raise FileNotFoundError(f'Missing annotation for {path}: {label}')
            lanes = self.load_annotation(label)
            self.max_lanes = max(self.max_lanes, len(lanes))
            self.annotations.append({'path': str(path), 'org_path': f'{self.split}/{path.name}',
                                     'annotation_path': str(label)})
        if not self.annotations:
            raise ValueError(f'No JPG images found in {image_dir}')
        self.logger.info('Indexed %d BDD100K images; max_lanes=%d', len(self), self.max_lanes)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        if 'lanes' in anno:
            return anno
        if self.multilabel:
            lanes, attributes = self.load_annotation(anno['annotation_path'], with_attributes=True)
            return dict(anno, lanes=lanes, lane_attributes=attributes)
        return dict(anno, lanes=self.load_annotation(anno['annotation_path']))

    def __len__(self):
        return len(self.annotations)

    def transform_annotations(self, transform):
        self.annotations = [transform(self[i]) for i in range(len(self))]

    def _prediction_points(self, predictions):
        ys = np.arange(self.img_h) / self.img_h
        result = []
        for lane in predictions:
            xs = np.asarray(lane(ys))
            valid = np.isfinite(xs) & (xs >= 0) & (xs < 1)
            result.append(list(zip(xs[valid]*self.img_w, ys[valid]*self.img_h)))
        return result

    def get_metrics(self, lanes, idx):
        from utils.culane_metric import culane_metric
        pred = self._prediction_points(lanes)
        valid = [i for i, lane in enumerate(pred) if len(lane) >= 2]
        _, fp, fn, ious, matches = culane_metric(
            [pred[i] for i in valid], self[idx]['lanes'],
            official=self.official_metric, img_shape=(self.img_h, self.img_w, 3))
        all_ious, all_matches = np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
        all_ious[valid], all_matches[valid] = ious, matches
        return fp + len(pred)-len(valid), fn, all_matches, all_ious

    def eval_predictions(self, predictions, output_basedir):
        if len(predictions) != len(self):
            raise ValueError('Expected one prediction per dataset image')
        output_dir = Path(output_basedir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tp = fp = fn = 0
        with open(output_dir / 'bdd100k_laneatt_predictions.jsonl', 'w') as handle:
            for idx, pred in enumerate(predictions):
                image_fp, image_fn, matches, _ = self.get_metrics(pred, idx)
                tp += int(matches.sum())
                fp += image_fp
                fn += image_fn
                record = {'raw_file': self.annotations[idx]['org_path'],
                          'lanes': self._prediction_points(pred)}
                if self.multilabel:
                    keys = ('lane_category', 'lane_color', 'lane_style',
                            'image_side', 'attribute_scores')
                    record['lane_attributes'] = [
                        {key: lane.metadata[key] for key in keys if key in lane.metadata}
                        for lane in pred]
                handle.write(json.dumps(record) + '\n')
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        return {'Precision': precision, 'Recall': recall,
                'F1': 2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.}


# Preserve the name used by the original unfinished file.
bdd100k = BDD100K
