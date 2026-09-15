"""Run from LaneATT: python -m unittest discover -s tests -p test_bdd100k_loader.py"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from lib.datasets.bdd100k import BDD100K
from lib.datasets.lane_dataset_loader import LaneDatasetLoader


class BDD100KTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        images = self.root / '100k_images/train'
        labels = self.root / '100k_json/train'
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / 'sample.jpg').touch()
        objects = [
            {'category': 'lane/single white', 'attributes': {'direction': 'parallel'},
             'poly2d': [[100, 100, 'L'], [100, 600, 'L']]},
            {'category': 'lane/single white', 'attributes': {'direction': 'vertical'},
             'poly2d': [[0, 300, 'L'], [1000, 300, 'L']]},
            {'category': 'area/drivable', 'poly2d': []},
            {'category': 'car', 'box2d': {}}]
        self.label = labels / 'sample.json'
        self.label.write_text(json.dumps({'frames': [{'objects': objects}]}))

    def test_interface_and_filtering(self):
        d = BDD100K(root=self.root)
        self.assertIsInstance(d, LaneDatasetLoader)
        self.assertEqual(len(d), 1)
        self.assertEqual(len(d[0]['lanes']), 1)
        self.assertEqual(d.max_lanes, 1)
        self.assertTrue(np.all(np.diff(np.array(d[0]['lanes'][0])[:, 1]) > 0))
        self.assertEqual(len(BDD100K(root=self.root/'100k_images')), 1)

    def test_cubic(self):
        p = BDD100K._sample_polyline([[0, 0, 'C'], [0, 100, 'C'],
                                     [100, 100, 'C'], [100, 200, 'L']])
        np.testing.assert_allclose(p[0], [0, 0])
        np.testing.assert_allclose(p[-1], [100, 200])
        np.testing.assert_allclose(p[16], [50, 100])

    def test_missing_labels_fail(self):
        self.label.unlink()
        with self.assertRaises(FileNotFoundError):
            BDD100K(root=self.root)

    def test_empty_and_matching_predictions(self):
        d = BDD100K(root=self.root)
        self.assertEqual(d.get_metrics([], 0)[:2], (0, 1))
        def lane(ys):
            return np.where((ys*720 >= 100) & (ys*720 <= 600), 100/1280, -1.)
        fp, fn, matches, _ = d.get_metrics([lane], 0)
        self.assertEqual((fp, fn), (0, 0))
        self.assertTrue(matches[0])
        result = d.eval_predictions([[lane]], self.root/'output')
        self.assertEqual(result['F1'], 1.)


if __name__ == '__main__':
    unittest.main()
