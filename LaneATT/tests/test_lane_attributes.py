import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from lib.datasets import LaneDataset
from lib.lane_attributes import NUM_ATTRIBUTES, encode_attributes, decode_attributes
from lib.models.laneatt import LaneATT


class AttributeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        freq = torch.zeros(2784)
        freq[1395:1403] = torch.arange(1, 9).float()
        cls.freq_path = cls.root/'anchors.pt'
        torch.save(freq, cls.freq_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, enabled):
        return LaneATT(backbone='resnet18', pretrained_backbone=False, img_w=128,
                       img_h=64, S=8, anchor_feat_channels=4,
                       anchors_freq_path=str(self.freq_path), topk_anchors=8,
                       multilabel=enabled)

    def test_encoding_and_unknowns(self):
        a = encode_attributes('lane/single yellow', {'style': 'dashed'})
        self.assertEqual(a[1], 1)
        self.assertEqual(a[-1], 1)
        self.assertEqual(a.sum(), 2)
        self.assertTrue((encode_attributes('lane/road curb', {})[-2:] == -1).all())
        meta = decode_attributes(np.maximum(a, 0)*.9, [[.1, .1], [.2, .5], [.8, 1.]])
        self.assertEqual((meta['lane_color'], meta['lane_style'], meta['image_side']),
                         ('yellow', 'dashed', 'left'))

    def test_crop_preserves_attributes(self):
        root = self.root/'data'
        for folder in ['100k_images/train', '100k_json/train']:
            (root/folder).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root/'100k_images/train/a.jpg'), np.zeros((720, 1280, 3), np.uint8))
        objects = [
            {'category': 'lane/single white', 'attributes': {'style': 'solid'},
             'poly2d': [[50, 200, 'L'], [50, 650, 'L']]},
            {'category': 'lane/double yellow', 'attributes': {'style': 'dashed'},
             'poly2d': [[1000, 200, 'L'], [1000, 650, 'L']]},
            {'category': 'lane/crosswalk', 'attributes': {},
             'poly2d': [[900, 200, 'L'], [900, 650, 'L']]}]
        (root/'100k_json/train/a.json').write_text(json.dumps({'frames':[{'objects': objects}]}))
        dataset = LaneDataset(dataset='bdd100k', root=root, multilabel=True,
                              augmentations=[{'name':'Crop','parameters':
                                  {'x_min':640,'y_min':0,'x_max':1280,'y_max':720}}])
        img, labels, _ = dataset[0]
        positive = labels[labels[:, 1] == 1]
        self.assertEqual(img.shape, (3, 360, 640))
        self.assertEqual(positive.shape, (1, 77+NUM_ATTRIBUTES))
        np.testing.assert_array_equal(positive[0,77:], encode_attributes('lane/double yellow', {'style':'dashed'}))
        self.assertEqual(len(dataset.label_to_lanes(labels)), 1)

    def test_training_and_decode_both_modes(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = self.model(enabled).eval()
                self.assertEqual(hasattr(model, 'attribute_layer'), enabled)
                output = model(torch.rand(1,3,64,128), nms_thres=0, nms_topk=8, conf_threshold=0.)
                self.assertEqual(len(output[0]), 5 if enabled else 4)
                torch.testing.assert_close(output[0][1], model.anchors[output[0][3]])
                targets = model.anchors[0:1].clone()
                targets[:, :2] = torch.tensor([0.,1.])
                targets[:, 4] = 8
                if enabled:
                    attr = torch.tensor(encode_attributes('lane/single yellow', {'style':'solid'}))[None]
                    targets = torch.cat((targets, attr),1)
                loss, metrics = model.loss(output, targets[None])
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(metrics['batch_positives'],0)
                loss.backward()
                self.assertGreater(model.cls_layer.weight.grad.abs().sum().item(),0)
                if enabled:
                    self.assertGreater(model.attribute_layer.weight.grad.abs().sum().item(),0)
                # Make valid decoded proposals and confident known attributes.
                proposals = model.anchors[:1].clone()
                proposals[:, :2] = torch.tensor([-4.,4.]); proposals[:,4]=8
                entry=(proposals,model.anchors[:1],torch.zeros(1,8),torch.tensor([0]))
                if enabled:
                    logits=torch.full((1,NUM_ATTRIBUTES),-8.)
                    logits[0,1]=8.; logits[0,-2]=8.
                    entry=(*entry,logits)
                lanes=model.decode([entry],as_lanes=True)[0]
                self.assertTrue(lanes)
                self.assertEqual('lane_color' in lanes[0].metadata,enabled)
                if enabled:
                    self.assertEqual(lanes[0].metadata['lane_color'],'yellow')
                # Decoding is repeatable and does not mutate stored logits.
                lanes2=model.decode([entry],as_lanes=True)[0]
                np.testing.assert_allclose(lanes[0].points,lanes2[0].points)
                # Same-mode checkpoint loading stays strict and complete.
                restored=self.model(enabled)
                restored.load_state_dict(model.state_dict())

    def test_nms_keeps_original_anchor_identity(self):
        model = self.model(True)
        proposals = model.anchors.clone().requires_grad_()
        # Only anchor 3 survives confidence filtering. Local filtered index 0
        # must never select original anchor 0 or its attribute logits.
        with torch.no_grad():
            proposals[:, :2] = torch.tensor([5., -5.])
            proposals[3, :2] = torch.tensor([-5., 5.])
            proposals[:, 4] = 8
        result = model.nms(proposals[None], torch.eye(8)[None], 0, 4, .5)[0]
        self.assertEqual(result[3].tolist(), [3])
        torch.testing.assert_close(result[1], model.anchors[3:4])
        result[0].sum().backward()
        self.assertEqual(proposals.grad[3].sum().item(), proposals.shape[1])
        self.assertEqual(proposals.grad[0].sum().item(), 0.)

    def test_empty_and_unknown_targets(self):
        model=self.model(True).eval()
        output=model(torch.rand(1,3,64,128),conf_threshold=1.)
        targets=torch.zeros(1,1,13+NUM_ATTRIBUTES)
        targets[:,:,13:]=-1
        loss,_=model.loss(output,targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.decode(output,as_lanes=True),[[]])
        output=model(torch.rand(1,3,64,128),nms_topk=8)
        targets=model.anchors[:1].clone(); targets[:,:2]=torch.tensor([0.,1.]); targets[:,4]=8
        targets=torch.cat((targets,torch.full((1,NUM_ATTRIBUTES),-1.)),1)[None]
        loss,metrics=model.loss(output,targets)
        self.assertEqual(metrics['attribute_loss'].item(),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
