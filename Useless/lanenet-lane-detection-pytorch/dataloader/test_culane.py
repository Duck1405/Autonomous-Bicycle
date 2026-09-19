"""Run from the project root: python -m unittest dataloader.test_culane."""

import tempfile
import unittest
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader.data_loaders import CULane, CULaneHNetDataset
from dataloader.hnet_dataset import hnet_collate
from dataloader.transformers import Rescale
from model.lanenet.backbone.H_Net import H_Net
from model.lanenet.hnet_loss import HNetLoss
from model.lanenet.train_lanenet import compute_loss


class CULaneContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'list').mkdir()
        (self.root / 'driver').mkdir()
        for name in ('lane', 'empty'):
            image = np.full((40, 80, 3), (10, 20, 200), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(self.root / 'driver' / (name + '.png')), image))
        self.annotation = self.root / 'driver/lane.lines.txt'
        self.annotation.write_text(
            '20 30 12 10 -2 5 16 20 20 30 8 0\n'
            '60 30 56 20 52 10 48 0\n'
            '5 5\n')
        (self.root / 'driver/empty.lines.txt').write_text('')
        (self.root / 'list/train.txt').write_text(
            '/driver/lane.png\n\ndriver/empty.png\n')

    def test_masks_and_rgb_and_empty_sample(self):
        dataset = CULane(self.root, lane_width=1)
        image, binary, instances = dataset[0]
        self.assertEqual(image.getpixel((0, 0)), (200, 20, 10))
        self.assertEqual(instances[20, 16], 1)
        self.assertEqual(instances[20, 56], 2)
        self.assertEqual(set(np.unique(instances)), {0, 1, 2})
        np.testing.assert_array_equal(binary, instances > 0)
        self.assertFalse(dataset[1][1].any())
        self.assertFalse(dataset[1][2].any())

    def test_joint_geometry_and_training_loss(self):
        joint = A.Compose([
            A.HorizontalFlip(p=1), A.Resize(20, 40),
            A.Normalize(), ToTensorV2(),
        ], additional_targets={'binary_mask': 'mask', 'instance_mask': 'mask'})
        dataset = CULane(self.root, lane_width=3, joint_transform=joint)
        images, binary, instances = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(images.shape), (2, 3, 20, 40))
        self.assertEqual(images.dtype, torch.float32)
        self.assertTrue(torch.equal(binary.bool(), instances > 0))
        original = CULane(self.root, lane_width=3)[0][2]
        expected = cv2.resize(original[:, ::-1], (40, 20), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(instances[0].numpy(), expected)
        logits = torch.randn(2, 2, 20, 40, requires_grad=True)
        embedding = torch.randn(2, 4, 20, 40, requires_grad=True)
        loss = compute_loss({'binary_seg_logits': logits, 'instance_embedding': embedding,
                             'binary_seg_pred': logits.argmax(1)},
                            binary.long(), instances.float())[0]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(embedding.grad).all())

    def test_separate_validation_transforms_and_gt_list(self):
        gt_list = self.root / 'list/val_gt.txt'
        gt_list.write_text('/driver/lane.png /unused-mask.png 1 1 0 0\n')
        dataset = CULane(self.root, list_file='list/val_gt.txt',
                         transform=transforms.Compose([
                             transforms.Resize((20, 40)), transforms.ToTensor()]),
                         target_transform=Rescale((40, 20)))
        image, binary, instances = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 20, 40))
        self.assertEqual(binary.shape, (20, 40))
        np.testing.assert_array_equal(binary, instances > 0)

    def test_hnet_coordinates_collation_and_backward(self):
        dataset = CULaneHNetDataset(self.root)
        image, lanes = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 64, 128))
        torch.testing.assert_close(lanes[0], torch.tensor([
            [8 / 80, 0], [12 / 80, 10 / 40],
            [16 / 80, 20 / 40], [20 / 80, 30 / 40]]))
        expected_rgb = (torch.tensor([200, 20, 10]) / 255 -
                        torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(image[:, 0, 0], expected_rgb)
        images, batch_lanes = next(iter(DataLoader(dataset, batch_size=2, collate_fn=hnet_collate)))
        self.assertEqual(batch_lanes[1], [])
        model = H_Net()
        loss = HNetLoss()(model(images), batch_lanes)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.regressor[-1].weight.grad).all())

    def test_malformed_and_missing_annotations_fail(self):
        dataset = CULane(self.root)
        for annotation in ('1 2 3', 'nan 1 2 3'):
            self.annotation.write_text(annotation)
            with self.assertRaisesRegex(ValueError, 'lane.lines.txt:1:'):
                dataset[0]
        self.annotation.unlink()
        with self.assertRaises(FileNotFoundError):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
