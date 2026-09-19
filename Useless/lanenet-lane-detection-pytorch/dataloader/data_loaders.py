# coding: utf-8
'''
Code is referred from https://github.com/klintan/pytorch-lanenet
delete the one-hot representation for instance output
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import random


class TusimpleSet(Dataset):
    def __init__(self, dataset, n_labels=3, transform=None, target_transform=None, joint_transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.n_labels = n_labels

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        # load all

        img = Image.open(self._gt_img_list[idx]).convert('RGB')
        label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)

        if self.joint_transform:
            image = np.array(img)
            binary_mask = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
            binary_mask = (binary_mask > 0).astype(np.uint8)

            augmented = self.joint_transform(
                image=image,
                binary_mask=binary_mask,
                instance_mask=label_instance_img,
            )

            return augmented['image'], augmented['binary_mask'], augmented['instance_mask']

        # optional transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label_img = self.target_transform(label_img)
            label_instance_img = self.target_transform(label_instance_img)

        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, label_binary, label_instance_img


class _CULaneBase(Dataset):
    """Lazy CULane image/point reader shared by the two training contracts."""

    SPLIT_FILES = {
        'train': 'list/train.txt', 'val': 'list/val.txt',
        'test': 'list/test.txt', 'test2': 'list/test2.txt',
        'normal': 'list/test_split/test0_normal.txt',
        'crowd': 'list/test_split/test1_crowd.txt',
        'hlight': 'list/test_split/test2_hlight.txt',
        'shadow': 'list/test_split/test3_shadow.txt',
        'noline': 'list/test_split/test4_noline.txt',
        'arrow': 'list/test_split/test5_arrow.txt',
        'curve': 'list/test_split/test6_curve.txt',
        'cross': 'list/test_split/test7_cross.txt',
        'night': 'list/test_split/test8_night.txt',
        'debug': 'list/debug.txt',
    }

    def __init__(self, root, split='train', list_file=None):
        self.root = os.fspath(root)
        self.split = split
        if list_file is None:
            if split not in self.SPLIT_FILES:
                raise ValueError('Unknown CULane split: {!r}'.format(split))
            list_file = self.SPLIT_FILES[split]
        self.list_file = os.fspath(list_file)
        if not os.path.isabs(self.list_file):
            self.list_file = os.path.join(self.root, self.list_file)

        # CULane paths beginning with '/' are still relative to the dataset.
        # *_gt.txt is also accepted: only its image column is needed because
        # supervision is generated from .lines.txt, not the optional PNG/flags.
        with open(self.list_file, 'r') as file:
            self.image_paths = [os.path.join(self.root, line.split()[0].lstrip('/'))
                                for line in file if line.strip()]
        if not self.image_paths:
            raise ValueError('No images listed in {}'.format(self.list_file))
        # Preserve list order; DataLoader/DistributedSampler owns shuffling.

    def __len__(self):
        return len(self.image_paths)

    def _load_sample(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError('Cannot read CULane image: {}'.format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation_path = os.path.splitext(image_path)[0] + '.lines.txt'
        lanes = []
        with open(annotation_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    values = [float(value) for value in line.split()]
                    if len(values) % 2:
                        raise ValueError('expected x/y pairs')
                    points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
                    if not np.isfinite(points).all():
                        raise ValueError('non-finite coordinates')
                except ValueError as exc:
                    raise ValueError('{}:{}: {}'.format(
                        annotation_path, line_number, exc)) from exc
                # Match the LaneATT reference: discard negative points,
                # deduplicate, sort by y, and discard lanes with <2 points.
                points = points[(points >= 0).all(axis=1)]
                points = np.unique(points, axis=0)
                if len(points) >= 2:
                    lanes.append(points[np.argsort(points[:, 1], kind='stable')])
        return image, lanes


class CULane(_CULaneBase):
    """CULane supervision with the same return contract as ``TusimpleSet``.

    Returns ``(image, binary_mask, instance_mask)``. Before transforms, image
    is RGB PIL and masks are HxW uint8: binary values 0/1 and instance values
    0 (background), 1, 2, ... (one ID per lane, not one-hot channels).
    Labels are rasterized lazily from each image's .lines.txt at ORIGINAL
    resolution using ``lane_width`` pixels (default 16); they are not exact
    copies of the distributed laneseg_label_w16 PNGs.

    Use train.py's ``build_train_transform`` as joint_transform. It must
    register binary_mask and instance_mask as mask targets so geometric
    augmentation stays aligned and mask interpolation is nearest-neighbor.
    Alternatively use an image transform and ``Rescale((width, height))`` as
    target_transform, like TuSimple validation. Image-only transforms must
    not introduce unpaired random geometry. With no transforms the image
    remains PIL, matching TusimpleSet (not ready for default batch collation).
    """

    def __init__(self, root, split='train', transform=None, target_transform=None,
                 joint_transform=None, lane_width=16, list_file=None):
        super().__init__(root, split, list_file)
        if not isinstance(lane_width, int) or isinstance(lane_width, bool) or lane_width < 1:
            raise ValueError('lane_width must be a positive integer')
        self.lane_width = lane_width
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def __getitem__(self, idx):
        image, lanes = self._load_sample(idx)
        if len(lanes) > 255:
            raise ValueError('More than 255 lanes cannot fit in a uint8 instance mask')
        instance_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lane_id, points in enumerate(lanes, 1):
            cv2.polylines(instance_mask, [np.rint(points).astype(np.int32)],
                          isClosed=False, color=lane_id,
                          thickness=self.lane_width, lineType=cv2.LINE_8)
        binary_mask = (instance_mask > 0).astype(np.uint8)
        if self.joint_transform is not None:
            augmented = self.joint_transform(
                image=image, binary_mask=binary_mask, instance_mask=instance_mask)
            return augmented['image'], augmented['binary_mask'], augmented['instance_mask']

        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            binary_mask = self.target_transform(binary_mask)
            instance_mask = self.target_transform(instance_mask)
        return image, binary_mask, instance_mask


class CULaneHNetDataset(_CULaneBase):
    """CULane equivalent of HNetDataset; batch with hnet_dataset.hnet_collate.

    Returns an ImageNet-normalized float32 RGB tensor (3, 64, 128) and a list
    of float32 (M, 2) lane tensors containing [x/original_width, y/original_height].
    Points are not quantized to the segmentation mask. As in the LaneATT
    reader, positive points beyond the image edge are retained, so normalized
    coordinates can slightly exceed 1. Empty annotations return an empty list.
    HNetLoss decides which lanes have enough points for its polynomial order.
    No geometric augmentation is applied, keeping images and points aligned.
    The current H_Net fully connected layer requires the default 128x64 size.
    """

    def __init__(self, root, split='train', list_file=None, resize_w=128, resize_h=64,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__(root, split, list_file)
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __getitem__(self, idx):
        image, lanes = self._load_sample(idx)
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.resize_w, self.resize_h),
                           interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).contiguous()
        scale = np.asarray([width, height], dtype=np.float32)
        lanes = [torch.from_numpy(points / scale) for points in lanes]
        return image, lanes
