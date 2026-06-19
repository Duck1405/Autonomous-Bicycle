import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

from model.utils.cli_helper import parse_args
from model.eval_function import Eval_Score
import sys
import numpy as np
import pandas as pd
import cv2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"You are using Device: {DEVICE} ")


def get_next_train_dir(save_root):
    if not os.path.isdir(save_root):
        print("Creating save root directory: {}".format(save_root))
        os.makedirs(save_root)

    run_numbers = []
    for name in os.listdir(save_root):
        path = os.path.join(save_root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith('train_'):
            continue

        run_number = name.split('train_', 1)[1]
        if run_number.isdigit():
            run_numbers.append(int(run_number))

    next_run_number = max(run_numbers, default=0) + 1
    train_dir = os.path.join(save_root, 'train_{}'.format(next_run_number))
    os.makedirs(train_dir)
    return train_dir


def _build_random_shadow():
    try:
        return A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.4,
        )
    except TypeError:
        return A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=5,
            p=0.4,
        )


def _build_random_fog():
    try:
        return A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.15)
    except TypeError:
        return A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15)


def _build_random_sun_flare():
    try:
        return A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0.5, 1.0),
            p=0.1,
        )
    except TypeError:
        return A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0.5,
            p=0.1,
        )


def _build_gauss_noise():
    try:
        return A.GaussNoise(std_range=(0.03, 0.08), p=0.2)
    except TypeError:
        return A.GaussNoise(var_limit=(10, 50), p=0.2)


def _build_image_compression():
    try:
        return A.ImageCompression(quality_range=(75, 95), p=0.2)
    except TypeError:
        return A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2)


def build_train_transform(resize_height, resize_width):
    if A is None or ToTensorV2 is None:
        raise ImportError(
            "albumentations is required for training augmentation. "
            "Install albumentations and albumentations[pytorch] in the training environment."
        )

    return A.Compose([
        A.Resize(height=resize_height, width=resize_width),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-10, 10),
            p=0.5,
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        _build_random_shadow(),
        _build_random_fog(),
        _build_random_sun_flare(),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        _build_gauss_noise(),
        _build_image_compression(),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'binary_mask': 'mask', 'instance_mask': 'mask'})



def train():
    args = parse_args()
    save_root = args.save
    save_path = get_next_train_dir(save_root)
    print("Starting LaneNet training")
    print("Dataset directory: {}".format(args.dataset))
    print("Save root directory: {}".format(save_root))
    print("Current training run directory: {}".format(save_path))
    print("Model type: {}".format(args.model_type))
    print("Loss type: {}".format(args.loss_type))
    print("Resize target: {}x{}".format(args.width, args.height))
    print("Batch size: {}".format(args.bs))
    print("Learning rate: {}".format(args.lr))

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')
    print("Training index: {}".format(train_dataset_file))
    print("Validation index: {}".format(val_dataset_file))

    resize_height = args.height
    resize_width = args.width
    print("Building transforms")

    train_transform = build_train_transform(resize_height, resize_width)
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    print("Loading training dataset")
    train_dataset = TusimpleSet(train_dataset_file, joint_transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    print("Loading validation dataset")
    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'], target_transform=target_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}
    print("Training samples: {}".format(dataset_sizes['train']))
    print("Validation samples: {}".format(dataset_sizes['val']))
    print("Training batches per epoch: {}".format(len(train_loader)))
    print("Validation batches per epoch: {}".format(len(val_loader)))

    print("Building model")
    model = LaneNet(arch=args.model_type)
    print("Moving model to device: {}".format(DEVICE))
    model.to(DEVICE)

    print("Creating optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")

    print("Entering train_model")
    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, loss_type=args.loss_type, num_epochs=args.epochs, save_path=save_path)
    print("Training loop finished; writing artifacts")
    df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))

if __name__ == '__main__':
    train()
