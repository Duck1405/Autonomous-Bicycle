import os
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import sys
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None
from model.utils.cli_helper import parse_args
import numpy as np
import pandas as pd


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


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_dist_initialized() or dist.get_rank() == 0


def rank0_print(message):
    if is_main_process():
        print(message)


def setup_distributed():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("setup_distributed RANK not in OS.Environ")
        sys.exit()
        # return {
        #     'distributed': False,
        #     'rank': 0,
        #     'local_rank': 0,
        #     'world_size': 1,
        #     'device': device,
        # }

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = 'nccl'
        device = torch.device('cuda', local_rank)
       
    else:
        backend = 'gloo'
        device = torch.device('cpu')
        print("Setup Distributed: CPU")
        sys.exit()

    dist.init_process_group(backend=backend, init_method='env://')
    return {
        'distributed': True,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device': device,
    }


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def get_shared_train_dir(save_root):
    save_path = [None]
    if is_main_process():
        save_path[0] = get_next_train_dir(save_root)
    if is_dist_initialized():
        dist.broadcast_object_list(save_path, src=0)
    return save_path[0]


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def train():
    args = parse_args()
    dist_info = setup_distributed()
    device = dist_info['device']
    rank = dist_info['rank']
    local_rank = dist_info['local_rank']
    world_size = dist_info['world_size']
    distributed = dist_info['distributed']


    save_root = args.save
    save_path = get_shared_train_dir(save_root)
    rank0_print("Starting LaneNet training")
    rank0_print("Distributed: {} rank={}/{} local_rank={}".format(distributed, rank, world_size, local_rank))
    rank0_print("Device: {}".format(device))
    rank0_print("Dataset directory: {}".format(args.dataset))
    rank0_print("Save root directory: {}".format(save_root))
    rank0_print("Current training run directory: {}".format(save_path))
    rank0_print("Model type: {}".format(args.model_type))
    rank0_print("Loss type: {}".format(args.loss_type))
    rank0_print("Resize target: {}x{}".format(args.width, args.height))
    rank0_print("Batch size per process: {}".format(args.bs))
    rank0_print("Learning rate: {}".format(args.lr))

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')
    rank0_print("Training index: {}".format(train_dataset_file))
    rank0_print("Validation index: {}".format(val_dataset_file))

    resize_height = args.height
    resize_width = args.width
    rank0_print("Building transforms")

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

    base_seed = 69
    random.seed(base_seed)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed + rank)

    rank0_print("Loading training dataset")
    train_dataset = TusimpleSet(train_dataset_file, joint_transform=train_transform)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=device.type == 'cuda',
    )

    rank0_print("Loading validation dataset")
    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'], target_transform=target_transforms)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=device.type == 'cuda',
    )

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    samplers = {
        'train': train_sampler,
        'val': val_sampler,
    }
    dataset_sizes = {'train': len(train_dataset), 'val' : len(val_dataset)}
    rank0_print("Training samples: {}".format(dataset_sizes['train']))
    rank0_print("Validation samples: {}".format(dataset_sizes['val']))
    rank0_print("Training batches per process per epoch: {}".format(len(train_loader)))
    rank0_print("Validation batches per process per epoch: {}".format(len(val_loader)))

    rank0_print("Building model")
    model = LaneNet(arch=args.model_type, device)
    rank0_print("Moving model to device: {}".format(device))
    model.to(device)

    if distributed:
        if device.type == 'cuda':
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DistributedDataParallel(model)

    rank0_print("Creating optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rank0_print("{} epochs {} training samples\n".format(args.epochs, len(train_dataset)))

    rank0_print("Entering train_model")
    model, log = train_model(
        model,
        optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        loss_type=args.loss_type,
        num_epochs=args.epochs,
        save_path=save_path,
        is_main_process=is_main_process(),
        samplers=samplers,
    )

    if is_main_process():
        print("Training loop finished; writing artifacts")
        df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
        df['epoch'] = log['epoch']
        df['training_loss'] = log['training_loss']
        df['val_loss'] = log['val_loss']

        train_log_save_filename = os.path.join(save_path, 'training_log.csv')
        df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss'], header=True,index=False,encoding='utf-8')
        print("training log is saved: {}".format(train_log_save_filename))

        model_save_filename = os.path.join(save_path, 'best_model.pth')
        torch.save(unwrap_model(model).state_dict(), model_save_filename)
        print("model is saved: {}".format(model_save_filename))

    if is_dist_initialized():
        dist.barrier()

if __name__ == '__main__':
    try:
        train()
    finally:
        cleanup_distributed()
