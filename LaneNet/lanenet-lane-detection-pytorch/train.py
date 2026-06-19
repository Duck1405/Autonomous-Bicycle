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

from model.utils.cli_helper import parse_args
from model.eval_function import Eval_Score
import sys
import numpy as np
import pandas as pd
import cv2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"You are using Device: {DEVICE} ")



def train():
    args = parse_args()
    save_path = args.save
    print("Starting LaneNet training")
    print("Dataset directory: {}".format(args.dataset))
    print("Save directory: {}".format(save_path))
    print("Model type: {}".format(args.model_type))
    print("Loss type: {}".format(args.loss_type))
    print("Resize target: {}x{}".format(args.width, args.height))
    print("Batch size: {}".format(args.bs))
    print("Learning rate: {}".format(args.lr))

    if not os.path.isdir(save_path):
        print("Creating save directory: {}".format(save_path))
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')
    print("Training index: {}".format(train_dataset_file))
    print("Validation index: {}".format(val_dataset_file))

    resize_height = args.height
    resize_width = args.width
    print("Building transforms")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ]),
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
    train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'], target_transform=target_transforms)
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
    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, loss_type=args.loss_type, num_epochs=args.epochs)
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
