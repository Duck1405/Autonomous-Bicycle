"""Train H-Net on TuSimple ground-truth lane points.

H-Net is tiny, so this is a simple single-device trainer (no DDP). Paper setup
(Neven et al., 2018, Sec. III-B): 128x64 input, Adam, batch size 10, lr 5e-5,
3rd-order polynomial, until convergence.

Example:
    python train_hnet.py \
        --tusimple_root /home/anindra/data/TUSimple/train_set \
        --epochs 200 --bs 10 --lr 5e-5 --save ./log_hnet

`--tusimple_root` must contain the clips/ folder and the label_data_*.json files
(the json `raw_file` paths are resolved relative to it).
"""

import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader

from dataloader.hnet_dataset import HNetDataset, hnet_collate
from model.lanenet.backbone.H_Net import H_Net
from model.lanenet.hnet_loss import HNetLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tusimple_root", required=True,
                   help="TuSimple train_set root (contains clips/ and label_data_*.json)")
    p.add_argument("--label_glob", default="label_data_*.json",
                   help="glob (relative to --tusimple_root) matching label json files")
    p.add_argument("--save", default="./log_hnet")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--poly_order", type=int, default=3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def find_label_files(root, label_glob):
    files = sorted(glob.glob(os.path.join(root, label_glob)))
    if not files:
        # tusimple_transform.py copies the jsons into a training/ subfolder.
        files = sorted(glob.glob(os.path.join(root, "**", label_glob), recursive=True))
    if not files:
        raise FileNotFoundError(
            "No label json matching {!r} under {}".format(label_glob, root))
    return files


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save, exist_ok=True)

    label_files = find_label_files(args.tusimple_root, args.label_glob)
    print("Device:", device)
    print("Label files:", label_files)

    dataset = HNetDataset(label_files, image_root=args.tusimple_root,
                          resize_w=args.width, resize_h=args.height)
    print("Training samples:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=hnet_collate,
        pin_memory=device.type == "cuda",
        drop_last=True,                 # keeps BatchNorm1d happy (no size-1 batch)
    )

    model = H_Net().to(device)
    criterion = HNetLoss(order=args.poly_order)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running, nb = 0.0, 0
        for images, batch_lanes in loader:
            images = images.to(device, non_blocking=True)
            params = model(images)
            loss = criterion(params, batch_lanes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            nb += 1

        epoch_loss = running / max(nb, 1)
        print("epoch {}/{}  loss {:.6f}".format(epoch + 1, args.epochs, epoch_loss))

        torch.save(model.state_dict(),
                   os.path.join(args.save, "hnet_epoch_{:03d}.pth".format(epoch + 1)))
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), os.path.join(args.save, "hnet_best.pth"))

    print("Best loss: {:.6f}".format(best))
    print("Saved checkpoints to:", args.save)


if __name__ == "__main__":
    main()
