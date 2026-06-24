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

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--poly_order", type=int, default=3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of data held out for validation (default 0.1)")
    return p.parse_args()


def find_label_files(root, label_glob):
    files = sorted(glob.glob(os.path.join(root, label_glob)))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", label_glob), recursive=True))
    if not files:
        raise FileNotFoundError(
            "No label json matching {!r} under {}".format(label_glob, root))
    return files


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one train or val pass. Pass optimizer=None for val."""
    training = optimizer is not None
    model.train() if training else model.eval()

    running, nb = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, batch_lanes in loader:
            images = images.to(device, non_blocking=True)
            params = model(images)
            loss = criterion(params, batch_lanes)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running += loss.item()
            nb += 1

    return running / max(nb, 1)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save, exist_ok=True)

    label_files = find_label_files(args.tusimple_root, args.label_glob)
    print("Device:", device)
    print("Label files:", label_files)

    full_dataset = HNetDataset(label_files, image_root=args.tusimple_root,
                               resize_w=args.width, resize_h=args.height)

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print("Train samples: {}  Val samples: {}".format(n_train, n_val))

    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=hnet_collate,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_dataset,  num_workers=32, batch_size=args.bs, 
                              shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, num_workers=32, batch_size=args.bs,
                            shuffle=False, drop_last=False, **loader_kwargs)

    model = H_Net().to(device)
    criterion = HNetLoss(order=args.poly_order)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val = float("inf")

    epoch_bar = tqdm(range(args.epochs), desc="H-Net", unit="epoch")
    for epoch in epoch_bar:

        # --- train ---
        model.train()
        running, nb = 0.0, 0
        batch_bar = tqdm(train_loader, desc="  train {:>3}".format(epoch + 1),
                         leave=False, unit="batch")
        for images, batch_lanes in batch_bar:
            images = images.to(device, non_blocking=True)
            params = model(images)
            loss = criterion(params, batch_lanes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            nb += 1
            batch_bar.set_postfix(loss="{:.6f}".format(running / nb))
        train_loss = running / max(nb, 1)

        # --- val ---
        model.eval()
        running, nb = 0.0, 0
        with torch.no_grad():
            batch_bar = tqdm(val_loader, desc="  val   {:>3}".format(epoch + 1),
                             leave=False, unit="batch")
            for images, batch_lanes in batch_bar:
                images = images.to(device, non_blocking=True)
                loss = criterion(model(images), batch_lanes)
                running += loss.item()
                nb += 1
                batch_bar.set_postfix(loss="{:.6f}".format(running / nb))
        val_loss = running / max(nb, 1)

        log["epoch"].append(epoch + 1)
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        epoch_bar.set_postfix(train="{:.6f}".format(train_loss),
                              val="{:.6f}".format(val_loss))
        tqdm.write("epoch {:>3}/{}  train {:.6f}  val {:.6f}{}".format(
            epoch + 1, args.epochs, train_loss, val_loss,
            "  *" if val_loss < best_val else "",
        ))

        torch.save(model.state_dict(),
                   os.path.join(args.save, "hnet_epoch_{:03d}.pth".format(epoch + 1)))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.save, "hnet_best.pth"))

    pd.DataFrame(log).to_csv(
        os.path.join(args.save, "hnet_training_log.csv"), index=False)
    print("Best val loss: {:.6f}".format(best_val))
    print("Saved checkpoints and log to:", args.save)


if __name__ == "__main__":
    main()
