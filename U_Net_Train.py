"""
U-Net cancer instance segmentation training script for VS Code/local Python.

Changes from the notebook version:
1. Removed final_weights.pt pre-computation.
2. Calculates weight map per batch during training/validation.
3. Uses 5 epochs by default.
4. Saves checkpoints to ./models.
5. Loads checkpoint automatically if ./models/checkpoint.pth exists.
6. Saves batch weight samples to ./weights for debugging/inspection.

Run:
    python U_Net_Train.py

Optional:
    python U_Net_Train.py --epochs 5 --batch_size 2
"""

from pathlib import Path
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# -----------------------------
# Dataset
# -----------------------------
class NumpySegDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, target_transform=None):
        self.images = np.load(images_path, mmap_mode="r")
        self.masks = np.load(masks_path, mmap_mode="r")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # image: (H, W, C) -> (C, H, W)
        # mask : (H, W, C) -> (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.int64).permute(2, 0, 1)

        # Optional normalization. If images are already normalized, remove this block.
        if image.max() > 1.0:
            image = image / 255.0

        return image, mask


# -----------------------------
# Model
# -----------------------------
class UNetConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes=6, in_channel=3):
        super().__init__()
        self.conv_1 = UNetConv2(in_channel, 64)
        self.conv_2 = UNetConv2(64, 128)
        self.conv_3 = UNetConv2(128, 256)
        self.conv_4 = UNetConv2(256, 512)

        self.mid_conv = UNetConv2(512, 1024)

        self.conv_5 = UNetConv2(1024, 512)
        self.conv_6 = UNetConv2(512, 256)
        self.conv_7 = UNetConv2(256, 128)
        self.conv_8 = UNetConv2(128, 64)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.end = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    @staticmethod
    def center_crop_like(src, target):
        """Center-crop src to target spatial size."""
        _, _, h, w = src.shape
        _, _, th, tw = target.shape
        top = max((h - th) // 2, 0)
        left = max((w - tw) // 2, 0)
        return src[:, :, top : top + th, left : left + tw]

    def forward(self, x):
        padded_x = F.pad(x, (92, 92, 92, 92), mode="reflect")

        conv_1 = self.conv_1(padded_x)
        if conv_1.size(2) % 2 != 0:
            conv_1 = F.pad(conv_1, (0, 1, 0, 1))
        pool1 = self.down(conv_1)

        conv_2 = self.conv_2(pool1)
        if conv_2.size(2) % 2 != 0:
            conv_2 = F.pad(conv_2, (0, 1, 0, 1))
        pool2 = self.down(conv_2)

        conv_3 = self.conv_3(pool2)
        if conv_3.size(2) % 2 != 0:
            conv_3 = F.pad(conv_3, (0, 1, 0, 1))
        pool3 = self.down(conv_3)

        conv_4 = self.conv_4(pool3)
        if conv_4.size(2) % 2 != 0:
            conv_4 = F.pad(conv_4, (0, 1, 0, 1))
        pool4 = self.down(conv_4)

        mid_conv = self.mid_conv(pool4)

        up_1 = self.up_1(mid_conv)
        cropped_conv_4 = self.center_crop_like(conv_4, up_1)
        up_1 = torch.cat([up_1, cropped_conv_4], dim=1)
        conv_5 = self.conv_5(up_1)

        up_2 = self.up_2(conv_5)
        cropped_conv_3 = self.center_crop_like(conv_3, up_2)
        up_2 = torch.cat([up_2, cropped_conv_3], dim=1)
        conv_6 = self.conv_6(up_2)

        up_3 = self.up_3(conv_6)
        cropped_conv_2 = self.center_crop_like(conv_2, up_3)
        up_3 = torch.cat([up_3, cropped_conv_2], dim=1)
        conv_7 = self.conv_7(up_3)

        up_4 = self.up_4(conv_7)
        cropped_conv_1 = self.center_crop_like(conv_1, up_4)
        up_4 = torch.cat([up_4, cropped_conv_1], dim=1)
        conv_8 = self.conv_8(up_4)

        out = self.end(conv_8)
        out = self.center_crop_like(out, x)
        return out


# -----------------------------
# Weight map and loss
# -----------------------------
def find_others(labels, i, j, k, b, d):
    """
    labels: (B, C, H, W)
    Returns local neighboring count information used for weight amplification.
    """
    _, _, height, width = labels.shape

    top = max(i - d, 0)
    bottom = min(i + d, height - 1)
    left = max(j - d, 0)
    right = min(j + d, width - 1)

    instance = labels[b, k, i, j]
    region = labels[b, k, top : bottom + 1, left : right + 1]

    other_classes = (region == 0).sum().item()
    other_instances = ((region != 0) & (region != instance)).sum().item()
    return other_classes, other_instances


def calculate_weights(masks):
    """
    Calculate weights only for the current batch.

    masks shape: (B, C, H, W)
    returns    : (B, H, W)
    """
    device = masks.device
    batch_size, num_classes, height, width = masks.shape
    weights = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)

    non_zero_counts = (masks != 0).sum(dim=(2, 3))  # (B, C)

    for b in range(batch_size):
        denom = non_zero_counts[b].sum().float().clamp_min(1.0)
        non_zero_ratio = non_zero_counts[b].float() / denom
        exp_non_zero_ratio = torch.exp(-non_zero_ratio)

        for k in range(num_classes):
            mask_k = masks[b, k]
            non_zero_mask = mask_k != 0
            weights[b][non_zero_mask] = exp_non_zero_ratio[k]

            # Same logic as original code, but applied only to current batch.
            for i in range(2, height, 5):
                for j in range(2, width, 5):
                    if non_zero_mask[i, j]:
                        other_classes, other_instances = find_others(masks, i, j, k, b, 2)
                        weights[b, i - 2 : i + 3, j - 2 : j + 3] *= (1.02) ** other_classes
                        weights[b, i - 2 : i + 3, j - 2 : j + 3] *= (1.05) ** other_instances

    return weights


def custom_loss(outputs, labels, weights):
    """
    outputs: (B, C, H, W), raw logits
    labels : (B, C, H, W), instance masks per channel
    weights: (B, H, W)
    """
    softmax_outputs = F.softmax(outputs, dim=1).clamp_min(1e-8)

    non_zero_mask = labels != 0
    selected_weights = weights.unsqueeze(1).expand_as(labels)[non_zero_mask]
    selected_softmax_outputs = softmax_outputs[non_zero_mask]

    if selected_softmax_outputs.numel() == 0:
        return outputs.sum() * 0.0

    loss = -selected_weights * torch.log(selected_softmax_outputs)
    loss = loss.sum()
    loss = loss / (labels.shape[0] * labels.shape[2] * labels.shape[3])
    return loss


# -----------------------------
# Checkpoint
# -----------------------------
def save_checkpoint(path, model, optimizer, epoch, best_val_loss=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = checkpoint.get("best_val_loss", None)
    return start_epoch, best_val_loss


# -----------------------------
# Training
# -----------------------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    models_dir = Path(args.models_dir)
    weights_dir = Path(args.weights_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / "checkpoint.pth"
    latest_model_path = models_dir / "model_latest.pth"
    final_model_path = models_dir / "model_final.pth"
    best_model_path = models_dir / "model_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = NumpySegDataset(args.images_path, args.masks_path)
    total_len = len(dataset)
    train_len = int(total_len * args.train_ratio)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(num_classes=args.num_classes, in_channel=args.in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = None

    if checkpoint_path.exists():
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Resume from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Start new training.")

    if start_epoch >= args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch}. No further training needed.")
        torch.save(model.state_dict(), final_model_path)
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"{phase.capitalize()} Phase")

            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    # Main change: calculate weights per batch, not precompute final_weights.pt.
                    batch_weights = calculate_weights(labels)

                    # Save only a small sample for inspection, not the entire dataset.
                    if args.save_weight_samples and batch_idx == 0:
                        weight_sample_path = weights_dir / f"{phase}_batch_weights_epoch_{epoch + 1}.pt"
                        torch.save(batch_weights.detach().cpu(), weight_sample_path)

                    loss = custom_loss(outputs, labels, batch_weights)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                del inputs, labels, outputs, batch_weights, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == "val":
                if best_val_loss is None or epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Saved best model: {best_model_path}")

        # Save checkpoint after every epoch.
        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss)
        torch.save(model.state_dict(), latest_model_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Saved latest model: {latest_model_path}")

    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining finished. Saved final model: {final_model_path}")


# -----------------------------
# CLI / main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="U-Net cancer instance segmentation training")

    parser.add_argument(
        "--images_path",
        type=str,
        default=r"D:\University\4-1.5\BOA_lab\Week1_260629_0705\Cancer_Instance_Segmentation\data\Part_1\Images\images.npy",
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        default=r"D:\University\4-1.5\BOA_lab\Week1_260629_0705\Cancer_Instance_Segmentation\data\Part_1\Masks\masks.npy",
    )
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--weights_dir", type=str, default="./weights")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_weight_samples", action="store_true", help="Save first batch weight map of each phase/epoch to ./weights")

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
