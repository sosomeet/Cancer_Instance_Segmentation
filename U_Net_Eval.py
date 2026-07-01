from pathlib import Path
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt


# -----------------------------
# Dataset
# -----------------------------
class NumpySegDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images = np.load(images_path, mmap_mode="r")
        self.masks = np.load(masks_path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.int64).permute(2, 0, 1)

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
        _, _, h, w = src.shape
        _, _, th, tw = target.shape
        top = max((h - th) // 2, 0)
        left = max((w - tw) // 2, 0)
        return src[:, :, top:top + th, left:left + tw]

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
        up_1 = torch.cat([up_1, self.center_crop_like(conv_4, up_1)], dim=1)
        conv_5 = self.conv_5(up_1)

        up_2 = self.up_2(conv_5)
        up_2 = torch.cat([up_2, self.center_crop_like(conv_3, up_2)], dim=1)
        conv_6 = self.conv_6(up_2)

        up_3 = self.up_3(conv_6)
        up_3 = torch.cat([up_3, self.center_crop_like(conv_2, up_3)], dim=1)
        conv_7 = self.conv_7(up_3)

        up_4 = self.up_4(conv_7)
        up_4 = torch.cat([up_4, self.center_crop_like(conv_1, up_4)], dim=1)
        conv_8 = self.conv_8(up_4)

        out = self.end(conv_8)
        out = self.center_crop_like(out, x)
        return out


# -----------------------------
# Weight map and loss
# -----------------------------
def find_others(labels, i, j, k, b, d):
    _, _, height, width = labels.shape

    top = max(i - d, 0)
    bottom = min(i + d, height - 1)
    left = max(j - d, 0)
    right = min(j + d, width - 1)

    instance = labels[b, k, i, j]
    region = labels[b, k, top:bottom + 1, left:right + 1]

    other_classes = (region == 0).sum().item()
    other_instances = ((region != 0) & (region != instance)).sum().item()
    return other_classes, other_instances


def calculate_weights(masks):
    device = masks.device
    batch_size, num_classes, height, width = masks.shape
    weights = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)

    non_zero_counts = (masks != 0).sum(dim=(2, 3))

    for b in range(batch_size):
        denom = non_zero_counts[b].sum().float().clamp_min(1.0)
        non_zero_ratio = non_zero_counts[b].float() / denom
        exp_non_zero_ratio = torch.exp(-non_zero_ratio)

        for k in range(num_classes):
            mask_k = masks[b, k]
            non_zero_mask = mask_k != 0
            weights[b][non_zero_mask] = exp_non_zero_ratio[k]

            for i in range(2, height, 5):
                for j in range(2, width, 5):
                    if non_zero_mask[i, j]:
                        other_classes, other_instances = find_others(masks, i, j, k, b, 2)
                        weights[b, i - 2:i + 3, j - 2:j + 3] *= (1.02) ** other_classes
                        weights[b, i - 2:i + 3, j - 2:j + 3] *= (1.05) ** other_instances

    return weights


def custom_loss(outputs, labels, weights):
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
# Metrics / Visualization
# -----------------------------
def labels_to_class_map(labels):
    b, c, h, w = labels.shape
    device = labels.device

    non_zero_indices = torch.nonzero(labels, as_tuple=False)
    result = torch.zeros((b, h, w), dtype=torch.long, device=device)

    if non_zero_indices.numel() > 0:
        batch_idx = non_zero_indices[:, 0]
        channel_idx = non_zero_indices[:, 1]
        height_idx = non_zero_indices[:, 2]
        width_idx = non_zero_indices[:, 3]
        result[batch_idx, height_idx, width_idx] = channel_idx

    return result


def compute_metrics(pred_map, target_map, num_classes):
    eps = 1e-8

    pixel_acc = (pred_map == target_map).float().mean().item()

    dice_scores = []
    iou_scores = []

    # Class 0 is treated as background here.
    for cls in range(1, num_classes):
        pred_cls = pred_map == cls
        target_cls = target_map == cls

        pred_sum = pred_cls.sum().float()
        target_sum = target_cls.sum().float()

        if pred_sum + target_sum == 0:
            continue

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        iou = (intersection + eps) / (union + eps)

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return pixel_acc, mean_dice, mean_iou


def save_visualization(images, target_map, pred_map, output_dir, batch_idx, max_items=4):
    output_dir.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu()
    target_map = target_map.detach().cpu()
    pred_map = pred_map.detach().cpu()

    n = min(images.shape[0], max_items)

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(target_map[i].numpy(), cmap="viridis", interpolation="nearest")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_map[i].numpy(), cmap="viridis", interpolation="nearest")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"val_batch{batch_idx:04d}_sample{i}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = NumpySegDataset(args.images_path, args.masks_path)

    total_len = len(dataset)
    train_len = int(total_len * args.train_ratio)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(num_classes=args.num_classes, in_channel=args.in_channels).to(device)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model: {model_path}")
    print(f"Validation samples: {len(val_dataset)}")

    total_loss = 0.0
    total_pixel_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    output_dir = Path(args.output_dir)
    visualized = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, total=len(val_loader), desc="Validation")

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            batch_weights = calculate_weights(labels)
            loss = custom_loss(outputs, labels, batch_weights)

            pred_map = torch.argmax(outputs, dim=1)
            target_map = labels_to_class_map(labels)

            pixel_acc, dice, iou = compute_metrics(
                pred_map=pred_map,
                target_map=target_map,
                num_classes=args.num_classes,
            )

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_pixel_acc += pixel_acc * batch_size
            total_dice += dice * batch_size
            total_iou += iou * batch_size
            total_samples += batch_size

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "iou": f"{iou:.4f}",
            })

            if visualized < args.num_visuals:
                n_save = min(args.num_visuals - visualized, batch_size)
                save_visualization(
                    images=inputs,
                    target_map=target_map,
                    pred_map=pred_map,
                    output_dir=output_dir,
                    batch_idx=batch_idx,
                    max_items=n_save,
                )
                visualized += n_save

    avg_loss = total_loss / total_samples
    avg_pixel_acc = total_pixel_acc / total_samples
    avg_dice = total_dice / total_samples
    avg_iou = total_iou / total_samples

    print("\\nEvaluation Results")
    print(f"Validation Loss     : {avg_loss:.6f}")
    print(f"Pixel Accuracy      : {avg_pixel_acc:.6f}")
    print(f"Mean Dice Score     : {avg_dice:.6f}")
    print(f"Mean IoU            : {avg_iou:.6f}")
    print(f"Saved visualizations: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Net cancer instance segmentation model")

    parser.add_argument(
        "--images_path",
        type=str,
        default=r"D:\\University\\4-1.5\\BOA_lab\\Week1_260629_0705\\Cancer_Instance_Segmentation\\data\\Part_1\\Images\\images.npy",
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        default=r"D:\\University\\4-1.5\\BOA_lab\\Week1_260629_0705\\Cancer_Instance_Segmentation\\data\\Part_1\\Masks\\masks.npy",
    )
    parser.add_argument("--model_path", type=str, default="./models/model_final.pth")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_visuals")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_visuals", type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
