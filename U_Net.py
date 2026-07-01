# 1. dataset introduction
import numpy as np

images = np.load(r'D:\University\4-1.5\BOA_lab\Week1_260629_0705\Unet_proj\data\Part_1\Images\images.npy', mmap_mode='r')
images = images.astype('int32')

masks = np.load(r'D:\University\4-1.5\BOA_lab\Week1_260629_0705\Unet_proj\data\Part_1\Masks\masks.npy', mmap_mode='r')
masks = masks.astype('int32')

print(images.shape)
print(masks.shape)

# 2. dataset definition and loading
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class NumpySegDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, target_transform=None):
        self.images = np.load(images_path, mmap_mode='r')
        self.masks = np.load(masks_path, mmap_mode='r')
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

        # image는 정규화하면 소수이므로 float
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        # mask는 정규화 미수행, int
        mask = torch.tensor(mask, dtype=torch.int64).permute(2,0,1)

        return image, mask

images_path=r'D:\University\4-1.5\BOA_lab\Week1_260629_0705\Unet_proj\data\Part_1\Images\images.npy'
masks_path=r'D:\University\4-1.5\BOA_lab\Week1_260629_0705\Unet_proj\data\Part_1\Masks\masks.npy'

dataset = NumpySegDataset(images_path, masks_path)

total_len = len(dataset)
train_len = int(total_len * 0.8)
val_len = total_len - train_len

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# 3. U-Net model architecture definition
class UNetConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConv2, self).__init__()
        # (입력크기+2*패딩-커널크기)/스트라이드+1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, num_classes=6, in_channel=3):
        # 부모 클래스 초기화되어 자식 클래스에서도 사용 가능
        super(UNet, self).__init__()
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

    def forward(self, x):
        padded_x = F.pad(x, (92, 92, 92, 92), mode='reflect')
        conv_1 = self.conv_1(padded_x) # output: 252*252
        if conv_1.size()[2] % 2 != 0:
            conv_1 = F.pad(conv_1, (0, 1, 0, 1))
        pool1 = self.down(conv_1) # output: 126*126

        conv_2 = self.conv_2(pool1) # output: 122*122
        if conv_2.size()[2] % 2 != 0:
            conv_2 = F.pad(conv_2, (0, 1, 0, 1))
        pool2 = self.down(conv_2) # output: 61*61

        conv_3 = self.conv_3(pool2) # output: 57*57
        if conv_3.size()[2] % 2 != 0:
            conv_3 = F.pad(conv_3, (0, 1, 0, 1))
        pool3 = self.down(conv_3) # output: 29*29

        conv_4 = self.conv_4(pool3) # output: 25*25
        if conv_4.size()[2] % 2 != 0:
            conv_4 = F.pad(conv_4, (0, 1, 0, 1))
        pool4 = self.down(conv_4) # output: 13*13

        mid_conv = self.mid_conv(pool4) # output: 9*9

        up_1 = self.up_1(mid_conv)
        scale_idx_1 = (conv_4.shape[2] - up_1.shape[2]) // 2
        cropped_conv_4 = conv_4[:, :, scale_idx_1:-scale_idx_1, scale_idx_1:-scale_idx_1]
        up_1 = torch.cat([up_1, cropped_conv_4], dim=1)
        conv_5 = self.conv_5(up_1)

        up_2 = self.up_2(conv_5)
        scale_idx_2 = (conv_3.shape[2] - up_2.shape[2]) // 2
        cropped_conv_3 = conv_3[:, :, scale_idx_2:-scale_idx_2, scale_idx_2:-scale_idx_2]
        up_2 = torch.cat([up_2, cropped_conv_3], dim=1)
        conv_6 = self.conv_6(up_2)

        up_3 = self.up_3(conv_6)
        scale_idx_3 = (conv_2.shape[2] - up_3.shape[2]) // 2
        cropped_conv_2 = conv_2[:, :, scale_idx_3:-scale_idx_3, scale_idx_3:-scale_idx_3]
        up_3 = torch.cat([up_3, cropped_conv_2], dim=1)
        conv_7 = self.conv_7(up_3)

        up_4 = self.up_4(conv_7)
        scale_idx_4 = (conv_1.shape[2] - up_4.shape[2]) // 2
        cropped_conv_1 = conv_1[:, :, scale_idx_4:-scale_idx_4, scale_idx_4:-scale_idx_4]
        up_4 = torch.cat([up_4, cropped_conv_1], dim=1)
        conv_8 = self.conv_8(up_4)

        end = self.end(conv_8)
        scale_idx_5 = (end.shape[2]-x.shape[2]) // 2
        end = end[:, :, scale_idx_5:-scale_idx_5, scale_idx_5:-scale_idx_5]

        return end
    
# 4. Loss function and optimizer definition
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def custom_loss(outputs, labels, weights):
    # softmax 계산
    softmax_outputs = F.softmax(outputs, dim=1)

    # CPU로 이동
    labels = labels.cpu()
    weights = weights.cpu()
    softmax_outputs = softmax_outputs.cpu()

    # 0이 아닌 위치를 찾기 위한 마스크 생성
    non_zero_mask = labels != 0

    # 마스크를 사용하여 필요한 값 선택 및 계산
    selected_weights = weights.unsqueeze(1).expand_as(labels)[non_zero_mask]
    selected_softmax_outputs = softmax_outputs[non_zero_mask]

    # 손실 계산
    running_loss = (-1) * selected_weights * torch.log(selected_softmax_outputs)
    running_loss = running_loss.sum()

    running_loss /= labels.shape[0] * labels.shape[2] * labels.shape[3]

    return running_loss.to(outputs.device)

import torch

def find_others(labels, i, j, k, b, d):
    left = max(i - d, 0)
    right = min(i + d, 255)  # 256이 아니라 255까지
    up = max(j - d, 0)
    down = min(j + d, 255)  # 256이 아니라 255까지
    instance = labels[b, k, i, j]

    region = labels[b, k, left:right+1, up:down+1]
    other_classes = (region == 0).sum().item()
    other_instances = ((region != 0) & (region != instance)).sum().item()

    return other_classes, other_instances

def calculate_weights(masks):
    device = masks.device
    batch_size, num_classes, height, width = masks.shape
    weights = torch.zeros((batch_size, height, width), device=device)
    non_zero_counts = (masks != 0).sum(dim=(2, 3))

    for b in range(batch_size):
        non_zero_ratio = non_zero_counts[b].float() / non_zero_counts[b].sum(dim=0, keepdim=True).float()
        exp_non_zero_ratio = torch.exp(-non_zero_ratio)
        
        for k in range(num_classes):
            mask_k = masks[b, k]
            non_zero_mask = mask_k != 0
            weights[b][non_zero_mask] = exp_non_zero_ratio[k]

            for i in range(2, height, 5):
                for j in range(2, width, 5):
                    if non_zero_mask[i, j]:
                        other_classes, other_instances = find_others(masks, i, j, k, b, 2)
                        weights[b, i-2:i+3, j-2:j+3] *= (1.02)**other_classes
                        weights[b, i-2:i+3, j-2:j+3] *= (1.05)**other_instances

    return weights

# 5. Train and validation loop
num_epochs = 50  # Number of epochs
batch_size = 4
val_idx_start = len(train_loader.dataset)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = train_loader
        else:
            model.eval()   # Set model to evaluate mode
            dataloader = val_loader

        running_loss = 0.0

        # Iterate over data with tqdm for the progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{phase.capitalize()} Phase")
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)

                if phase == 'train':
                    batch_weights = weights[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                else:
                    batch_weights = weights[val_idx_start + batch_idx * batch_size : val_idx_start + (batch_idx + 1) * batch_size]

                loss = custom_loss(outputs, labels, batch_weights)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)

            # Update the progress bar with the current loss value
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if phase == 'val':
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')
        print()
