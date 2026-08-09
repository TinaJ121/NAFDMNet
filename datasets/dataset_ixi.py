import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ==========================
# 固定随机种子，保证实验可复现
# ==========================
def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =====================================================
# Rician Noise
# =====================================================
def add_rician_noise(img, sigma=0.06):
    """
    img : numpy [H,W], 0~1
    """
    noise_real = np.random.randn(*img.shape)
    noise_imag = np.random.randn(*img.shape)

    noisy = np.sqrt(
        (img + sigma * noise_real) ** 2 +
        (sigma * noise_imag) ** 2
    )
    noisy = np.clip(noisy, 0, 1)
    return noisy

# =====================================================
# Data Augmentation
# =====================================================
def random_flip(img, gt):
    if random.random() < 0.5:
        img = np.fliplr(img)
        gt = np.fliplr(gt)
    if random.random() < 0.5:
        img = np.flipud(img)
        gt = np.flipud(gt)
    return img.copy(), gt.copy()

def random_rotate(img, gt):
    k = random.randint(0, 3)
    img = np.rot90(img, k)
    gt = np.rot90(gt, k)
    return img.copy(), gt.copy()

# =====================================================
# Dataset
# =====================================================
class IXIDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sigma=0.06,
        target_size=256,
        augment=False
    ):
        self.root_dir = root_dir
        self.sigma = sigma
        self.augment = augment

        self.files = []
        exts = ['png', 'jpg', 'jpeg', 'bmp']
        for f in os.listdir(root_dir):
            ext = f.split('.')[-1].lower()
            if ext in exts:
                self.files.append(os.path.join(root_dir, f))

        self.files.sort()
        # 第八处：增加断言，防止文件夹为空
        assert len(self.files) > 0, f'No image found in {root_dir}'

        # 第二处：移除Resize，仅保留ToTensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'Loaded {len(self.files)} images')
        print(f'Rician Sigma base = {sigma}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name = os.path.basename(path)

        img = Image.open(path).convert('L')
        gt = self.transform(img).squeeze(0).numpy()

        # 第九处：训练时sigma随机扰动
        if self.augment:
            sigma = np.random.uniform(self.sigma * 0.8, self.sigma * 1.2)
        else:
            sigma = self.sigma

        noisy = add_rician_noise(gt, sigma=sigma)

        if self.augment:
            noisy, gt = random_flip(noisy, gt)
            noisy, gt = random_rotate(noisy, gt)

        noisy = torch.FloatTensor(noisy).unsqueeze(0)
        gt = torch.FloatTensor(gt).unsqueeze(0)

        # 第七处：返回path
        return {
            'noisy': noisy,
            'gt': gt,
            'filename': name,
            'path': path
        }

# =====================================================
# Loader 【已移出class，和类平级！】
# =====================================================
def get_ixi_loaders(
    train_dir,
    val_dir,
    batch_size=4,
    sigma=0.06,
    target_size=256,
    num_workers=0):  # Windows默认0，Linux改成4

    train_set = IXIDataset(
        train_dir,
        sigma=sigma,
        target_size=target_size,
        augment=True
    )

    val_set = IXIDataset(
        val_dir,
        sigma=sigma,
        target_size=target_size,
        augment=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # 第四处
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def get_ixi_test_loader(
    test_dir,
    sigma=0.06,
    batch_size=1,
    target_size=256,
    num_workers=0):

    dataset = IXIDataset(
        test_dir,
        sigma=sigma,
        target_size=target_size,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader