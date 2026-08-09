import sys
sys.path.insert(0, r'E:\Restormer-main\Denoising')

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from metrics_net import MetricTracker


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class N2VUNet(nn.Module):
    """
    U-Net for Noise2Void
    depth=2, initial_features=96 (paper BSD68 setting)
    linear activation in last layer (original paper)
    """
    def __init__(self, in_ch=1, base_ch=96, depth=2, kernel_size=3):
        super().__init__()
        self.depth = depth

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch = in_ch
        enc_chs = []
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.enc.append(ConvBlock(ch, out_ch, kernel_size))
            self.pool.append(nn.MaxPool2d(2))
            enc_chs.append(out_ch)
            ch = out_ch

        bot_ch = base_ch * (2 ** depth)
        self.bottleneck = ConvBlock(ch, bot_ch, kernel_size)
        ch = bot_ch

        self.up   = nn.ModuleList()
        self.dec  = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = enc_chs[i]
            out_ch  = base_ch * (2 ** i)
            self.up.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.dec.append(ConvBlock(out_ch + skip_ch, out_ch, kernel_size))
            ch = out_ch

        self.out_conv = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x):
        skips = []
        h = x
        for enc, pool in zip(self.enc, self.pool):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:])
            h = dec(torch.cat([h, skip], dim=1))
        return self.out_conv(h)


def n2v_mask(patch, n_masked=64):
    """
    Apply blind-spot masking to N randomly selected pixels per patch.
    Replace center pixel value with a randomly selected neighboring pixel.
    Returns: masked_patch, mask
    """
    B, C, H, W = patch.shape
    masked = patch.clone()
    mask   = torch.zeros(B, C, H, W, device=patch.device)

    for b in range(B):
        grid_size = max(1, int(np.sqrt(H * W / n_masked)))
        coords = []
        for gy in range(0, H, grid_size):
            for gx in range(0, W, grid_size):
                py = np.random.randint(gy, min(gy + grid_size, H))
                px = np.random.randint(gx, min(gx + grid_size, W))
                coords.append((py, px))

        np.random.shuffle(coords)
        coords = coords[:n_masked]

        for (py, px) in coords:
            ry = np.random.randint(max(0, py-2), min(H, py+3))
            rx = np.random.randint(max(0, px-2), min(W, px+3))
            masked[b, :, py, px] = patch[b, :, ry, rx]
            mask[b, :, py, px]   = 1.0

    return masked, mask


def n2v_loss(pred, target, mask):
    """
    N2V loss: MSE calculated only on masked pixels (Eq.10)
    """
    diff = (pred - target) ** 2
    loss = (diff * mask).sum() / (mask.sum() + 1e-8)
    return loss


class M4RawN2VDataset(Dataset):
    """
    N2V Training Dataset: uses only noisy images, no GT required.
    Extracts patches from single-scan M4Raw (T101, T201, FLAIR01).
    """
    def __init__(self, data_dir, patch_size=64,
                 sequences=('T1', 'T2', 'FLAIR'),
                 frames_per_seq=18, augment=True):
        self.patch_size = patch_size
        self.augment    = augment

        SEQ_MAP = {
            'T1'   : 'T101',
            'T2'   : 'T201',
            'FLAIR': 'FLAIR01',
        }

        all_files = os.listdir(data_dir)
        patients  = sorted(set(
            f.split('_')[0] for f in all_files if f.endswith('.h5')))

        self.samples = []
        for pid in patients:
            for seq in sequences:
                suf = SEQ_MAP.get(seq)
                if suf is None:
                    continue
                fp = os.path.join(data_dir, f'{pid}_{suf}.h5')
                if not os.path.exists(fp):
                    continue
                try:
                    with h5py.File(fp, 'r') as f:
                        n = f['reconstruction_rss'].shape[0]
                    center = n // 2
                    half   = min(frames_per_seq, n) // 2
                    for t in range(max(0, center-half),
                                   min(n, center+half)):
                        self.samples.append((fp, t))
                except:
                    continue

        print(f'N2V Dataset: {len(self.samples)} slices')

    def __len__(self):
        return len(self.samples) * 16

    def __getitem__(self, idx):
        sample_idx = idx // 16
        fp, t = self.samples[sample_idx % len(self.samples)]

        with h5py.File(fp, 'r') as f:
            rss = f['reconstruction_rss'][t].astype(np.float32)

        rss_max = rss.max() + 1e-8
        img = np.clip(rss / rss_max, 0, 1)

        H, W = img.shape
        ps   = self.patch_size

        if H > ps and W > ps:
            y = np.random.randint(0, H - ps)
            x = np.random.randint(0, W - ps)
            patch = img[y:y+ps, x:x+ps]
        else:
            patch = img[:ps, :ps]

        if self.augment:
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k).copy()
            if np.random.rand() > 0.5:
                patch = np.fliplr(patch).copy()

        return torch.from_numpy(patch).unsqueeze(0).float()


class M4RawN2VTestDataset(Dataset):
    """Test Dataset: requires GT (multi-average)"""
    def __init__(self, data_dir, sequences=('T1','T2','FLAIR'),
                 frames_per_seq=18):
        SEQ_CAND = {
            'T1'   : ('T101', ['T101','T102','T103','T104','T105','T106']),
            'T2'   : ('T201', ['T201','T202','T203','T204','T205','T206']),
            'FLAIR': ('FLAIR01', ['FLAIR01','FLAIR02','FLAIR03','FLAIR04']),
        }
        all_files = os.listdir(data_dir)
        patients  = sorted(set(
            f.split('_')[0] for f in all_files if f.endswith('.h5')))

        pid0 = patients[0]
        self.seq_cfg = {}
        for seq in sequences:
            inp_suf, gt_sufs = SEQ_CAND[seq]
            actual_gts = [s for s in gt_sufs
                          if os.path.exists(
                              os.path.join(data_dir, f'{pid0}_{s}.h5'))]
            if actual_gts:
                self.seq_cfg[seq] = (inp_suf, actual_gts)

        self.samples  = []
        self.data_dir = data_dir
        for pid in patients:
            for seq, (inp_suf, gt_sufs) in self.seq_cfg.items():
                fp = os.path.join(data_dir, f'{pid}_{inp_suf}.h5')
                if not os.path.exists(fp):
                    continue
                with h5py.File(fp, 'r') as f:
                    n = f['reconstruction_rss'].shape[0]
                center = n // 2
                half   = min(frames_per_seq, n) // 2
                for t in range(max(0, center-half), min(n, center+half)):
                    self.samples.append((pid, seq, t, inp_suf, gt_sufs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, seq, t, inp_suf, gt_sufs = self.samples[idx]
        fp = os.path.join(self.data_dir, f'{pid}_{inp_suf}.h5')
        with h5py.File(fp, 'r') as f:
            inp = f['reconstruction_rss'][t].astype(np.float32)

        gt_list = []
        for suf in gt_sufs:
            fp2 = os.path.join(self.data_dir, f'{pid}_{suf}.h5')
            if os.path.exists(fp2):
                with h5py.File(fp2, 'r') as f:
                    gt_list.append(f['reconstruction_rss'][t].astype(np.float32))
        gt = np.mean(gt_list, axis=0).astype(np.float32)

        gt_max = gt.max() + 1e-8
        inp_n  = np.clip(inp / gt_max, 0, 1).astype(np.float32)
        gt_n   = np.clip(gt  / gt_max, 0, 1).astype(np.float32)

        return (torch.from_numpy(inp_n).unsqueeze(0),
                torch.from_numpy(gt_n).unsqueeze(0),
                pid, seq, t, len(gt_sufs))


if __name__ == '__main__':
    TRAIN_DIR = r'E:\V1.6\M4RawV1.5_multicoil_train\multicoil_train'
    VAL_DIR   = r'E:\V1.6\M4Raw_multicoil_test\multicoil_test'
    OUT_DIR   = r'./checkpoints/n2v_m4raw'
    EPOCHS    = 150
    BATCH     = 128
    LR        = 4e-4
    N_MASKED  = 64
    PATCH_SIZE = 64

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    model = N2VUNet(in_ch=1, base_ch=96, depth=2, kernel_size=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_params/1e6:.2f}M')

    tr_ds = M4RawN2VDataset(
        TRAIN_DIR, patch_size=PATCH_SIZE,
        sequences=('T1','T2','FLAIR'), frames_per_seq=18)
    va_ds = M4RawN2VTestDataset(
        VAL_DIR, sequences=('T1','T2','FLAIR'), frames_per_seq=18)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH,
                            shuffle=True, num_workers=0,
                            pin_memory=(device.type=='cuda'))
    va_loader = DataLoader(va_ds, batch_size=1,
                            shuffle=False, num_workers=0)
    print(f'Train : {len(tr_loader)} batches')
    print(f'Val   : {len(va_loader)} samples')

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=10)

    print(f'\n{"="*60}')
    print(f'  Noise2Void Training on M4Raw  [Official Hyperparameters]')
    print(f'  CVPR 2019 | Krull et al.')
    print(f'  Loss: N2V-MSE  lr={LR}  batch={BATCH}  epochs={EPOCHS}')
    print(f'  patch={PATCH_SIZE}x{PATCH_SIZE}  masked_pixels={N_MASKED}')
    print(f'{"="*60}\n')

    log_f     = open(os.path.join(OUT_DIR, 'train_log.jsonl'), 'a')
    best_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        model.train()
        run_loss, n = 0.0, 0

        if HAS_TQDM:
            pbar = tqdm(tr_loader,
                        desc=f'Epoch {epoch:3d}/{EPOCHS}',
                        leave=False, unit='batch',
                        bar_format='{l_bar}{bar:32}{r_bar}')
        else:
            pbar = tr_loader

        for patch in pbar:
            patch = patch.to(device)

            masked, mask = n2v_mask(patch, n_masked=N_MASKED)

            optimizer.zero_grad(set_to_none=True)
            pred = model(masked)
            loss = n2v_loss(pred, patch, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run_loss += loss.item()
            n += 1

            if HAS_TQDM and n % 50 == 0:
                pbar.set_postfix(
                    loss=f'{run_loss/n:.5f}',
                    lr=f'{lr:.1e}')

        if HAS_TQDM:
            pbar.close()

        train_loss = run_loss / max(n, 1)

        scheduler.step(train_loss)

        model.eval()
        tracker = MetricTracker(use_lpips=False)
        with torch.no_grad():
            for noisy, gt, *_ in va_loader:
                noisy = noisy.to(device)
                gt    = gt.to(device)
                pred  = model(noisy)
                tracker.update(torch.clamp(pred, 0, 1), gt)

        val_m   = tracker.summary()
        elapsed = time.time() - t0

        print(f'[{epoch:3d}/{EPOCHS}] lr={lr:.1e}  '
            f'loss={train_loss:.5f}  |  '
            f'Val: PSNR={val_m["psnr"]:.2f}  '
            f'SSIM={val_m["ssim"]:.4f}  '
            f'[{elapsed:.0f}s]')

        log_f.write(json.dumps({
            'epoch'     : epoch,
            'lr'        : lr,
            'train_loss': train_loss,
            'val'       : val_m,
            'elapsed'   : elapsed,
        }) + '\n')
        log_f.flush()

        if val_m['psnr'] > best_psnr:
            best_psnr = val_m['psnr']
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'best_psnr': best_psnr,
                'val'      : val_m,
                'config'   : {
                    'loss'       : 'N2V-MSE',
                    'lr'         : LR,
                    'batch'      : BATCH,
                    'patch_size' : PATCH_SIZE,
                    'n_masked'   : N_MASKED,
                }
            }, os.path.join(OUT_DIR, 'best.pth'))
            print(f'  * New best PSNR: {best_psnr:.2f} dB -> best.pth')

        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'best_psnr': best_psnr,
        }, os.path.join(OUT_DIR, 'last.pth'))

        if epoch % 25 == 0:
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(OUT_DIR, f'epoch_{epoch:03d}.pth'))
            print(f'  Saved epoch_{epoch:03d}.pth')

    log_f.close()
    print(f'\n[Done] Best val PSNR: {best_psnr:.2f} dB')
    print(f'Checkpoints: {OUT_DIR}')