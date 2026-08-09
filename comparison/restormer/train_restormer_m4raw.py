import sys
sys.path.insert(0, r'D:\Restormer\Restormer-main')
sys.path.insert(0, r'E:\Restormer-main\Denoising')

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from basicsr.models.archs.restormer_arch import Restormer
from dataset_strong import get_m4raw_loaders
from metrics_net import MetricTracker

DATA_DIR = r'E:\V1.6\M4RawV1.5_multicoil_train\multicoil_train'
OUT_DIR = r'./checkpoints/restormer_m4raw_official'
EPOCHS = 150
BATCH = 4       
LR = 3e-4      
MIN_LR = 1e-6
WARMUP = 10

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

model = Restormer(
    inp_channels=1,
    out_channels=1,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='BiasFree',
    dual_pixel_task=False,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params : {n_params/1e6:.2f}M')

tr_loader, va_loader, _ = get_m4raw_loaders(
    DATA_DIR,
    batch_size=BATCH,
    num_workers=0,
    sequences=('T1', 'T2', 'FLAIR'),
    frames_per_seq=18,
    pin_memory=(device.type == 'cuda'),
)
print(f'Train  : {len(tr_loader)} batches')
print(f'Val    : {len(va_loader)} batches')

criterion = nn.L1Loss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)
scaler = GradScaler()

def get_lr(epoch):
    if epoch <= WARMUP:
        return LR * epoch / WARMUP
    progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * progress))

print(f'\n{"="*65}')
print(f'  Restormer Training on M4Raw  [Official Hyperparameters]')
print(f'  Loss: L1  LR: {LR}  Batch: {BATCH}  Epochs: {EPOCHS}')
print(f'  Output: {OUT_DIR}')
print(f'{"="*65}\n')

log_f = open(os.path.join(OUT_DIR, 'train_log.jsonl'), 'a')
best_psnr = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    lr = get_lr(epoch)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    model.train()
    run_loss, run_psnr, n = 0.0, 0.0, 0

    if HAS_TQDM:
        pbar = tqdm(tr_loader,
                    desc=f'Epoch {epoch:3d}/{EPOCHS}',
                    leave=False, unit='batch',
                    bar_format='{l_bar}{bar:32}{r_bar}')
    else:
        pbar = tr_loader

    for batch in pbar:
        noisy = batch['noisy'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            pred = model(noisy)
            loss = criterion(pred, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        scaler.step(optimizer)
        scaler.update()

        run_loss += loss.item()
        with torch.no_grad():
            pc = torch.clamp(pred, 0, 1)
            mse = ((pc - gt)**2).mean()
            run_psnr += (10 * torch.log10(1.0 / (mse + 1e-10))).item()
        n += 1

        if HAS_TQDM and n % 20 == 0:
            pbar.set_postfix(
                loss=f'{run_loss/n:.4f}',
                psnr=f'{run_psnr/n:.2f}',
                lr=f'{lr:.1e}')

    if HAS_TQDM:
        pbar.close()

    train_loss = run_loss / max(n, 1)
    train_psnr = run_psnr / max(n, 1)

    model.eval()
    tracker = MetricTracker(use_lpips=False)
    with torch.no_grad():
        for batch in va_loader:
            noisy = batch['noisy'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                pred = model(noisy)
            tracker.update(torch.clamp(pred, 0, 1), gt)

    val_m = tracker.summary()
    elapsed = time.time() - t0

    print(f'[{epoch:3d}/{EPOCHS}] lr={lr:.2e}  '
          f'loss={train_loss:.4f}  '
          f'train_PSNR={train_psnr:.2f}  |  '
          f'Val: PSNR={val_m["psnr"]:.2f}  '
          f'SSIM={val_m["ssim"]:.4f}  '
          f'[{elapsed:.0f}s]')

    log_f.write(json.dumps({
        'epoch': epoch,
        'lr': lr,
        'train_loss': train_loss,
        'train_psnr': train_psnr,
        'val': val_m,
        'elapsed': elapsed,
    }) + '\n')
    log_f.flush()

    if val_m['psnr'] > best_psnr:
        best_psnr = val_m['psnr']
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'val': val_m,
            'config': {
                'loss': 'L1',
                'lr': LR,
                'batch': BATCH,
            }
        }, os.path.join(OUT_DIR, 'best.pth'))
        print(f'  * New best PSNR: {best_psnr:.2f} dB -> best.pth')

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'best_psnr': best_psnr,
    }, os.path.join(OUT_DIR, 'last.pth'))

    if epoch % 25 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_psnr': best_psnr,
        }, os.path.join(OUT_DIR, f'epoch_{epoch:03d}.pth'))
        print(f'  Saved epoch_{epoch:03d}.pth')

log_f.close()
print(f'\n[Done] Best val PSNR: {best_psnr:.2f} dB')
print(f'Checkpoints: {OUT_DIR}')