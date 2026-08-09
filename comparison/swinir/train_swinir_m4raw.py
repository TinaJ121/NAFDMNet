import sys
import os
import importlib.util

sys.path.insert(0, r'E:\SwinIR-main')
sys.path.insert(0, r'E:\Restormer-main\Denoising')

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_swinir_cls():
    spec = importlib.util.spec_from_file_location(
        "network_swinir",
        r"E:\SwinIR-main\models\network_swinir.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SwinIR

SwinIR = load_swinir_cls()
print("SwinIR loaded OK")

from dataset_strong import get_m4raw_loaders
from metrics_net    import MetricTracker


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        diff = pred - gt
        return torch.mean(torch.sqrt(diff**2 + self.eps**2))

DATA_DIR = r'E:\V1.6\M4RawV1.5_multicoil_train\multicoil_train'
OUT_DIR  = r'./checkpoints/swinir_m4raw_official'
EPOCHS   = 150
BATCH    = 1      
LR       = 2e-4    

MILESTONES = [60, 80, 90, 95]
GAMMA      = 0.5

os.makedirs(OUT_DIR, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

model = SwinIR(
    upscale=1,
    in_chans=1,
    img_size=128,
    window_size=8,
    img_range=1.,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='',
    resi_connection='1conv',
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


criterion = CharbonnierLoss(eps=1e-3)  

optimizer = optim.Adam(                
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.99),
)

scheduler = optim.lr_scheduler.MultiStepLR(  
    optimizer,
    milestones=MILESTONES,
    gamma=GAMMA,
)

print(f'\n{"="*65}')
print(f'  SwinIR Training on M4Raw  [Official Hyperparameters]')
print(f'  Loss: Charbonnier  LR: {LR}  Batch: {BATCH}  Epochs: {EPOCHS}')
print(f'  Milestones: {MILESTONES}  Gamma: {GAMMA}')
print(f'  Output: {OUT_DIR}')
print(f'{"="*65}\n')

log_f     = open(os.path.join(OUT_DIR, 'train_log.jsonl'), 'a')
best_psnr = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    lr = optimizer.param_groups[0]['lr']

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
        gt    = batch['gt'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda',
                                 enabled=(device.type == 'cuda')):
            pred = model(noisy)
            loss = criterion(pred, gt)  

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        run_loss += loss.item()
        with torch.no_grad():
            pc  = torch.clamp(pred, 0, 1)
            mse = ((pc - gt)**2).mean()
            run_psnr += (10 * torch.log10(
                1.0 / (mse + 1e-10))).item()
        n += 1

        if HAS_TQDM and n % 20 == 0:
            pbar.set_postfix(
                loss=f'{run_loss/n:.4f}',
                psnr=f'{run_psnr/n:.2f}',
                lr=f'{lr:.1e}')

    if HAS_TQDM:
        pbar.close()

    scheduler.step()

    train_loss = run_loss  / max(n, 1)
    train_psnr = run_psnr  / max(n, 1)

    model.eval()
    tracker = MetricTracker(use_lpips=False)
    with torch.no_grad():
        for batch in va_loader:
            noisy = batch['noisy'].to(device, non_blocking=True)
            gt    = batch['gt'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda',
                                     enabled=(device.type == 'cuda')):
                pred = model(noisy)
            tracker.update(torch.clamp(pred, 0, 1), gt)

    val_m   = tracker.summary()
    elapsed = time.time() - t0

    print(f'[{epoch:3d}/{EPOCHS}] lr={lr:.2e}  '
          f'loss={train_loss:.4f}  '
          f'train_PSNR={train_psnr:.2f}  |  '
          f'Val: PSNR={val_m["psnr"]:.2f}  '
          f'SSIM={val_m["ssim"]:.4f}  '
          f'[{elapsed:.0f}s]')

    log_f.write(json.dumps({
        'epoch'      : epoch,
        'lr'         : lr,
        'train_loss' : train_loss,
        'train_psnr' : train_psnr,
        'val'        : val_m,
        'elapsed'    : elapsed,
    }) + '\n')
    log_f.flush()

    if val_m['psnr'] > best_psnr:
        best_psnr = val_m['psnr']
        torch.save({
            'epoch'     : epoch,
            'model'     : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_psnr' : best_psnr,
            'val'       : val_m,
            'config'    : {
                'loss'  : 'Charbonnier',
                'lr'    : LR,
                'batch' : BATCH,
            }
        }, os.path.join(OUT_DIR, 'best.pth'))
        print(f'  * New best PSNR: {best_psnr:.2f} dB -> best.pth')

    torch.save({
        'epoch'     : epoch,
        'model'     : model.state_dict(),
        'best_psnr' : best_psnr,
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