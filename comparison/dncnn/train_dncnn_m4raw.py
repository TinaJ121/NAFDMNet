import sys
sys.path.insert(0, r'E:\Restormer-main\Denoising')

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from dataset_strong import get_m4raw_loaders
from metrics_net    import MetricTracker

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super().__init__()
        padding = 1
        layers  = []
        layers += [nn.Conv2d(image_channels, n_channels, 3,
                            padding=padding, bias=True),
                   nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(n_channels, n_channels, 3,
                                padding=padding, bias=False),
                       nn.BatchNorm2d(n_channels,
                                     eps=0.0001, momentum=0.95),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(n_channels, image_channels, 3,
                            padding=padding, bias=False)]
        self.dncnn = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return x - self.dncnn(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


DATA_DIR = r'E:\V1.6\M4RawV1.5_multicoil_train\multicoil_train'
OUT_DIR  = r'./checkpoints/dncnn_m4raw_official'
EPOCHS   = 150
BATCH    = 16      
LR       = 1e-3   
DEPTH    = 17      
N_CH     = 64      

MILESTONES = [50, 75]
GAMMA      = 0.1

os.makedirs(OUT_DIR, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

model = DnCNN(depth=DEPTH, n_channels=N_CH).to(device)
print(f'Params : {count_params(model)/1e6:.2f}M')

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

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=MILESTONES,
    gamma=GAMMA,
)

print(f'\n{"="*65}')
print(f'  DnCNN Training on M4Raw  [Official Hyperparameters]')
print(f'  Loss: MSE  LR: {LR} (x{GAMMA} at {MILESTONES})')
print(f'  Batch: {BATCH}  Epochs: {EPOCHS}  Depth: {DEPTH}')
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

        pred = model(noisy)
        loss = criterion(pred, gt)   

        loss.backward()
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
                loss=f'{run_loss/n:.6f}',
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
            pred  = model(noisy)
            tracker.update(torch.clamp(pred, 0, 1), gt)

    val_m   = tracker.summary()
    elapsed = time.time() - t0

    print(f'[{epoch:3d}/{EPOCHS}] lr={lr:.2e}  '
          f'loss={train_loss:.6f}  '
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
                'loss'  : 'MSE',
                'lr'    : LR,
                'batch' : BATCH,
                'depth' : DEPTH,
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