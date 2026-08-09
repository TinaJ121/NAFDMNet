

import os
import sys
import math
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

sys.path.insert(0, r'E:\Restormer-main\Denoising')
from models.ablation_models import build_ablation_model, ABLATION_DESCRIPTIONS, count_params
from dataset_net_v3  import get_m4raw_loaders
from metrics_net_v3  import MetricTracker


def parse_args():
    p = argparse.ArgumentParser('NAFDMNet Ablation Training')
    p.add_argument('--variant',       type=str, required=True,
                   choices=['A1_Baseline','A2_PlusSwin','A3_PlusCBAM',
                            'A4_NAFDnoAdapt','A5_NAFDFull','A6_Full'])
    p.add_argument('--data_dir',      type=str, required=True)
    p.add_argument('--val_dir',       type=str, default=None)
    p.add_argument('--output_dir',    type=str,
                   default='./checkpoints/ablation')
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--min_lr',        type=float, default=1e-6)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int,   default=5)
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--hidden_dim',    type=int,   default=160)
    p.add_argument('--latent_dim',    type=int,   default=320)
    p.add_argument('--amp',           action='store_true', default=True)
    p.add_argument('--resume',        type=str,   default=None)
    return p.parse_args()


class WarmupCosineScheduler:
    def __init__(self, opt, warmup, total, base_lr, min_lr=1e-6):
        self.opt=opt; self.warmup=warmup; self.total=total
        self.base_lr=base_lr; self.min_lr=min_lr; self.epoch=0
    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup:
            lr = self.base_lr * self.epoch / self.warmup
        else:
            p  = (self.epoch-self.warmup)/max(1, self.total-self.warmup)
            lr = self.min_lr+(self.base_lr-self.min_lr)*0.5*(1+math.cos(math.pi*p))
        for pg in self.opt.param_groups: pg['lr'] = lr
        return lr


def charbonnier(pred, gt, eps=1e-3):
    return torch.mean(torch.sqrt((pred-gt)**2 + eps**2))

def ssim_loss(pred, gt):
    from metrics_net_v3 import compute_ssim
    return 1.0 - compute_ssim(pred.clamp(0,1), gt.clamp(0,1))


@torch.no_grad()
def validate(model, loader, device, tracker, n_samples=100):
    model.eval(); tracker.reset()
    for i, batch in enumerate(loader):
        if i >= n_samples: break
        noisy = batch['noisy'].to(device)
        gt    = batch['gt'].to(device)
        out, _ = model(noisy)
        tracker.update(out.clamp(0,1), gt)
    model.train()
    return tracker.summary()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Ablation variant: {}".format(args.variant))
    print("Description: {}".format(ABLATION_DESCRIPTIONS[args.variant]))
    print("=" * 60)

   
    out_dir = Path(args.output_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

  
    tr_loader, va_loader, _ = get_m4raw_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, sequences=('T1','T2','FLAIR'),
        pin_memory=(device.type=='cuda'), val_data_dir=args.val_dir)


    kwargs = {'hidden_dim': args.hidden_dim}
    if args.variant in ('A5_NAFDFull', 'A6_Full'):
        kwargs['latent_dim'] = args.latent_dim
    model = build_ablation_model(args.variant, **kwargs).to(device)
    print("Params: {:.2f}M".format(count_params(model)/1e6))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.lr, args.min_lr)
    scaler  = torch.cuda.amp.GradScaler(enabled=args.amp)
    tracker = MetricTracker(use_lpips=False)

    start_epoch, best_psnr = 1, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr   = ckpt.get('best_psnr', 0.0)
        print("Resumed from epoch {}, best_psnr={:.2f}".format(start_epoch, best_psnr))

    log_f = open(out_dir / 'log.jsonl', 'a')

    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        t0 = time.time()
        lr = scheduler.step()
        running_loss = 0.0; n = 0

        pbar = tqdm(tr_loader,
                    desc="[{}] Ep{:3d}/{} lr={:.1e}".format(
                        args.variant, epoch, args.epochs, lr),
                    leave=False) if HAS_TQDM else tr_loader

        for batch in pbar:
            noisy = batch['noisy'].to(device, non_blocking=True)
            gt    = batch['gt'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                out, _ = model(noisy)
                
                loss = charbonnier(out, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            running_loss += loss.item(); n += 1
            if HAS_TQDM:
                pbar.set_postfix(loss="{:.5f}".format(running_loss/n),
                                 best="{:.2f}".format(best_psnr))

        avg_loss = running_loss / max(n, 1)

    
        val_m = validate(model, va_loader, device, tracker)
        elapsed = time.time() - t0
        psnr_val = val_m['psnr']

        print("[{}] Ep{:3d}/{} lr={:.1e} loss={:.5f} | "
              "PSNR={:.2f} SSIM={:.4f} NRMSE={:.4f} [{:.0f}s]".format(
            args.variant, epoch, args.epochs, lr, avg_loss,
            psnr_val, val_m['ssim'], val_m['nrmse'], elapsed))

        log_f.write(json.dumps({
            'epoch':epoch, 'lr':lr, 'loss':avg_loss,
            'val':val_m, 'elapsed':elapsed}) + '\n')
        log_f.flush()

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save({'epoch':epoch, 'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'best_psnr':best_psnr, 'variant':args.variant},
                       out_dir / 'best.pth')
            print("  * New best PSNR={:.2f} -> best.pth".format(best_psnr))

        torch.save({'epoch':epoch, 'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'best_psnr':best_psnr},
                   out_dir / 'last.pth')

        if epoch % 10 == 0:
            torch.save({'epoch':epoch, 'model':model.state_dict(),
                        'best_psnr':best_psnr},
                       out_dir / 'epoch_{:03d}.pth'.format(epoch))

    log_f.close()
    print("Done. Best PSNR={:.2f}".format(best_psnr))


if __name__ == "__main__":
    main()