import os
import json
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

try:
    from tqdm import tqdm
    HAS_TQDM = True
except:
    HAS_TQDM = False

from model import (
    NAFDMNet,
    NAFDMLoss,
    count_params,
)

from datasets.dataset_ixi import get_ixi_loaders
from metrics_net import MetricTracker


# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=0.06)  # 论文σ=0.06
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--latent_dim", type=int, default=320)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/nafdm_ixi")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--use_lpips", action="store_true", default=False)

    return parser.parse_args()


# ============================================================
# Warmup Cosine
# ============================================================
class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        base_lr,
        min_lr,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * self.epoch / self.warmup_epochs
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


# ============================================================
# Save Checkpoint
# ============================================================
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ============================================================
# Load Checkpoint
# ============================================================
def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.epoch = ckpt.get("scheduler_epoch", 0)

    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr = ckpt.get("best_psnr", 0.0)
    print(f"Resume from epoch {start_epoch}")
    return start_epoch, best_psnr


# ============================================================
# Train One Epoch
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    total_epochs,
):
    model.train()
    tracker = MetricTracker(use_lpips=False)
    running_loss = 0.0

    if HAS_TQDM:
        loader = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for batch in loader:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            pred, _ = model(noisy)
            loss, loss_dict = criterion(pred, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pred = pred.detach().clamp(0, 1)
        tracker.update(pred, gt)

        if HAS_TQDM and tracker.count > 0:
            loader.set_postfix(
                loss=f"{running_loss / tracker.count:.4f}",
                psnr=f"{tracker.summary()['psnr']:.2f}",
            )

    metrics = tracker.summary()
    metrics["loss"] = running_loss / max(tracker.count, 1)
    return metrics


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    tracker = MetricTracker(use_lpips=False)
    running_loss = 0.0

    if HAS_TQDM:
        loader = tqdm(loader, desc="Validation", leave=False)

    for batch in loader:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)

        pred, _ = model(noisy)
        loss, _ = criterion(pred, gt)
        running_loss += loss.item()

        pred = pred.clamp(0, 1)
        tracker.update(pred, gt)

    metrics = tracker.summary()
    metrics["loss"] = running_loss / max(tracker.count, 1)
    return metrics


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("NAFDMNet v3 Training (IXI)")
    print("=" * 60)
    print("Device       :", device)
    if device.type == "cuda":
        print("GPU          :", torch.cuda.get_device_name(0))
    print("Rician σ     :", args.sigma)
    print("=" * 60)

    # Dataset
    train_loader, val_loader = get_ixi_loaders(
        train_dir=args.data_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        sigma=args.sigma,
        target_size=args.target_size,
        num_workers=args.num_workers,
    )

    print(f"Train images : {len(train_loader.dataset)}")
    print(f"Val images   : {len(val_loader.dataset)}")

    # Model
    model = NAFDMNet(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    print(f"Params       : {count_params(model)/1e6:.2f} M")

    criterion = NAFDMLoss(use_lpips=args.use_lpips).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

    scaler = GradScaler(enabled=args.amp)
    start_epoch = 1
    best_psnr = 0.0

    # Resume
    if args.resume is not None:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler)

    print("\n" + "="*60)

    # Train Loop
    for epoch in range(start_epoch, args.epochs+1):
        lr = scheduler.step()
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, epoch, args.epochs
        )

        val_metrics = validate(model, val_loader, criterion, device)
        cost = time.time() - t0

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"LR={lr:.2e} | "
            f"TrainLoss={train_metrics['loss']:.4f} "
            f"TrainPSNR={train_metrics['psnr']:.2f} | "
            f"ValPSNR={val_metrics['psnr']:.2f} "
            f"SSIM={val_metrics['ssim']:.4f} "
            f"NRMSE={val_metrics['nrmse']:.4f} "
            f"({cost:.1f}s)"
        )

        # Save best
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler_epoch": scheduler.epoch,
                    "best_psnr": best_psnr,
                },
                os.path.join(args.output_dir, "best.pth")
            )
            print(f"==> Best PSNR updated: {best_psnr:.2f}")

        # Save last
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_epoch": scheduler.epoch,
                "best_psnr": best_psnr,
            },
            os.path.join(args.output_dir, "last.pth")
        )

    print("\n" + "="*60)
    print("Training Finished")
    print(f"Best PSNR: {best_psnr:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()