import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from noise2noise_unet import UNet
from rep2rep_dataset_m4raw import M4RawRep2RepDataset, M4RawRep2RepTestDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",       type=str, required=True)
    p.add_argument("--val_dir",         type=str, default=None)
    p.add_argument("--save_dir",        type=str,
                   default="./checkpoints/noise2noise_m4raw")
    p.add_argument("--contrast_list",   nargs="+",
                   default=["T1", "T2", "FLAIR"])
    p.add_argument("--epochs",          type=int,   default=80)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr_max",          type=float, default=0.001)
    p.add_argument("--adam_beta2",      type=float, default=0.99)
    p.add_argument("--rampup_epochs",   type=int,   default=10)
    p.add_argument("--rampdown_epochs", type=int,   default=30)
    p.add_argument("--patch_size",      type=int,   default=128)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--resume",          type=str,   default=None)
    return p.parse_args()


def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    return 1.0

def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0

def get_lr_beta1(epoch, num_epochs, lr_max, rampup_len, rampdown_len,
                 beta1_initial=0.9, beta1_rampdown=0.5):
    ru = rampup(epoch, rampup_len)
    rd = rampdown(epoch, num_epochs, rampdown_len)
    lr = ru * rd * lr_max
    beta1 = rd * beta1_initial + (1.0 - rd) * beta1_rampdown
    return lr, beta1


@torch.no_grad()
def validate(model, val_loader, device, n_samples=50):
    from rep2rep_inference import compute_metrics
    model.eval()
    psnrs, ssims, nrmses = [], [], []
    for i, batch in enumerate(val_loader):
        if i >= n_samples:
            break
        y, sigma, gt, *_ = batch
        y  = y.to(device) - 0.5
        pred = (model(y) + 0.5).clamp(0., 1.)
        pred_np = pred[0, 0].cpu().numpy()
        gt_np   = gt[0, 0].numpy()
        m = compute_metrics(pred_np, gt_np)
        psnrs.append(m["PSNR"])
        ssims.append(m["SSIM"])
        nrmses.append(m["NRMSE"])
    model.train()
    return (float(np.mean(psnrs)) if psnrs else 0.0,
            float(np.mean(ssims)) if ssims else 0.0,
            float(np.mean(nrmses)) if nrmses else 0.0)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet(in_channels=1, base_ch=48).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()

    train_dataset = M4RawRep2RepDataset(
        data_dir=args.train_dir,
        patch_size=args.patch_size,
        augment=True,
        contrast_list=args.contrast_list,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = M4RawRep2RepTestDataset(
            data_dir=args.val_dir,
            n_avg=1,
            contrast_list=args.contrast_list,
        )
        val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_psnr   = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        best_psnr   = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_max,
        betas=(0.9, args.adam_beta2),
    )

    print(f"Training for {args.epochs} epochs  "
          f"(batch={args.batch_size}, lr_max={args.lr_max})")

    for epoch in range(start_epoch, args.epochs):
        lr, beta1 = get_lr_beta1(
            epoch, args.epochs, args.lr_max,
            args.rampup_epochs, args.rampdown_epochs,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["betas"] = (beta1, args.adam_beta2)

        model.train()
        running_loss = 0.0
        n_batches    = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch [{epoch+1:3d}/{args.epochs}]",
                    dynamic_ncols=True, unit="batch")

        for y1, sigma1, y2 in pbar:
            y1 = y1.to(device) - 0.5
            y2 = y2.to(device) - 0.5

            pred = model(y1)
            loss = criterion(pred, y2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1
            pbar.set_postfix(
                loss=f"{running_loss/n_batches:.5f}",
                lr=f"{lr:.2e}",
                best=f"{best_psnr:.2f}dB",
            )

        avg_loss = running_loss / max(n_batches, 1)

        val_str = ""
        if val_loader:
            psnr, ssim, nrmse = validate(model, val_loader, device)
            val_str = (f"  Val PSNR={psnr:.2f}dB  "
                       f"SSIM={ssim:.4f}  NRMSE={nrmse:.4f}")
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({"epoch": epoch+1, "model": model.state_dict(),
                            "best_psnr": best_psnr},
                           save_dir / "best.pth")
                val_str += " -> saved best.pth"

        tqdm.write(f"Epoch {epoch+1:3d}/{args.epochs}  "
                   f"loss={avg_loss:.5f}  lr={lr:.2e}{val_str}")

        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": epoch+1, "model": model.state_dict(),
                        "best_psnr": best_psnr},
                       save_dir / f"epoch_{epoch+1:03d}.pth")

    torch.save({"epoch": args.epochs, "model": model.state_dict(),
                "best_psnr": best_psnr},
               save_dir / "final.pth")
    print(f"Done. Best PSNR={best_psnr:.2f} dB")


if __name__ == "__main__":
    main()