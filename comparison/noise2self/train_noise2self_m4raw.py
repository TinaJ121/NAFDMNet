
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import h5py

sys.path.insert(0, r'E:\noise2self-master')
from mask import Masker
from models.unet import Unet


class M4RawSingleDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, augment=True,
                 contrast_list=('T1', 'T2', 'FLAIR')):
        self.patch_size = patch_size
        self.augment    = augment
        self.samples    = []

        import re
        pattern = re.compile(r'^\d+_(T1|T2|FLAIR)\d+\.h5$', re.IGNORECASE)
        for fname in sorted(os.listdir(data_dir)):
            if not pattern.match(fname):
                continue
            contrast = None
            for c in contrast_list:
                if c.upper() in fname.upper():
                    contrast = c
                    break
            if contrast is None:
                continue
            fp = os.path.join(data_dir, fname)
            try:
                with h5py.File(fp, 'r') as f:
                    n_frames = f['reconstruction_rss'].shape[0]
                for t in range(n_frames):
                    self.samples.append((fp, t))
            except Exception as e:
                print(f"[Skip] {fname}: {e}")

        print(f"[Noise2Self Dataset] {len(self.samples)} samples | {data_dir}")

    def __len__(self):
        return len(self.samples)

    def _load(self, fp, t):
        with h5py.File(fp, 'r') as f:
            img = f['reconstruction_rss'][t].astype(np.float32)
        vmax = img.max()
        return img / vmax if vmax > 0 else img

    def _crop(self, img):
        H, W = img.shape
        p    = self.patch_size
        if H < p:
            img = np.pad(img, ((0,p-H),(0,0)), 'reflect'); H = p
        if W < p:
            img = np.pad(img, ((0,0),(0,p-W)), 'reflect'); W = p
        t = np.random.randint(0, H-p+1)
        l = np.random.randint(0, W-p+1)
        return img[t:t+p, l:l+p]

    def _augment(self, img):
        if np.random.rand() > 0.5: img = np.fliplr(img).copy()
        if np.random.rand() > 0.5: img = np.flipud(img).copy()
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        return img

    def __getitem__(self, idx):
        fp, t = self.samples[idx]
        img   = self._load(fp, t)
        img   = self._crop(img)
        if self.augment:
            img = self._augment(img)
        return torch.from_numpy(img[np.newaxis])


class M4RawTestDataset(Dataset):
    TEST_CFG = {
        'T1':    {'input': 'T101',    'gt_list': ['T102', 'T103']},
        'T2':    {'input': 'T201',    'gt_list': ['T202', 'T203']},
        'FLAIR': {'input': 'FLAIR01', 'gt_list': ['FLAIR02']},
    }

    def __init__(self, data_dir, contrast_list=('T1', 'T2', 'FLAIR')):
        self.data_dir = data_dir
        self.samples  = []
        import re
        pattern  = re.compile(r'^(\d+)_(T101|T201|FLAIR01)\.h5$')
        patients = sorted(set(
            pattern.match(f).group(1)
            for f in os.listdir(data_dir)
            if pattern.match(f)))

        for pid in patients:
            for contrast in contrast_list:
                if contrast not in self.TEST_CFG: continue
                cfg     = self.TEST_CFG[contrast]
                inp_suf = cfg['input']
                gt_sufs = [s for s in cfg['gt_list']
                           if os.path.exists(
                               os.path.join(data_dir, f"{pid}_{s}.h5"))]
                if not gt_sufs: continue
                fp = os.path.join(data_dir, f"{pid}_{inp_suf}.h5")
                if not os.path.exists(fp): continue
                with h5py.File(fp, 'r') as f:
                    n_frames = f['reconstruction_rss'].shape[0]
                for t in range(n_frames):
                    self.samples.append((pid, contrast, t, inp_suf, gt_sufs))

        print(f"[Test Dataset] {len(patients)} patients, "
              f"{len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, contrast, t, inp_suf, gt_sufs = self.samples[idx]
        with h5py.File(
                os.path.join(self.data_dir, f"{pid}_{inp_suf}.h5"), 'r') as f:
            inp = f['reconstruction_rss'][t].astype(np.float32)
        gt_imgs = []
        for suf in gt_sufs:
            fp = os.path.join(self.data_dir, f"{pid}_{suf}.h5")
            if os.path.exists(fp):
                with h5py.File(fp, 'r') as f:
                    gt_imgs.append(f['reconstruction_rss'][t].astype(np.float32))
        gt  = np.mean(gt_imgs, axis=0)
        gt_max = gt.max() + 1e-8
        inp_n  = np.clip(inp / gt_max, 0, 1).astype(np.float32)
        gt_n   = np.clip(gt  / gt_max, 0, 1).astype(np.float32)
        return (torch.from_numpy(inp_n[np.newaxis]),
                torch.from_numpy(gt_n[np.newaxis]),
                pid, contrast, t)


@torch.no_grad()
def validate(model, masker, val_loader, device, n_samples=100):
    from rep2rep_inference import compute_metrics
    model.eval()
    psnrs, ssims, nrmses = [], [], []

    for i, (inp, gt, *_) in enumerate(val_loader):
        if i >= n_samples: break
        inp = inp.to(device)
        pred = masker.infer_full_image(inp, model)
        pred = pred.clamp(0., 1.)

        pred_np = pred[0, 0].cpu().numpy()
        gt_np   = gt[0, 0].numpy()
        m = compute_metrics(pred_np, gt_np)
        psnrs.append(m['PSNR'])
        ssims.append(m['SSIM'])
        nrmses.append(m['NRMSE'])

    model.train()
    return (float(np.mean(psnrs)),
            float(np.mean(ssims)),
            float(np.mean(nrmses)))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",       type=str, required=True)
    p.add_argument("--val_dir",         type=str, default=None)
    p.add_argument("--save_dir",        type=str,
                   default="./checkpoints/noise2self_m4raw")
    p.add_argument("--contrast_list",   nargs="+",
                   default=["T1", "T2", "FLAIR"])
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=0.001)
    p.add_argument("--patch_size",      type=int,   default=128)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--mask_width",      type=int,   default=3,
                   help="grid size for J-invariant masking (default 3)")
    p.add_argument("--mask_mode",       type=str,   default="interpolate",
                   choices=["zero", "interpolate"],
                   help="zero or interpolate (recommended interpolate)")
    p.add_argument("--save_every",      type=int,   default=10)
    p.add_argument("--val_every",       type=int,   default=1)
    p.add_argument("--resume",          type=str,   default=None)
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    masker = Masker(width=args.mask_width,
                    mode=args.mask_mode,
                    infer_single_pass=False)
    print(f"Masker: width={args.mask_width}, mode={args.mask_mode}, "
          f"n_masks={len(masker)}")

    model = Unet(n_channel_in=1, n_channel_out=1,
                 residual=False, down='conv',
                 up='tconv', activation='selu').to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = M4RawSingleDataset(
        data_dir=args.train_dir,
        patch_size=args.patch_size,
        augment=True,
        contrast_list=args.contrast_list)
    train_loader  = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True)

    val_loader = None
    if args.val_dir:
        val_dataset = M4RawTestDataset(
            data_dir=args.val_dir,
            contrast_list=args.contrast_list)
        val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_psnr   = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_psnr   = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    print(f"\nTraining for {args.epochs} epochs")
    print(f"Batch={args.batch_size}, LR={args.lr}, "
          f"Patch={args.patch_size}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch [{epoch+1:3d}/{args.epochs}]",
                    dynamic_ncols=True, unit="batch")

        for noisy in pbar:
            noisy = noisy.to(device)

            mask_i = np.random.randint(0, len(masker))
            net_input, mask = masker.mask(noisy, mask_i)

            net_output = model(net_input)

            loss = criterion(net_output * mask, noisy * mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1
            pbar.set_postfix(
                loss=f"{running_loss/n_batches:.5f}",
                best=f"{best_psnr:.2f}dB")

        avg_loss = running_loss / max(n_batches, 1)

        val_str = ""
        if val_loader and (epoch + 1) % args.val_every == 0:
            psnr, ssim, nrmse = validate(model, masker, val_loader, device)
            val_str = (f"   Val PSNR={psnr:.2f}dB  "
                       f"SSIM={ssim:.4f}  NRMSE={nrmse:.4f}")
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({"epoch": epoch+1,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_psnr": best_psnr},
                           save_dir / "best.pth")
                val_str += " -> saved best.pth"

        tqdm.write(f"Epoch {epoch+1:3d}/{args.epochs}  "
                   f"loss={avg_loss:.5f}{val_str}")

        if (epoch + 1) % args.save_every == 0:
            torch.save({"epoch": epoch+1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_psnr": best_psnr},
                       save_dir / f"epoch_{epoch+1:03d}.pth")

    torch.save({"epoch": args.epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_psnr},
               save_dir / "final.pth")
    print(f"Done. Best PSNR={best_psnr:.2f} dB")


if __name__ == "__main__":
    main()