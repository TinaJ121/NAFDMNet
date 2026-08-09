"""
test_ixi.py

NAFDMNet on IXI Dataset
Rician Noise σ=0.06

Output
PSNR
SSIM
NRMSE
LPIPS

Mean ± Std

Save:
Input
Noisy
Output
GT
Residual
Heatmap
CSV
JSON
"""

import os
import json
import argparse
import numpy as np

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import (
    NAFDMNet,
    count_params
)

from datasets.dataset_ixi import get_ixi_test_loader

from metrics_net import (
    compute_psnr,
    compute_ssim,
    compute_nrmse
)

try:
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg")
    HAS_LPIPS = True
except:
    HAS_LPIPS = False
    print("LPIPS not installed.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_ixi"
    )

    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.06
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=160
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=320
    )

    parser.add_argument(
        "--num_vis",
        type=int,
        default=10
    )

    return parser.parse_args()


def compute_lpips(pred, gt, device):
    if not HAS_LPIPS:
        return 0

    lpips_fn.to(device)

    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
        gt = gt.repeat(1, 3, 1, 1)

    pred = pred * 2 - 1
    gt = gt * 2 - 1

    with torch.no_grad():
        score = lpips_fn(pred, gt)

    return score.mean().item()


def save_visualization(
    clean,
    noisy,
    pred,
    save_path,
    psnr,
    ssim,
    nrmse
):
    residual = np.abs(pred - clean)

    fig, ax = plt.subplots(1, 5, figsize=(22, 5))

    ax[0].imshow(clean, cmap="gray")
    ax[0].set_title("Ground Truth")

    ax[1].imshow(noisy, cmap="gray")
    ax[1].set_title("Noisy")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")

    ax[3].imshow(residual, cmap="hot")
    ax[3].set_title("Residual")

    ax[4].imshow(residual, cmap="jet")
    ax[4].set_title("Heatmap")

    for a in ax:
        a.axis("off")

    plt.suptitle(
        f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  NRMSE={nrmse:.4f}"
    )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def main():
    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("Device :", device)
    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))
    print("=" * 70)

    ####################################################
    # Model
    ####################################################
    model = NAFDMNet(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print("Checkpoint :", args.checkpoint)
    print("Params : %.2f M" % (count_params(model) / 1e6))

    ####################################################
    # Dataset
    ####################################################
    loader = get_ixi_test_loader(
        test_dir=args.data_dir,
        sigma=args.noise_sigma,
        batch_size=args.batch_size,
        num_workers=0
    )

    print("Test images :", len(loader))

    ####################################################
    # Results
    ####################################################
    results = []
    vis_num = 0

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)
            filename = batch["filename"][0]

            pred, _ = model(noisy)
            pred = torch.clamp(pred, 0, 1)

            psnr = compute_psnr(pred, gt)
            ssim = compute_ssim(pred, gt)
            nrmse = compute_nrmse(pred, gt)
            lpips_score = compute_lpips(pred, gt, device)

            results.append({
                "filename": filename,
                "psnr": float(psnr),
                "ssim": float(ssim),
                "nrmse": float(nrmse),
                "lpips": float(lpips_score)
            })

            if vis_num < args.num_vis:
                save_visualization(
                    gt[0, 0].cpu().numpy(),
                    noisy[0, 0].cpu().numpy(),
                    pred[0, 0].cpu().numpy(),
                    os.path.join(
                        args.output_dir,
                        "vis",
                        filename
                    ),
                    psnr,
                    ssim,
                    nrmse
                )
                vis_num += 1

    ####################################################
    # Statistics
    ####################################################
    psnrs = [x["psnr"] for x in results]
    ssims = [x["ssim"] for x in results]
    nrmses = [x["nrmse"] for x in results]
    lpipses = [x["lpips"] for x in results]

    print("\n")
    print("=" * 70)
    print("NAFDMNet on IXI")
    print("=" * 70)

    print("PSNR  : %.4f ± %.4f dB" % (np.mean(psnrs), np.std(psnrs)))
    print("SSIM  : %.4f ± %.4f" % (np.mean(ssims), np.std(ssims)))
    print("NRMSE : %.4f ± %.4f" % (np.mean(nrmses), np.std(nrmses)))
    print("LPIPS : %.4f ± %.4f" % (np.mean(lpipses), np.std(lpipses)))

    ####################################################
    # Save JSON
    ####################################################
    json.dump(
        {
            "PSNR_mean": float(np.mean(psnrs)),
            "PSNR_std": float(np.std(psnrs)),
            "SSIM_mean": float(np.mean(ssims)),
            "SSIM_std": float(np.std(ssims)),
            "NRMSE_mean": float(np.mean(nrmses)),
            "NRMSE_std": float(np.std(nrmses)),
            "LPIPS_mean": float(np.mean(lpipses)),
            "LPIPS_std": float(np.std(lpipses)),
            "results": results
        },
        open(
            os.path.join(args.output_dir, "results.json"),
            "w"
        ),
        indent=4
    )

    ####################################################
    # Save CSV
    ####################################################
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("filename,psnr,ssim,nrmse,lpips\n")
        for r in results:
            f.write(f"{r['filename']},{r['psnr']:.4f},{r['ssim']:.4f},{r['nrmse']:.4f},{r['lpips']:.4f}\n")

    print("\nSaved to")
    print(args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()