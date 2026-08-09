"""
=========================================================
Test Noise2Self on IXI Dataset
=========================================================
"""

import os
import csv

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import cfg, make_dirs

from datasets.ixi_dataset import IXIDataset

from models.unet import UNet

from utils.metrics import MetricTracker


# =====================================================
# Result Directory
# =====================================================

make_dirs()

os.makedirs(

    os.path.join(

        cfg.result_dir,

        "denoised"

    ),

    exist_ok=True

)


# =====================================================
# Dataset
# =====================================================

test_dataset = IXIDataset(

    root_dir=cfg.test_dir,

    sigma=cfg.sigma,

    mask_ratio=cfg.mask_ratio,

    training=False

)

test_loader = DataLoader(

    test_dataset,

    batch_size=1,

    shuffle=False,

    num_workers=cfg.num_workers,

    pin_memory=True

)


# =====================================================
# Model
# =====================================================

model = UNet(

    in_channels=1,

    out_channels=1

)

checkpoint = torch.load(

    cfg.best_model,

    map_location=cfg.device

)

model.load_state_dict(

    checkpoint["model"]

)

model.to(

    cfg.device

)

model.eval()


# =====================================================
# Metric
# =====================================================

tracker = MetricTracker()


# =====================================================
# CSV
# =====================================================

csv_path = os.path.join(

    cfg.result_dir,

    "metrics.csv"

)

csv_file = open(

    csv_path,

    "w",

    newline=""

)

writer = csv.writer(

    csv_file

)

writer.writerow(

    [

        "Image",

        "PSNR",

        "SSIM",

        "NRMSE"

    ]

)

print("=" * 60)

print("Testing Started")

print("=" * 60)
# =====================================================
# Test
# =====================================================

with torch.no_grad():

    for index, batch in enumerate(test_loader):

        # ---------------------------------------------
        # Load Data
        # ---------------------------------------------

        inputs = batch["input"].to(cfg.device)

        clean = batch["clean"].to(cfg.device)

        image_name = batch["name"][0]

        # ---------------------------------------------
        # Forward
        # ---------------------------------------------

        outputs = model(inputs)

        outputs = torch.clamp(
            outputs,
            0.0,
            1.0
        )

        # ---------------------------------------------
        # Tensor -> NumPy
        # ---------------------------------------------

        prediction = outputs.squeeze().cpu().numpy()

        target = clean.squeeze().cpu().numpy()

        # ---------------------------------------------
        # Calculate Metrics
        # ---------------------------------------------

        tracker.update(
            prediction,
            target
        )

        psnr = tracker.psnr[-1]

        ssim = tracker.ssim[-1]

        nrmse = tracker.nrmse[-1]

        # ---------------------------------------------
        # Save CSV
        # ---------------------------------------------

        writer.writerow(

            [

                image_name,

                "{:.6f}".format(psnr),

                "{:.6f}".format(ssim),

                "{:.6f}".format(nrmse)

            ]

        )

        # ---------------------------------------------
        # Save Image
        # ---------------------------------------------

        save_image = (

            prediction * 255.0

        ).astype(

            np.uint8

        )

        save_path = os.path.join(

            cfg.result_dir,

            "denoised",

            image_name

        )

        cv2.imwrite(

            save_path,

            save_image

        )

        # ---------------------------------------------
        # Print
        # ---------------------------------------------

        print(

            "[{}/{}] {} | PSNR {:.4f} | SSIM {:.4f} | NRMSE {:.4f}".format(

                index + 1,

                len(test_loader),

                image_name,

                psnr,

                ssim,

                nrmse

            )

        )
# =====================================================
# Summary
# =====================================================

mean_result, std_result = tracker.summary()

summary_path = os.path.join(

    cfg.result_dir,

    "summary.txt"

)

with open(

    summary_path,

    "w"

) as f:

    f.write("=" * 60 + "\n")

    f.write("Noise2Self Evaluation Results\n")

    f.write("=" * 60 + "\n\n")

    f.write(
        "Total Test Images : {}\n\n".format(
            len(test_loader)
        )
    )

    f.write(
        "PSNR Mean : {:.6f}\n".format(
            mean_result["PSNR"]
        )
    )

    f.write(
        "PSNR Std  : {:.6f}\n\n".format(
            std_result["PSNR"]
        )
    )

    f.write(
        "SSIM Mean : {:.6f}\n".format(
            mean_result["SSIM"]
        )
    )

    f.write(
        "SSIM Std  : {:.6f}\n\n".format(
            std_result["SSIM"]
        )
    )

    f.write(
        "NRMSE Mean : {:.6f}\n".format(
            mean_result["NRMSE"]
        )
    )

    f.write(
        "NRMSE Std  : {:.6f}\n".format(
            std_result["NRMSE"]
        )
    )

csv_file.close()

print("\n")

print("=" * 60)

print("Testing Finished.")

print("=" * 60)

print(
    "Results Saved To : {}".format(
        cfg.result_dir
    )
)

print(
    "Metrics CSV : {}".format(
        csv_path
    )
)

print(
    "Summary TXT : {}".format(
        summary_path
    )
)

print("=" * 60)