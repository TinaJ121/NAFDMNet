import torch
import torch.nn.functional as F
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ===========================================
# PSNR (支持 batch [B,1,H,W])
# ===========================================
def compute_psnr(pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    # 处理 batch: [B,1,H,W]
    if pred.ndim == 4:
        pred = pred[:, 0]
        gt = gt[:, 0]
        scores = []
        for i in range(pred.shape[0]):
            scores.append(peak_signal_noise_ratio(gt[i], pred[i], data_range=1.0))
        return float(np.mean(scores))

    # 单张图
    pred = pred.squeeze()
    gt = gt.squeeze()
    return peak_signal_noise_ratio(gt, pred, data_range=1.0)


# ===========================================
# SSIM (修复 win_size 报错，支持 batch)
# ===========================================
def compute_ssim(pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    # 处理 batch: [B,1,H,W]
    if pred.ndim == 4:
        pred = pred[:, 0]
        gt = gt[:, 0]
        scores = []
        for i in range(pred.shape[0]):
            scores.append(structural_similarity(
                gt[i], pred[i], data_range=1.0, win_size=7
            ))
        return float(np.mean(scores))

    # 单张图
    pred = pred.squeeze()
    gt = gt.squeeze()
    return structural_similarity(gt, pred, data_range=1.0, win_size=7)


# ===========================================
# NRMSE
# ===========================================
def compute_nrmse(pred, gt):
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()

    mse = F.mse_loss(pred, gt)
    rmse = torch.sqrt(mse)
    norm = gt.max() - gt.min()

    return (rmse / (norm + 1e-8)).item()


# ===========================================
# MetricTracker
# ===========================================
class MetricTracker:
    def __init__(self, use_lpips=False):
        self.use_lpips = use_lpips
        self.reset()

    def reset(self):
        self.psnr = []
        self.ssim = []
        self.nrmse = []
        self.count = 0

    def update(self, pred, gt):
        self.psnr.append(compute_psnr(pred, gt))
        self.ssim.append(compute_ssim(pred, gt))
        self.nrmse.append(compute_nrmse(pred, gt))
        self.count += 1

    def summary(self):
        if self.count == 0:
            return {"psnr": 0, "ssim": 0, "nrmse": 0}

        return {
            "psnr": float(np.mean(self.psnr)),
            "ssim": float(np.mean(self.ssim)),
            "nrmse": float(np.mean(self.nrmse))
        }

    def summary_str(self):
        m = self.summary()
        return f"PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} NRMSE={m['nrmse']:.4f}"