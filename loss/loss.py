"""
损失函数 — 支持 L1 + SSIM + Freq + Edge + LPIPS
LPIPS需要: pip install lpips
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════
# SSIM
# ═══════════════════════════════════════

def _gaussian_window(size=11, sigma=1.5):
    x = torch.arange(size).float() - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5,
                 C1=0.01 ** 2, C2=0.03 ** 2):
        super().__init__()
        self.ws = window_size
        self.C1, self.C2 = C1, C2
        w = _gaussian_window(window_size, sigma)
        self.register_buffer('window', w)

    def forward(self, pred, target):
        w = self.window.expand(pred.shape[1], 1, -1, -1)
        pad = self.ws // 2
        mu1 = F.conv2d(pred,   w, padding=pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, w, padding=pad, groups=pred.shape[1])
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu12 = mu1 * mu2
        s1 = F.conv2d(pred * pred,     w, padding=pad,
                      groups=pred.shape[1]) - mu1_sq
        s2 = F.conv2d(target * target, w, padding=pad,
                      groups=pred.shape[1]) - mu2_sq
        s12= F.conv2d(pred * target,   w, padding=pad,
                      groups=pred.shape[1]) - mu12
        ssim_map = ((2 * mu12 + self.C1) * (2 * s12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2))
        return 1 - ssim_map.mean()

# ═══════════════════════════════════════
# 频域Loss
# ═══════════════════════════════════════

class FrequencyLoss(nn.Module):
    def __init__(self, log_scale=True):
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred, target):
        pf = torch.fft.fft2(pred,   norm='ortho')
        tf = torch.fft.fft2(target, norm='ortho')
        if self.log_scale:
            pm = torch.log1p(pf.abs())
            tm = torch.log1p(tf.abs())
        else:
            pm, tm = pf.abs(), tf.abs()
        mag_loss   = F.l1_loss(pm, tm)
        phase_loss = F.l1_loss(torch.angle(pf), torch.angle(tf))
        return mag_loss + 0.1 * phase_loss

# ═══════════════════════════════════════
# 边缘Loss
# ═══════════════════════════════════════

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def _grad(self, x):
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._grad(pred), self._grad(target))

# ═══════════════════════════════════════
# Charbonnier Loss（比L1更平滑）
# ═══════════════════════════════════════

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))

# ═══════════════════════════════════════
# LPIPS（感知损失）
# ═══════════════════════════════════════

class LPIPSLoss(nn.Module):
    def __init__(self, net='vgg'):
        super().__init__()
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net)
            self.available = True
            print(f"[LPIPS] 使用 {net} 网络")
        except ImportError:
            print("[LPIPS] 未安装lpips，将跳过。pip install lpips")
            self.available = False

    def forward(self, pred, target):
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        # LPIPS需要输入在[-1,1]
        p = pred   * 2 - 1
        t = target * 2 - 1
        # 如果是单通道，重复3次
        if p.shape[1] == 1:
            p = p.repeat(1, 3, 1, 1)
            t = t.repeat(1, 3, 1, 1)
        return self.lpips_fn(p, t).mean()

# ═══════════════════════════════════════
# 总损失
# ═══════════════════════════════════════

class NAFDMLoss(nn.Module):
    """
    L = λ_char * L_char
    + λ_ssim * L_ssim
    + λ_freq * L_freq
    + λ_edge * L_edge
    + λ_lpips * L_lpips

    注意: Charbonnier代替原来的L1，效果更好
    """

    def __init__(self,
                 lambda_char=1.0,
                 lambda_ssim=0.5,
                 lambda_freq=0.2,
                 lambda_edge=0.1,
                 lambda_lpips=0.1,
                 use_lpips=True):
        super().__init__()
        self.lw = dict(
            char=lambda_char,
            ssim=lambda_ssim,
            freq=lambda_freq,
            edge=lambda_edge,
            lpips=lambda_lpips,
        )
        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.freq_loss = FrequencyLoss()
        self.edge_loss = EdgeLoss()
        self.lpips_loss = LPIPSLoss() if use_lpips else None

    def forward(self, pred, target):
        char = self.char_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        freq = self.freq_loss(pred, target)
        edge = self.edge_loss(pred, target)

        total = (self.lw['char'] * char +
                 self.lw['ssim'] * ssim +
                 self.lw['freq'] * freq +
                 self.lw['edge'] * edge)

        ld = {
            'char': char.item(),
            'ssim': ssim.item(),
            'freq': freq.item(),
            'edge': edge.item(),
            'lpips': 0.0,
            'total': 0.0,
        }

        if self.lpips_loss is not None and self.lpips_loss.available:
            lp = self.lpips_loss(pred.detach(), target.detach())
            # LPIPS只用于监控，不加入反传（防止显存爆炸）
            ld['lpips'] = lp.item()

        ld['total'] = total.item()
        return total, ld