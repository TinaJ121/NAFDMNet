import os
import sys
sys.path.insert(0, r'F:\IXI')

import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import bm3d
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False
    print('[Error] pip install bm3d')

from dataset_test import get_m4raw_test_loader
from metrics_net import compute_psnr, compute_ssim, compute_nrmse

try:
    import lpips
    loss_fn_lpips = lpips.LPIPS(net='vgg')
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       type=str, required=True)
    p.add_argument('--output_dir',     type=str, default='./test_results/bm3d_m4raw')
    p.add_argument('--sequences',      type=str, nargs='+', default=['T1', 'T2', 'FLAIR'])
    p.add_argument('--frames_per_seq', type=int, default=18)
    p.add_argument('--target_size',    type=int, default=256)
    p.add_argument('--n_vis',          type=int, default=20)
    p.add_argument('--sigma_mode',     type=str, default='auto',
                   choices=['auto', 'fixed'],
                   help='auto=estimate sigma via MAD, fixed=fixed sigma=0.1')
    p.add_argument('--sigma_fixed',    type=float, default=0.1)
    return p.parse_args()

def estimate_sigma_mad(img):
    """Estimate noise sigma via MAD (consistent with NAFDMNet)"""
    from scipy.ndimage import laplace
    filtered = laplace(img)
    med = np.median(filtered)
    mad = np.median(np.abs(filtered - med))
    return mad / 0.6745

def save_vis(noisy, pred, ref, save_dir, idx, psnr, ssim, nrmse,
             patient='', seq='', sigma=0.0):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0b0f1a')
    titles = [
        'Input (single scan)',
        f'BM3D (σ={sigma:.4f})',
        'GT (multi-average)',
        '|Diff| BM3D-GT'
    ]
    imgs = [noisy, pred, ref, np.abs(pred - ref)]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
    fig.suptitle(
        f'BM3D | Patient={patient}  Seq={seq}\n'
        f'PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  NRMSE={nrmse:.4f}',
        color='#60a5fa', fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'vis_{idx:04d}.png'),
                dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

def run_bm3d(loader, sigma_mode, sigma_fixed, n_vis, vis_dir):
    assert HAS_BM3D, 'pip install bm3d'

    results = []
    vis_n = 0

    for i, batch in enumerate(loader):
        noisy_t = batch['noisy']  # [1,1,H,W]
        gt_t    = batch['gt']

        noisy_np = noisy_t[0, 0].numpy().astype(np.float64)
        gt_np    = gt_t[0, 0].numpy().astype(np.float32)

        # Estimate sigma
        if sigma_mode == 'auto':
            sigma = float(estimate_sigma_mad(noisy_np))
            sigma = np.clip(sigma, 1e-6, 1.0)
        else:
            sigma = sigma_fixed

        # BM3D denoising
        try:
            denoised = bm3d.bm3d(noisy_np, sigma_psd=sigma,
                                  stage_arg=bm3d.BM3DStages.ALL_STAGES)
            denoised = np.clip(denoised, 0, 1).astype(np.float32)
        except Exception as e:
            print(f'  [Skip {i}] BM3D error: {e}')
            denoised = noisy_np.astype(np.float32)

        # Compute metrics
        pred_t = torch.from_numpy(denoised).unsqueeze(0).unsqueeze(0)
        gt_tt  = gt_t

        p = compute_psnr(pred_t, gt_tt)
        s = compute_ssim(pred_t, gt_tt)
        n = compute_nrmse(pred_t, gt_tt)

        # LPIPS
        lpips_val = 0.0
        if HAS_LPIPS:
            try:
                pred_3c = pred_t.repeat(1, 3, 1, 1) * 2 - 1
                gt_3c   = gt_tt.repeat(1, 3, 1, 1) * 2 - 1
                lpips_val = float(loss_fn_lpips(pred_3c, gt_3c).item())
            except Exception:
                pass

        results.append({
            'idx'    : i,
            'psnr'   : p,
            'ssim'   : s,
            'nrmse'  : n,
            'lpips'  : lpips_val,
            'sigma'  : float(sigma),
            'patient': batch.get('patient', [''])[0],
            'seq'    : batch.get('seq',     [''])[0],
            'frame'  : int(batch.get('frame', [0])[0]),
            'n_avg'  : int(batch.get('n_avg',  [0])[0]),
        })

        if vis_dir and vis_n < n_vis:
            save_vis(
                noisy_np.astype(np.float32), denoised, gt_np,
                vis_dir, vis_n, p, s, n,
                patient=batch.get('patient', [''])[0],
                seq=batch.get('seq', [''])[0],
                sigma=sigma,
            )
            vis_n += 1

        if (i + 1) % 50 == 0:
            recent = results[-50:]
            print(f'  [{i+1}/{len(loader)}] '
                  f'PSNR={np.mean([r["psnr"] for r in recent]):.2f}  '
                  f'SSIM={np.mean([r["ssim"] for r in recent]):.4f}  '
                  f'sigma={sigma:.4f}')

    return results

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loader = get_m4raw_test_loader(
        args.data_dir,
        batch_size=1,
        num_workers=0,
        sequences=tuple(args.sequences),
        frames_per_seq=args.frames_per_seq,
        target_size=args.target_size,
        pin_memory=False,
    )
    print(f'Test batches: {len(loader)}')
    print(f'Sigma mode  : {args.sigma_mode}')
    if args.sigma_mode == 'fixed':
        print(f'Sigma fixed : {args.sigma_fixed}')
    print()

    vis_dir = os.path.join(args.output_dir, 'vis')
    print('Running BM3D...')
    results = run_bm3d(
        loader, args.sigma_mode, args.sigma_fixed,
        args.n_vis, vis_dir,
    )

    # Summary
    all_psnr  = [r['psnr']  for r in results]
    all_ssim  = [r['ssim']  for r in results]
    all_nrmse = [r['nrmse'] for r in results]
    all_lpips = [r['lpips'] for r in results]

    print(f'\n{"="*60}')
    print(f'  BM3D Test Results [M4Raw]')
    print(f'  Samples : {len(results)}')
    print(f'  PSNR    = {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f} dB')
    print(f'  SSIM    = {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}')
    print(f'  NRMSE   = {np.mean(all_nrmse):.4f} ± {np.std(all_nrmse):.4f}')
    print(f'  LPIPS   = {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}')
    print(f'{"="*60}')

    # Per-sequence breakdown
    print('\n  Per-sequence breakdown:')
    for seq in args.sequences:
        seq_res = [r for r in results if r['seq'] == seq]
        if seq_res:
            psnrs  = [r['psnr']  for r in seq_res]
            ssims  = [r['ssim']  for r in seq_res]
            nrmses = [r['nrmse'] for r in seq_res]
            n_a    = seq_res[0]['n_avg']
            print(f'  {seq:6s} (GT={n_a} averages): '
                  f'PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}  '
                  f'SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}  '
                  f'NRMSE={np.mean(nrmses):.4f}±{np.std(nrmses):.4f}  '
                  f'(n={len(seq_res)})')

    print(f'\n  [Paper Table Format]')
    print(f'  BM3D | '
          f'PSNR={np.mean(all_psnr):.2f}±{np.std(all_psnr):.2f} | '
          f'SSIM={np.mean(all_ssim):.4f}±{np.std(all_ssim):.4f} | '
          f'NRMSE={np.mean(all_nrmse):.4f}±{np.std(all_nrmse):.4f} | '
          f'LPIPS={np.mean(all_lpips):.4f}')

    # Save JSON and CSV
    out_json = os.path.join(args.output_dir, 'results_bm3d.json')
    with open(out_json, 'w') as f:
        json.dump({
            'summary': {
                'psnr' : float(np.mean(all_psnr)),
                'ssim' : float(np.mean(all_ssim)),
                'nrmse': float(np.mean(all_nrmse)),
                'lpips': float(np.mean(all_lpips)),
                'psnr_std' : float(np.std(all_psnr)),
                'ssim_std' : float(np.std(all_ssim)),
                'nrmse_std': float(np.std(all_nrmse)),
            },
            'per_sample': results,
        }, f, indent=2)

    csv_path = os.path.join(args.output_dir, 'results_bm3d.csv')
    with open(csv_path, 'w') as f:
        f.write('idx,patient,seq,frame,n_avg,psnr,ssim,nrmse,lpips,sigma\n')
        for r in results:
            f.write(f"{r['idx']},{r['patient']},{r['seq']},"
                    f"{r['frame']},{r['n_avg']},"
                    f"{r['psnr']:.4f},{r['ssim']:.4f},"
                    f"{r['nrmse']:.4f},{r['lpips']:.4f},"
                    f"{r['sigma']:.6f}\n")

    print(f'\nResults → {out_json}')
    print(f'CSV     → {csv_path}')
    print(f'Vis     → {vis_dir}/')

if __name__ == '__main__':
    main()