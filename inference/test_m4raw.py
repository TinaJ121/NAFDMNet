import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'E:\Restormer-main\Denoising')

from models.nafdm_net_  import NAFDMNet, count_params
from dataset_m4raw       import get_m4raw_test_loader
from metrics_net          import compute_psnr, compute_ssim, compute_nrmse

try:
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg')
    HAS_LPIPS = True
except:
    HAS_LPIPS = False
    print("[Warning] lpips not installed, skipping LPIPS calculation")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',     type=str, required=True)
    p.add_argument('--data_dir',       type=str, required=True)
    p.add_argument('--output_dir',     type=str, default='./test_results/nafdm_v3')
    p.add_argument('--sequences',      type=str, nargs='+', default=['T1','T2','FLAIR'])
    p.add_argument('--frames_per_seq', type=int, default=18)
    p.add_argument('--hidden_dim',     type=int, default=160)
    p.add_argument('--latent_dim',     type=int, default=320)
    p.add_argument('--n_vis',          type=int, default=10,
                   help='Number of visualization images to save per sequence')
    return p.parse_args()


def save_vis(noisy, pred, gt, save_path, psnr, ssim, nrmse, pid, seq, frame):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0d1117')
    titles = ['Input (single scan)', 'NAFDMNet v3 Output',
              'GT (multi-average)',  '|Residual|']
    imgs   = [noisy, pred, gt, np.abs(pred - gt)]
    vmaxs  = [1, 1, 1, 0.1]
    for ax, img, title, vmax in zip(axes, imgs, titles, vmaxs):
        ax.imshow(img, cmap='gray', vmin=0, vmax=vmax, interpolation='bicubic')
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
    fig.suptitle(
        f'NAFDMNet v3 | {pid}  {seq}  frame={frame}\n'
        f'PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  NRMSE={nrmse:.4f}',
        color='#58a6ff', fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


def compute_lpips(pred, gt, device):
    if not HAS_LPIPS:
        return 0.0
    lpips_fn.to(device)
    p = (pred * 2 - 1).repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred*2-1
    t = (gt   * 2 - 1).repeat(1, 3, 1, 1) if gt.shape[1]   == 1 else gt*2-1
    with torch.no_grad():
        return lpips_fn(p, t).mean().item()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model = NAFDMNet(hidden_dim=args.hidden_dim,
                     latent_dim=args.latent_dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded: {args.checkpoint}')
    print(f'Best val PSNR: {ckpt.get("best_psnr", "N/A")}')
    print(f'Params: {count_params(model)/1e6:.2f}M\n')

    # Load test dataset
    loader = get_m4raw_test_loader(
        args.data_dir,
        batch_size=1,
        sequences=tuple(args.sequences),
        frames_per_seq=args.frames_per_seq,
    )
    print(f'Test samples: {len(loader)}\n')

    # Collect results by sequence
    seq_results = {seq: [] for seq in args.sequences}
    vis_count   = {seq: 0  for seq in args.sequences}

    print('Running inference...')
    for i, batch in enumerate(loader):
        noisy = batch['noisy'].to(device)
        gt    = batch['gt'].to(device)
        pid   = batch['patient'][0] if isinstance(batch['patient'], (list,tuple)) else batch['patient']
        seq   = batch['seq'][0]     if isinstance(batch['seq'],     (list,tuple)) else batch['seq']
        frame = int(batch['frame'][0]) if isinstance(batch['frame'], (list,tuple)) else int(batch['frame'])
        n_avg = int(batch['n_avg'][0])  if isinstance(batch['n_avg'],  (list,tuple)) else int(batch['n_avg'])

        with torch.no_grad():
            pred, inter = model(noisy)
        pred_c = torch.clamp(pred, 0, 1)

        p  = compute_psnr(pred_c, gt)
        s  = compute_ssim(pred_c, gt)
        n  = compute_nrmse(pred_c, gt)
        lp = compute_lpips(pred_c, gt, device)

        seq_results[seq].append({
            'patient': pid, 'frame': frame, 'n_avg': n_avg,
            'psnr': float(p), 'ssim': float(s),
            'nrmse': float(n), 'lpips': float(lp),
        })

        # Save visualizations
        if seq in vis_count and vis_count[seq] < args.n_vis:
            save_path = os.path.join(
                args.output_dir, 'vis', seq,
                f'{pid}_frame{frame:02d}_psnr{p:.2f}.png')
            save_vis(
                noisy[0,0].cpu().numpy(),
                pred_c[0,0].cpu().numpy(),
                gt[0,0].cpu().numpy(),
                save_path, p, s, n, pid, seq, frame)
            vis_count[seq] += 1

        if (i+1) % 100 == 0:
            print(f'  [{i+1}/{len(loader)}]')

    # ══════════════════════════════════
    # Print results
    # ══════════════════════════════════
    print(f'\n{"="*65}')
    print(f'  NAFDMNet v3 Test Results  [M4Raw]')
    print(f'{"="*65}')

    all_psnr, all_ssim, all_nrmse, all_lpips = [], [], [], []

    for seq in args.sequences:
        res = seq_results[seq]
        if not res:
            continue
        psnrs  = [r['psnr']  for r in res]
        ssims  = [r['ssim']  for r in res]
        nrmses = [r['nrmse'] for r in res]
        lpipss = [r['lpips'] for r in res]
        n_avg  = res[0]['n_avg']

        print(f'\n  {seq} (GT={n_avg} averages, n={len(res)}):')
        print(f'    PSNR  = {np.mean(psnrs):.4f} ± {np.std(psnrs):.4f} dB')
        print(f'    SSIM  = {np.mean(ssims):.4f} ± {np.std(ssims):.4f}')
        print(f'    NRMSE = {np.mean(nrmses):.4f} ± {np.std(nrmses):.4f}')
        print(f'    LPIPS = {np.mean(lpipss):.4f} ± {np.std(lpipss):.4f}')

        all_psnr.extend(psnrs)
        all_ssim.extend(ssims)
        all_nrmse.extend(nrmses)
        all_lpips.extend(lpipss)

    print(f'\n  {"─"*50}')
    print(f'  Overall (n={len(all_psnr)}):')
    print(f'    PSNR  = {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f} dB')
    print(f'    SSIM  = {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}')
    print(f'    NRMSE = {np.mean(all_nrmse):.4f} ± {np.std(all_nrmse):.4f}')
    print(f'    LPIPS = {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}')

    # Paper Table format
    print(f'\n  [Paper Table Format]')
    for seq in args.sequences:
        res = seq_results[seq]
        if res:
            psnrs  = [r['psnr']  for r in res]
            ssims  = [r['ssim']  for r in res]
            nrmses = [r['nrmse'] for r in res]
            lpipss = [r['lpips'] for r in res]
            print(f'  {seq:6s}: PSNR={np.mean(psnrs):.2f}  '
                  f'SSIM={np.mean(ssims):.4f}  '
                  f'NRMSE={np.mean(nrmses):.4f}  '
                  f'LPIPS={np.mean(lpipss):.4f}')
    print(f'  {"─"*50}')
    print(f'  Avg   : PSNR={np.mean(all_psnr):.2f}  '
          f'SSIM={np.mean(all_ssim):.4f}  '
          f'NRMSE={np.mean(all_nrmse):.4f}  '
          f'LPIPS={np.mean(all_lpips):.4f}')
    print(f'{"="*65}')

    # Save JSON
    out_json = os.path.join(args.output_dir, 'results_nafdm_v3.json')
    with open(out_json, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'best_psnr' : ckpt.get('best_psnr'),
            'per_seq'   : {seq: {
                'psnr' : float(np.mean([r['psnr']  for r in seq_results[seq]])),
                'ssim' : float(np.mean([r['ssim']  for r in seq_results[seq]])),
                'nrmse': float(np.mean([r['nrmse'] for r in seq_results[seq]])),
                'lpips': float(np.mean([r['lpips'] for r in seq_results[seq]])),
                'n'    : len(seq_results[seq]),
            } for seq in args.sequences if seq_results[seq]},
            'overall'   : {
                'psnr' : float(np.mean(all_psnr)),
                'ssim' : float(np.mean(all_ssim)),
                'nrmse': float(np.mean(all_nrmse)),
                'lpips': float(np.mean(all_lpips)),
                'n'    : len(all_psnr),
            },
            'per_sample': seq_results,
        }, f, indent=2)

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'results_nafdm_v3.csv')
    with open(csv_path, 'w') as f:
        f.write('patient,seq,frame,n_avg,psnr,ssim,nrmse,lpips\n')
        for seq in args.sequences:
            for r in seq_results[seq]:
                f.write(f"{r['patient']},{seq},{r['frame']},{r['n_avg']},"
                        f"{r['psnr']:.4f},{r['ssim']:.4f},"
                        f"{r['nrmse']:.4f},{r['lpips']:.4f}\n")

    print(f'\nResults -> {out_json}')
    print(f'CSV     -> {csv_path}')
    print(f'Vis     -> {args.output_dir}/vis/')


if __name__ == '__main__':
    main()