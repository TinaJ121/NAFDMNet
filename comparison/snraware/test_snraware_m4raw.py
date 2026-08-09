import sys
sys.path.insert(0, r'E:\SNRAware-main\src')
sys.path.insert(0, r'E:\Restormer-main\Denoising')

import os
import json
import argparse
import numpy as np
import torch
import h5py
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from metrics_net import compute_psnr, compute_ssim, compute_nrmse

try:
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg')
    HAS_LPIPS = True
except:
    HAS_LPIPS = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_pts',      type=str, required=True)
    p.add_argument('--model_yaml',     type=str, required=True)
    p.add_argument('--data_dir',       type=str, required=True)
    p.add_argument('--output_dir',     type=str, default='./test_results/snraware_m4raw')
    p.add_argument('--sequences',      type=str, nargs='+', default=['T1', 'T2', 'FLAIR'])
    p.add_argument('--frames_per_seq', type=int, default=18)
    p.add_argument('--n_vis',          type=int, default=20)
    p.add_argument('--max_samples',    type=int, default=None)
    p.add_argument('--batch_size',     type=int, default=1)
    return p.parse_args()

SEQ_CANDIDATES = {
    'T1'   : {'input': 'T101',
              'gt_candidates': ['T101','T102','T103','T104','T105','T106']},
    'T2'   : {'input': 'T201',
              'gt_candidates': ['T201','T202','T203','T204','T205','T206']},
    'FLAIR': {'input': 'FLAIR01',
              'gt_candidates': ['FLAIR01','FLAIR02','FLAIR03','FLAIR04']},
}

def build_samples(data_dir, sequences, frames_per_seq):
    all_files    = os.listdir(data_dir)
    all_patients = sorted(set(f.split('_')[0] for f in all_files if f.endswith('.h5')))

    pid0 = all_patients[0]
    seq_config = {}
    for seq in sequences:
        if seq not in SEQ_CANDIDATES:
            continue
        cand    = SEQ_CANDIDATES[seq]
        inp_suf = cand['input']
        actual_gts = [s for s in cand['gt_candidates']
                      if os.path.exists(
                          os.path.join(data_dir, f'{pid0}_{s}.h5'))]
        if actual_gts and os.path.exists(
                os.path.join(data_dir, f'{pid0}_{inp_suf}.h5')):
            seq_config[seq] = (inp_suf, actual_gts)
            print(f'  {seq}: Input={inp_suf}, GT={actual_gts}')

    samples = []
    for pid in all_patients:
        for seq, (inp_suf, gt_sufs) in seq_config.items():
            fp = os.path.join(data_dir, f'{pid}_{inp_suf}.h5')
            if not os.path.exists(fp):
                continue
            try:
                with h5py.File(fp, 'r') as f:
                    n_frames = f['reconstruction_rss'].shape[0]
                n_use  = min(n_frames, frames_per_seq)
                center = n_frames // 2
                half   = n_use // 2
                for t in range(max(0, center - half),
                               min(n_frames, center + half)):
                    samples.append((pid, seq, t, inp_suf, gt_sufs))
            except:
                continue
    return samples, seq_config

def kspace_to_complex_image(kspace_frame):
    import numpy.fft as npfft
    imgs = []
    for c in range(kspace_frame.shape[0]):
        img_c = npfft.ifft2(npfft.ifftshift(kspace_frame[c]))
        imgs.append(img_c)
    imgs = np.stack(imgs, axis=0)

    rss = np.sqrt(np.sum(np.abs(imgs)**2, axis=0))

    return rss.astype(np.float32)

def load_snraware_model(model_pts, model_yaml, device="cuda"):
    from snraware.projects.mri.denoising.inference_model import load_scripted_model
    from omegaconf import OmegaConf
    model  = load_scripted_model(model_pts)
    model  = model.to(device)
    config = OmegaConf.load(model_yaml)
    return model, config

def snraware_denoise(model, config, image_np, device):
    from snraware.projects.mri.denoising.inference import apply_model
    from omegaconf import OmegaConf

    H, W = image_np.shape

    inp_complex = (image_np.astype(np.float32)
                   + 1j * np.zeros_like(image_np))[:, :, np.newaxis]

    gmap = np.ones((H, W, 1), dtype=np.float32)

    cutout = tuple(config.dataset.cutout_shape)

    output = apply_model(
        model=model,
        data=inp_complex,
        gmap=gmap,
        scaling_factor=1.0,
        cutout=cutout,
        overlap=(16, 16, 8),
        batch_size=1,
        device='cuda',
        verbose=False,
    )

    pred = np.abs(output[:, :, 0]).astype(np.float32)
    return pred

def save_vis(noisy, pred, ref, save_dir, idx,
             psnr, ssim, nrmse, patient='', seq=''):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0b0f1a')
    titles = ['Input (single scan)', 'SNRAware (output)',
              'GT (multi-average)',   '|Diff| Output-GT']
    imgs   = [noisy, pred, ref, np.abs(pred - ref)]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
    fig.suptitle(
        f'SNRAware-small | Patient={patient}  Seq={seq}\n'
        f'PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  NRMSE={nrmse:.4f}',
        color='#f59e0b', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'vis_{idx:04d}.png'),
                dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

def compute_lpips_score(pred_np, gt_np):
    if not HAS_LPIPS:
        return 0.0
    p = torch.from_numpy(pred_np).float().unsqueeze(0).unsqueeze(0)
    g = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0)
    p = p.repeat(1,3,1,1)*2-1
    g = g.repeat(1,3,1,1)*2-1
    with torch.no_grad():
        return lpips_fn(p, g).item()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'vis')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CUDA available:{torch.cuda.is_available()}')

    print('\n[1] Loading SNRAware model...')
    model, config = load_snraware_model(args.model_pts, args.model_yaml, device=str(device))
    print('Model loaded successfully')

    print('\n[2] Building sample list...')
    samples, seq_config = build_samples(
        args.data_dir, args.sequences, args.frames_per_seq)
    print(f'Total samples: {len(samples)}')

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f'Limited to first {args.max_samples} samples')

    results   = []
    all_lpips = []
    vis_n     = 0

    pbar = tqdm(samples, desc='SNRAware Testing', unit='img')

    for i, (pid, seq, t, inp_suf, gt_sufs) in enumerate(pbar):
        try:
            fp_inp = os.path.join(args.data_dir, f'{pid}_{inp_suf}.h5')
            with h5py.File(fp_inp, 'r') as f:
                rss_inp = f['reconstruction_rss'][t].astype(np.float32)

            gt_list = []
            for suf in gt_sufs:
                fp2 = os.path.join(args.data_dir, f'{pid}_{suf}.h5')
                if os.path.exists(fp2):
                    with h5py.File(fp2, 'r') as f:
                        gt_list.append(
                            f['reconstruction_rss'][t].astype(np.float32))
            rss_gt = np.mean(gt_list, axis=0).astype(np.float32)

            gt_max   = rss_gt.max() + 1e-8
            inp_norm = np.clip(rss_inp / gt_max, 0, 1).astype(np.float32)
            gt_norm  = np.clip(rss_gt  / gt_max, 0, 1).astype(np.float32)

            pred_raw = snraware_denoise(model, config, inp_norm, device)

            if pred_raw.max() > 1.1:
                pred_norm = np.clip(
                    pred_raw / (pred_raw.max() + 1e-8), 0, 1)
            else:
                pred_norm = np.clip(pred_raw, 0, 1)

            pred_t = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0)
            gt_t   = torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0)

            p = compute_psnr(pred_t, gt_t)
            s = compute_ssim(pred_t, gt_t)
            n = compute_nrmse(pred_t, gt_t)
            l = compute_lpips_score(pred_norm, gt_norm)
            all_lpips.append(l)

            results.append({
                'idx'    : i,
                'psnr'   : float(p),
                'ssim'   : float(s),
                'nrmse'  : float(n),
                'lpips'  : float(l),
                'patient': pid,
                'seq'    : seq,
                'frame'  : t,
                'n_avg'  : len(gt_sufs),
            })

            pbar.set_postfix(
                PSNR=f'{p:.2f}',
                SSIM=f'{s:.4f}',
                seq=seq)

            if vis_n < args.n_vis:
                save_vis(inp_norm, pred_norm, gt_norm,
                         vis_dir, vis_n, p, s, n,
                         patient=pid, seq=seq)
                vis_n += 1

        except Exception as e:
            print(f'\n[Skip] {pid}_{seq}_t{t}: {e}')
            continue

    if not results:
        print('No results!')
        return

    all_psnr  = [r['psnr']  for r in results]
    all_ssim  = [r['ssim']  for r in results]
    all_nrmse = [r['nrmse'] for r in results]

    print(f'\n{"="*60}')
    print(f'  SNRAware-small Test Results [M4Raw]')
    print(f'  Samples : {len(results)}')
    print(f'  PSNR    = {np.mean(all_psnr):.4f} dB')
    print(f'  SSIM    = {np.mean(all_ssim):.4f}')
    print(f'  NRMSE   = {np.mean(all_nrmse):.4f}')
    print(f'  LPIPS   = {np.mean(all_lpips):.4f}')
    print(f'{"="*60}')

    print('\n  Per-sequence breakdown:')
    seq_all_psnr, seq_all_ssim, seq_all_nrmse = [], [], []
    for seq in args.sequences:
        seq_res = [r for r in results if r['seq'] == seq]
        if seq_res:
            psnrs  = [r['psnr']  for r in seq_res]
            ssims  = [r['ssim']  for r in seq_res]
            nrmses = [r['nrmse'] for r in seq_res]
            n_a    = seq_res[0]['n_avg']
            print(f'  {seq:6s} (GT={n_a}-average): '
                  f'PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}  '
                  f'SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}  '
                  f'NRMSE={np.mean(nrmses):.4f}±{np.std(nrmses):.4f}  '
                  f'(n={len(seq_res)})')
            seq_all_psnr.extend(psnrs)
            seq_all_ssim.extend(ssims)
            seq_all_nrmse.extend(nrmses)

    print(f'\n  [Paper Table Format]')
    print(f'  SNRAware | '
          f'PSNR={np.mean(seq_all_psnr):.2f}±{np.std(seq_all_psnr):.2f} | '
          f'SSIM={np.mean(seq_all_ssim):.4f}±{np.std(seq_all_ssim):.4f} | '
          f'NRMSE={np.mean(seq_all_nrmse):.4f}±{np.std(seq_all_nrmse):.4f} | '
          f'LPIPS={np.mean(all_lpips):.4f}')

    out_json = os.path.join(args.output_dir, 'results_snraware.json')
    with open(out_json, 'w') as f:
        json.dump({
            'summary': {
                'psnr' : float(np.mean(all_psnr)),
                'ssim' : float(np.mean(all_ssim)),
                'nrmse': float(np.mean(all_nrmse)),
                'lpips': float(np.mean(all_lpips)),
            },
            'per_sample': results
        }, f, indent=2)

    csv_path = out_json.replace('.json', '.csv')
    with open(csv_path, 'w') as f:
        f.write('idx,patient,seq,frame,n_avg,psnr,ssim,nrmse,lpips\n')
        for r in results:
            f.write(f"{r['idx']},{r['patient']},{r['seq']},"
                    f"{r['frame']},{r['n_avg']},"
                    f"{r['psnr']:.4f},{r['ssim']:.4f},"
                    f"{r['nrmse']:.4f},{r['lpips']:.4f}\n")

    print(f'\nResults → {out_json}')
    print(f'CSV     → {csv_path}')
    print(f'Vis     → {vis_dir}/')

if __name__ == '__main__':
    main()
