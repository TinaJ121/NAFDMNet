import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Training set: Input and GT configuration for each sequence
TRAIN_SEQ_CONFIG = {
    'T1': {
        'input': 'T101',
        'gt_list': ['T101', 'T102', 'T103'],  # 3-average
    },
    'T2': {
        'input': 'T201',
        'gt_list': ['T201', 'T202', 'T203'],  # 3-average
    },
    'FLAIR': {
        'input': 'FLAIR01',
        'gt_list': ['FLAIR01', 'FLAIR02'],    # 2-average
    },
}

# Test set: More averages (cleaner GT)
TEST_SEQ_CONFIG = {
    'T1': {
        'input': 'T101',
        'gt_list': ['T101','T102','T103','T104','T105','T106'],  # 6-average
    },
    'T2': {
        'input': 'T201',
        'gt_list': ['T201','T202','T203','T204','T205','T206'],  # 6-average
    },
    'FLAIR': {
        'input': 'FLAIR01',
        'gt_list': ['FLAIR01','FLAIR02','FLAIR03','FLAIR04'],    # 4-average
    },
}

def augment_pair(a, b):
    """Random flip and rotation data augmentation"""
    if np.random.rand() < 0.5:
        a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
    if np.random.rand() < 0.5:
        a, b = np.flipud(a).copy(), np.flipud(b).copy()
    if np.random.rand() < 0.3:
        k = np.random.randint(1, 4)
        a, b = np.rot90(a, k).copy(), np.rot90(b, k).copy()
    return a, b

def load_rss(fp, frame_idx):
    """Load single-frame RSS image from h5 file"""
    with h5py.File(fp, 'r') as f:
        return f['reconstruction_rss'][frame_idx].astype(np.float32)

def load_avg_rss(data_dir, patient, suffixes, frame_idx):
    """Load multiple repeated scans and compute their average"""
    imgs = []
    for suf in suffixes:
        fp = os.path.join(data_dir, f'{patient}_{suf}.h5')
        if os.path.exists(fp):
            imgs.append(load_rss(fp, frame_idx))
    if not imgs:
        return None
    return np.mean(imgs, axis=0).astype(np.float32)


class M4RawTrainDataset(Dataset):
    """
    M4Raw Training Dataset
    Input: Single-scan RSS (Real low-field noise)
    GT:    Multi-average RSS (Higher SNR reference)
    """
    def __init__(self,
                 data_dir,
                 sequences=('T1', 'T2', 'FLAIR'),
                 frames_per_seq=18,
                 augment=True,
                 target_size=256):
        self.data_dir    = data_dir
        self.augment     = augment
        self.target_size = target_size

        # Detect actual existing sequence configurations
        all_files = os.listdir(data_dir)
        all_patients = sorted(set(
            f.split('_')[0] for f in all_files if f.endswith('.h5')))

        # Use the first patient to check which files exist
        pid0 = all_patients[0]
        self.seq_cfg = {}
        for seq in sequences:
            if seq not in TRAIN_SEQ_CONFIG:
                continue
            cfg = TRAIN_SEQ_CONFIG[seq]
            inp_suf  = cfg['input']
            gt_sufs  = [s for s in cfg['gt_list']
                        if os.path.exists(
                            os.path.join(data_dir, f'{pid0}_{s}.h5'))]
            if gt_sufs and os.path.exists(
                    os.path.join(data_dir, f'{pid0}_{inp_suf}.h5')):
                self.seq_cfg[seq] = (inp_suf, gt_sufs)

        # Build sample list
        self.samples = []
        for pid in all_patients:
            for seq, (inp_suf, gt_sufs) in self.seq_cfg.items():
                fp = os.path.join(data_dir, f'{pid}_{inp_suf}.h5')
                if not os.path.exists(fp):
                    continue
                try:
                    with h5py.File(fp, 'r') as f:
                        n_frames = f['reconstruction_rss'].shape[0]
                    n_use  = min(n_frames, frames_per_seq)
                    center = n_frames // 2
                    half   = n_use // 2
                    for t in range(max(0, center-half),
                                   min(n_frames, center+half)):
                        self.samples.append((pid, seq, t, inp_suf, gt_sufs))
                except Exception as e:
                    print(f'[Skip] {pid}_{seq}: {e}')

        print(f'[Train] {len(all_patients)} patients, '
              f'{len(self.samples)} samples | {data_dir}')
        for seq, (inp, gts) in self.seq_cfg.items():
            print(f'  {seq}: input={inp}, GT={gts}({len(gts)}-average)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, seq, t, inp_suf, gt_sufs = self.samples[idx]

        # Read input (single scan)
        fp_inp = os.path.join(self.data_dir, f'{pid}_{inp_suf}.h5')
        rss_inp = load_rss(fp_inp, t)

        # Read GT (multi-average)
        rss_gt = load_avg_rss(self.data_dir, pid, gt_sufs, t)
        if rss_gt is None:
            rss_gt = rss_inp.copy()

        # Normalization (use GT max value to maintain consistent scale)
        gt_max = rss_gt.max() + 1e-8
        inp_n  = np.clip(rss_inp / gt_max, 0, 1).astype(np.float32)
        gt_n   = np.clip(rss_gt  / gt_max, 0, 1).astype(np.float32)

        # Data augmentation
        if self.augment:
            inp_n, gt_n = augment_pair(inp_n, gt_n)

        return {
            'noisy'  : torch.from_numpy(inp_n).unsqueeze(0),
            'gt'     : torch.from_numpy(gt_n).unsqueeze(0),
            'patient': pid,
            'seq'    : seq,
            'frame'  : t,
        }


class M4RawValDataset(Dataset):
    """
    M4Raw Validation/Test Dataset
    Uses higher-order averages as GT (cleaner)
    """
    def __init__(self,
                 data_dir,
                 sequences=('T1', 'T2', 'FLAIR'),
                 frames_per_seq=18,
                 target_size=256,
                 is_test=False):
        self.data_dir    = data_dir
        self.target_size = target_size
        seq_cfg_src      = TEST_SEQ_CONFIG if is_test else TRAIN_SEQ_CONFIG

        all_files    = os.listdir(data_dir)
        all_patients = sorted(set(
            f.split('_')[0] for f in all_files if f.endswith('.h5')))

        pid0 = all_patients[0]
        self.seq_cfg = {}
        for seq in sequences:
            if seq not in seq_cfg_src:
                continue
            cfg     = seq_cfg_src[seq]
            inp_suf = cfg['input']
            gt_sufs = [s for s in cfg['gt_list']
                       if os.path.exists(
                           os.path.join(data_dir, f'{pid0}_{s}.h5'))]
            if gt_sufs and os.path.exists(
                    os.path.join(data_dir, f'{pid0}_{inp_suf}.h5')):
                self.seq_cfg[seq] = (inp_suf, gt_sufs)

        self.samples = []
        for pid in all_patients:
            for seq, (inp_suf, gt_sufs) in self.seq_cfg.items():
                fp = os.path.join(data_dir, f'{pid}_{inp_suf}.h5')
                if not os.path.exists(fp):
                    continue
                try:
                    with h5py.File(fp, 'r') as f:
                        n_frames = f['reconstruction_rss'].shape[0]
                    n_use  = min(n_frames, frames_per_seq)
                    center = n_frames // 2
                    half   = n_use // 2
                    for t in range(max(0, center-half),
                                   min(n_frames, center+half)):
                        self.samples.append(
                            (pid, seq, t, inp_suf, gt_sufs))
                except Exception as e:
                    print(f'[Skip] {pid}_{seq}: {e}')

        tag = 'Test' if is_test else 'Val'
        print(f'[{tag}] {len(all_patients)} patients, '
              f'{len(self.samples)} samples | {data_dir}')
        for seq, (inp, gts) in self.seq_cfg.items():
            print(f'  {seq}: GT={len(gts)}-average')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, seq, t, inp_suf, gt_sufs = self.samples[idx]

        fp_inp  = os.path.join(self.data_dir, f'{pid}_{inp_suf}.h5')
        rss_inp = load_rss(fp_inp, t)
        rss_gt  = load_avg_rss(self.data_dir, pid, gt_sufs, t)
        if rss_gt is None:
            rss_gt = rss_inp.copy()

        gt_max = rss_gt.max() + 1e-8
        inp_n  = np.clip(rss_inp / gt_max, 0, 1).astype(np.float32)
        gt_n   = np.clip(rss_gt  / gt_max, 0, 1).astype(np.float32)

        return {
            'noisy'  : torch.from_numpy(inp_n).unsqueeze(0),
            'gt'     : torch.from_numpy(gt_n).unsqueeze(0),
            'patient': pid,
            'seq'    : seq,
            'frame'  : t,
            'n_avg'  : len(gt_sufs),
        }

def get_m4raw_loaders(data_dir,
                      batch_size=4,
                      num_workers=0,
                      sequences=('T1', 'T2', 'FLAIR'),
                      frames_per_seq=18,
                      target_size=256,
                      pin_memory=True,
                      val_data_dir=None,
                      **kwargs):
    """
    M4Raw Training/Validation DataLoader
    val_data_dir: Validation directory; if None, takes from training directory
    """
    tr_ds = M4RawTrainDataset(
        data_dir=data_dir,
        sequences=sequences,
        frames_per_seq=frames_per_seq,
        augment=True,
        target_size=target_size,
    )

    va_dir = val_data_dir if val_data_dir else data_dir
    va_ds  = M4RawValDataset(
        data_dir=va_dir,
        sequences=sequences,
        frames_per_seq=frames_per_seq,
        target_size=target_size,
        is_test=(val_data_dir is not None),
    )

    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True)
    va_loader = DataLoader(
        va_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    # Third return value is None (compatible with legacy 3-tuple interface)
    return tr_loader, va_loader, None

def get_m4raw_test_loader(data_dir,
                          batch_size=1,
                          num_workers=0,
                          sequences=('T1', 'T2', 'FLAIR'),
                          frames_per_seq=18,
                          target_size=256,
                          pin_memory=True,
                          **kwargs):
    """M4Raw Test DataLoader (uses 6/4-average GT)"""
    te_ds = M4RawValDataset(
        data_dir=data_dir,
        sequences=sequences,
        frames_per_seq=frames_per_seq,
        target_size=target_size,
        is_test=True,
    )
    return DataLoader(
        te_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)


if __name__ == '__main__':
    import sys

    train_dir = r'E:\V1.6\M4RawV1.5_multicoil_train\multicoil_train'
    test_dir  = r'E:\V1.6\M4Raw_multicoil_test\multicoil_test'

    print('='*50)
    print('Testing training set...')
    tr, va, _ = get_m4raw_loaders(
        train_dir,
        val_data_dir=test_dir,
        batch_size=2,
        frames_per_seq=4,
    )
    b = next(iter(tr))
    print(f'noisy: {b["noisy"].shape} [{b["noisy"].min():.3f},{b["noisy"].max():.3f}]')
    print(f'gt:    {b["gt"].shape}    [{b["gt"].min():.3f},{b["gt"].max():.3f}]')
    print(f'seq:   {b["seq"]}')

    print('\nTesting test set...')
    te = get_m4raw_test_loader(test_dir, frames_per_seq=4)
    b  = next(iter(te))
    print(f'noisy: {b["noisy"].shape}')
    print(f'gt:    {b["gt"].shape}')
    print(f'n_avg: {b["n_avg"]}')
    print(f'seq:   {b["seq"]}')