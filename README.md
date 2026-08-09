# NAFDMNet: Noise-Adaptive Frequency-Domain Modulation Network for MRI Denoising


Official PyTorch implementation of **NAFDMNet** for MRI denoising.

This repository provides the complete implementation of the proposed **Noise-Adaptive Frequency-Domain Modulation Network (NAFDMNet)**, including training, testing, comparison experiments, visualization, and complexity analysis.

NAFDMNet is designed for MRI denoising under both simulated noise conditions and real low-field MRI acquisition scenarios.


---

# Overview


MRI denoising is a challenging task because noise suppression and anatomical structure preservation need to be considered simultaneously.

This work proposes NAFDMNet, which introduces a noise-adaptive frequency-domain modulation strategy to exploit image-specific noise characteristics and dynamically adjust frequency responses.

The proposed network consists of:

- Noise-aware adaptive modulation
- Frequency-domain feature processing
- CNN and Transformer-based feature extraction
- Feature fusion and reconstruction


The repository contains:

- NAFDMNet training code
- IXI and M4Raw dataset processing
- Comparison methods implementation
- Quantitative and qualitative evaluation
- Visualization tools
- Computational complexity analysis


---

# Environment


The experiments are implemented using:


```
Python >= 3.8

PyTorch >= 1.8

CUDA >= 11.0
```


Recommended GPU:

```
NVIDIA GPU with CUDA support
```



---

# Repository Structure


```
NAFDMNet
│
├── checkpoint
│   └── best.pth
│
│
├── compared_results
│
│   ├── IXI_results
│   │
│   │   ├── bm3d.png
│   │   ├── dncnn.png
│   │   ├── nafdm.png
│   │   ├── nlm.png
│   │   ├── noise2noise.png
│   │   ├── noise2self.png
│   │   ├── noise2void.png
│   │   ├── restormer.png
│   │   ├── snraware.png
│   │   └── swinir.png
│   │
│   │
│   └── M4Raw_results
│
│       ├── Flair.png
│       ├── T1.png
│       └── T2.png
│
│
├── comparison
│
│   ├── bm3d
│   │
│   │   ├── test_bm3d_ixi.py
│   │   └── test_bm3d_m4raw.py
│   │
│   │
│   ├── dncnn
│   │
│   │   └── train_dncnn_m4raw.py
│   │
│   │
│   ├── nlm
│   │
│   │   ├── test_nlm_ixi.py
│   │   └── test_nlm_m4raw.py
│   │
│   │
│   ├── noise2noise
│   │
│   │   ├── train_noise2noise_ixi.py
│   │   └── train_noise2noise_m4raw.py
│   │
│   │
│   ├── noise2self
│   │
│   │   ├── test_noise2self_ixi.py
│   │   └── train_noise2self_m4raw.py
│   │
│   │
│   ├── noise2void
│   │
│   │   ├── train_n2v_ixi.py
│   │   └── train_n2v_m4raw.py
│   │
│   │
│   ├── restormer
│   │
│   │   ├── test_restormer_ixi.py
│   │   └── train_restormer_m4raw.py
│   │
│   │
│   ├── snraware
│   │
│   │   ├── test_snraware_m4raw.py
│   │   └── train_snraware_ixi.py
│   │
│   │
│   └── swinir
│
│       ├── train_swinir_ixi.py
│       └── train_swinir_m4raw.py
│
│
├── datasets
│
│   ├── dataset_ixi.py
│   └── dataset_m4raw.py
│
│
├── figures
│
│   ├── make_residual_figure.py
│   ├── make_single_method_vis.py
│   └── make_tmi_figure.py
│
│
├── inference
│
│   ├── test_ixi.py
│   └── test_m4raw.py
│
│
├── loss
│
│   └── loss.py
│
│
├── metrics
│
│   └── metrics_net.py
│
│
├── models
│
│   ├── ablation_models.py
│   └── nafdm_net_.py
│
│
├── pretrained_models
│
│   ├── nafdm_ixi
│   │
│   │   ├── best.pth
│   │   ├── config.json
│   │   └── last.pth
│   │
│   ├── nafdm_ixi_final
│   │
│   │   └── config.json
│   │
│   ├── nafdm_ixi_safe
│   │
│   │   └── config.json
│   │
│   ├── nafdm_ixi_test
│   │
│   │   └── config.json
│   │
│   └── test_safe
│
│       └── config.json
│
│
├── single_vis
│
│   ├── bm3d_visualization.png
│   ├── dncnn_visualization.png
│   ├── nafdm_visualization.png
│   ├── nlm_visualization.png
│   ├── noise2noise_visualization.png
│   ├── noise2self_visualization.png
│   ├── noise2void_visualization.png
│   ├── restormer_visualization.png
│   ├── snraware_visualization.png
│   └── swinir_visualization.png
│
│
├── train
│
│   ├── train.py
│   └── train_ablation.py
│
│
└── complexity.py

```


---

# Dataset


NAFDMNet is evaluated on two MRI datasets:

- IXI dataset
- M4Raw dataset


---

# IXI Dataset


The IXI dataset is used for simulated MRI denoising experiments.


The dataset processing code is:


```
datasets/dataset_ixi.py
```


The IXI experiments include:

- MRI slice loading
- Data preprocessing
- Noise generation
- Training and testing


---

# M4Raw Dataset


The M4Raw dataset is used for real low-field MRI denoising experiments.


The dataset processing code is:


```
datasets/dataset_m4raw.py
```


The experiments include:

- T1-weighted MRI
- T2-weighted MRI
- FLAIR MRI


---

# Training


## NAFDMNet Training


The main training script:


```
train/train.py
```


Run:


```bash
python train/train.py
```



---

## Ablation Training


The ablation experiments are implemented in:


```
train/train_ablation.py
```


Run:


```bash
python train/train_ablation.py
```



---

# Testing


The testing codes are provided in:


```
inference
```


## IXI Testing


```bash
python inference/test_ixi.py
```



## M4Raw Testing


```bash
python inference/test_m4raw.py
```



The evaluation metrics include:


- PSNR
- SSIM
- NRMSE
- LPIPS


---

# Comparison Experiments


The comparison experiments are located in:


```
comparison
```


The implemented comparison methods include:


## Traditional Methods

- BM3D
- NLM


## CNN-based Methods

- DnCNN


## Transformer-based Methods

- SwinIR
- Restormer


## Self-supervised Methods

- Noise2Noise
- Noise2Self
- Noise2Void


## Noise-aware Method

- SNRAware


The comparison results are stored in:


```
compared_results
```


---

# Visualization


Visualization scripts:


```
figures
```


including:


```
make_residual_figure.py

make_single_method_vis.py

make_tmi_figure.py
```


The visualization results are saved in:


```
single_vis
```


---

# Model


The NAFDMNet model is implemented in:


```
models/nafdm_net_.py
```


The ablation models are provided in:


```
models/ablation_models.py
```


---

# Loss Function


The loss implementation is provided in:


```
loss/loss.py
```


---

# Metrics


Evaluation metrics are implemented in:


```
metrics/metrics_net.py
```


The evaluation includes:


- PSNR
- SSIM
- NRMSE
- LPIPS


---

# Pretrained Models


Pretrained models and configurations are provided in:


```
pretrained_models
```


Available models:


```
nafdm_ixi

nafdm_ixi_final

nafdm_ixi_safe

nafdm_ixi_test

test_safe
```


---

# Computational Complexity


The complexity evaluation script:


```
complexity.py
```


Run:


```bash
python complexity.py
```


The complexity analysis includes:


- Parameters
- FLOPs
- Inference time
- GPU memory consumption


---

# Citation


If you find this work useful, please cite:


```bibtex
@article{NAFDMNet2026,
title={Noise-Adaptive Frequency-Domain Modulation Network for MRI Denoising},
author={},
journal={IEEE Transactions on Medical Imaging},
year={2026}
}
```


---

# Acknowledgement


This project is built upon several excellent open-source projects:


- SwinIR
- Restormer
- BM3D
- Noise2Void


We sincerely thank the authors for their contributions.
