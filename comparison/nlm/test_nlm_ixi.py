import os
import sys

sys.path.insert(
    0,
    r'E:\Restormer-main\Denoising'
)


import json
import numpy as np

import torch

import nibabel as nib

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


from tqdm import tqdm


from skimage.transform import resize


from metrics_net import (
    compute_psnr,
    compute_ssim,
    compute_nrmse
)


import bm3d



try:
    import lpips

    lpips_fn = lpips.LPIPS(net='vgg')

    HAS_LPIPS=True

except:

    HAS_LPIPS=False



# ==================================
# IXI Dataset
# ==================================

def build_ixi_samples(
        data_dir,
        sequence='T1'
):


    folder=os.path.join(
        data_dir,
        'IXI-'+sequence
    )


    files=[]


    for f in os.listdir(folder):

        if f.endswith('.nii') or f.endswith('.nii.gz'):

            files.append(
                os.path.join(folder,f)
            )


    files.sort()


    print("==========================")
    print("IXI sequence:",sequence)
    print("Volumes:",len(files))
    print("==========================")


    return files





def load_center_slice(
        path,
        size=256
):


    volume=nib.load(path).get_fdata()


    z=volume.shape[2]//2


    img=volume[:,:,z]


    img=resize(
        img,
        (size,size),
        preserve_range=True
    )


    img=img.astype(np.float32)



    img=(img-img.min())/(
        img.max()-img.min()+1e-8
    )


    return img




# ==================================
# Rician Noise
# ==================================

def add_rician_noise(
        img,
        sigma=0.06
):


    n1=np.random.normal(
        0,
        sigma,
        img.shape
    )


    n2=np.random.normal(
        0,
        sigma,
        img.shape
    )


    noisy=np.sqrt(
        (img+n1)**2+n2**2
    )


    return np.clip(
        noisy,
        0,
        1
    )





# ==================================
# LPIPS
# ==================================

def compute_lpips(
        pred,
        gt
):


    if not HAS_LPIPS:

        return 0


    p=torch.from_numpy(
        pred
    ).float().unsqueeze(0).unsqueeze(0)


    g=torch.from_numpy(
        gt
    ).float().unsqueeze(0).unsqueeze(0)


    p=p.repeat(1,3,1,1)*2-1

    g=g.repeat(1,3,1,1)*2-1


    with torch.no_grad():

        return float(
            lpips_fn(p,g)
        )





# ==================================
# Visualization
# ==================================

def save_vis(
        noisy,
        pred,
        gt,
        save_dir,
        idx,
        psnr,
        ssim,
        nrmse
):


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    fig,axs=plt.subplots(
        1,
        4,
        figsize=(16,4)
    )


    imgs=[
        noisy,
        pred,
        gt,
        np.abs(pred-gt)
    ]


    titles=[
        'Noisy',
        'BM3D',
        'GT',
        'Difference'
    ]


    for ax,img,title in zip(
        axs,
        imgs,
        titles
    ):


        ax.imshow(
            img,
            cmap='gray',
            vmin=0,
            vmax=1
        )


        ax.set_title(title)

        ax.axis('off')



    plt.suptitle(
        f'BM3D | PSNR={psnr:.2f} SSIM={ssim:.4f} NRMSE={nrmse:.4f}'
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            save_dir,
            f'vis_{idx:04d}.png'
        ),
        dpi=150
    )


    plt.close()






# ==================================
# Main Test
# ==================================

def main():


    data_dir=r'F:\IXI'


    sequence='T1'


    sigma=0.06



    files=build_ixi_samples(
        data_dir,
        sequence
    )



    results=[]



    vis_dir='./bm3d_ixi_vis'



    for i,path in enumerate(
        tqdm(files)
    ):


        gt=load_center_slice(
            path
        )


        noisy=add_rician_noise(
            gt,
            sigma
        )



        # BM3D

        pred=bm3d.bm3d(
            noisy,
            sigma_psd=sigma
        )


        pred=np.clip(
            pred,
            0,
            1
        )



        pred_t=torch.tensor(
            pred
        ).unsqueeze(0).unsqueeze(0)



        gt_t=torch.tensor(
            gt
        ).unsqueeze(0).unsqueeze(0)



        psnr=compute_psnr(
            pred_t,
            gt_t
        )


        ssim=compute_ssim(
            pred_t,
            gt_t
        )


        nrmse=compute_nrmse(
            pred_t,
            gt_t
        )


        lp=compute_lpips(
            pred,
            gt
        )



        results.append(
            {

            'name':
            os.path.basename(path),

            'psnr':
            float(psnr),

            'ssim':
            float(ssim),

            'nrmse':
            float(nrmse),

            'lpips':
            float(lp)

            }
        )



        if i<20:

            save_vis(
                noisy,
                pred,
                gt,
                vis_dir,
                i,
                psnr,
                ssim,
                nrmse
            )



    # ==========================
    # Summary
    # ==========================


    psnr=[
        x['psnr']
        for x in results
    ]

    ssim=[
        x['ssim']
        for x in results
    ]

    nrmse=[
        x['nrmse']
        for x in results
    ]

    lpips=[
        x['lpips']
        for x in results
    ]



    print("\n============================")

    print("BM3D IXI Result")

    print(
        f"PSNR={np.mean(psnr):.4f} ± {np.std(psnr):.4f}"
    )


    print(
        f"SSIM={np.mean(ssim):.4f} ± {np.std(ssim):.4f}"
    )


    print(
        f"NRMSE={np.mean(nrmse):.4f} ± {np.std(nrmse):.4f}"
    )


    print(
        f"LPIPS={np.mean(lpips):.4f}"
    )


    print("============================")



    with open(
        'bm3d_ixi_result.json',
        'w'
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )




if __name__=='__main__':

    main()