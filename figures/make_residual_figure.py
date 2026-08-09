import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ==========================
# path
# ==========================

img_dir = r"F:\IXI\我们的方法\compare_results"

save_path = r"F:\IXI\我们的方法\TMI_residual_compare.png"



# ==========================
# methods
# ==========================

methods=[
    "bm3d",
    "nlm",
    "dncnn",
    "noise2void",
    "noise2self",
    "restormer",
    "snraware",
    "swinir",
    "noise2noise",
    "nafdm"
]


titles=[
    "BM3D",
    "NLM",
    "DnCNN",
    "Noise2Void",
    "Noise2Self",
    "Restormer",
    "SNRAware",
    "Noise2noise"
    "SwinIR",
    "NAFDMNet"
]



def read_img(name):

    path=os.path.join(
        img_dir,
        name+".png"
    )

    img=np.array(
        Image.open(path)
        .convert("L")
    )

    return img.astype(
        np.float32
    )/255.



gt=read_img("gt")



# ==========================
# Figure
# ==========================


fig,axs=plt.subplots(
    2,
    len(methods)+1,
    figsize=(24,7)
)



# 第一行
for i,(m,t) in enumerate(zip(methods,titles)):


    pred=read_img(m)


    axs[0,i].imshow(
        pred,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axs[0,i].set_title(
        t,
        fontsize=12
    )

    axs[0,i].axis("off")


    residual=np.abs(
        pred-gt
    )


    axs[1,i].imshow(
        residual,
        cmap="hot",
        vmin=0,
        vmax=0.3
    )

    axs[1,i].set_title(
        "Residual",
        fontsize=10
    )

    axs[1,i].axis("off")



# GT列

axs[0,-1].imshow(
    gt,
    cmap="gray",
    vmin=0,
    vmax=1
)

axs[0,-1].set_title(
    "Ground Truth"
)

axs[0,-1].axis("off")


axs[1,-1].axis("off")



plt.tight_layout()


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("Saved:")
print(save_path)