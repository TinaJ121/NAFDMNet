import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


img_dir = r"F:\IXI\我们的方法\compare_results"

save_dir = r"F:\IXI\我们的方法\single_vis"

os.makedirs(save_dir,exist_ok=True)



methods = [
    "bm3d",
    "nlm",
    "dncnn",
    "noise2void",
    "noise2self",
    "restormer",
    "snraware",
    "noise2noise",
    "swinir",
    "nafdm"
]


titles = {
    "bm3d":"BM3D",
    "nlm":"NLM",
    "dncnn":"DnCNN",
    "noise2void":"Noise2Void",
    "noise2self":"Noise2Self",
    "restormer":"Restormer",
    "snraware":"SNRAware",
    "noise2noise":"Noise2Noise",
    "swinir":"SwinIR",
    "nafdm":"NAFDMNet"
}



def read_img(name):

    path=os.path.join(
        img_dir,
        name+".png"
    )

    img=np.array(
        Image.open(path)
        .convert("L")
    )

    return img.astype(np.float32)/255.



gt=read_img("gt")

noisy=read_img("noisy")



for method in methods:


    pred=read_img(method)


    residual=np.abs(
        pred-gt
    )

    # residual归一化
    residual=residual/(residual.max()+1e-8)



    fig,axs=plt.subplots(
        1,
        4,
        figsize=(16,4)
    )


    imgs=[
        noisy,
        pred,
        gt,
        residual
    ]


    names=[
        "Rician Noisy",
        titles[method],
        "Ground Truth",
        "Residual"
    ]


    for ax,img,name in zip(
        axs,
        imgs,
        names
    ):


        ax.imshow(
            img,
            cmap="gray",
            vmin=0,
            vmax=1
        )


        ax.set_title(
            name,
            fontsize=14
        )


        ax.axis("off")



    plt.tight_layout()


    save_path=os.path.join(
        save_dir,
        method+"_visualization.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "saved:",
        save_path
    )


print("Finished")