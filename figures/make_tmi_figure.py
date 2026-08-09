import os
import matplotlib.pyplot as plt
from PIL import Image


# =========================
# 图片目录
# =========================

img_dir = r"F:\IXI\我们的方法\compare_results"


save_path = r"F:\IXI\我们的方法\TMI_compare_0004.png"



# =========================
# 顺序
# =========================

names = [
    "noisy",
    "bm3d",
    "nlm",
    "dncnn",
    "noise2void",
    "noise2self",
    "restormer",
    "snraware",
    "swinir",
    "nafdm",
    "gt"
]


titles = [
    "Noisy",
    "BM3D",
    "NLM",
    "DnCNN",
    "Noise2Void",
    "Noise2Self",
    "Restormer",
    "SNRAware",
    "SwinIR",
    "NAFDMNet",
    "GT"
]



# =========================
# figure
# =========================


fig, axes = plt.subplots(
    2,
    6,
    figsize=(18,6)
)


axes = axes.flatten()



for i,name in enumerate(names):

    path=os.path.join(
        img_dir,
        name+".png"
    )


    img=Image.open(path)


    axes[i].imshow(
        img,
        cmap="gray"
    )


    axes[i].set_title(
        titles[i],
        fontsize=12
    )


    axes[i].axis("off")



# 多出来一个空位置

for i in range(len(names),len(axes)):

    axes[i].axis("off")



plt.tight_layout()


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("Saved:")
print(save_path)