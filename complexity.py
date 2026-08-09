import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Rectangle, FancyArrowPatch



# =====================================================
# Paths
# =====================================================

noise_path = r"F:\IXI\我们的方法\1\noise.png"
swinir_path = r"F:\IXI\我们的方法\1\swinir.png"
nafdm_path = r"F:\IXI\我们的方法\1\nafdm.png"



# =====================================================
# Read PNG
# =====================================================

def read_img(path):

    img = Image.open(path)

    img = np.array(img)


    if img.ndim == 3:
        img = img[:,:,0]


    img = img.astype(np.float32)


    img = (
        img-img.min()
    )/(img.max()-img.min()+1e-8)


    return img



noise = read_img(noise_path)
swinir = read_img(swinir_path)
nafdm = read_img(nafdm_path)



print("Noise:",noise.shape)
print("SwinIR:",swinir.shape)
print("NAFDM:",nafdm.shape)



imgs=[
    noise,
    swinir,
    nafdm
]


titles=[
    "Low-field Input",
    "SwinIR",
    "NAFDMNet"
]



# =====================================================
# Display range
# =====================================================

pixels=np.concatenate(
    [
        x.flatten()
        for x in imgs
    ]
)


vmin=np.percentile(pixels,1)
vmax=np.percentile(pixels,99)



# =====================================================
# ROI
# =====================================================

def get_roi(img):

    h,w=img.shape


    x1=int(w*0.35)
    y1=int(h*0.35)

    x2=int(w*0.65)
    y2=int(h*0.65)


    return x1,y1,x2,y2



# =====================================================
# Figure
# =====================================================

plt.rcParams["font.family"]="Arial"


fig=plt.figure(
    figsize=(6.8,5.0),
    facecolor="white"
)



# =====================================================
# Compact layout
# =====================================================


# main MRI images

main_pos=[

    [0.13,0.60,0.20,0.23],

    [0.40,0.60,0.20,0.23],

    [0.67,0.60,0.20,0.23]

]



# zoom images

zoom_pos=[

    [0.12,0.37,0.22,0.18],

    [0.39,0.37,0.22,0.18],

    [0.66,0.37,0.22,0.18]

]



# =====================================================
# Draw images
# =====================================================


for i,img in enumerate(imgs):


    x1,y1,x2,y2=get_roi(img)



    # --------------------------
    # Full image
    # --------------------------

    ax=fig.add_axes(
        main_pos[i]
    )


    ax.imshow(
        img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )


    ax.axis("off")


    ax.set_title(
        titles[i],
        fontsize=11,
        fontweight="bold",
        pad=5
    )



    rect=Rectangle(
        (x1,y1),
        x2-x1,
        y2-y1,
        linewidth=1.5,
        edgecolor="yellow",
        facecolor="none"
    )


    ax.add_patch(rect)



    # --------------------------
    # ROI zoom
    # --------------------------

    ax2=fig.add_axes(
        zoom_pos[i]
    )


    crop=img[
        y1:y2,
        x1:x2
    ]


    ax2.imshow(
        crop,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )


    ax2.axis("off")



# =====================================================
# Short arrows
# =====================================================


fig.add_artist(
    FancyArrowPatch(
        (0.35,0.70),
        (0.385,0.70),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.1
    )
)


fig.add_artist(
    FancyArrowPatch(
        (0.62,0.70),
        (0.655,0.70),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.1
    )
)



# =====================================================
# Text
# =====================================================


# Title

fig.text(
    0.5,
    0.94,
    "Motivation of Noise-Aware Low-field MRI Restoration",
    ha="center",
    fontsize=12,
    fontweight="bold"
)



# Description

fig.text(
    0.23,
    0.27,
    "Noise corruption\n"
    "Intensity fluctuation",
    ha="center",
    fontsize=8
)


fig.text(
    0.50,
    0.27,
    "Existing restoration\n"
    "Fine detail loss",
    ha="center",
    fontsize=8
)


fig.text(
    0.77,
    0.27,
    "Noise-aware modulation\n"
    "Structure recovery",
    ha="center",
    fontsize=8
)



# Bottom two-line statement

fig.text(
    0.5,
    0.11,
    "NAFDMNet dynamically adapts frequency responses according to noise characteristics\n"
    "to preserve high-frequency anatomical details in low-field MRI.",
    ha="center",
    fontsize=8.5,
    linespacing=1.4
)



# =====================================================
# Save
# =====================================================

save_path=r"F:\IXI\我们的方法\TMI_Motivation_Final_v4.png"


plt.savefig(
    save_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)


plt.show()


print("Saved:")
print(save_path)