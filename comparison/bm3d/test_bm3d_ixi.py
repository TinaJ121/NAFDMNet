import os
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader

from skimage.transform import resize



class IXIBM3DDataset(Dataset):

    def __init__(
            self,
            data_dir,
            sequence="T1",
            sigma=0.06,
            target_size=256
    ):

        self.data_dir = data_dir
        self.sequence = sequence
        self.sigma = sigma
        self.target_size = target_size


        seq_dir = os.path.join(
            data_dir,
            "IXI-" + sequence
        )


        if not os.path.exists(seq_dir):
            raise FileNotFoundError(
                f"Cannot find {seq_dir}"
            )


        self.files = []


        for f in os.listdir(seq_dir):

            if f.endswith(".nii") or f.endswith(".nii.gz"):

                self.files.append(
                    os.path.join(seq_dir,f)
                )


        self.files.sort()


        print("==============================")
        print("IXI Dataset")
        print("Sequence :",sequence)
        print("Images   :",len(self.files))
        print("==============================")


        self.images=[]


        for f in self.files:

            img=nib.load(f).get_fdata()


            # 中间slice
            z=img.shape[2]//2


            img=img[:,:,z]


            img=resize(
                img,
                (
                    target_size,
                    target_size
                ),
                preserve_range=True
            )


            img=img.astype(np.float32)


            # normalize

            img=(img-img.min())/(
                img.max()-img.min()+1e-8
            )


            self.images.append(img)



    def add_rician_noise(self,img):

        n1=np.random.normal(
            0,
            self.sigma,
            img.shape
        )

        n2=np.random.normal(
            0,
            self.sigma,
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



    def __len__(self):

        return len(self.images)



    def __getitem__(self,index):

        clean=self.images[index]


        noisy=self.add_rician_noise(
            clean
        )


        return {

            "noisy":
                torch.from_numpy(noisy)
                .unsqueeze(0)
                .float(),


            "gt":
                torch.from_numpy(clean)
                .unsqueeze(0)
                .float(),


            "name":
                os.path.basename(
                    self.files[index]
                )

        }



def get_ixi_loader(
        data_dir,
        sequence="T1",
        sigma=0.06,
        batch_size=1
):


    dataset=IXIBM3DDataset(
        data_dir,
        sequence,
        sigma
    )


    loader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


    return loader