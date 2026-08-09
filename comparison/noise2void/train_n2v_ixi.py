import os
import numpy as np

import nibabel as nib

import torch

from torch.utils.data import Dataset

from skimage.transform import resize



class IXIN2VDataset(Dataset):


    def __init__(
        self,
        data_dir,
        sequence="T1",
        patch_size=64,
        augment=True
    ):


        self.patch_size=patch_size
        self.augment=augment


        folder=os.path.join(
            data_dir,
            "IXI-"+sequence
        )


        self.files=[]


        for f in os.listdir(folder):

            if f.endswith(".nii") or f.endswith(".nii.gz"):

                self.files.append(
                    os.path.join(folder,f)
                )


        self.files.sort()


        self.images=[]


        for f in self.files:


            img=nib.load(f).get_fdata()


            # central slice

            z=img.shape[2]//2

            img=img[:,:,z]


            img=resize(
                img,
                (256,256),
                preserve_range=True
            )


            img=(img-img.min())/(
                img.max()-img.min()+1e-8
            )


            self.images.append(
                img.astype(np.float32)
            )


        print(
            "IXI N2V images:",
            len(self.images)
        )



    def __len__(self):

        # 增加patch数量

        return len(self.images)*32



    def __getitem__(self,index):


        img=self.images[
            index % len(self.images)
        ]



        ps=self.patch_size


        H,W=img.shape


        y=np.random.randint(
            0,
            H-ps
        )

        x=np.random.randint(
            0,
            W-ps
        )


        patch=img[
            y:y+ps,
            x:x+ps
        ]



        if self.augment:


            k=np.random.randint(0,4)

            patch=np.rot90(
                patch,
                k
            ).copy()


            if np.random.rand()>0.5:

                patch=np.flipud(
                    patch
                ).copy()



        return torch.from_numpy(
            patch
        ).unsqueeze(0).float()