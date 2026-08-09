import os
import numpy as np

import nibabel as nib

import torch

from torch.utils.data import Dataset

from skimage.transform import resize



class IXIN2NDataset(Dataset):


    def __init__(
        self,
        data_dir,
        sequence="T1",
        sigma=0.06,
        patch_size=128,
        augment=True
    ):


        self.sigma=sigma
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
            "IXI images:",
            len(self.images)
        )



    def add_rician(self,img):


        n1=np.random.randn(
            *img.shape
        )*self.sigma


        n2=np.random.randn(
            *img.shape
        )*self.sigma



        return np.sqrt(
            (img+n1)**2+n2**2
        ).clip(0,1)



    def random_crop(self,img):


        h,w=img.shape


        ps=self.patch_size


        y=np.random.randint(
            0,
            h-ps
        )

        x=np.random.randint(
            0,
            w-ps
        )


        return img[
            y:y+ps,
            x:x+ps
        ]



    def __len__(self):

        return len(self.images)*20



    def __getitem__(self,index):


        img=self.images[
            index%len(self.images)
        ]


        if self.augment:

            img=self.random_crop(img)


            if np.random.rand()>0.5:

                img=np.flip(
                    img,
                    axis=0
                )


        noisy1=self.add_rician(img)

        noisy2=self.add_rician(img)



        return (

            torch.from_numpy(noisy1)
            .unsqueeze(0)
            .float(),


            torch.tensor(
                self.sigma
            ).float(),


            torch.from_numpy(noisy2)
            .unsqueeze(0)
            .float()

        )