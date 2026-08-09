import os
import numpy as np

import nibabel as nib

import torch

from torch.utils.data import Dataset, DataLoader

from skimage.transform import resize



class IXISwinIRDataset(Dataset):

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


            volume=nib.load(f).get_fdata()


            # central slice
            z=volume.shape[2]//2


            img=volume[:,:,z]


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


        print("========================")
        print("IXI SwinIR Dataset")
        print("Sequence:",sequence)
        print("Volumes :",len(self.images))
        print("========================")



    def add_rician_noise(
            self,
            img
    ):


        n1=np.random.randn(
            *img.shape
        )*self.sigma


        n2=np.random.randn(
            *img.shape
        )*self.sigma


        noisy=np.sqrt(
            (img+n1)**2+n2**2
        )


        return np.clip(
            noisy,
            0,
            1
        )



    def random_crop(self,img):

        ps=self.patch_size


        h,w=img.shape


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
            index % len(self.images)
        ]


        if self.augment:

            img=self.random_crop(img)


            if np.random.rand()>0.5:

                img=np.flip(
                    img,
                    0
                ).copy()


            if np.random.rand()>0.5:

                img=np.flip(
                    img,
                    1
                ).copy()



        noisy=self.add_rician_noise(
            img
        )


        return {

            "noisy":
            torch.from_numpy(noisy)
            .unsqueeze(0)
            .float(),


            "gt":
            torch.from_numpy(img)
            .unsqueeze(0)
            .float()

        }





def get_ixi_loaders(
        data_dir,
        batch_size=4,
        num_workers=0
):


    dataset=IXISwinIRDataset(
        data_dir,
        sequence="T1",
        sigma=0.06,
        patch_size=128,
        augment=True
    )


    n=len(dataset)


    train_len=int(
        n*0.8
    )


    val_len=n-train_len


    train_set,val_set=torch.utils.data.random_split(
        dataset,
        [train_len,val_len],
        generator=torch.Generator().manual_seed(42)
    )


    train_loader=DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )


    val_loader=DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


    return train_loader,val_loader