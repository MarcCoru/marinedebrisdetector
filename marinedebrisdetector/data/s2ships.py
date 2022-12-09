from torch.utils.data import Dataset
import os
import torch
import numpy as np

regions = ['01_mask_rome.npy',
     '07_mask_suez6.npy',
     '02_mask_suez1.npy',
     '08_mask_brest1.npy',
     '13_mask_rotterdam1.npy',
     '03_mask_suez2.npy',
     '11_mask_marseille.npy',
     '16_mask_southampton.npy',
     '05_mask_suez4.npy',
     '14_mask_rotterdam2.npy',
     '10_mask_toulon.npy',
     '09_mask_panama.npy',
     '06_mask_suez5.npy',
     '04_mask_suez3.npy',
     '15_mask_rotterdam3.npy',
     '12_mask_portsmouth.npy']

class S2shipsRegion(Dataset):
    def __init__(self, root, region, transform=None, imagesize=64):
        assert region in regions

        f = np.load(os.path.join(root, "dataset_npy", region), allow_pickle=True)
        self.image = f.item().get("data")
        self.label = f.item().get("label")

        H,W,C = self.image.shape

        x,y,_ = np.where(self.label)
        in_x_bounds = (x > imagesize//2) & (x < H-imagesize//2)
        in_y_bounds = (y > imagesize//2) & (y < W - imagesize//2)
        in_bounds = in_x_bounds & in_y_bounds

        self.region = region
        self.transform = transform
        self.imagesize = imagesize
        self.points = list(zip(x[in_bounds], y[in_bounds]))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        x,y = self.points[item]

        sz = self.imagesize // 2
        x_image = self.image[x - sz:x + sz, y - sz:y + sz]
        mask = np.zeros_like(x_image[:,:,0])

        x_image = x_image.transpose(2,0,1)

        if self.transform is not None:
            x_image, mask = self.transform(x_image, mask)

        return x_image, mask, f"{self.region}-{item}"

class S2Ships(torch.utils.data.ConcatDataset):
    def __init__(self, root, **kwargs):

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [S2shipsRegion(root, region, **kwargs) for region in regions]
        )

if __name__ == '__main__':


    import matplotlib.pyplot as plt
    from visualization import rgb, fdi, ndvi
    from transforms import get_transform

    ds = S2Ships(root="/ssd/marinedebris/S2SHIPS", imagesize=80, transform=get_transform("train", cropsize=64))

    for idx in np.random.randint(0,len(ds),10):
        image, mask, id = ds[idx]
        image = image.numpy()
        fig, axs = plt.subplots(1,3, figsize=(3*3,3))
        axs[0].imshow(rgb(image).transpose(1, 2, 0))
        axs[1].imshow(fdi(image), cmap="magma")
        axs[2].imshow(ndvi(image), cmap="viridis")
        fig.suptitle(id)

        for ax, title in zip(axs,["rgb","fdi","ndvi"]):
            ax.axis("off")
            ax.set_title(title)

        plt.tight_layout()
    plt.show()
