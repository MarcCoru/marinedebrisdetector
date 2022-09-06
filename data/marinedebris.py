import torch
from torch.utils.data import Dataset, ConcatDataset
import geopandas as gpd
import os
import rasterio as rio
import pandas as pd
import numpy as np

REGIONS = [
    "accra_20181031",
    "lagos_20190101",
    "marmara_20210519",
    "neworleans_20200202",
    "venice_20180630"
]

class MarineDebrisRegionDataset(Dataset):

    def __init__(self, root, region, imagesize=1280, shuffle=False):
        self.points = gpd.read_file(os.path.join(root, region + ".shp"))
        if shuffle:
            self.points = self.points.sample(frac=1)
        self.tifffile = os.path.join(root, region + ".tif")
        self.imagesize = imagesize

        with rio.open(self.tifffile) as src:
            self.crs = src.crs
            self.transform = src.transform
            left, bottom, right, top = src.bounds

        self.points = self.points.to_crs(self.crs)

        # remove points that are too close to the image border
        image_bounds = self.points.buffer(self.imagesize//2).bounds
        out_of_bounds = pd.concat(
            [image_bounds.minx < left, image_bounds.miny < bottom, image_bounds.maxx > right, image_bounds.maxy > top],
            axis=1).any(1)
        self.points = self.points.loc[~out_of_bounds]



    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        point = self.points.iloc[item]

        left, bottom, right, top = point.geometry.buffer(self.imagesize//2).bounds
        window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

        with rio.open(self.tifffile) as src:
            image = src.read(window=window)

            if image.shape[0] == 13: # L1C data <- drop B10 band (index )
                image = image[np.array([0,1,2,3,4,5,6,7,8,9,11,12])]

        image = (image * 1e-4).astype(rio.float32)

        return image, point.type

class MarineDebrisDataset(ConcatDataset):
    def __init__(self, root, fold="train", **kwargs):
        assert fold in ["train", "val"], f"fold {fold} not in train or val"

        if fold == "train":
            self.regions = ["lagos_20190101",
                            "marmara_20210519",
                            "neworleans_20200202",
                            "venice_20180630"]
        elif fold == "val":
            self.regions = ["accra_20181031"]

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [MarineDebrisRegionDataset(root, region, **kwargs) for region in self.regions]
        )

def main():
    ds = MarineDebrisRegionDataset(root="/ssd/marinedebris/marinedebris_refined" ,region="accra_20181031")

    import matplotlib.pyplot as plt
    from skimage.exposure import equalize_hist
    import numpy as np

    image, label = ds[0]
    rgb = equalize_hist(image[np.array([3, 2, 1])]).transpose(1,2,0)
    plt.imshow(rgb)

    plt.show()
    print()



if __name__ == '__main__':
    main()
