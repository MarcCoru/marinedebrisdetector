import torch
from torch.utils.data import Dataset, ConcatDataset
import geopandas as gpd
import os
import rasterio as rio
import pandas as pd
import numpy as np
from data.utils import read_tif_image

REGIONS = [
    #"accra_20181031", # remove accra, as will be in test
    "lagos_20190101",
    "marmara_20210519",
    "neworleans_20200202",
    "venice_20180630"
]

val_regions = ["lagos_20190101",
                "marmara_20210519",
                "neworleans_20200202",
                "venice_20180630"]

test_regions = ["accra_20181031"]

class RefinedFlobsRegionDataset(Dataset):

    def __init__(self, root, region, imagesize=1280, shuffle=False, transform=None):
        self.points = gpd.read_file(os.path.join(root, region + ".shp"))
        if shuffle:
            self.points = self.points.sample(frac=1, random_state=0)
        self.tifffile = os.path.join(root, region + ".tif")
        self.imagesize = imagesize
        self.data_transform = transform

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

        image = torch.from_numpy((image * 1e-4).astype(rio.float32))

        if self.data_transform is not None:
            image = self.data_transform(image)

        return image, point.type, item

class RefinedFlobsDataset(ConcatDataset):
    def __init__(self, root, fold="val", **kwargs):
        assert fold in ["val", "test"], f"fold {fold} not in val or test"

        if fold == "val":
            self.regions = val_regions
        elif fold == "test": # may not be used
            self.regions = test_regions
        else:
            raise NotImplementedError()

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [RefinedFlobsRegionDataset(root, region, **kwargs) for region in self.regions]
        )

class RefinedFlobsQualitativeRegionDataset(Dataset):
    def __init__(self, root, region, output_size=None):
        self.tifffile = os.path.join(root, region + ".tif")
        self.region = region
        self.output_size = output_size # overwrite geometry and fix a squared output size

        bboxes = gpd.read_file(os.path.join(root, region+"_qualitative_bbox.shp"))

        with rio.open(self.tifffile) as src:
            self.crs = src.crs
            self.transform = src.transform

        bboxes = bboxes.to_crs(self.crs)

        if output_size is not None:
            bboxes.geometry = bboxes.geometry.centroid.buffer((output_size * 10)//2)

        self.images = []
        for i, row in bboxes.iterrows():
            left, bottom, right, top = row.geometry.bounds
            window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

            self.images.append(read_tif_image(self.tifffile, window)[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item] * 1e-4
        return image, f"{self.region}-{item}"

class RefinedFlobsQualitativeDataset(ConcatDataset):
    def __init__(self, root, fold="val", **kwargs):
        assert fold in ["val", "test"], f"fold {fold} not in val or test"

        if fold == "val":
            self.regions = val_regions
        elif fold == "test": # may not be used
            self.regions = test_regions
        else:
            raise NotImplementedError()

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [RefinedFlobsQualitativeRegionDataset(root, region, **kwargs) for region in self.regions]
        )

def main():

    ds = RefinedFlobsQualitativeDataset(root="/data/marinedebris/marinedebris_refined", fold="val")
    for i,id in ds:
        print(i.shape, id)

    ds = RefinedFlobsQualitativeDataset(root="/data/marinedebris/marinedebris_refined", fold="test")
    for i,id in ds:
        print(i.shape, id)

    return
    ds = RefinedFlobsRegionDataset(root="/data/marinedebris/marinedebris_refined" ,region="accra_20181031")

    import matplotlib.pyplot as plt
    from skimage.exposure import equalize_hist
    import numpy as np

    image, label, id = ds[0]
    rgb = equalize_hist(image.numpy()[np.array([3, 2, 1])]).transpose(1,2,0)
    plt.imshow(rgb)

    plt.show()
    print()



if __name__ == '__main__':
    main()
