import torch
from torch.utils.data import Dataset, ConcatDataset
import geopandas as gpd
import os
import rasterio as rio
import pandas as pd
from .utils import read_tif_image
from rasterio import features

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

test_regions = ["accra_20181031",
                "durban_20190424"]

class RefinedFlobsRegionDataset(Dataset):

    def __init__(self, root, region, imagesize=1280, shuffle=False, transform=None):
        self.points = gpd.read_file(os.path.join(root, region + ".shp"))

        self.region = region
        if shuffle:
            self.points = self.points.sample(frac=1, random_state=0)
        self.tifffile = os.path.join(root, region + ".tif")
        self.imagesize = imagesize
        self.data_transform = transform

        with rio.open(self.tifffile) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.height, self.width = src.height, src.width
            profile = src.profile
            left, bottom, right, top = src.bounds

        self.points = self.points.to_crs(self.crs)

        # remove points that are too close to the image border
        image_bounds = self.points.buffer(self.imagesize//2).bounds
        out_of_bounds = pd.concat(
            [image_bounds.minx < left, image_bounds.miny < bottom, image_bounds.maxx > right, image_bounds.maxy > top],
            axis=1).any(axis=1)
        self.points = self.points.loc[~out_of_bounds]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        point = self.points.iloc[item]

        left, bottom, right, top = point.geometry.buffer(self.imagesize//2).bounds
        window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

        image, _ = read_tif_image(self.tifffile, window)

        image = torch.from_numpy((image * 1e-4).astype(rio.float32))

        if self.data_transform is not None:
            image = self.data_transform(image)

        return image, point.type, f"{self.region}-{item}"

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

        segments_file = os.path.join(root, region + "_qualitative_poly.shp")
        self.segments = gpd.read_file(segments_file) if os.path.exists(segments_file) else None
        self.segments_tiffile = os.path.join(root, region + "_qualitative_poly.tif") if os.path.exists(segments_file) else None

        with rio.open(self.tifffile) as src:
            self.crs = src.crs
            self.transform = src.transform
            profile = src.profile

        bboxes = bboxes.to_crs(self.crs)

        if output_size is not None:
            bboxes.geometry = bboxes.geometry.centroid.buffer((output_size * 10)//2)

        # segmentation polygons
        if self.segments is not None:
            profile.update(
                count=1,
                dtype="uint8"
            )
            self.segments = self.segments.to_crs(self.crs)

            if not os.path.exists(self.segments_tiffile):
                mask = features.rasterize(self.segments.geometry, all_touched=True,
                                          transform=self.transform, out_shape=(self.height, self.width))

                with rio.open(self.segments_tiffile, "w", **profile) as dst:
                    dst.write(mask[None])

        self.images = []
        for i, row in bboxes.iterrows():
            left, bottom, right, top = row.geometry.bounds
            window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

            image = read_tif_image(self.tifffile, window)[0]

            if self.segments_tiffile is not None:
                with rio.open(self.segments_tiffile) as src:
                    mask = src.read(1, window=window)
            else:
                mask = None

            self.images.append((image, mask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image, mask = self.images[item]
        image = image * 1e-4
        return image, mask, f"{self.region}-{item}"

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
