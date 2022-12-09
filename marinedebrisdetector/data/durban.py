import torch
from rasterio.windows import from_bounds
import rasterio as rio
from rasterio import features
import geopandas as gpd
import os
import numpy as np
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt

bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

annotated_objects = ['debris', 'ships', 'land', 'coastline', 'cummulus_clouds',
       'haze_dense', 'haze_transparent']

class DurbanDataset(torch.utils.data.Dataset):
    def __init__(self, root, output_size=64,
                 transform=None, objects="debris"):
        if objects == "debris":
            self.shapefile = os.path.join(root, "durban_20190424_debris.shp")
        elif objects == "ships":
            self.shapefile = os.path.join(root, "durban_20190424_ships.shp")
        else:
            raise ValueError(f"specified objects {objects} not valid. either `debris` or ships")

        self.imagefile = os.path.join(root, "durban_20190424.tif")
        self.polygons = gpd.read_file(self.shapefile)
        self.transform = transform
        self.output_size=output_size

        with rio.open(self.imagefile) as src:
            self.imagemeta = src.meta
            self.imagebounds = tuple(src.bounds)

        self.polygons = self.polygons.to_crs(self.imagemeta["crs"])

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, index):
        line = self.polygons.iloc[index]
        left, bottom, right, top = line.geometry.bounds

        width = right - left
        height = top - bottom

        # buffer_left_right = (self.output_size[0] * 10 - width) / 2
        buffer_left_right = (self.output_size * 10 - width) / 2
        left -= buffer_left_right
        right += buffer_left_right

        # buffer_bottom_top = (self.output_size[1] * 10 - height) / 2
        buffer_bottom_top = (self.output_size * 10 - height) / 2
        bottom -= buffer_bottom_top
        top += buffer_bottom_top

        window = from_bounds(left, bottom, right, top, self.imagemeta["transform"])

        with rio.open(self.imagefile) as src:
            image = src.read(window=window)
            win_transform = src.window_transform(window)

        mask = features.rasterize(self.polygons.geometry, all_touched=True,
                                  transform=win_transform, out_shape=image[0].shape)

        # if feature is near the image border, image wont be the desired output size
        H, W = self.output_size, self.output_size
        c, h, w = image.shape
        dh = (H - h) / 2
        dw = (W - w) / 2
        image = np.pad(image, [(0, 0), (int(np.ceil(dh)), int(np.floor(dh))),
                               (int(np.ceil(dw)), int(np.floor(dw)))])

        mask = np.pad(mask, [(int(np.ceil(dh)), int(np.floor(dh))),
                             (int(np.ceil(dw)), int(np.floor(dw)))])

        mask = mask.astype(float)
        image = (image * 1e-4).astype(float)

        image = np.nan_to_num(image)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, line.name

def s2_to_FDI(scene):

    NIR = scene[bands.index("B8")]
    RED2 = scene[bands.index("B6")]

    SWIR1 = scene[bands.index("B11")]

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    return NIR - NIR_prime

def s2_to_RGB(scene):
    tensor = np.stack([scene[bands.index('B4')],scene[bands.index('B3')],scene[bands.index('B2')]])
    return equalize_hist(tensor.swapaxes(0,1).swapaxes(1,2))

def s2_to_NDVI(scene):
    NIR = scene[bands.index("B8")]
    RED = scene[bands.index("B4")]
    return (NIR - RED) / (NIR + RED + 1e-12)

def write_annotations_file(
        objects_file = "/data/marinedebris/durban/durban_20190424_objects.shp",
        tiffile = "/data/marinedebris/durban/durban_20190424.tif",
        annotationfile = "/data/marinedebris/durban/durban_20190424_annotated.tif"
    ):

    gdf = gpd.read_file(objects_file)

    with rio.open(tiffile, "r") as src:
        gdf = gdf.to_crs(src.crs)
        profile = src.profile

    profile.update(
        count=len(annotated_objects),
        dtype="uint8"
    )

    with rio.open(annotationfile, "w", **profile) as dst:
        for i, object_name in enumerate(annotated_objects):
            gdf_ = gdf.loc[gdf["name"] == object_name]

            with rio.open(tiffile, "r") as src:
                mask = features.rasterize(gdf_.geometry, all_touched=True,
                                          transform=src.transform, out_shape=(src.height, src.width))

                dst.write_band(i + 1, mask)

                dst.set_band_description(i + 1, object_name)

if __name__ == '__main__':
    write_annotations_file()

    ds = DurbanDataset(root="/data/marinedebris/durban")
    for image, mask, id in ds:
        fig, axs = plt.subplots(1,4,figsize=(3*4,3))

        ax = axs[0]
        ax.imshow(s2_to_RGB(image))
        ax.set_title("RGB")
        ax.axis("off")

        ax = axs[1]
        ax.imshow(s2_to_FDI(image))
        ax.set_title("FDI")
        ax.axis("off")

        ax = axs[2]
        ax.imshow(s2_to_NDVI(image))
        ax.set_title("NDVI")
        ax.axis("off")

        ax = axs[3]
        ax.imshow(mask)
        ax.set_title("FDI")
        ax.axis("off")

        fig.suptitle(id)
        fig.tight_layout()

        break

    plt.show()
