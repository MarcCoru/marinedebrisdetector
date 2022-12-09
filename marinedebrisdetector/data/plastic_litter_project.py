import os

import torch
import rasterio as rio
from rasterio import features
import geopandas as gpd
import numpy as np
from skimage.exposure import equalize_hist

bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
bands_ps = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
bands_ps_used = ["blue", "green", "red", "nir"] # use BGR+NIR for compatibility

data_url = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/PLP.zip"

def download(data_path):
    import urllib.request
    import zipfile

    if not os.path.exists("PLP.zip"):
        urllib.request.urlretrieve(data_url, "PLP.zip")

    with zipfile.ZipFile("PLP.zip", 'r') as zip_ref:
        zip_ref.extractall(data_path)

class PLPDataset(torch.utils.data.Dataset):
    def __init__(self, root, year=2021, output_size=64):
        self.year = year
        self.image_root = os.path.join(root, f"PLP{year}", "Sentinel-2")
        targets_shapefile = os.path.join(root, f"PLP{year}", f"PLP{year}_targets.shp")
        self.targets_gdf = gpd.read_file(targets_shapefile)
        self.s2 = sorted([img for img in os.listdir(self.image_root) if img.endswith(".tif")])
        self.dates = [i[:8] for i in self.s2]
        self.output_size = output_size

    def __len__(self):
        return len(self.s2)

    def __getitem__(self, index):
        date = self.dates[index]
        img = self.s2[index]
        with rio.open(os.path.join(self.image_root, img), "r") as src:
            df = self.targets_gdf.loc[self.targets_gdf.date == date]
            buffered = df.copy()
            buffered["geometry"] = buffered.apply(buffer, axis=1)

            shapes = ((geom, value) for geom, value in zip(buffered.geometry, buffered.id))

            rasterized = features.rasterize(shapes,
                                            out_shape=src.shape,
                                            fill=0,
                                            out=None,
                                            transform=src.transform,
                                            all_touched=True,
                                            default_value=-1,
                                            dtype=None)

            arr = src.read()

            arr = center_crop(arr, image_size=self.output_size)
            rasterized = center_crop(rasterized[None], image_size=self.output_size)[0]

            arr = np.nan_to_num(arr)

        arr = arr * 1e-4

        return arr, rasterized, date

class PlanetScopePLPDataset(torch.utils.data.Dataset):
    def __init__(self, root, year=2022, output_size=64):
        assert year == 2022, "only 2022 available"

        self.year = year
        self.image_root = os.path.join(root, f"PLP{year}", "PlanetScope", "files")

        self.ps = sorted([img for img in os.listdir(self.image_root) if img.endswith("8b_harmonized_clip.tif")])
        self.dates = [i[:8] for i in self.ps]
        self.output_size = output_size

    def __len__(self):
        return len(self.ps)

    def __getitem__(self, index):
        date = self.dates[index]
        img = self.ps[index]
        with rio.open(os.path.join(self.image_root, img), "r") as src:

            arr = src.read()

            arr = center_crop(arr, image_size=self.output_size)

            arr = np.nan_to_num(arr)

        arr = arr * 1e-4

        arr = arr[np.array([bands_ps.index(b) for b in bands_ps_used])]

        return arr, np.zeros_like(arr[0]), self.ps[index]

def center_crop(arr, image_size=128):
    w = image_size // 2
    D, H, W = arr.shape
    cx = H // 2
    cy = W // 2
    return arr[:, cx - w:cx + w, cy - w: cy + w]

def buffer(row):
    if not np.isnan(row.radius):
        return row.geometry.buffer(row.radius)
    else:
        return row.geometry.buffer(row.width, cap_style=3) # rectangular


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

def ps_to_RGB(scene):
    tensor = np.stack([scene[bands_ps_used.index('red')], scene[bands_ps_used.index('green')], scene[bands_ps_used.index('blue')]])
    return equalize_hist(tensor.swapaxes(0,1).swapaxes(1,2))

def s2_to_NDVI(scene):
    NIR = scene[bands.index("B8")]
    RED = scene[bands.index("B4")]
    return (NIR - RED) / (NIR + RED)

def main():
    import matplotlib.pyplot as plt

    # download to /tmp/PLP
    download("/tmp")

    year = 2022
    ds = PlanetScopePLPDataset(root="/tmp/PLP", year=year, output_size=32)
    for x,y,id in ds:
        print(x.shape, id)
        for image, mask, id in ds:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(ps_to_RGB(image))
            ax.set_title(id)

    plt.show()

    year = 2022
    ds = PLPDataset(root="/tmp/PLP", year=year, output_size=32)

    for image, mask, id in ds:
        fig, axs = plt.subplots(1,3,figsize=(3*3,3))

        ax = axs[0]
        ax.imshow(s2_to_RGB(image))
        ax.set_title("RGB")
        ax.axis("off")

        ax = axs[1]
        ax.imshow(s2_to_FDI(image))
        ax.set_title("FDI")
        ax.axis("off")

        ax = axs[2]
        ax.imshow(mask)
        ax.set_title("FDI")
        ax.axis("off")

        fig.suptitle(id)
        fig.tight_layout()

        os.makedirs(f"/tmp/PLP{year}", exist_ok=True)
        fig.savefig(f"/tmp/PLP{year}/{id}.png")

    plt.show()

if __name__ == '__main__':
    main()
