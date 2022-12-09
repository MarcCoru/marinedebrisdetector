import rasterio.windows
from torch.utils.data import Dataset, ConcatDataset
import geopandas as gpd
import os
import rasterio as rio
import pandas as pd
from rasterio.features import rasterize
import numpy as np
from .utils import read_tif_image, pad
import torch

# regions where we could not re-download the corresponding tif image

EXCLUDED = [
    'S2_24-4-19_36JUN', # durban (mexcluded)
]

REGIONS = ['S2_1-12-19_48MYU',
     'S2_11-1-19_19QDA',
     'S2_11-6-18_16PCC',
     'S2_12-12-20_16PCC',
     'S2_13-12-18_16PCC',
     'S2_14-11-18_48PZC',
     'S2_14-12-20_18QYF',
     'S2_14-3-20_18QYF',
     'S2_14-9-18_16PCC',
     'S2_15-10-20_18QYF',
     'S2_15-11-20_16PCC',
     'S2_15-9-20_18QYF',
     'S2_16-2-18_16PEC',
     'S2_17-7-16_51PTS',
     'S2_18-1-18_48PZC',
     'S2_18-5-19_51PTS',
     'S2_19-3-20_18QYF',
     'S2_20-10-20_18QYF',
     'S2_20-4-18_30VWH',
     'S2_21-2-17_16PCC',
     'S2_21-2-18_16PCC',
     'S2_22-12-20_18QYF',
     'S2_22-3-20_18QWF',
     'S2_23-1-21_18QYF',
     'S2_23-9-20_16PCC',
     'S2_24-10-18_16PDC',
     'S2_24-11-19_48PZC',
     'S2_24-3-20_18QYF',
     'S2_24-8-20_16PCC',
     'S2_25-5-19_48MXU',
     'S2_26-2-18_16PCC',
     'S2_27-1-19_16QED',
     'S2_29-11-15_16PEC',
     'S2_29-11-20_18QYF',
     'S2_29-12-20_18QYF',
     'S2_29-8-17_51RVQ',
     'S2_3-1-21_18QYF',
     'S2_3-11-16_16PDC',
     'S2_3-11-18_16PDC',
     'S2_30-8-17_16PCC',
     'S2_30-8-18_16PCC',
     'S2_4-12-20_18QYF',
     'S2_4-9-16_16PCC',
     'S2_4-9-19_16PCC',
     'S2_6-12-17_48MYU',
     'S2_6-12-18_48MXU',
     'S2_7-10-18_52SDD',
     'S2_7-3-20_18QYG',
     'S2_9-10-17_16PEC'    
     'S2_12-1-17_16PEC',
    'S2_12-1-17_16PCC',
    'S2_19-9-18_16PCC',
    'S2_19-9-18_16PDC',
    'S2_4-3-18_50LLR',
    'S2_8-3-18_16PEC',
    'S2_8-3-18_16QED',
    'S2_27-1-19_16PCC',
    'S2_28-9-20_16PCC',
    'S2_18-9-20_16PDC',
    'S2_12-1-19_16PEC',
    'S2_28-9-20_16PDC',
    'S2_18-9-20_16PCC']

CLASS_MAPPING = {
     1: 'Marine Debris',
     2: 'Dense Sargassum',
     3: 'Sparse Sargassum',
     4: 'Natural Organic Material',
     5: 'Ship',
     6: 'Clouds',
     7: 'Marine Water',
     8: 'Sediment-Laden Water',
     9: 'Foam',
     10: 'Turbid Water',
     11: 'Shallow Water',
     12: 'Waves',
     13: 'Cloud Shadows',
     14: 'Wakes',
     15: 'Mixed Water'
}

# the paper uses fewer classes in the evaluation tables
CLASS_MAPPING_USED = {
     1: 'Marine Debris',
     2: 'Dense Sargassum',
     3: 'Sparse Sargassum',
     4: 'Natural Organic Material',
     5: 'Ship',
     6: 'Clouds',
     7: 'Marine Water',
     8: 'Sediment-Laden Water',
     9: 'Foam',
     10: 'Turbid Water',
     11: 'Shallow Water'
}

KEEP_CLASSES = [1, 7]

DEBRIS_CLASSES = [1,2,3,4,9]

class MaridaRegionDataset(Dataset):
     def __init__(self,path,region, imagesize=128, data_transform=None, classification=False):
         self.imagesize = imagesize * 10
         self.data_transform = data_transform
         self.region = region
         self.classification = classification

         tile = region[-5:]

         gdf = gpd.read_file(os.path.join(path, "shapefiles", region + ".shp"))

         # keep only classes in keep classes
         gdf = gdf.loc[gdf["id"].isin(KEEP_CLASSES)]

         self.maskpath = os.path.join(path, "masks", region + ".tif")
         os.makedirs(os.path.dirname(self.maskpath), exist_ok=True)

         mapping = pd.read_csv(os.path.join(path,"marida_mapping.csv"))
         m = mapping.loc[mapping.region == region]

         # keep only image that matches the tile of the region
         m = m.loc[m.tifpath.apply(lambda x: x.endswith(tile + ".tif"))]

         # pick L2A if available randomly
         if "S2_SR" in list(m["mod"]):
             m = m.loc[m["mod"] == "S2_SR"]

         if len(m) == 0:
             print(region)

         assert len(m) == 1

         self.tifpath = os.path.join(path, "scenes", m.tifpath.iloc[0])
         with rio.open(self.tifpath) as src:
              crs = src.crs
              width = src.width
              height = src.height
              transform = src.transform
              profile = src.profile

         self.gdf = gdf.to_crs(crs)
         self.transform = transform

         if not os.path.exists(self.maskpath):
             mask = rasterize(zip(self.gdf.geometry, self.gdf.id), all_touched=True,
                                       transform=transform, out_shape=(height, width))

             profile["count"] = 1
             profile["dtype"] = "uint8"

             print(f"writing mask to {self.maskpath}")
             with rio.open(self.maskpath, "w", **profile) as dst:
                dst.write(mask[None])

     def __len__(self):
        return len(self.gdf)

     def __getitem__(self, item):
          row = self.gdf.iloc[item]
          minx, miny, maxx, maxy = row.geometry.centroid.buffer(self.imagesize // 2).bounds
          window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=self.transform)

          image, _ = read_tif_image(self.tifpath, window)
          image = image.astype("float")

          with rasterio.open(self.maskpath, "r") as src:
              mask = src.read(window=window)[0]

          # agreggate labels to binary classes
          mask = (np.stack([mask == c for c in DEBRIS_CLASSES]).sum(0) > 0).astype(int)

          image, mask = pad(image, mask, self.imagesize // 10)

          if self.data_transform is not None:
              image, mask = self.data_transform(image, mask)

          if self.classification:
              mask = torch.tensor(row.id in DEBRIS_CLASSES).long()

          return image, mask, f"marida-{item}"

class MaridaDataset(ConcatDataset):
    def __init__(self, path, fold="train", **kwargs):
        assert fold in ["train", "val","test"]

        with open(os.path.join(path,"splits",f"{fold}_X.txt")) as f:
            lines = f.readlines()

        self.regions = list(set(["S2_" + "_".join(l.replace("\n","").split("_")[:-1]) for l in lines]))
        self.regions = [r for r in self.regions if r in REGIONS]

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [MaridaRegionDataset(path, region, **kwargs) for region in self.regions]
        )



if __name__ == '__main__':
    #ds = MaridaRegionDataset(path="/data/marinedebris/MARIDA", region="S2_28-9-20_16PCC")
    #ds[14]



    ds = MaridaDataset(path="/ssd/marinedebris/MARIDA", fold="train")
    print(len(ds))

    ds = MaridaDataset(path="/ssd/marinedebris/MARIDA", fold="val")
    print(len(ds))

    ds = MaridaDataset(path="/ssd/marinedebris/MARIDA", fold="test")
    print(len(ds))

    import matplotlib.pyplot as plt
    from visualization import rgb

    N = 5
    fig, axs_all = plt.subplots(N,2, figsize=(3*2,3*N))
    idxs = np.random.RandomState(1).randint(0,len(ds),N)

    for axs, idx in zip(axs_all, idxs):
        image, mask, id = ds[idx]

        axs[0].imshow(rgb(image).transpose(1,2,0))
        axs[1].imshow(mask)
        [ax.axis("off") for ax in axs]

    plt.show()

    """
    from tqdm import tqdm
    region = MISSING_REGIONS[0]
    for region in tqdm(MISSING_REGIONS):
        ds = MaridaRegionDataset(path="/ssd/marinedebris/MARIDA", region=region)
        for image, mask, id in ds:
            assert image.shape == (12,128,128), f"{region}-{id} image wrong size {image.shape}"
            assert mask.shape == (128, 128), f"{region}-{id} mask wrong size {mask.shape}"
    """
