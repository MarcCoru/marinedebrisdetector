from torch.utils.data import Dataset
import geopandas as gpd
import os
from glob import glob
import rasterio as rio
from rasterio.merge import merge

regions = ['S2_1-12-19_48MYU',
     'S2_11-1-19_19QDA',
     'S2_11-6-18_16PCC',
     'S2_12-1-17_16PCC',
     'S2_12-1-17_16PEC',
     'S2_12-1-19_16PEC',
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
     'S2_18-9-20_16PCC',
     'S2_18-9-20_16PDC',
     'S2_19-3-20_18QYF',
     'S2_19-9-18_16PCC',
     'S2_19-9-18_16PDC',
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
     'S2_24-4-19_36JUN',
     'S2_24-8-20_16PCC',
     'S2_25-5-19_48MXU',
     'S2_26-2-18_16PCC',
     'S2_27-1-19_16PCC',
     'S2_27-1-19_16QED',
     'S2_28-9-20_16PCC',
     'S2_28-9-20_16PDC',
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
     'S2_4-3-18_50LLR',
     'S2_4-9-16_16PCC',
     'S2_4-9-19_16PCC',
     'S2_6-12-17_48MYU',
     'S2_6-12-18_48MXU',
     'S2_7-10-18_52SDD',
     'S2_7-3-20_18QYG',
     'S2_8-3-18_16PEC',
     'S2_8-3-18_16QED',
     'S2_9-10-17_16PEC']

class MaridaRegionDataset(Dataset):
    def __init__(self,path,region):
        gdf = gpd.read_file(os.path.join(path, "shapefiles", region + ".shp"))
        tiles = glob(os.path.join(path, "patches", region, f"{region}_?.tif"))
        arr, transform = merge(tiles)

        print()
        pass

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return


if __name__ == '__main__':
    ds = MaridaRegionDataset(path="/data/marinedebris/MARIDA", region=regions[0])
    ds[0]