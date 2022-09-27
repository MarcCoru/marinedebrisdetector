import torch
from rasterio.windows import from_bounds
import rasterio as rio
from rasterio import features
from shapely.geometry import LineString, Polygon
import geopandas as gpd
import os
import numpy as np
import pandas as pd
from data.utils import get_window, read_tif_image, pad, line_is_closed, \
    split_line_gdf_into_segments, remove_lines_outside_bounds
from data.label_refinement import refine_masks

L1CBANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# offset from image border to sample hard negative mining samples
HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET = 1000  # meter

trainregions = [
    "biscay_20180419",
    "danang_20181005",
    "kentpointfarm_20180710",
    "kolkata_20201115",
    "lagos_20200505",
    "london_20180611",
    "longxuyen_20181102",
    "mandaluyong_20180314",
    "panama_20190425",
    "portalfredSouthAfrica_20180601",
    "riodejaneiro_20180504",
    "sandiego_20180804",
    "sanfrancisco_20190219",
    "shengsi_20190615",
    "suez_20200403",
    "tangshan_20180130",
    "toledo_20191221",
    "tungchungChina_20190922",
    "tunisia_20180715",
    "turkmenistan_20181030",
    "venice_20180928",
    "vungtau_20180423"
    ]

# same as regions in marinedebris.py
valregions = [
    "accra_20181031",
    "lagos_20190101",
    "neworleans_20200202",
    "venice_20180630"
]


def get_region_split(seed=0, fractions=(0.6, 0.2, 0.2)):

    # fix random state
    random_state = np.random.RandomState(seed)

    # shuffle sequence of regions
    shuffled_regions = random_state.permutation(allregions)

    # determine first N indices for training
    train_idxs = np.arange(0, np.floor(len(shuffled_regions) * fractions[0]).astype(int))

    # next for validation
    idx = np.ceil(len(shuffled_regions) * (fractions[0] + fractions[1])).astype(int)
    val_idxs = np.arange(train_idxs.max() + 1, idx)

    # the remaining for test
    test_idxs = np.arange(val_idxs.max() + 1, len(shuffled_regions))

    return dict(train=list(shuffled_regions[train_idxs]),
                val=list(shuffled_regions[val_idxs]),
                test=list(shuffled_regions[test_idxs]))



class FloatingSeaObjectRegionDataset(torch.utils.data.Dataset):
    def __init__(self, root, region, output_size=64,
                 transform=None, hard_negative_mining=True,
                 refine_labels=True, cache_to_npy=True):

        shapefile = os.path.join(root, region + ".shp")

        imagefile = os.path.join(root, region + ".tif")
        imagefilel2a = os.path.join(root, region + "_l2a.tif")
        if os.path.exists(imagefilel2a):
            imagefile = imagefilel2a # use l2afile if exists

        self.refine_labels = refine_labels

        self.transform = transform
        self.region = region

        self.imagefile = imagefile
        self.output_size = output_size

        with rio.open(imagefile) as src:
            self.imagemeta = src.meta
            self.imagebounds = tuple(src.bounds)

        lines = gpd.read_file(shapefile)
        lines = lines.to_crs(self.imagemeta["crs"])

        # find closed lines, convert them to polygons and store them separately for later rasterization
        is_closed_line = lines.geometry.apply(line_is_closed)
        rasterize_polygons = lines.loc[is_closed_line].geometry.apply(Polygon)

        self.lines = split_line_gdf_into_segments(lines)

        self.lines["is_hnm"] = False
        if hard_negative_mining:
            random_points = self.sample_points_for_hard_negative_mining()
            random_points["is_hnm"] = True
            self.lines = pd.concat([self.lines, random_points]).reset_index(drop=True)

        # remove line segments that are outside the image bounds
        self.lines = remove_lines_outside_bounds(self.lines, self.imagebounds)

        # take lines to rasterize
        rasterize_lines = self.lines.loc[~self.lines["is_hnm"]].geometry

        # combine with polygons to rasterize
        self.rasterize_geometries = pd.concat([rasterize_lines, rasterize_polygons])

        if cache_to_npy:
            reflab_suffix = "_reflab" if refine_labels else ""
            self.npyfolder = os.path.join(root, f"npy_{output_size}"+reflab_suffix, region)
            os.makedirs(self.npyfolder, exist_ok=True)



    def sample_points_for_hard_negative_mining(self):
        # hard negative mining:
        # get some random negatives from the image bounds to ensure that the model can learn on negative examples
        # e.g. land, clouds, etc

        with rio.open(self.imagefile) as src:
            left, bottom, right, top = src.bounds

        offset = HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET  # m
        assert top - bottom > 2 * HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET, f"Hard Negative Mining offset 2x{HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET}m too large for the image height: {top - bottom}m"
        assert right - left > 2 * HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET, f"Hard Negative Mining offset 2x{HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET}m too large for the image width: {right - left}m"
        N_random_points = len(self.lines)

        # sample random x positions within bounds
        zx = np.random.rand(N_random_points)
        zx *= ((right - offset) - (left + offset))
        zx += left + offset

        # sample random y positions within bounds
        zy = np.random.rand(N_random_points)
        zy *= ((top - offset) - (bottom + offset))
        zy += bottom + offset

        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(zx, zy))

    def __len__(self):
        return len(self.lines)

    def item_in_cache(self, index):
        return os.path.exists(os.path.join(self.npyfolder, str(index) + ".npz"))
    def get_item_from_cache(self, index):
        # CACHING to Npyfolder for faster loading (20 seconds versus 3 minutes)
        npzfile = os.path.join(self.npyfolder,str(index)+".npz")

        with np.load(npzfile) as f:
            image = f["image"]
            mask = f["mask"]
            id = f["id"]

        return image, mask, str(id)

    def __getitem__(self, index):
        if hasattr(self, 'npyfolder') and self.item_in_cache(index):
            image, mask, id = self.get_item_from_cache(index)
        else:
            image, mask, id = self.get_item_from_image(index)

            # save to numpy
            if hasattr(self, 'npyfolder'):
                np.savez(os.path.join(self.npyfolder, str(index) + ".npz"),
                         image=image,
                         mask=mask,
                         id=id)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, id

    def get_item_from_image(self, index):
        line = self.lines.iloc[index]

        window = get_window(line, output_size=self.output_size, transform=self.imagemeta["transform"])

        image, win_transform = read_tif_image(self.imagefile, window)

        # rasterize geometries to mask
        mask = features.rasterize(self.rasterize_geometries, all_touched=True,
                                  transform=win_transform, out_shape=image[0].shape)

        # pad image if at the border
        image, mask = pad(image, mask, self.output_size)

        # to float
        image, mask = image.astype(float), mask.astype(float)


        # mark random points form hard negative mining with a suffix
        # to distinguish them from actual labels
        hard_negative_mining_suffix = "-hnm" if line["is_hnm"] else ""
        id = f"{self.region}-{index}" + hard_negative_mining_suffix

        image = np.nan_to_num(image)

        if self.refine_labels:
            if len(np.unique(mask)) > 1: # only if labels are present to be refined
                mask = refine_masks(image, mask)

        return image, mask, id


class FloatingSeaObjectDataset(torch.utils.data.ConcatDataset):
    def __init__(self, root, fold="train", **kwargs):
        assert fold in ["train", "val"]

        if fold=="train":
            self.regions = trainregions
        elif fold == "val":
            self.regions = valregions
        else:
            raise NotImplementedError()

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [FloatingSeaObjectRegionDataset(root, region, **kwargs) for region in self.regions]
        )


