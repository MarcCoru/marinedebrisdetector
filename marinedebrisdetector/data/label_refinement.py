import os.path

import rasterio
from skimage.segmentation import random_walker
from marinedebrisdetector.visualization import fdi
from skimage.morphology import dilation, disk
from skimage.filters import threshold_otsu
from rasterio import features
import geopandas as gpd
from tqdm.contrib.concurrent import process_map
from shapely.geometry import Polygon
import numpy as np
from itertools import product

from .utils import get_window, read_tif_image, line_is_closed

buffersizes_water = [0,1,2]
rw_beta = [1,10]
object_seed_probability = [0.25, 0.5, 0.75, 0.95]

refinement_args_list = [dict(buffersize_water=w, rw_beta=beta, object_seed_probability=seed)
    for w, beta, seed in product(buffersizes_water, rw_beta, object_seed_probability)]

#print(f"arguments")
#print(refinement_args_list)

def main(root="/data/marinedebris/floatingobjects"):
    scenes_path = os.path.join(root, "scenes")
    shapefile_path = os.path.join(root, "shapefiles")
    masks_path = os.path.join(root, "masks", "refined2")

    shapefiles = [os.path.join(shapefile_path, shp) for shp in os.listdir(shapefile_path) if shp.endswith("shp")]
    regions = [os.path.basename(shp).replace(".shp","") for shp in shapefiles]

    imagetiffiles = []
    for region in regions:
        if os.path.exists(os.path.join(scenes_path, region+"_l2a.tif")):
            imagetiffiles.append(os.path.join(scenes_path, region+"_l2a.tif"))
        else:
            imagetiffiles.append(os.path.join(scenes_path, region + ".tif"))

    masktifffiles = [os.path.join(masks_path, region + ".tif") for region in regions]

    r = process_map(refine_worker, zip(shapefiles, imagetiffiles, masktifffiles), max_workers=16, total=len(shapefiles), desc="refining labels")

    #with Pool(16) as p:
    #    p.map(refine_worker, zip(shapefiles, imagetiffiles, masktifffiles))

    #for shapefile, imagetiffile, masktifffile in zip(shapefiles, imagetiffiles, masktifffiles):
    #    refine_worker((shapefile, imagetiffile, masktifffile))
    #    break

def refine_worker(args_tuple):
    shapefile, imagetiffile, masktifffile = args_tuple

    gdf = gpd.read_file(shapefile)

    # close lines to polygons
    is_closed_line = gdf.geometry.apply(line_is_closed)
    if is_closed_line.any():
        rasterize_polygons = gdf.loc[is_closed_line].geometry.apply(Polygon)
        gdf.loc[is_closed_line, "geometry"] = rasterize_polygons # replace

    with rasterio.open(imagetiffile, "r") as src:
        gdf = gdf.to_crs(src.crs)

    refine_masks_iterative(imagetiffile, masktifffile, gdf.geometry, patch_size=128)


def refine_masks(image, mask,
                 buffersize_water=3,
                 water_seed_probability=0.90,
                 object_seed_probability=0.2,
                 rw_beta=10,
                 return_all=False):
    """
    refines a coarse label mask given an image with otsu thresholding on the FDI representation of the image and
    """

    out_shape = image.shape[1:]

    # rasterize geometries to mask
    mask_lines = mask
    mask_water = dilation(mask_lines, footprint=disk(buffersize_water)) == 0

    random_seeds = np.random.rand(*out_shape) > water_seed_probability
    seeds_water = random_seeds * mask_water

    fdi_image = fdi(image * 1e-4)
    fdi_image = (fdi_image - fdi_image.min())
    fdi_image = fdi_image / fdi_image.max() * 255
    thresh = threshold_otsu(fdi_image)
    otsu_segments = fdi_image > thresh

    mask_lines = (~mask_water)

    # if there is overlap with otsu segments, refined mask lines further by otsu segments
    if (otsu_segments * mask_lines).sum() > 0:
        seeds_lines = otsu_segments * mask_lines * (np.random.rand(*out_shape) > object_seed_probability)
    else:  # otherwise ignore otsu segments
        seeds_lines = mask_lines * (np.random.rand(*out_shape) > object_seed_probability)

    if seeds_lines.sum() > 0: # if line seeds present, otherwise fall back to returning the original mask
        markers = seeds_lines * 1 + seeds_water * 2
        labels = random_walker(fdi_image, markers, beta=rw_beta, mode='bf', return_full_prob=False) == 1
    else:
        print(f"could not refine sample, returning original mask")
        labels = mask
        markers = None

    if return_all:
        return labels, otsu_segments, markers, fdi_image, mask_lines
    else:
        return labels

def refine_masks_iterative(imagetiffile, masktifffile, geometries,
                 patch_size=256):

    maskprofile = initialize_mask(imagetiffile, masktifffile)

    with rasterio.open(imagetiffile, "r") as src:
        transform = src.transform

    with rasterio.open(masktifffile, "w+", **maskprofile) as dst:
        for geometry in geometries:
            window = get_window(geometry, output_size=patch_size, transform=transform)

            image, win_transform = read_tif_image(imagetiffile, window)

            mask_rasterized = features.rasterize(geometries, all_touched=True,
                                      transform=win_transform, out_shape=(patch_size, patch_size))

            if mask_rasterized.shape[0] != patch_size or mask_rasterized.shape[1] != patch_size:
                continue # skip if at the border

            if image.shape[1] != patch_size or image.shape[2] != patch_size:
                continue # skip if at the border


            mask_refined = np.vstack([refine_masks(image, mask_rasterized, **refinement_args)[None] for refinement_args in refinement_args_list])
            mask_previous = dst.read(window=window)[1:] # take previously written refined masks (excluding first band containing the original mask)

            # combine by or-ing both together
            new_mask = mask_refined | mask_previous

            # stack unrefined mask at first position
            all_masks = np.vstack([mask_rasterized[None], new_mask])

            dst.write(all_masks.astype(np.uint8), window=window)

            dst.set_band_description(1, f"original")
            for i, desc in enumerate(refinement_args_list):
                dst.set_band_description(i+2, f"buff{desc['buffersize_water']}_beta{desc['rw_beta']}_pseed{desc['object_seed_probability']}")

        print(f"writing {masktifffile}")



def initialize_mask(imagetiffile, masktifffile):
    os.makedirs(os.path.dirname(masktifffile), exist_ok=True)

    if os.path.exists(masktifffile):
        os.remove(masktifffile)

    with rasterio.open(imagetiffile, "r") as src:
        profile = src.meta
        profile.update(
            count=len(refinement_args_list) + 1,
            dtype="uint8",
            compression="lzw"
        )

    with rasterio.open(masktifffile, "w+", **profile) as dst:
        dst.write(np.zeros((len(refinement_args_list) + 1, profile["height"], profile["width"])))

    return profile


if __name__ == '__main__':
    main()
