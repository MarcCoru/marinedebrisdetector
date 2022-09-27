import numpy as np
from skimage.segmentation import random_walker
from visualization import fdi
from skimage.morphology import dilation, disk
from skimage.filters import threshold_otsu

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

    seeds_lines = otsu_segments * (~mask_water) * (np.random.rand(*out_shape) > object_seed_probability)

    markers = seeds_lines * 1 + seeds_water * 2

    labels = random_walker(fdi_image, markers, beta=rw_beta, mode='bf', return_full_prob=False) == 1

    if return_all:
        return labels, otsu_segments, markers, fdi_image, mask_lines
    else:
        return labels
