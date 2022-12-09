from skimage.exposure import equalize_hist
import numpy as np

bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
def ndvi(scene):
    NIR = scene[bands.index("B8")]
    RED = scene[bands.index("B4")]
    return (NIR - RED) / (NIR + RED + 1e-12)

def fdi(scene):
    # tbd

    NIR = scene[bands.index("B8")]
    RED2 = scene[bands.index("B6")]
#    RED2 = cv2.resize(RED2, NIR.shape)

    SWIR1 = scene[bands.index("B11")]
    #SWIR1 = cv2.resize(SWIR1, NIR.shape)

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    return NIR - NIR_prime

def rgb(scene):
    return equalize_hist(scene[np.array([3,2,1])])
