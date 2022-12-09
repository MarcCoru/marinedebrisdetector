import numpy as np
from marinedebrisdetector.data import L2ABANDS as bands

"""
def center_crop(image,mask):
    size_crop = 56
    R = transforms.CenterCrop(size_crop)
    image = R(image)
    mask = R(mask)
    return image, mask
"""

def get_transform(mode, intensity=0, add_fdi_ndvi=False, cropsize = 224, hr_only=False):
    assert mode in ["train", "test"]
    if mode in ["train"]:
        def train_transform(image, mask=None):

            if add_fdi_ndvi:
                fdi = np.expand_dims(calculate_fdi(image),0)
                ndvi = np.expand_dims(calculate_ndvi(image),0)
                image = np.vstack([image,ndvi,fdi])

            image *= 1e-4

            # return image, mask
            data_augmentation = get_data_augmentation(intensity=intensity, cropsize=cropsize)
            return data_augmentation(image, mask)
        return train_transform
    else:
        def test_transform(image, mask=None):
            if add_fdi_ndvi:
                fdi = np.expand_dims(calculate_fdi(image),0)
                ndvi = np.expand_dims(calculate_ndvi(image),0)
                image = np.vstack([image,ndvi,fdi])

            image *= 1e-4

            image, mask = center_crop(image, mask, cropsize)

            image = torch.Tensor(image)
            if mask is not None:
                mask = torch.Tensor(mask)
                return image, mask
            else:
                return image
        return test_transform


def calculate_fdi(scene):
    # scene values [0,1e4]

    NIR = scene[bands.index("B8")] * 1e-4
    RED2 = scene[bands.index("B6")] * 1e-4
#    RED2 = cv2.resize(RED2, NIR.shape)

    SWIR1 = scene[bands.index("B11")] * 1e-4
    #SWIR1 = cv2.resize(SWIR1, NIR.shape)

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    return NIR - NIR_prime

def calculate_ndvi(scene):
    NIR = scene[bands.index("B8")].astype(np.float)
    RED = scene[bands.index("B4")].astype(np.float)
    return (NIR - RED) / (NIR + RED + 1e-12)

def get_data_augmentation(intensity, cropsize):
    """
    do data augmentation:
    model
    """
    def data_augmentation(image, mask=None):
        image = torch.Tensor(image)
        if mask is not None:
            mask = torch.Tensor(mask).unsqueeze(0)

        if random.random() < 0.5:
            # flip left right
            image = torch.fliplr(image)
            if mask is not None:
                mask = torch.fliplr(mask)

        rot = np.random.choice([0,1,2,3])
        image = torch.rot90(image, rot, [1, 2])
        if mask is not None:
            mask = torch.rot90(mask, rot, [1, 2])

        if random.random() < 0.5:
            # flip up-down
            image = torch.flipud(image)
            if mask is not None:
                mask = torch.flipud(mask)

        if intensity >= 1:

            # a slight rescaling
            scale_factor = np.random.normal(1, 1e-1)
            min_scale_factor = (cropsize + 5) / image.shape[1] # clamp scale factor so that random crop to certain cropsize is still possible
            scale_factor = np.max([scale_factor, min_scale_factor])

            image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode="bilinear", align_corners=True,recompute_scale_factor=True).squeeze(0)
            if mask is not None:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=scale_factor, mode="bilinear", align_corners=True,recompute_scale_factor=True).squeeze(0)

            image, mask = random_crop(image, mask, cropsize=cropsize)

            std_noise = 1 * image.std()
            if random.random() < 0.5:
                # add noise per pixel and per channel
                pixel_noise = torch.rand(image.shape[1], image.shape[2])
                pixel_noise = torch.repeat_interleave(pixel_noise.unsqueeze(0), image.size(0), dim=0)
                image = image + pixel_noise*std_noise

            if random.random() < 0.5:
                channel_noise = torch.rand(image.shape[0]).unsqueeze(1).unsqueeze(2)
                channel_noise = torch.repeat_interleave(torch.repeat_interleave(channel_noise, image.shape[1], 1),
                                                        image.shape[2], 2)
                image = image + channel_noise*std_noise

            if random.random() < 0.5:
                # add noise
                noise = torch.rand(image.shape[0], image.shape[1], image.shape[2]) * std_noise
                image = image + noise

        if intensity >= 2:
            # channel shuffle
            if random.random() < 0.5:
                idxs = np.arange(image.shape[0])
                np.random.shuffle(idxs) # random band indixes
                image = image[idxs]

        if mask is not None:
            mask = mask.squeeze(0)
            return image, mask
        else:
            return image
    return data_augmentation

def random_crop(image, mask=None, cropsize=64):
    C, W, H = image.shape
    w, h = cropsize, cropsize

    # distance from image border
    dh, dw = h // 2, w // 2

    # sample some point inside the valid square
    x = np.random.randint(dw, W - dw)
    y = np.random.randint(dh, H - dh)

    # crop image
    image = image[:, x - dw:x + dw, y - dh:y + dh]
    if mask is not None:
        mask = mask[:, x - dw:x + dw, y - dh:y + dh]

    return image, mask


def center_crop(image, mask=None, size=64):
    D, H, W = image.shape

    cx = W // 2
    cy = H // 2

    image = image[:, cx-size//2:cx+size//2, cy-size//2:cy+size//2]
    if mask is not None:
        mask = mask[cx-size//2:cx+size//2, cy-size//2:cy+size//2]
    return image, mask

import torch
from torch import nn
import torchvision.transforms as T
import random

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class RandomRot90(nn.Module):
    def __init__(self, dims=[2, 3]):
        super().__init__()
        self.dims = dims

    def forward(self,x):
        rot = torch.randint(high=4,size=(1,))
        return torch.rot90(x, int(rot), self.dims)

class PixelNoise(nn.Module):
    """
    for each pixel, same across all bands
    """
    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x):
        N, C, H, W = x.shape
        noise_level = x.std() * self.std_noise
        pixel_noise = torch.rand(H, W, device=x.device)
        return x + pixel_noise.view(1,1,H,W) * noise_level

class ChannelNoise(nn.Module):
    """
    for each channel
    """
    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x):
        N, C, H, W = x.shape
        noise_level = x.std() * self.std_noise

        channel_noise = torch.rand(C, device=x.device)
        return x + channel_noise.view(1,-1,1,1).to(x.device) * noise_level

class Noise(nn.Module):
    """
    for each channel
    """
    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x):
        noise_level = x.std() * self.std_noise
        noise = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
        return x + noise * noise_level


import torchvision.transforms as T

def get_train_transform(crop_size=64):
    return torch.nn.Sequential(
            #RandomApply(
            #    T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            #    p = 0.3
            #),
            #T.RandomGrayscale(p=0.2),
            RandomRot90(),
            #T.RandomRotation(90),
            T.RandomResizedCrop(crop_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.2),
            RandomApply(PixelNoise(std_noise=0.25),p = 0.2),
            RandomApply(ChannelNoise(std_noise=0.25),p = 0.2),
            RandomApply(Noise(std_noise=0.25),p = 0.2),
        )
