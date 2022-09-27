import torch
import rasterio
from rasterio.windows import Window
import numpy as np
from itertools import product
import os
from tqdm import tqdm
import argparse
from transforms import get_transform
from scipy.ndimage.filters import gaussian_filter
from data import L2ABANDS, L1CBANDS

class PythonPredictor():

    #def __init__(self, modelname, modelpath, image_size=(480,480), device="cpu", offset=64, use_test_aug=2, add_fdi_ndvi=False):
    def __init__(self, image_size=(480,480), device="cpu", offset=64, use_test_aug=2, add_fdi_ndvi=False):
        self.image_size = image_size
        self.device = device
        self.offset = offset # remove border effects from the CNN
        self.use_test_aug = use_test_aug

        #self.model = UNet(n_channels=12, n_classes=1, bilinear=False)
        self.transform = get_transform("test", add_fdi_ndvi=add_fdi_ndvi, cropsize=image_size[0])

    def predict(self, model, path, predpath):
        with rasterio.open(path, "r") as src:
            meta = src.meta
        self.model = model.to(self.device)
        self.model.eval()

        # for prediction
        predimage = predpath
        meta["count"] = 1
        meta["dtype"] = "uint8" # storing as uint8 saves a lot of storage space

        #Window(col_off, row_off, width, height)
        H, W = self.image_size

        rows = np.arange(0, meta["height"], H)
        cols = np.arange(0, meta["width"], W)

        image_window = Window(0, 0, meta["width"], meta["height"])

        with rasterio.open(predimage, "w+", **meta) as dst:

            for r, c in tqdm(product(rows, cols), total=len(rows) * len(cols), leave=False):

                window = image_window.intersection(
                    Window(c-self.offset, r-self.offset, W+self.offset, H+self.offset))

                with rasterio.open(path) as src:
                    image = src.read(window=window)

                # if L1C image (13 bands). read only the 12 bands compatible with L2A data
                if (image.shape[0] == 13):
                    image = image[[L1CBANDS.index(b) for b in L2ABANDS]]

                # to torch + normalize
                image = torch.from_numpy(image.astype(np.float32))
                image = image.to(self.device) * 1e-4

                # predict
                with torch.no_grad():
                    x = image.unsqueeze(0)
                    #import pdb; pdb.set_trace()
                    y_logits = torch.sigmoid(self.model(x).squeeze(0))
                    if self.use_test_aug > 0:
                        y_logits += torch.sigmoid(torch.fliplr(self.model(torch.fliplr(x)))).squeeze(0) # fliplr)
                        y_logits += torch.sigmoid(torch.flipud(self.model(torch.flipud(x)))).squeeze(0) # flipud
                        if self.use_test_aug > 1:
                            for rot in [1, 2, 3]: # 90, 180, 270 degrees
                                y_logits += torch.sigmoid(torch.rot90(self.model(torch.rot90(x, rot, [2, 3])),-rot,[2,3]).squeeze(0))
                            y_logits /= 6
                        else:
                            y_logits /= 3

                    y_score = y_logits.cpu().detach().numpy()[0]
                    #y_score = y_score[:,self.offset:-self.offset, self.offset:-self.offset]

                data = dst.read(window=window)[0] / 255
                overlap = data > 0

                if overlap.any():
                    # smooth transition in overlapping regions
                    dx, dy = np.gradient(overlap.astype(float)) # get border
                    g = np.abs(dx) + np.abs(dy)
                    transition = gaussian_filter(g, sigma=self.offset / 2)
                    transition /= transition.max()
                    transition[~overlap] = 1.# normalize to 1

                    y_score = transition * y_score + (1-transition) * data

                # write
                writedata = (np.expand_dims(y_score, 0).astype(np.float32) * 255).astype(np.uint8)
                dst.write(writedata, window=window)

from train_segmentation import SegmentationModel

def main(args):

    device = "cuda"
    predictor = PythonPredictor(image_size=(480,480), device=device, add_fdi_ndvi=False)
    model = SegmentationModel()
    model.load_state_dict(torch.load("checkpoints/unet/last.ckpt", map_location=device)["state_dict"])
    #predictor = PythonPredictor(args.model, args.snapshot_path, args.image_size, device="cuda", add_fdi_ndvi=args.add_fdi_ndvi)

    predictor.predict(model.model, args.image_path, args.prediction_path)
    print(f"read image file from {os.path.abspath(args.image_path)}")
    print(f"wrote prediction file to {os.path.abspath(args.prediction_path)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default="/data/marinedebris/durban/durban_20190424.tif")
    parser.add_argument('--prediction-path', type=str, default="checkpoints/unet/durban.tif")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--snapshot-path', type=str)
    parser.add_argument('--add-fdi-ndvi', action="store_true")

    args = parser.parse_args()
    args.image_size = (args.image_size,args.image_size)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
