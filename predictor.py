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

class ScenePredictor():

    #def __init__(self, modelname, modelpath, image_size=(480,480), device="cpu", offset=64, use_test_aug=2, add_fdi_ndvi=False):
    def __init__(self, image_size=(480,480), device="cpu", offset=64,
                 use_test_aug=2, add_fdi_ndvi=False, activation="sigmoid"):
        self.image_size = image_size
        self.activation = activation
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
                H, W = self.image_size

                window = image_window.intersection(
                    Window(c-self.offset, r-self.offset, W+self.offset, H+self.offset))

                with rasterio.open(path) as src:
                    image = src.read(window=window)

                # if L1C image (13 bands). read only the 12 bands compatible with L2A data
                if (image.shape[0] == 13):
                    image = image[[L1CBANDS.index(b) for b in L2ABANDS]]

                # pad with zeros
                H, W = self.image_size
                H, W = H + self.offset*2, W + self.offset*2

                bands, h, w = image.shape
                dh = (H - h) / 2
                dw = (W - w) / 2
                image = np.pad(image, [(0, 0), (int(np.ceil(dh)), int(np.floor(dh))),
                                       (int(np.ceil(dw)), int(np.floor(dw)))])

                # to torch + normalize
                image = torch.from_numpy(image.astype(np.float32))
                image = image.to(self.device) * 1e-4

                # predict
                with torch.no_grad():
                    x = image.unsqueeze(0)

                    y_logits = self.model(x).squeeze(0)

                    if self.activation == "sigmoid":
                        y_logits = torch.sigmoid(y_logits)

                    y_score = y_logits.cpu().detach().numpy()[0]
                    #y_score = y_score[:,self.offset:-self.offset, self.offset:-self.offset]

                # unpad
                y_score=y_score[int(np.ceil(dh)):y_score.shape[0]-int(np.floor(dh)), int(np.ceil(dw)):y_score.shape[1]-int(np.floor(dw))]
                assert y_score.shape[0] == window.height, "unpadding size mismatch"
                assert y_score.shape[1] == window.width, "unpadding size mismatch"

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
    model = SegmentationModel.load_from_checkpoint(args.checkpoint_path)
    #model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)["state_dict"])
    #predictor = PythonPredictor(args.model, args.snapshot_path, args.image_size, device="cuda", add_fdi_ndvi=args.add_fdi_ndvi)

    predictor.predict(model.model, args.image_path, args.prediction_path)
    print(f"read image file from {os.path.abspath(args.image_path)}")
    print(f"wrote prediction file to {os.path.abspath(args.prediction_path)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default="/data/marinedebris/durban/durban_20190424.tif")
    parser.add_argument('--prediction-path', type=str, default="checkpoints/unet/durban.tif")
    parser.add_argument('--checkpoint-path', type=str, default="checkpoints/flobs-segm-weight-decay/wde-12/epoch=54-val_loss=0.52.ckpt")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--add-fdi-ndvi', action="store_true")

    args = parser.parse_args()
    args.image_size = (args.image_size,args.image_size)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
