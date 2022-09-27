
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import pandas as pd
import pytorch_lightning as pl
import wandb
from matplotlib import cm, colors
import numpy as np
import torch

class PlotPredictionsCallback(pl.Callback):
    def __init__(self, logger, dataset, indices):
        super().__init__()
        self.logger = logger
        self.dataset = dataset
        self.indices = indices

    def on_validation_epoch_end(self, trainer, model):
        images, masks, id  = map(np.stack, zip(*[self.dataset[i] for i in self.indices]))
        images = torch.from_numpy(images).to(model.device)
        masks = torch.from_numpy(masks).to(model.device)

        y_scores = torch.sigmoid(model(images))

        norm = colors.Normalize(vmin=0, vmax=1, clip=True)
        scmap = cm.ScalarMappable(norm=norm, cmap="viridis")
        predictions = [wandb.Image(scmap.to_rgba(i)*255) for i in y_scores.squeeze(1).detach().cpu().numpy()]

        targets = [wandb.Image(cm.viridis(i)) for i in masks.detach().cpu().numpy()]

        rgb = equalize_hist(images[:, np.array([3, 2, 1])].detach().cpu().numpy())
        rgb_images = [wandb.Image(i) for i in rgb.transpose(0,2,3,1)]

        df = pd.DataFrame([predictions, rgb_images, targets, id], index=["predictions","images", "targets","id"]).T
        self.logger.log_table(key="predictions", dataframe=df, step=trainer.global_step)

class PLPCallback(pl.Callback):
    def __init__(self, logger, dataset):
        super().__init__()
        self.logger = logger
        self.dataset = dataset

    def on_validation_epoch_end(self, trainer, model):

        images, masks, years = [], [], []
        for image, mask, year in self.dataset:
            images.append(image)
            masks.append(mask)
            years.append(year)

        images = torch.from_numpy(np.stack(images)).to(model.device) * 1e-4
        masks = torch.from_numpy(np.stack(masks)).to(model.device) > 0

        y_probs = torch.sigmoid(model(images))
        avg_probability = (y_probs.squeeze(1) * masks).mean()

        self.logger.log_metrics({"avg_plp_probability": avg_probability.detach().cpu().numpy()})

"""
from predictor import PythonPredictor
class PredictDurbanCallback(pl.Callback):
    def __init__(self, imagepath, predpath):
        super().__init__()
        self.imagepath = imagepath
        self.predpath = predpath
        self.predictor = PythonPredictor(image_size=(480,480), device="cuda")

    def on_validation_epoch_end(self, trainer, model):
        self.predictor.predict(model.model, self.imagepath, self.predpath)
        print()
"""
