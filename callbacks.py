
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import pandas as pd
import pytorch_lightning as pl
import wandb
from matplotlib import cm
import numpy as np
import torch

class PlotPredictionsCallback_old(pl.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.last_outputs = outputs
        self.last_batch = batch

    def on_validation_epoch_end(self, trainer, pl_module):
        images, masks, id = self.last_batch
        y_scores = self.last_outputs["y_scores"]

        predictions = [wandb.Image(cm.viridis(i)*256) for i in y_scores.squeeze(1).detach().cpu().numpy()]

        rgb = equalize_hist(images[:, np.array([3, 2, 1])].detach().cpu().numpy())
        rgb_images = [wandb.Image(i) for i in rgb.transpose(0,2,3,1)]

        df = pd.DataFrame([predictions, rgb_images], index=["predictions","images"]).T
        self.logger.log_table(key="predictions", dataframe=df, step=trainer.global_step)

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

        predictions = [wandb.Image(cm.viridis(i)*256) for i in y_scores.squeeze(1).detach().cpu().numpy()]

        targets = [wandb.Image(cm.viridis(i)) for i in masks.detach().cpu().numpy()]

        rgb = equalize_hist(images[:, np.array([3, 2, 1])].detach().cpu().numpy())
        rgb_images = [wandb.Image(i) for i in rgb.transpose(0,2,3,1)]

        df = pd.DataFrame([predictions, rgb_images, targets, id], index=["predictions","images", "targets","id"]).T
        self.logger.log_table(key="predictions", dataframe=df, step=trainer.global_step)