from skimage.exposure import equalize_hist
import pandas as pd
import pytorch_lightning as pl
import wandb
from matplotlib import cm, colors
import numpy as np
import torch
from visualization import rgb, fdi, ndvi

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

class RefinedRegionsQualitativeCallback(pl.Callback):
    def __init__(self, logger, dataset):
        super().__init__()
        self.logger = logger
        self.dataset = dataset

    def on_validation_epoch_end(self, trainer, model):
        stats = []
        for idx, (x, y, id) in enumerate(self.dataset):
            stat = dict(id=id)

            image = torch.from_numpy(x).unsqueeze(0).to(model.device).float()
            y_prob = torch.sigmoid(model(image).squeeze())
            pred = y_prob > model.threshold

            norm = colors.Normalize(vmin=-0.1, vmax=0.1, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="magma")
            stat["fdi"] = wandb.Image(scmap.to_rgba(fdi(image.squeeze(0)).cpu()))

            norm = colors.Normalize(vmin=-0.5, vmax=0.5, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="viridis")
            stat["ndvi"] = wandb.Image(scmap.to_rgba(ndvi(image.squeeze(0)).cpu()))

            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="viridis")
            stat["prob"] = wandb.Image(scmap.to_rgba(y_prob.cpu()))

            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="tab10")
            stat["annot"] = wandb.Image(scmap.to_rgba(pred.cpu()))

            stat["rgb"] = wandb.Image(rgb(image.squeeze(0).cpu().numpy()).transpose(1, 2, 0))

            stats.append(stat)

        df = pd.DataFrame(stats)

        self.logger.log_table(key="qualitative", dataframe=df, step=trainer.global_step)

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

        images = torch.from_numpy(np.stack(images)).to(model.device)
        masks = torch.from_numpy(np.stack(masks)).to(model.device)

        y_probs = torch.sigmoid(model(images)).squeeze(1)

        pred = y_probs > model.threshold
        msk = (masks > 0)

        recall= msk[pred].float().mean().cpu().detach().numpy()
        precision = pred[msk].float().mean().cpu().detach().numpy()
        fscore = 2 * (precision*recall) / (precision+recall+1e-12)
        self.log(f"PLP{self.dataset.year}", dict(
            recall=float(recall),
            precision=float(precision),
            fscore=float(fscore)
        ))

        stats = []
        for image, mask, y_prob, year in zip(images, masks, y_probs, years):
            stat = dict(year=year)

            for cl in np.unique(mask.cpu()):
                m = mask == cl
                avg_prob_c = y_prob[m].mean()
                stat[f"cl_{int(cl)}"] = float(avg_prob_c)

            norm = colors.Normalize(vmin=-0.1, vmax=0.1, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="magma")
            stat["fdi"] = wandb.Image(scmap.to_rgba(fdi(image).cpu()))

            norm = colors.Normalize(vmin=-0.5, vmax=0.5, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="viridis")
            stat["ndvi"] = wandb.Image(scmap.to_rgba(ndvi(image).cpu()))

            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="viridis")
            stat["prob"] = wandb.Image(scmap.to_rgba(y_prob.cpu()))

            maxcl = torch.unique(mask).max().cpu().numpy()
            norm = colors.Normalize(vmin=0, vmax=maxcl, clip=True)
            scmap = cm.ScalarMappable(norm=norm, cmap="tab10")
            stat["class"] = wandb.Image(scmap.to_rgba(mask.cpu()))

            stat["rgb"] = wandb.Image(rgb(image.cpu().numpy()).transpose(1, 2, 0))

            stats.append(stat)

        df = pd.DataFrame(stats)

        self.logger.log_table(key=f"PLP{self.dataset.year}", dataframe=df, step=trainer.global_step)
