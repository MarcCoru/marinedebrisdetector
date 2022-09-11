from data import MarineDebrisRegionDataset, MarineDebrisDataset
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import numpy as np
import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.callbacks import Callback
from transforms import augment
from pytorch_lightning.loggers import WandbLogger

from model import get_model
class ResNetClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        if False:
            self.model = models.swin_t(weights=models.Swin_T_Weights(models.Swin_T_Weights.DEFAULT))

            w = self.model.features[0][0].weight
            w = torch.nn.Parameter(torch.repeat_interleave(torch.tensor(w), 4, 1))
            b = self.model.features[0][0].bias

            newconv = nn.Conv2d(12, 96, kernel_size=(4, 4), stride=(4, 4))
            newconv.weight = w
            newconv.bias = b

            self.model.features[0][0] = newconv
            self.model.head = nn.Linear(in_features=768, out_features=1, bias=True)
        else:
            self.model = models.resnet18()
            self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512,1)

        """
        self.model = nn.Sequential(get_model("unet"),
                                   nn.AdaptiveAvgPool2d(output_size=(1, 1))
                                   )
        """

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        return loss

    def training_step_end(self, losses):
        self.log("train_loss", losses.cpu().detach().mean())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        y_scores = torch.sigmoid(y_hat)
        return {"y_scores":y_scores.cpu().detach(), "y_true":y.cpu().detach(), "loss":loss.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_scores = y_scores.reshape(-1)
        y_pred = y_scores > 0.5

        print()
        self.log("val_loss", loss.mean())
        self.log("val_accuracy", (y_true == y_pred).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-8)

def main():

    ds = MarineDebrisDataset(root="/ssd/marinedebris/marinedebris_refined", fold="train", shuffle=True,
                             imagesize=640, transform=augment)

    val_ds = MarineDebrisDataset(root="/ssd/marinedebris/marinedebris_refined", fold="val", shuffle=True,
                             imagesize=640, transform=augment)

    checkpoint_callback = ModelCheckpoint(dirpath='/tmp/checkpoints', save_last=True)


    model = ResNetClassifier()
    #model = ResNetClassifier.load_from_checkpoint("/tmp/checkpoints/last-v2.ckpt")
    train_loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=16, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=16, drop_last=True)

    wandb_logger = WandbLogger(project="marinedebris", log_model=True)
    wandb_logger.watch(model)

    trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", callbacks=[checkpoint_callback], logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__ == '__main__':
    main()
