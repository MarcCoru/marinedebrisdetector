import torch
import pytorch_lightning
from data.floatingobjects import FloatingSeaObjectDataset
from model.swin_transformer import SwinTransformer
from collections import OrderedDict
import gdown
import os
import wandb
from torch import nn
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        #vits8.patch_embed.proj = nn.Conv2d(13, 384, kernel_size=(8, 8), stride=(8, 8))

        #self.model = vits8


        self.model = SwinTransformer(in_chans=13)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, ids = batch
        logits = self.model(images)
        return

    def training_step_end(self, losses):
        self.log("train_loss", losses.cpu().detach().mean())

    def validation_step(self, batch, batch_idx):
        images, label, ids = batch

        logits = self.model(images)
        y_pred = logits.argmax(1)
        loss = self.criterion(logits, label)
        return {"features": logits.cpu().detach().numpy(),
                "val_loss": loss.cpu().detach().numpy(),
                "y_pred": y_pred.cpu().detach().numpy(),
                "y_true": label.cpu().detach().numpy()}

    def validation_epoch_end(self, outputs):
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_pred = np.hstack([o["y_pred"] for o in outputs])
        metrics = classification_report(y_true, y_pred,output_dict=True,labels=np.arange(10),target_names=IGBP_simplified_classes,zero_division=0)

        class_wise_metrics = pd.DataFrame([metrics[n] for n in IGBP_simplified_classes], index=IGBP_simplified_classes)
        class_wise_metrics["classname"] = class_wise_metrics.index

        cols = class_wise_metrics.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        class_wise_metrics = class_wise_metrics[cols]

        global_metrics = pd.DataFrame([metrics[n] for n in metrics.keys() if n not in IGBP_simplified_classes], index=[n for n in metrics.keys() if n not in IGBP_simplified_classes])

        self.logger.log_table(key="class_wise_metrics", dataframe=class_wise_metrics, step=self.trainer.global_step)
        self.logger.log_table(key="global_metrics", dataframe=global_metrics, step=self.trainer.global_step)

        self.log("f1-score", global_metrics["f1-score"].to_dict())
        self.log("recall", global_metrics["f1-score"].to_dict())
        self.log("precision", global_metrics["f1-score"].to_dict())
        self.log("val_loss", np.hstack([o["val_loss"] for o in outputs]).mean())

        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=y_true, preds=y_pred,
                                                           class_names=IGBP_simplified_classes)})


    def test_step(self, batch, batch_idx):
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr = 0.001, weight_decay=0.4)


def main():
    #ds = FloatingSeaObjectDataset(root="/ssd/floatingObjects/data")



    net = LitModel()
    trainer = Trainer()
    trainer.fit(net)
    print()
    pass

if __name__ == '__main__':
    main()
