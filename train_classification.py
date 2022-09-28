from data import MarineDebrisDataset
import numpy as np
import torch
from datetime import datetime

import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint

from transforms import get_train_transform
from pytorch_lightning.loggers import WandbLogger

from torch import nn

class Classifier(pl.LightningModule):
    def __init__(self, model="torchvit"):
        super().__init__()

        """
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
        elif False:
            self.model = ResNet50(use_cbam=True, image_depth=12, num_classes=1)
        elif False:
            self.model = TinyCBAM(image_depth=12, num_classes=1, return_attention=return_attention)
        elif False:
            self.model = SwinTransformer(
                patch_size=[1, 1],
                embed_dim=96,
                depths=[2],
                num_heads=[3],
                window_size=[7, 7],
                stochastic_depth_prob=0.2)
            self.model.features[0][0] = nn.Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
            self.model.head = nn.Linear(in_features=96, out_features=1, bias=True)
            print()


        elif False:
            self.model = JustCBAM(image_depth=12, num_classes=1)
            """
        if model == "torchvit":
            from torchvision.models import VisionTransformer
            self.model = VisionTransformer(
                    image_size=32,
                    patch_size=1,
                    num_layers=1,
                    num_heads=4,
                    hidden_dim=64,
                    mlp_dim=64)
            self.model.conv_proj = nn.Conv2d(12, 64, kernel_size=(1, 1), stride=(1, 1))
            self.model.heads.head = nn.Linear(in_features=64, out_features=1, bias=True)
        elif model == "lrpvit":
            from model.classification.explLRP import VisionTransformer
            self.model = VisionTransformer(
                in_chans=12,
                img_size=32,
                patch_size=1,
                depth=1,
                num_heads=4,
                embed_dim=64,
                num_classes=1,
                mlp_ratio=1)

            print()
        elif model == "resnet18":
            self.model = models.resnet18()
            self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512,1)
        else:
            return NotImplementedError()

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
        return torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-8)
        #return torch.optim.Adam(
        #    [
        #        {"params":list({k:v for k,v in self.model.named_parameters() if "head" not in k}.values())},
        #        {"params":self.model.head.parameters(), "weight_decay":1e-8} # lower weight decay for head
        #    ], lr=1e-4, weight_decay=1e-8)


def main():

    imagesize = 64
    crop_size = 32
    workers = 16

    model = "lrpvit"

    ds = MarineDebrisDataset(root="/data/marinedebris/marinedebris_refined", fold="train", shuffle=True,
                             imagesize=imagesize * 10, transform=get_train_transform(crop_size=crop_size))

    val_ds = MarineDebrisDataset(root="/data/marinedebris/marinedebris_refined", fold="val", shuffle=True,
                             imagesize=crop_size * 10, transform=None)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"{model}_{ts}"
    logger = WandbLogger(project="marinedebris", name=run_name, log_model=True, save_code=True)

    checkpointer = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename="{epoch}-{val_accuracy:.2f}",
        monitor="val_accuracy",
        mode="max",
        save_last=True,
    )

    model = Classifier(model=model)
    #model.load_state_dict(torch.load("checkpoints/SwinT_2022-09-14_22:14:42/epoch=292-val_accuracy=0.89.ckpt")["state_dict"])

    train_loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=workers, drop_last=True)


    #checkpoint = "checkpoints/Vit_2022-09-15_21:18:37/epoch=591-val_accuracy=0.95.ckpt"
    checkpoint = None
    trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", callbacks=[checkpointer],
                         logger=logger, fast_dev_run=False, resume_from_checkpoint=checkpoint)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__ == '__main__':
    main()
