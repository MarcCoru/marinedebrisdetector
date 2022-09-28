import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import get_model
from data import FloatingSeaObjectDataset
from data.marinedebris import MarineDebrisRegionDataset
from data.plastic_litter_project import PLPDataset
from transforms import get_transform
from sklearn.metrics import roc_curve
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix, jaccard_score
from loss import get_loss
from visualization import fdi, ndvi
from callbacks import PlotPredictionsCallback, PLPCallback
from data.s2ships import S2Ships

from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data")
    parser.add_argument('--snapshot-path', type=str, default="/tmp/snapshot")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=1e-12)
    parser.add_argument('--seed', type=int, default=1, help="random seed for train/test region split")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--augmentation-intensity', type=int, default=1, help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)")
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--add-fdi-ndvi', action="store_true")
    parser.add_argument('--cache-to-numpy', action="store_true", help="performance optimization: caches images to npz files in a npy folder within data-path.")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('--no-pretrained', action="store_true")

    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--pos-weight', type=float, default=1, help="positional weight for the floating object class, large values counteract")

    args = parser.parse_args()
    # args.image_size = (args.image_size,args.image_size)

    return args

def calculate_metrics(targets, scores, optimal_threshold):
    predictions = scores > optimal_threshold

    auroc = roc_auc_score(targets, scores)
    p, r, f, s = precision_recall_fscore_support(y_true=targets,
                                                 y_pred=predictions, zero_division=0, average="binary")
    kappa = cohen_kappa_score(targets, predictions)

    jaccard = jaccard_score(targets, predictions)

    summary = dict(
        auroc=auroc,
        precision=p,
        recall=r,
        fscore=f,
        kappa=kappa,
        jaccard=jaccard,
        threshold=optimal_threshold
    )

    return summary

class SegmentationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = get_model("unet", inchannels=12, pretrained=False)

        self.criterion = get_loss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        im, target, id = batch
        y_pred = self.model(im)
        loss = self.criterion(y_pred.squeeze(1), target)
        return loss
    def training_step_end(self, losses):
        self.log("train_loss", losses.cpu().detach().mean())

    def validation_step(self, batch, batch_idx):
        images, masks, id = batch
        logits = self.model(images)
        valid_data = images.sum(1) != 0  # all pixels > 0
        loss = self.criterion(logits.squeeze(1), target=masks, mask=valid_data)
        y_scores = torch.sigmoid(logits)
        return {"y_scores":y_scores.cpu().detach(), "y_true":masks.cpu().detach(), "loss":loss.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_scores = y_scores.reshape(-1)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        optimal_threshold = thresholds[ix]

        wandb.log({"roc_curve": wandb.plot.roc_curve(y_true, np.stack([y_scores, 1 - y_scores]).T)})

        metrics = calculate_metrics(y_true, y_scores, optimal_threshold)
        self.log("val_loss", loss.mean())
        self.log("validation", {k:torch.tensor(v) for k,v in metrics.items()})

    def test_step(self, batch, batch_idx):
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

from data import MarineDebrisRegionDataset, MarineDebrisDataset
def main(args):

    model = SegmentationModel(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    #model = model.load_from_checkpoint("/home/marc/projects/marinedetector/floatingobjects/hs5rnyqw/checkpoints/epoch=323-step=169776.ckpt")

    train_transform = get_transform("train", intensity=args.augmentation_intensity, cropsize=args.image_size)
    image_load_size = int(args.image_size * 1.2) # load images slightly larger to be cropped later to image_size

    flobs_dataset = FloatingSeaObjectDataset(args.data_path, fold="train",
                                       transform=train_transform,refine_labels=True,
                                       output_size=image_load_size, cache_to_npy=True)
    shipsdataset = S2Ships("/data/marinedebris/S2SHIPS", imagesize=image_load_size, transform=train_transform)
    train_dataset = torch.utils.data.ConcatDataset([flobs_dataset, shipsdataset])

    valid_dataset = FloatingSeaObjectDataset(args.data_path, fold="val",
                                             transform=get_transform("test", add_fdi_ndvi=args.add_fdi_ndvi, cropsize=args.image_size), refine_labels=True,
                                             output_size=args.image_size, hard_negative_mining=False,
                                             cache_to_npy=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"unet_{ts}"
    logger = WandbLogger(project="flobs-segm", name=run_name, log_model=True, save_code=True)
    #logger.watch(model)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoints", run_name),
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    plot_dataset = FloatingSeaObjectDataset(args.data_path, fold="val",
                                             transform=get_transform("test", add_fdi_ndvi=False),
                                             output_size=128, hard_negative_mining=False,
                                             cache_to_npy=True, refine_labels=True)

    plot_indices = np.random.randint(len(plot_dataset), size=6)
    plot_predictions = PlotPredictionsCallback(logger=logger, dataset=plot_dataset, indices=plot_indices)

    plp_dataset = PLPDataset(root="/data/marinedebris/PLP", year=2022, output_size=32)
    plp_callback = PLPCallback(logger, plp_dataset)
    #durban_callback = PredictDurbanCallback(imagepath="/data/marinedebris/durban/durban_20190424.tif", predpath="checkpoints/durban.tif")

    trainer = pl.Trainer(accelerator="gpu", logger=logger,devices=1,
                         callbacks=[plot_predictions,
                                    checkpointer,
                                    plp_callback],
                         fast_dev_run=False)
    trainer.fit(model, train_loader, val_loader)# , ckpt_path=f"checkpoints/{run_name}/last.ckpt")

if __name__ == '__main__':
    main(parse_args())
