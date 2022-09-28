import numpy as np
import torch
from torch.optim import Adam
from model import get_model
import wandb
import pytorch_lightning as pl
from metrics import get_loss, calculate_metrics
from sklearn.metrics import precision_recall_curve

class SegmentationModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = get_model(model, inchannels=12, pretrained=False)

        self.criterion = get_loss()

        # store a threshold paramter that will be updated based on the validation set
        self.register_buffer("threshold", torch.tensor(0.5))

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def predict(self, x, return_probs=False):
        probs = torch.sigmoid(self.model(x))
        if return_probs:
            return (probs > self.threshold).long(), probs
        else:
            return (probs > self.threshold).long()

    def training_step(self, batch, batch_idx):
        im, target, id = batch
        y_pred = self.model(im)
        loss = self.criterion(y_pred.squeeze(1), target)
        return loss

    def training_step_end(self, losses):
        self.log("train_loss", losses.cpu().detach().mean())

    def common_step(self, batch, batch_idx):
        images, masks, id = batch
        logits = self.model(images)
        N, _, H, W = logits.shape
        h, w = H//2, W // 2
        logits = logits.squeeze(1)[:, h, w] # keep only center
        loss = self.criterion(logits, target=masks.float())
        y_scores = torch.sigmoid(logits)
        return {"y_scores":y_scores.cpu().detach(), "y_true":masks.cpu().detach(), "loss":loss.cpu().numpy()}

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_scores = y_scores.reshape(-1)

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ix = np.abs(precision - recall).argmin()
        optimal_threshold = thresholds[ix]
        self.threshold = torch.tensor(optimal_threshold)

        wandb.log({"roc_curve": wandb.plot.roc_curve(y_true, np.stack([1-y_scores, y_scores]).T,
                                                     labels=[0,1], classes_to_plot=[1])})

        wandb.log({"pr": wandb.plot.pr_curve(y_true, np.stack([1-y_scores, y_scores]).T,
                                             labels=[0, 1], classes_to_plot=[1])})

        metrics = calculate_metrics(y_true, y_scores, optimal_threshold)
        self.log("val_loss", loss.mean())
        self.log("validation", {k:torch.tensor(v) for k,v in metrics.items()})

    def test_epoch_end(self, outputs) -> None:
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])

        metrics = calculate_metrics(y_true, y_scores, self.threshold)
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
