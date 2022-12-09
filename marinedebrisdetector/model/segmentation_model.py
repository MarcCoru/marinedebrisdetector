import numpy as np
import torch
from torch.optim import Adam
import wandb
import pytorch_lightning as pl
from marinedebrisdetector.model import get_model
from marinedebrisdetector.metrics import get_loss, calculate_metrics
from sklearn.metrics import precision_recall_curve

class SegmentationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        model = args.model
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.hr_only = args.hr_only # keep only HR bands R-G-B-NIR
        if self.hr_only:
            self.inchannels = 4
        else:
            self.inchannels = 12

        self.model = get_model(model, inchannels=self.inchannels, pretrained=False)

        self.criterion = get_loss()

        # store a threshold paramter that will be updated based on the validation set
        self.register_buffer("threshold", torch.tensor(0.5))

        self.save_hyperparameters()

    def forward(self, x):

        if x.shape[1] > self.inchannels:
            x = x[:, np.array([1, 2, 3, 7])]

        return self.model(x)

    def predict(self, x, return_probs=False):
        probs = torch.sigmoid(self(x))
        if return_probs:
            return (probs > self.threshold).long(), probs
        else:
            return (probs > self.threshold).long()

    def training_step(self, batch, batch_idx):
        im, target, id = batch
        y_pred = self(im)
        loss = self.criterion(y_pred.squeeze(1), target)
        return loss

    def training_step_end(self, losses):
        self.log("train_loss", losses.cpu().detach().mean())

    def common_step(self, batch, batch_idx):
        images, masks, id = batch
        logits = self(images)
        N, _, H, W = logits.shape
        h, w = H//2, W // 2
        logits = logits.squeeze(1)[:, h, w] # keep only center
        loss = self.criterion(logits, target=masks.float())
        y_scores = torch.sigmoid(logits)
        return {"y_scores":y_scores.cpu().detach(), "y_true":masks.cpu().detach(), "loss":loss.cpu().numpy(), "id":id}

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])
        ids = np.hstack([o["id"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_scores = y_scores.reshape(-1)

        #is_marida = np.array(["marida" in i for i in ids])
        #y_true_marida = y_true[is_marida]
        #y_scores_marida = y_scores[is_marida]

        #y_true = y_true[~is_marida]
        #y_scores = y_scores[~is_marida]

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ix = np.abs(precision - recall).argmin()
        optimal_threshold = thresholds[ix]
        self.threshold = torch.tensor(optimal_threshold)

        wandb.log({"roc_curve": wandb.plot.roc_curve(y_true, np.stack([1-y_scores, y_scores]).T,
                                                     labels=[0,1], classes_to_plot=[1])})

        wandb.log({"pr": wandb.plot.pr_curve(y_true, np.stack([1-y_scores, y_scores]).T,
                                             labels=[0, 1], classes_to_plot=[1])})

        #if len(y_scores_marida) > 100: # only if some samples available to calculate metrics
        #    metrics_marida = calculate_metrics(y_true_marida, y_scores_marida, optimal_threshold)
        #    self.log("validation-marida", {k: torch.tensor(v) for k, v in metrics_marida.items()})

        metrics = calculate_metrics(y_true, y_scores, optimal_threshold)
        self.log("validation", {k: torch.tensor(v) for k, v in metrics.items()})
        self.log("auroc", metrics["auroc"])

        self.log("val_loss", loss.mean())


    def test_epoch_end(self, outputs) -> None:
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])
        ids = np.hstack([o["id"] for o in outputs])
        regions = [id.split("-")[0] for id in ids]

        y_true = torch.from_numpy(y_true)
        y_scores = torch.from_numpy(y_scores)

        metrics = calculate_metrics(y_true, y_scores, self.threshold.cpu())
        metrics["loss"] = loss.mean()

        metrics = {"test_"+k:v for k,v in metrics.items()}
        self.log_dict(metrics)

        for r in np.unique(regions):
            mask = np.array([r_ == r for r_ in regions])
            metrics = calculate_metrics(y_true[mask], y_scores[mask], self.threshold.cpu())
            metrics = {f"test_{r}_" + k: v for k, v in metrics.items()}
            self.log_dict(metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
