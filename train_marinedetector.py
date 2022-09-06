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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCamCallback(Callback):
    def __init__(self):
        super().__init__()
        self.batch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.batch = batch

    def on_train_epoch_end(self, trainer, pl_module):
        images, labels = self.batch

        model = pl_module
        target_layers = [model.model.layer4[-1]]

        act = ActivationsAndGradients(model, [model.model.layer4[-1]], reshape_transform=lambda x: x)
        logits = act(images)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward()

        alpha = act.gradients[0].mean(-1).mean(-1)

        A = act.activations[0].permute(0,2,3,1)

        gradcam = (alpha[0] * A[0]).sum(-1)

        input_tensor =  images
        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        targets = [ClassifierOutputTarget(1)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        print()

from model import get_model
class ResNetClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(get_model("unet"),
                                   nn.AdaptiveAvgPool2d(output_size=(1, 1))
                                   )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    ds = MarineDebrisDataset(root="/ssd/marinedebris/marinedebris_refined", fold="train", shuffle=True, imagesize=640)

    checkpoint_callback = ModelCheckpoint(dirpath='/tmp/checkpoints', save_last=True)

    trainer = pl.Trainer(max_epochs=100, accelerator="gpu", callbacks=[checkpoint_callback])
    model = ResNetClassifier()
    #model = ResNetClassifier.load_from_checkpoint("/tmp/checkpoints/last.ckpt")
    train_loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=16, drop_last=True)

    trainer.fit(model, train_dataloaders=train_loader)



if __name__ == '__main__':
    main()
