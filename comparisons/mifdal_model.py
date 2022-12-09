from model.segmentation_model import SegmentationModel
import torch
from argparse import Namespace

from model.unet import UNet

def load_mifdal_model():
    unet_model = UNet(n_channels=12,
                      n_classes=1,
                      bilinear=False)

    args = Namespace(
        model = "unet",
        learning_rate = 0.001,
        weight_decay = 1e-4,
        hr_only = False
    )
    model = SegmentationModel(args)

    """unet weights from 
    https://drive.google.com/uc?export=download&id=1uZkaj7MPubCqCzSTTYS_57vbxKpgbOig"""
    state_dict = torch.load("/data/marinedebris/results/mifdal/unet-posweight1-lr001-bs160-ep50-aug1-seed0/unet-posweight1-lr001-bs160-ep50-aug1-seed0.pth.tar")[
        "model_state_dict"]
    unet_model.load_state_dict(state_dict)

    # value determined by mifdal_validation.py
    model.register_buffer("threshold", torch.tensor(0.038854))

    model.model = unet_model

    return model
