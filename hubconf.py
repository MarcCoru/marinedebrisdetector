dependencies = ["torch", "pytorch_lightning"]

from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.checkpoints import CHECKPOINTS

def unetpp(seed):
    assert seed in [1,2,3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}"])

def unet(seed):
    assert seed in [1, 2, 3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet{seed}"])

if __name__ == '__main__':
    unetpp(1)
