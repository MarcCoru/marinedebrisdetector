dependencies = ["torch", "pytorch_lightning", "segmentation_models_pytorch"]

from marinedebrisdetector import SegmentationModel
from marinedebrisdetector import CHECKPOINTS

def unetpp(seed=1):
    assert seed in [1,2,3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}"], trust_repo=True)

def unetppnoref(seed=1):
    assert seed in [1,2,3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}_no_label_refinement"], trust_repo=True)

def unet(seed=1):
    assert seed in [1, 2, 3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet{seed}"], trust_repo=True)

if __name__ == '__main__':
    unetpp(1)
