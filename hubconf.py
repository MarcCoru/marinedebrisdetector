dependencies = ["torch", "pytorch_lightning", "segmentation_models_pytorch"]

from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.checkpoints import CHECKPOINTS

def unetpp(seed):
    assert seed in [1,2,3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}"], trust_repo=True)

def unetpp_no_label_refinement(seed):
    assert seed in [1,2,3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}_no_label_refinement"], trust_repo=True)

def unet(seed):
    assert seed in [1, 2, 3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet{seed}"], trust_repo=True)

if __name__ == '__main__':
    unetpp(1)
