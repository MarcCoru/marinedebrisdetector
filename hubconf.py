dependencies = ["torch", "pytorch_lightning", "segmentation_models_pytorch"]

from marinedebrisdetector.model.segmentation_model import SegmentationModel as _SegmentationModel
from marinedebrisdetector.checkpoints import CHECKPOINTS

def unetpp(seed=1):
    assert seed in [1,2,3]
    return _SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}"], trust_repo=True)

def unetppnoref(seed=1):
    assert seed in [1,2,3]
    return _SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}_no_label_refinement"], trust_repo=True)

def unet(seed=1):
    assert seed in [1, 2, 3]
    return _SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet{seed}"], trust_repo=True)

if __name__ == '__main__':
    unetpp(1)
