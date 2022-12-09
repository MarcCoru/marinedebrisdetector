from .unet import UNet
import segmentation_models_pytorch as smp

def get_model(modelname, inchannels=12, pretrained=True):

    if modelname == "unet":
        # initialize model (random weights)
        return UNet(n_channels=inchannels,
                     n_classes=1,
                     bilinear=False)
    if modelname == "unet++":
        return smp.UnetPlusPlus(in_channels=inchannels, classes=1)
    if modelname == "manet":
        return smp.MAnet(in_channels=inchannels, classes=1)
    else:
        raise NotImplementedError()

