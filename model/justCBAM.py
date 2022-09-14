from torch import nn
from .cbamresnet import Channel_Attention, Spatial_Attention, CBAM

class JustCBAM(nn.Module):
    '''Bottleneck modules
    '''

    def __init__(self, image_depth=12, num_classes=1, return_attention=False):
        '''Param init.
        '''
        super(JustCBAM, self).__init__()

        self.cbam = CBAM(image_depth, return_attention=return_attention, reduction_ratio=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(image_depth, num_classes)
        )


    def forward(self, x):
        x = self.cbam(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.classifier(x)
        return x
