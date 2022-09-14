from torch import nn
from .cbamresnet import Channel_Attention, Spatial_Attention, CBAM

class TinyCBAM(nn.Module):
    '''Bottleneck modules
    '''

    def __init__(self, image_depth=12, num_classes=1, return_attention=False):
        '''Param init.
        '''
        super(TinyCBAM, self).__init__()

        self.in_channels = in_channels = 64
        self.out_channels = out_channels = 128

        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=3, stride=1, in_channels=image_depth, out_channels=self.in_channels, padding=3, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True))

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=True, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels, return_attention=return_attention)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        )


    def forward(self, x):
        x = self.conv_block1(x)
        x = self.convs(x)
        x = self.cbam(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.classifier(x)
        return x
