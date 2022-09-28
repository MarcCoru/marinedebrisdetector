
print()
from torchvision.models import swin_t
from torchvision.models.swin_transformer import SwinTransformer
from torch import nn

model = SwinTransformer(
        patch_size=[1, 1],
        embed_dim=96,
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        window_size=[7,7],
        stochastic_depth_prob=0.2)
model = model.features[0][0] = nn.Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))

print()
