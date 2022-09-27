print()
from data import MarineDebrisRegionDataset, MarineDebrisDataset
from train_classification import ResNetClassifier
import torch

model = ResNetClassifier()
checkpoint_file = "/home/marc/projects/marinedetector/checkpoints/epoch=591-val_accuracy=0.95.ckpt"
model.load_state_dict(torch.load(checkpoint_file)["state_dict"])

import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import numpy as np
imagesize = 32

val_ds = MarineDebrisDataset(root="/ssd/marinedebris/marinedebris_refined", fold="val", shuffle=False,
                             imagesize=imagesize * 10, transform=None)


def hook(module, input, output):
    x = input[0]
    #module(*input)
    output, attns = model.model.encoder.layers.encoder_layer_0.self_attention(query=x, key=x, value=x,
                                                                             need_weights=True, average_attn_weights=False)

    fig, axs = plt.subplots(1,attns.shape[1], figsize=(12,3))
    for i, (ax, attn) in enumerate(zip(axs, attns[0])):
        a = attn[0,1:].view(32,32)
        ax.imshow(a.detach().numpy())
        ax.set_title(f"head {i}")
    return


model.eval()
model.model.encoder.layers.encoder_layer_0.ln_1.register_forward_hook(hook)


for i in range(10):
    x, y = val_ds[i]

    y_pred = model(x.unsqueeze(0))>0
    print(y_pred, y)
    plt.figure()
    plt.imshow(equalize_hist(x.numpy()[np.array([3,2,1])]).transpose(1,2,0))
    plt.show()
