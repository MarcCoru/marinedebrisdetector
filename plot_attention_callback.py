
import torch
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import wandb
import matplotlib as mpl
from skimage.exposure import equalize_hist
import numpy as np

class PlotAttentionCallback(Callback):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.last_outputs = outputs
        self.last_batch = batch

    def on_validation_epoch_end(self, trainer, pl_module):
        model = pl_module
        logger = self.logger

        def hook(module, input, output):
            x = output
            output, attns = model.model.encoder.layers.encoder_layer_0.self_attention(query=x, key=x, value=x,
                                                                                      need_weights=True,
                                                                                      average_attn_weights=False)

            viridis = mpl.colormaps['viridis']

            for idx in range(3):
                images = []
                for i, attn in enumerate(attns[idx]):
                    a_flat = attn[0, 1:]
                    sz = torch.sqrt(torch.tensor(a_flat.shape[0])).long()
                    a = attn[0, 1:].view(sz, sz)
                    images.append(wandb.Image(viridis(a.cpu().detach().numpy())))

                wandb.log({f"attn{idx}": images})
            """
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            for i, (ax, attn) in enumerate(zip(axs, attns[0])):
                a = attn[0, 1:].view(32, 32)
                ax.imshow(a.cpu().detach().numpy())
                ax.set_title(f"head {i}")
            return
            """

        x,y = self.last_batch

        for idx in range(3):
            rgb = equalize_hist(x.detach().cpu().numpy()[0, np.array([3, 2, 1])])
            rgb_image = wandb.Image(rgb.transpose(1,2,0))
            wandb.log({f"rgb{idx}": rgb_image})

        handle = model.model.encoder.layers.encoder_layer_0.ln_1.register_forward_hook(hook)
        model(x)
        handle.remove()

def main():
    from data import MarineDebrisRegionDataset, MarineDebrisDataset
    from train_marinedetector import Classifier

    model = Classifier()
    checkpoint_file = "/home/marc/projects/marinedetector/checkpoints/Vit_2022-09-15_21:18:37/epoch=591-val_accuracy=0.95.ckpt"
    model.load_state_dict(torch.load(checkpoint_file)["state_dict"])

    import matplotlib.pyplot as plt
    from skimage.exposure import equalize_hist
    import numpy as np
    imagesize = 32

    val_ds = MarineDebrisDataset(root="/data/marinedebris/marinedebris_refined", fold="val", shuffle=False,
                                 imagesize=imagesize * 10, transform=None)

    def hook(module, input, output):
        x = output
        output, attns = model.model.encoder.layers.encoder_layer_0.self_attention(query=x, key=x, value=x,
                                                                                 need_weights=True, average_attn_weights=False)

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        for i, (ax, attn) in enumerate(zip(axs, attns[0])):
            a = attn[0,1:].view(32,32)
            ax.imshow(a.detach().numpy())
            ax.set_title(f"head {i}")
        return


    model.eval()
    #model.model.encoder.layers.encoder_layer_0.ln_1.register_forward_hook(hook)

    idxs = np.random.RandomState(0).randint(len(val_ds), size=20)
    for i in idxs:
        x, y = val_ds[i]

        logits, attns = model.model(x.unsqueeze(0), need_weights=True)
        logits.backward(retain_graph=True)

        a_matrix = torch.stack([attn[0, 1:].view(32, 32) for attn in attns[0]])

        fig, axs = plt.subplots(1, 5, figsize=(3*5, 3))

        axs[0].imshow(equalize_hist(x.numpy()[np.array([3,2,1])]).transpose(1,2,0))

        for h, (ax, a) in enumerate(zip(axs[1:], a_matrix)):
            ax.imshow(a.detach().numpy())
            ax.set_title(f"head {h}")

        y_pred = logits>0
        fig.suptitle(f"idx {i} pred {int(y_pred)}, true {int(y)}")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

if __name__ == '__main__':
    main()
