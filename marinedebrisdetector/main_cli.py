import torch
from torch import nn
import argparse
from tqdm import tqdm

from marinedebrisdetector.predictor import ScenePredictor

class MarineDebrisDetector(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.device = args.device

        if not args.ensemble:
            self.model = torch.hub.load("marccoru/marinedebrisdetector", args.model, int(args.seed)).to(self.device).eval()
        if args.ensemble:
            self.model = [torch.hub.load("marccoru/marinedebrisdetector", args.model, seed).to(self.device).eval() for seed in [1,2,3]]

        if args.test_time_augmentation:
            if not args.ensemble:
                self.model = TestTimeAugmentation_wapper(self.model)
            else:
                self.model = [TestTimeAugmentation_wapper(model) for model in self.model]
    def forward(self, X):
        if isinstance(self.model, list): # ensemble
            y_score = [torch.sigmoid(model(X.to(self.device))) for model in self.model]

            # normalize scores to be at threshold 0.5
            y_pred = torch.median(torch.stack([y_sc > model.threshold for y_sc, model in zip(y_score, self.model)]),dim=0).values

            return y_score, y_pred

        else:
            y_score = torch.sigmoid(self.model(X.to(self.device)))

            # re-normalize scores to be at 0.5
            #y_score = normalize(y_score, self.model)

            return y_score, y_score > self.model.threshold


class TestTimeAugmentation_wapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threshold = model.threshold

    def forward(self, x):
        y_logits = self.model(x)

        y_logits += torch.fliplr(self.model(torch.fliplr(x)))  # fliplr)
        y_logits += torch.flipud(self.model(torch.flipud(x)))  # flipud

        for rot in [1, 2, 3]:  # 90, 180, 270 degrees
            y_logits += torch.rot90(self.model(torch.rot90(x, rot, [2, 3])), -rot, [2, 3])
        y_logits /= 6

        return y_logits

def normalize(score, model):
    return score * 0.5 / model.threshold

def plot_qualitative(detector):
    import numpy as np
    download_qualitative()

    X = np.load("qualitative_test.npz")["X"]
    Y = np.load("qualitative_test.npz")["Y"]
    ids = np.load("qualitative_test.npz")["ids"]

    X = torch.from_numpy(X).float()

    with torch.no_grad():
        y_pred, y_score = detector(X)

    import matplotlib.pyplot as plt
    from marinedebrisdetector.visualization import rgb, fdi

    N = X.shape[0]
    fig, axs = plt.subplots(N, 5, figsize=(5*3,N*3))

    for ax, title in zip(axs[0],["rgb", "fdi", "mask", "y_pred", "y_score"]):
        ax.set_title(title)

    for x, y, y_score_, y_pred_, id, ax_row in zip(X, Y, y_score.cpu(), y_pred.cpu(), ids, axs):
        ax_row[0].imshow(rgb(x.numpy()).transpose(1, 2, 0))
        ax_row[1].imshow(fdi(x.numpy()))
        ax_row[2].imshow(y)
        ax_row[3].imshow(y_score_.squeeze().numpy())
        ax_row[4].imshow(y_pred_.squeeze().numpy())

        ax_row[0].set_ylabel(id)

        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def download(url):
    import urllib
    import os
    if not os.path.exists(os.path.basename(url)):
        output_path = os.path.basename(url)
        print(f"downloading {url} to {output_path}")
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"{os.path.basename(url)} exists. skipping...")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
def download_qualitative():
    download("https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/qualitative_test.npz")

def download_accra():
    download("https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/accra_20181031.tif")

def download_durban():
    download("https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/durban_20190424.tif")

def parse_args():
    parser = argparse.ArgumentParser(prog='MarineDebrisDetector')
    parser.add_argument('scene', default=None, nargs='?')
    parser.add_argument('--model', type=str, choices=["unetpp", "unetpp_no_label_refinement", "unet"], default="unetpp")
    parser.add_argument('--plot-qualitative', action="store_true")
    parser.add_argument('--seed', default="1", choices=["1","2","3"])
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument('--ensemble', action="store_true")
    parser.add_argument('--test-time-augmentation', action="store_true")

    # download test data
    parser.add_argument('--download-accra', action="store_true")
    parser.add_argument('--download-durban', action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.download_accra:
        download_accra()

    if args.download_durban:
        download_durban()

    detector = MarineDebrisDetector(args)

    if args.scene is not None:
        predictor = ScenePredictor(device=args.device)
        predictor.predict(detector, args.scene, args.scene.replace(".tif", "_prediction.tif"))

    if args.plot_qualitative:
        plot_qualitative(detector)

if __name__ == '__main__':
    main()
