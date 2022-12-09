import pandas as pd

from model.segmentation_model import SegmentationModel
from data.marinedebrisdatamodule import MarineDebrisDataModule
import argparse
import pytorch_lightning as pl
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from predictor import ScenePredictor
import torch
from visualization import rgb, fdi, ndvi
from torch import nn
from checkpoints import CHECKPOINT_FOLDERS

pl.seed_everything(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=str, default="ours", choices=["ours", "mifdal"])
    parser.add_argument('--ckpt-folder', type=str, default="/data/marinedebris/checkpoints/unet++1")
    parser.add_argument('--data-path', type=str, default="/data/marinedebris")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--calculate_qualitative', action='store_true')
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")

    args = parser.parse_args()

    return args

def parse_checkpoint_files_return_best(ckpt_files, metric="auroc", top_k=1, lower_better=False):
    """
    expects checkpoint files of format 'epoch=0-val_loss=0.77-auroc=0.852.ckpt'
    with metrics separated by - and key-values by =
    """
    top_k-=1
    stats = []
    for ckpt_file in ckpt_files:
        ckpt_file = ckpt_file.replace(".ckpt", "")
        stat = {f.split("=")[0]: float(f.split("=")[1]) for f in ckpt_file.split("-")}
        stat["filename"] = ckpt_file
        stats.append(stat)
    df = pd.DataFrame(stats)
    fname = df.sort_values(metric, ascending=lower_better).iloc[top_k].filename
    return fname + ".ckpt"

from copy import copy
class EnsembleModel(nn.Module):
    def __init__(self, checkpoint_files):
        super().__init__()
        self.models = nn.ModuleList([SegmentationModel.load_from_checkpoint(checkpoint_path=ckpt).model for ckpt in checkpoint_files])
    def forward(self, x):
        return torch.stack([model(x) for model in self.models]).mean(0)

def main(args):
    ensemble = False

    if args.comparison == "ours":
        ckpt_files = [f for f in os.listdir(args.ckpt_folder) if f.endswith(".ckpt") and f != "last.ckpt"]

        ckpt_file = parse_checkpoint_files_return_best(ckpt_files, top_k=args.topk)
        print(f"using checkpoint {ckpt_file}")
        model = SegmentationModel.load_from_checkpoint(checkpoint_path=os.path.join(args.ckpt_folder, ckpt_file))

        if ensemble:
            checkpoint_files = [os.path.join(args.ckpt_folder, ckpt_file) for ckpt_file in ckpt_files]
            model.model = EnsembleModel(checkpoint_files)

    elif args.comparison == "mifdal":
        from comparisons.mifdal_model import load_mifdal_model
        model = load_mifdal_model()
    else:
        raise ValueError()

    model = model.eval()


    marinedebris_datamodule = MarineDebrisDataModule(data_root=args.data_path,
                                        image_size=args.image_size,
                                        workers=args.workers,
                                        batch_size=args.batch_size)

    logger = pl.loggers.csv_logs.CSVLogger(args.ckpt_folder, name="test_log", version=args.topk-1)
    trainer = pl.Trainer(logger=logger, accelerator="gpu")
    trainer.test(model, marinedebris_datamodule)

    if args.calculate_qualitative:

        model = model.eval()
        model = TestTimeAugmentation_wapper(model)


        write_qualitative(model,
                  dataset=marinedebris_datamodule.get_plp_dataset(2022, output_size=64),
                  path=os.path.join(args.ckpt_folder, "plp2022"),
                  cut_border=16)

        write_qualitative(model,
                  dataset=marinedebris_datamodule.get_plp_dataset(2021, output_size=64),
                  path=os.path.join(args.ckpt_folder, "plp2021"),
                  cut_border=16)

        write_qualitative(model,
                              dataset=marinedebris_datamodule.get_qualitative_test_dataset(),
                              path=os.path.join(args.ckpt_folder, "qualitative"))


        predictor = ScenePredictor(device="cuda")
        for name, path in marinedebris_datamodule.get_prediction_scene_paths():
            predpath = os.path.join(args.ckpt_folder, "test_scenes", name + "_prediction.tif")
            os.makedirs(os.path.dirname(predpath), exist_ok=True)
            print(f"writing {os.path.abspath(predpath)}")
            predictor.predict(TestTimeAugmentation_wapper(model), path, predpath)

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

def write_plp(model, dataset, path):
    model.eval()
    os.makedirs(path, exist_ok=True)
    for image, mask, id in dataset:
        x = torch.from_numpy(image).float().unsqueeze(0)
        y_score = torch.sigmoid(model(x)).squeeze().detach().numpy()

        write_predictions(model, y_score, x, path, mask, id)

def write_qualitative(model, dataset, path, cut_border=0):
    os.makedirs(path, exist_ok=True)
    for image, mask, id in dataset:
        x = torch.from_numpy(image).float().unsqueeze(0)
        y_score = torch.sigmoid(model(x)).squeeze().detach().numpy()

        if cut_border > 0:
            y_score = y_score[cut_border:-cut_border, cut_border:-cut_border]
            x = x[:,:,cut_border:-cut_border,cut_border:-cut_border]
            mask = mask[cut_border:-cut_border, cut_border:-cut_border]

        write_predictions(model, y_score, x, path, mask, id)

def write_predictions(model, y_score, x, path, mask, id):
    y_pred = y_score > model.threshold.numpy()

    fig, ax = plt.subplots()
    ax.imshow(y_score, cmap="Reds") # vmin=0, vmax=1,
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_yscore.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    fig, ax = plt.subplots()
    ax.imshow(y_pred, vmin=0, vmax=1, cmap="Reds")
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_ypred.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    norm = colors.Normalize(vmin=-0.5, vmax=0.5, clip=True)
    scmap = cm.ScalarMappable(norm=norm, cmap="viridis")
    fig, ax = plt.subplots()
    ax.imshow(scmap.to_rgba(ndvi(x.squeeze(0)).cpu()))
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_ndvi.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    norm = colors.Normalize(vmin=-0.1, vmax=0.1, clip=True)
    scmap = cm.ScalarMappable(norm=norm, cmap="magma")
    fig, ax = plt.subplots()
    ax.imshow(scmap.to_rgba(fdi(x.squeeze(0)).cpu()))
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_fdi.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    fig, ax = plt.subplots()
    ax.imshow(rgb(x.squeeze(0).cpu().numpy()).transpose(1, 2, 0))
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_rgb.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    fig, ax = plt.subplots()
    ax.imshow(rgb(x.squeeze(0).cpu().numpy()).transpose(1, 2, 0))
    ax.contour(y_score, cmap="Reds", vmin=0, vmax=1, levels=8)
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_yscore_overlay.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    fig, ax = plt.subplots()
    ax.imshow(rgb(x.squeeze(0).cpu().numpy()).transpose(1, 2, 0))
    ax.contour(y_pred, cmap="Reds", vmin=0, vmax=1, levels=8)
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_ypred_overlay.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    fig, ax = plt.subplots()
    ax.imshow(mask, vmin=0, vmax=1, cmap="Reds", interpolation="none")
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_mask.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

    fig, ax = plt.subplots()
    ax.imshow(rgb(x.squeeze(0).cpu().numpy()).transpose(1, 2, 0))
    ax.contour(mask, cmap="Reds", vmin=0, vmax=1, levels=8)
    ax.axis("off")
    write_path = os.path.join(path, f"{id}_mask_overlay.png")
    fig.savefig(write_path, bbox_inches="tight", pad_inches=0)
    print(f"writing {os.path.abspath(write_path)}")

if __name__ == '__main__':
    main(parse_args())
