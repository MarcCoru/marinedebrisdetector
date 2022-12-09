from model.mifdal_model import load_mifdal_model
from data.marinedebrisdatamodule import MarineDebrisDataModule
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import numpy as np
import pytorch_lightning as pl
pl.seed_everything(0)

@torch.no_grad()
def main():
    model = load_mifdal_model()

    model.eval()
    dm = MarineDebrisDataModule("/data/marinedebris", image_size=128)
    dm.setup("fit")
    dl = dm.val_dataloader()

    y_true, y_scores = [], []
    for images, mask, id in tqdm(dl, total=len(dl)):
        logits = model(images)
        N, _, H, W = logits.shape
        h, w = H // 2, W // 2
        logits = logits.squeeze(1)[:, h, w]  # keep only ce
        y_scores.append(torch.sigmoid(logits))
        y_true.append(mask)

    y_true = torch.hstack(y_true).numpy()
    y_scores = torch.hstack(y_scores).numpy()

    y_true = y_true.reshape(-1).astype(int)
    y_scores = y_scores.reshape(-1)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ix = np.abs(precision - recall).argmin()
    optimal_threshold = thresholds[ix]

    print(f"optimal threshold is {optimal_threshold}")

if __name__ == '__main__':
    main()
