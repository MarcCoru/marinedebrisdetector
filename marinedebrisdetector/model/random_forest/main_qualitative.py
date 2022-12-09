import sys
sys.path.append("/home/marc/projects/marinedetector")
import joblib
from data.marinedebrisdatamodule import MarineDebrisDataModule
import numpy as np
from random_forest import get_random_forest

import os

import matplotlib
matplotlib.use("pdf")

from predictor import ScenePredictor

from sklearn.metrics import precision_recall_curve
from test_kikaki import extract_feature_image
import torch

def main():
    root = "/data/marinedebris/results/kikaki/randomforest"
    rf_path = os.path.join(root, "random_forest.joblib")

    marinedebris_datamodule = MarineDebrisDataModule("/data/marinedebris", no_label_refinement=True)
    marinedebris_datamodule.setup("fit")

    if os.path.exists(rf_path):
        rf_classifier = joblib.load(rf_path)
    else:
        rf_classifier = get_random_forest()
        # TRAIN
        f = np.load(os.path.join(root, "train.npz"))
        X = f["X"]
        y = f["y"]

        rf_classifier.fit(X, y)
        joblib.dump(rf_classifier, rf_path)

    # VAL
    f = np.load(os.path.join(root, "val.npz"))
    X = f["X"]
    y = f["y"]

    yscore = rf_classifier.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, yscore)
    ix = np.abs(precision - recall).argmin()
    optimal_threshold = thresholds[ix]

    model = Imageprediction_wapper(rf_classifier, threshold=optimal_threshold)

    predictor = ScenePredictor(device="cpu", offset=0, activation="none")
    for ds in marinedebris_datamodule.get_qualitative_test_dataset().datasets:
        path = ds.tifffile
        predpath = os.path.join(root, "test_scenes", os.path.basename(path))
        os.makedirs(os.path.dirname(predpath), exist_ok=True)
        print(f"writing {os.path.abspath(predpath)}")
        predictor.predict(model, path, predpath)
    print()


class Imageprediction_wapper(torch.nn.Module):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, x):
        feat = extract_feature_image(x.squeeze().cpu())

        D, H, W = feat.shape
        feat_flat = np.nan_to_num(feat.reshape(D, H*W).T)

        y_proba = self.model.predict_proba(feat_flat)[:, 1]

        y_proba = y_proba.reshape(H, W)

        return torch.from_numpy(y_proba).unsqueeze(0).unsqueeze(0)

if __name__ == '__main__':
    main()
