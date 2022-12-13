"""
This script downloads trainind, validation, and test data from our dataset, fits a random forest classifier,
and reports results.
It corresponds to RF - trained on our dataset in the paper
"""

import pandas as pd
import numpy as np

from marinedebrisdetector.data.marinedebrisdatamodule import MarineDebrisDataModule
from marinedebrisdetector.visualization import rgb, fdi, ndvi
import marinedebrisdetector.model.random_forest.engineering_patches as eng
from marinedebrisdetector.model.random_forest.random_forest import get_random_forest
from marinedebrisdetector.metrics import calculate_metrics
from marinedebrisdetector.data.utils import download

from functools import partial
from tqdm import tqdm
import os

import matplotlib

import skimage.color
import skimage
from skimage import feature
from sklearn.metrics import classification_report, precision_recall_curve
import torch

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import argparse

TRAINDATA_URL = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/randomforest/train.npz"
VALDATA_URL = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/randomforest/val.npz"
TESTDATA_URL = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/randomforest/test.npz"

DEFAULT_DATA_PATH = "randomforest_data"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-folder', type=str, default=".") # current folder
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH) # download data

    args = parser.parse_args()

    return args


def main(args):
    root = args.ckpt_folder
    data_path = args.data_path

    N_images = 100000

    # container for metrics.csv will be updated
    stats = dict()

    if data_path is not DEFAULT_DATA_PATH:
        dm = MarineDebrisDataModule(data_path, no_label_refinement=True)
        dm.setup("fit")

        # this function writes out train.npz, val.npz and test.npz.
        # but it takes several hours.
        extract_and_save_features(root, dm, N_images=N_images)
    else:
        download(TRAINDATA_URL, output_path=root)
        download(VALDATA_URL, output_path=root)
        download(TESTDATA_URL, output_path=root)

    rf_classifier = get_random_forest()
    # TRAIN
    f = np.load(os.path.join(root, "train.npz"))
    X = f["X"]
    y = f["y"]

    rf_classifier.fit(X, y)

    # VAL
    f = np.load(os.path.join(root, "val.npz"))
    X = f["X"]
    y = f["y"]

    yscore = rf_classifier.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, yscore)
    ix = np.abs(precision - recall).argmin()
    optimal_threshold = thresholds[ix]

    print(f"optimal threshold according to validation set is {optimal_threshold:.3f}")

    # TEST
    f = np.load(os.path.join(root, "test.npz"))
    X = f["X"]
    y = f["y"]
    id = f["id"]
    region = np.array([i.split("-")[0] for i in id])
    is_accra = region == "accra_20181031"
    is_durban = region == "durban_20190424"
    is_marida = region == "marida"

    y_pred = rf_classifier.predict(X)
    yscore = rf_classifier.predict_proba(X)[:,1]

    print("combined")
    print(classification_report(y, y_pred))

    metrics = calculate_metrics(targets=y, scores=yscore, optimal_threshold=optimal_threshold)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # store in dict for metrics.csv
    stats.update(**{f"test_{k}": v for k, v in metrics.items()})

    print("accra_20181031")
    print(classification_report(y[is_accra], y_pred[is_accra]))

    metrics = calculate_metrics(targets=y[is_accra], scores=yscore[is_accra], optimal_threshold=optimal_threshold)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # store in dict for metrics.csv
    stats.update(**{f"test_accra_20181031_{k}": v for k, v in metrics.items()})

    print("durban_20190424")
    print(classification_report(y[is_durban], y_pred[is_durban]))

    metrics = calculate_metrics(targets=y[~is_accra], scores=yscore[~is_accra], optimal_threshold=optimal_threshold)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # store in dict for metrics.csv
    stats.update(**{f"test_durban_20190424_{k}": v for k, v in metrics.items()})

    print("marida")
    print(classification_report(y[is_marida], y_pred[is_marida]))

    metrics = calculate_metrics(targets=y[~is_accra], scores=yscore[~is_accra], optimal_threshold=optimal_threshold)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # store in dict for metrics.csv
    stats.update(**{f"test_marida_{k}": v for k, v in metrics.items()})

    # this path emulates the pytorch lightning default logger (consistent with other models)
    metrics_file = os.path.join(root, "test_log", "version_0", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    pd.DataFrame([stats]).to_csv(metrics_file)

    if data_path is DEFAULT_DATA_PATH:
        print(f"skipping qualitative predictions, as data_path is not specified. "
              f"please point this to the data root folder where PLP and qualitative data are located")
    else:
        path = root + "/qualitative"
        qual_test_dataset = dm.get_qualitative_test_dataset()
        write_qualitative(rf_classifier, qual_test_dataset, path, optimal_threshold=optimal_threshold)

        path = root + "/plp2021"
        qual_test_dataset = dm.get_plp_dataset(2021, output_size=64)
        write_qualitative(rf_classifier, qual_test_dataset, path, cut_border=16, optimal_threshold=optimal_threshold)

        path = root + "/plp2022"
        qual_test_dataset = dm.get_plp_dataset(2022, output_size=64)
        write_qualitative(rf_classifier, qual_test_dataset, path, cut_border=16, optimal_threshold=optimal_threshold)
def write_qualitative(rf_classifier, qual_test_dataset, path, cut_border=0, optimal_threshold=0.5):
    os.makedirs(path, exist_ok=True)

    for x, mask, id in qual_test_dataset:
        feat = extract_feature_image(torch.from_numpy(x))

        D, H, W = feat.shape
        feat_flat = feat.reshape(D, H*W).T # HW x D
        feat_flat = np.nan_to_num(feat_flat)
        pred_flat = rf_classifier.predict_proba(feat_flat)[:,1]
        y_score = pred_flat.reshape(H,W)

        x = torch.from_numpy(x).unsqueeze(0)

        if cut_border > 0:
            y_score = y_score[cut_border:-cut_border, cut_border:-cut_border]
            x = x[:,:,cut_border:-cut_border,cut_border:-cut_border]
            mask = mask[cut_border:-cut_border, cut_border:-cut_border]

        threshold = optimal_threshold
        y_pred = y_score > threshold

        fig, ax = plt.subplots()
        ax.imshow(y_score, vmin=0, vmax=1, cmap="Reds")
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


def extract_and_save_features(root, dm, N_images = 10000, N_pts = 5):
    """
    generates train.npz. val.npz and test.noz in root if they dont exist
    takes several hours
    """

    os.makedirs(root, exist_ok=True)

    train_path = os.path.join(root,"train.npz")
    val_path = os.path.join(root, "val.npz")
    test_path = os.path.join(root, "test.npz")



    if not os.path.exists(train_path):
        train_dataset = dm.train_dataset
        idxs = np.random.RandomState(0).randint(0, len(train_dataset), N_images)

        X, Y, ids = [], [], []
        for idx in tqdm(idxs):
            img, mask, id = train_dataset[idx]
            x,y = extract_features(img, mask, N_pts=N_pts)
            if x is None:
                continue
            if y is None:
                continue
            X.append(x)
            Y.append(y)
            ids.append(id)

        X = np.nan_to_num(np.vstack(X))
        y = np.hstack(Y)
        id = np.hstack(ids)

        np.savez(train_path, X=X, y=y, id=id)

    if not os.path.exists(val_path):
        valid_dataset = dm.valid_dataset
        X_val, y_val, id_val = [],[],[]
        for img, mask, id in tqdm(valid_dataset):
            X_val.append(extract_feature_center(img))
            y_val.append(mask)
            id_val.append(id)

        X = np.nan_to_num(np.vstack(X_val))
        y = np.hstack(y_val)
        id = np.hstack(id_val)
        np.savez(val_path, X=X, y=y, id=id)

    if not os.path.exists(test_path):
        test_dataset = dm.test_dataset
        X_test, y_test, id_test = [], [], []
        for img, mask, id in tqdm(test_dataset):
            X_test.append(extract_feature_center(img))
            y_test.append(mask)
            id_test.append(id)

        X = np.nan_to_num(np.vstack(X_test))
        y = np.hstack(y_test)
        id = np.hstack(id_test)
        np.savez(test_path, X=X, y=y, id=id)

def extract_feature_image(img):
    indices = calculate_indices(img)
    texture = calculate_texture(img)
    # spatial = calculate_spatial(img) # spatial feature are not used in the paper

    return np.vstack([img.numpy(), indices, texture])

def extract_feature_center(img):
    image_features = extract_feature_image(img)

    c, h, w = image_features.shape
    return image_features[:, h // 2, w // 2]

def extract_features(img, mask, N_pts=5):
    # calculates features and takes N_pts random positive and negative pixels

    image_features = extract_feature_image(img)

    x_pos, y_pos = np.where(mask)
    if len(x_pos) > 0:
        idxs = np.random.RandomState(0).randint(0,len(x_pos), size=N_pts)
        v_pos = image_features[:,x_pos[idxs],y_pos[idxs]].T
        target_pos = np.ones(v_pos.shape[0])
    else:
        v_pos = None
        target_pos = None

    x_neg, y_neg = np.where(~mask.numpy().astype(bool))
    if len(x_neg) > 0:
        idxs = np.random.RandomState(0).randint(0,len(x_neg), size=N_pts)
        v_neg = image_features[:,x_neg[idxs],y_neg[idxs]].T
        target_neg = np.zeros(v_neg.shape[0])
    else:
        return None, None

    if v_pos is not None:
        feat = np.vstack([v_pos, v_neg])
        target = np.hstack([target_pos, target_neg])
    else:
        feat = v_neg
        target = target_neg

    return feat, target


def calculate_spatial(img, sigma_min=1, sigma_max=16):

    rgb_image = rgb(img.numpy())
    gray = skimage.color.rgb2gray(rgb_image.transpose(1,2,0))

    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max)
    return features_func(gray).astype(gray.dtype).transpose(2,0,1)

def calculate_texture(img, window_size = 13, max_value = 16):

    rgb_image = rgb(img.numpy())
    gray = skimage.color.rgb2gray(rgb_image.transpose(1,2,0))

    bins = np.linspace(0.00, 1.00, max_value)
    num_levels = max_value + 1

    temp_gray = np.pad(gray, (window_size - 1) // 2, mode='reflect')

    features_results = np.zeros((gray.shape[0], gray.shape[1], 6), dtype=temp_gray.dtype)

    for col in range((window_size - 1) // 2, gray.shape[0] + (window_size - 1) // 2):
        for row in range((window_size - 1) // 2, gray.shape[0] + (window_size - 1) // 2):
            temp_gray_window = temp_gray[row - (window_size - 1) // 2: row + (window_size - 1) // 2 + 1,
                               col - (window_size - 1) // 2: col + (window_size - 1) // 2 + 1]

            inds = np.digitize(temp_gray_window, bins)

            # Calculate on E, NE, N, NW as well as symmetric. So calculation on all directions and with 1 pixel offset-distance
            matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=num_levels,
                                              normed=True, symmetric=True)

            # Aggregate all directions
            matrix_coocurrence = matrix_coocurrence.mean(3)[:, :, :, np.newaxis]

            con, dis, homo, ener, cor, asm = eng.glcm_feature(matrix_coocurrence)
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 0] = con
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 1] = dis
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 2] = homo
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 3] = ener
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 4] = cor
            features_results[row - (window_size - 1) // 2, col - (window_size - 1) // 2, 5] = asm

    return np.stack(features_results).transpose(2,0,1)

def calculate_indices(img):
    NDVI = eng.ndvi(img[4], img[8])
    FAI = eng.fai(img[4], img[8], img[10])
    FDI = eng.fdi(img[6], img[8], img[10])
    SI = eng.si(img[2], img[3], img[4])
    NDWI = eng.ndwi(img[3], img[8])
    NRD = eng.nrd(img[4], img[8])
    NDMI = eng.ndmi(img[8], img[10])
    BSI = eng.bsi(img[2], img[4], img[8], img[10])

    return np.stack([NDVI, FAI, FDI, SI, NDWI, NRD, NDMI, BSI])

if __name__ == '__main__':
    main(parse_args())
