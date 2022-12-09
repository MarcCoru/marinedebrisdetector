import sys
sys.path.append("..")

import rasterio as rio
from data.durban import annotated_objects
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.feature
import torch
from visualization import rgb


def get_confusions(prediction_file, annotation_file, threshold):
    with rio.open(prediction_file, "r") as src:
        predictions = src.read(1) / 255

    prediction = predictions > threshold

    peaks = skimage.feature.peak_local_max(predictions, threshold_abs=threshold, min_distance=3)

    with rio.open(annotation_file, "r") as src:
        annotations = src.read().transpose(1, 2, 0)

    # add water (not annotated at all to the annotations)
    water = annotations.sum(2) == 0

    annotations = np.dstack([annotations, water[:, :, None]])

    annotated_objects_all = annotated_objects + ["water"]

    prediction[peaks[:, 0], peaks[:, 1]]
    annotated_peaks = annotations[peaks[:, 0], peaks[:, 1]]

    instance, class_ids = np.where(annotated_peaks)

    anot, counts = np.unique(class_ids, return_counts=True)

    # initialize empty
    df = pd.DataFrame([annotated_objects_all, np.zeros(len(annotated_objects_all))],
                      index=["name", "counts"]).T.set_index("name")

    # fill with values
    for a, c in zip(np.array(annotated_objects_all)[anot], counts):
        df.loc[a] = c

    p_annot = annotated_peaks.argmax(1)
    p_peaks = np.hstack([peaks, p_annot[:, None]])

    return df, p_peaks, annotated_objects_all

def plot_detections(p_peaks, image_file):
    with rio.open(image_file, "r") as src:
        img = src.read()

    from matplotlib import colors
    debriscolor = "#FF0000"
    shipscolor = "#FBEE66"
    coastlinecolor = "#5C2483"
    landcolor = "#00A79F"
    cloudscolor = "#C8D300"
    hazedense = "#5B3428"
    hazetrans = "#CAC7C7"
    watercolor = "#4F8FCC"
    cmapcolors = [debriscolor, shipscolor, landcolor, coastlinecolor, cloudscolor, hazedense, hazetrans, watercolor]

    cmap = colors.LinearSegmentedColormap.from_list("debris", cmapcolors)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb(img).transpose(1, 2, 0))
    ax.scatter(p_peaks[:, 1], p_peaks[:, 0], c=p_peaks[:, 2], s=15, cmap=cmap, linewidths=0.5)
    ax.axis("off")
    return fig

def main(datapath = "/data/marinedebris/durban",
         threshold = None,
         checkpoint_file = "/data/marinedebris/results/ours/unet++_2022-10-03_04:00:27/epoch=89-val_loss=0.52.ckpt"):

    checkpoint_path = os.path.dirname(checkpoint_file)

    prediction_file = os.path.join(checkpoint_path, "test_scenes", "durban_prediction.tif")
    prediction_file_l1c = os.path.join(checkpoint_path, "test_scenes", "durban_l1c_prediction.tif")

    write_path = "/data/marinedebris/results/ours/unet++_2022-10-03_04:00:27/confusiondurban"
    os.makedirs(write_path, exist_ok=True)

    image_file = os.path.join(datapath,"durban_20190424.tif")
    image_file_l1c = os.path.join(datapath, "durban_20190424_l1c.tif")
    annotation_file = os.path.join(datapath, "durban_20190424_annotated.tif")

    if threshold is None:
        threshold = float(torch.load(checkpoint_file, map_location="cpu")["state_dict"]["threshold"])
        print(threshold)

    df, p_peaks, annotated_objects_all = get_confusions(prediction_file, annotation_file, threshold)
    df.loc[["debris", "haze_transparent", "haze_dense", "cummulus_clouds", "ships", "land", "coastline", "water"]]
    print(df)
    df.to_csv(os.path.join(write_path, "counts.csv"))
    print(df.sum())
    fig = plot_detections(p_peaks, image_file)

    writefile = os.path.join(write_path, "detections.png")
    print(f"writing plot to {writefile}")
    fig.savefig(writefile, bbox_inches="tight", pad_inches=0)

    df, p_peaks, annotated_objects_all = get_confusions(prediction_file_l1c, annotation_file, threshold)
    df.loc[["debris", "haze_transparent", "haze_dense", "cummulus_clouds", "ships", "land", "coastline", "water"]]

    print(df)
    df.to_csv(os.path.join(write_path, "counts_l1c.csv"))
    print(df.sum())
    fig = plot_detections(p_peaks, image_file_l1c)

    writefile = os.path.join(write_path, "detections_l1c.png")
    print(f"writing plot to {writefile}")
    fig.savefig(writefile, bbox_inches="tight", pad_inches=0)

def download(data_path):
    import urllib.request
    import zipfile

    if not os.path.exists("PLP.zip"):
        urllib.request.urlretrieve(data_url, "PLP.zip")

    with zipfile.ZipFile("PLP.zip", 'r') as zip_ref:
        zip_ref.extractall(data_path)

if __name__ == '__main__':

    import urllib.request
    import zipfile

    data_url = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/durban.zip"
    checkpoint_url = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/models/unet%2B%2B_durban.zip"
    mifdalmodel = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/models/unet-posweight1-lr001-bs160-ep50-aug1-seed0.zip"

    if not os.path.exists("durban.zip"):
        urllib.request.urlretrieve(data_url, "durban.zip")

    with zipfile.ZipFile("durban.zip", 'r') as zip_ref:
        zip_ref.extractall(".")


    if not os.path.exists("unet++_durban.zip"):
        urllib.request.urlretrieve(checkpoint_url, "unet++_durban.zip")

        with zipfile.ZipFile("unet++_durban.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    main(datapath="./durban",
         threshold=None,
         checkpoint_file="unet++_durban/epoch=89-val_loss=0.52.ckpt")

    os.makedirs("mifdal", exist_ok=True)
    if not os.path.exists("unet-posweight1-lr001-bs160-ep50-aug1-seed0.zip"):
        urllib.request.urlretrieve(mifdalmodel, "unet-posweight1-lr001-bs160-ep50-aug1-seed0.zip")

        with zipfile.ZipFile("unet-posweight1-lr001-bs160-ep50-aug1-seed0.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    main(datapath="./durban",
         threshold=0.038854,
         checkpoint_file="unet-posweight1-lr001-bs160-ep50-aug1-seed0/unet-posweight1-lr001-bs160-ep50-aug1-seed0.pth.tar")
