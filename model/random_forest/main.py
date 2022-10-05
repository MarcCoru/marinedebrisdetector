import sys
sys.path.append("/home/marc/projects/marinedetector")

from data.marinedebrisdatamodule import MarineDebrisDataModule
from visualization import rgb
import model.random_forest.engineering_patches as eng
import numpy as np

from functools import partial
from tqdm import tqdm
import os

import skimage.color
import skimage
from skimage import feature


def main():
    dm = MarineDebrisDataModule("/data/marinedebris", no_label_refinement=True)
    dm.setup("fit")

    root = "/data/marinedebris/randomforest"
    os.makedirs(root, exist_ok=True)

    train_path = os.path.join(root,"train.npz")
    val_path = os.path.join(root, "val.npz")
    test_path = os.path.join(root, "test.npz")

    N_images = 10000
    N_pts = 5

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


def extract_feature_center(img):
    indices = calculate_indices(img)
    texture = calculate_texture(img)
    spatial = calculate_spatial(img)

    image_features = np.vstack([img.numpy(), indices, texture, spatial])

    c, h, w = image_features.shape
    return image_features[:, h // 2, w // 2]

def extract_features(img, mask, N_pts=5):
    # calculates features and takes N_pts random positive and negative pixels

    indices = calculate_indices(img)
    texture = calculate_texture(img)
    spatial = calculate_spatial(img)

    image_features = np.vstack([img.numpy(),indices,texture,spatial])

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
    main()
