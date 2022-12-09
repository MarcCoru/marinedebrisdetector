import sys
sys.path.append("..")
sys.path.append("/home/sushen/marine_debris_semester_project")
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from data.refined_floatingobjects import RefinedFlobsRegionDataset
from data.marida import MaridaDataset, CLASS_MAPPING_USED
CLASS_MAPPING_USED_INV = {v:k for k,v in  CLASS_MAPPING_USED.items()}
from metrics import calculate_metrics

L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
marida_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
marida_band_idxs = np.array([L2ABANDS.index(b) for b in marida_bands])

def transform(image):
    img = image[marida_band_idxs]
    C, H, W = img.shape

    # center pixels
    x = img[:, H // 2, W // 2]
    return x

def transform_image(image, msk):
    img = image[marida_band_idxs] * 1e-4

    C, H, W = img.shape

    # center pixels
    x = img[:, H // 2, W // 2]
    y = msk[H // 2, W // 2]

    return x, y

from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, jaccard_score, accuracy_score, roc_auc_score
def calculate_metrics(targets, predictions, scores):

    auroc = roc_auc_score(targets, scores)

    p, r, f, s = precision_recall_fscore_support(y_true=targets,
                                                 y_pred=predictions, zero_division=0, average="binary")
    kappa = cohen_kappa_score(targets, predictions)

    jaccard = jaccard_score(targets, predictions)
    accuracy = accuracy_score(targets, predictions)

    return dict(
        auroc=auroc,
        accuracy=accuracy,
        precision=p,
        recall=r,
        fscore=f,
        kappa=kappa,
        jaccard=jaccard
    )

def predict_refined_floatingobjects(model, region):
    # Load test dataset
    f = np.load('/data/sushen/marinedebris/results/kikaki/randomforest/randomforest/test.npz')
    X = f["X"]
    y = f["y"]
    id = f["id"]

    # Difference between l1c and l2a is that we remove Band 10. So now need to remove Band 9 to use Marida RF.
    X = np.delete(X, 9, axis=1)

    # Must keep features based on type of random forest trained on Marida
    if rf_number == 0:
        X = X[:, :11]
    if rf_number == 1:
        X = X[:, :19]

    datapoint_region = np.array([i.split("-")[0] for i in id])
    if region == "accra_20181031":
        indexes = datapoint_region == "accra_20181031"
    if region == "durban_20190424":
        indexes = datapoint_region == "durban_20190424"
    X = X[indexes]
    y = y[indexes]

    y_pred = model.predict(X)
    y_score = model.predict_proba(X)
    y_score = y_score[:, 0] # 0 is index of marine debris

    y_binary = np.array([y == "Marine Debris" for y in y_pred])

    metrics = calculate_metrics(targets=y, predictions=y_binary, scores=y_score)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    print((y_binary == y).mean())
    print(classification_report(y_true=y, y_pred=y_binary))

def predict_marida(model):
    # Load test dataset
    f = np.load('/data/sushen/marinedebris/results/kikaki/randomforest/randomforest/test.npz')
    X = f["X"]
    y = f["y"]
    id = f["id"]

    # Difference between l1c and l2a is that we remove Band 10. So now need to remove Band 9 to use Marida RF.
    X = np.delete(X, 9, axis=1)

    # Must keep features based on type of random forest trained on Marida
    if rf_number == 0:
        X = X[:, :11]
    if rf_number == 1:
        X = X[:, :19]

    datapoint_region = np.array([i.split("-")[0] for i in id])
    indexes = datapoint_region == "marida"
    X = X[indexes]
    y = y[indexes]

    y_pred = model.predict(X)
    y_score = model.predict_proba(X)
    y_score = y_score[:, 0] # 0 is index of marine debris

    y_binary = np.array([y == "Marine Debris" for y in y_pred])

    y = np.array(y)
    y_binary = np.array(y_binary)

    metrics = calculate_metrics(targets=y, predictions=y_binary, scores=y_score)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    print(classification_report(y_true=y, y_pred=y_binary))

    print((y_binary == y).mean())

# 0, 1 or 2 to choose which rf model to be used
rf_number = 2
rf_classifier_paths = ['/data/sushen/marinedebris/MARIDA/rf_classifier_ss.joblib', 
    '/data/sushen/marinedebris/MARIDA/rf_classifier_ss_si.joblib', 
    '/data/sushen/marinedebris/MARIDA/rf_classifier_ss_si_glcm.joblib']
rf_classifier_path = rf_classifier_paths[rf_number]
model = joblib.load(rf_classifier_path)

print("durban")
predict_refined_floatingobjects(model, "durban_20190424")

print("accra")
predict_refined_floatingobjects(model, "accra_20181031")

print("marida")
predict_marida(model)