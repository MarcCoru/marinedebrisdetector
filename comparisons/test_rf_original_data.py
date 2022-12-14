"""
This script downloads the random forest classifier fitted on the original MARIDA data and tests it on our
re-downloaded marida data
It corresponds to RF - trained on the original MARIDA dataset in the paper
"""

import sys
sys.path.append("..")
import os
import joblib
import numpy as np
from sklearn.metrics import classification_report
import argparse
from marinedebrisdetector.data.utils import download

RANDOMFOREST_URL = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/models/rf_classifier_marida_original.joblib"
TESTDATA_URL = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/randomforest/test.npz"

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

def predict_refined_floatingobjects(model, region, test_data_path):
    # Load test dataset
    f = np.load(test_data_path)
    X = f["X"]
    y = f["y"]
    id = f["id"]

    # Difference between l1c and l2a is that we remove Band 10. So now need to remove Band 9 to use Marida RF.
    X = np.delete(X, 9, axis=1)

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

def predict_marida(model, test_data_path):
    # Load test dataset
    f = np.load(test_data_path)
    X = f["X"]
    y = f["y"]
    id = f["id"]

    # Difference between l1c and l2a is that we remove Band 10. So now need to remove Band 9 to use Marida RF.
    X = np.delete(X, 9, axis=1)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, default=".")
    parser.add_argument('--data-path', type=str, default=".")

    args = parser.parse_args()

    return args

def main(args):
    download(RANDOMFOREST_URL, args.ckpt_path)
    download(TESTDATA_URL, output_path=args.data_path)

    rf_classifier_path = os.path.join(args.ckpt_path,"rf_classifier_marida_original.joblib")
    test_data_path = os.path.join(args.data_path, "test.npz")
    model = joblib.load(rf_classifier_path)

    print("durban")
    predict_refined_floatingobjects(model, "durban_20190424", test_data_path)

    print("accra")
    predict_refined_floatingobjects(model, "accra_20181031", test_data_path)

    print("marida")
    predict_marida(model, test_data_path)

if __name__ == '__main__':
    main(parse_args())
