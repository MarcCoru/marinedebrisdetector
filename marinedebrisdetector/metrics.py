from torch import nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, \
    cohen_kappa_score, jaccard_score, accuracy_score

def get_loss(pos_weight=None):
    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    def criterion(y_pred, target, mask=None):
        """a wrapper around BCEWithLogitsLoss that ignores no-data
        mask provides a boolean mask on valid data"""
        loss = bcecriterion(y_pred, target)
        if mask is not None:
            return (loss * mask.double()).mean()
        else:
            return loss.mean()
    return criterion


def calculate_metrics(targets, scores, optimal_threshold):
    predictions = scores > optimal_threshold

    auroc = roc_auc_score(targets, scores)
    p, r, f, s = precision_recall_fscore_support(y_true=targets,
                                                 y_pred=predictions, zero_division=0, average="binary")
    kappa = cohen_kappa_score(targets, predictions)

    jaccard = jaccard_score(targets, predictions)

    accuracy = accuracy_score(targets, predictions)

    summary = dict(
        auroc=auroc,
        precision=p,
        accuracy=accuracy,
        recall=r,
        fscore=f,
        kappa=kappa,
        jaccard=jaccard,
        threshold=optimal_threshold
    )

    return summary
