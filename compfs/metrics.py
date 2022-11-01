"""The possible metrics that can be used by the models."""

# stdlib
from functools import reduce

# third party
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(x, y):
    # Accuracy.
    acc = 100 * torch.sum(torch.argmax(x, dim=-1) == y) / len(y)
    return acc.item()


def lasso_accuracy(x, y):
    # Accuracy for lasso, does not use softmax.
    return (100 * torch.sum((x == y)) / len(y)).item()


def sklearn_accuracy(x, y):
    x = x > 0.5
    return 100 * accuracy_score(y, x)


def return_0(x, y):
    # Place-holder function if a metric doesn't exist.
    return 0


def mse(x, y):
    # MSE for regression.
    return 0.5 * torch.mean((x - y) ** 2).item()


def auroc(x, y):
    # Area under roc curve.
    return roc_auc_score(
        y.detach().cpu().numpy(),
        torch.softmax(x, dim=-1).detach().cpu().numpy()[:, 1],
    )


def lasso_auroc(x, y):
    # Area under roc curve for Lasso.
    return roc_auc_score(
        y.detach().cpu().numpy(),
        torch.sigmoid(x).detach().cpu().numpy(),
    )


def sklearn_auroc(x, y):
    return roc_auc_score(y, x)


def gsim(true_groups, predicted_groups):
    # Returns gsim, number of true groups, and number of discovered groups, given
    # true groups and predicted groups as input.
    gsim = 0
    if len(true_groups) == 0:  # i.e. we don't know the ground truth.
        return -1, len(true_groups), len(predicted_groups)
    if len(predicted_groups) > 0:
        for g in true_groups:
            current_max = 0
            for g_hat in predicted_groups:
                jac = np.intersect1d(g, g_hat).size / np.union1d(g, g_hat).size
                if jac == 1:
                    current_max = 1
                    break
                if jac > current_max:
                    current_max = jac
            gsim += current_max
        gsim /= max(len(true_groups), len(predicted_groups))
        return gsim, len(true_groups), len(predicted_groups)
    else:  # We didn't find anything.
        return 0, len(true_groups), len(predicted_groups)


def tpr_fdr(true_groups, predicted_groups):
    # True positive rate and false discovery rate.

    if len(true_groups) == 0:  # Ground truth not known.
        return -1, -1

    if len(predicted_groups) == 0:
        return 0.0, 0.0

    predicted_features = np.unique(reduce(np.union1d, predicted_groups))
    true_features = np.unique(reduce(np.union1d, true_groups))

    overlap = np.intersect1d(predicted_features, true_features).size
    tpr = 100 * overlap / len(true_features)
    fdr = (
        100 * (len(predicted_features) - overlap) / len(predicted_features)
    )  # If len(predicted_features) != 0 else 0.0.
    return tpr, fdr
