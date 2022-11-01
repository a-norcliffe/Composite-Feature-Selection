import numpy as np
import torch
import torch.nn as nn

from compfs.datasets import NumpyDataset
from compfs.metrics import accuracy, gsim, tpr_fdr
from compfs.models import CompFS, TorchModel
from compfs.thresholding_functions import make_lambda_threshold

# Set and print device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# These can be changed to run your own data.
X_train = np.random.normal(size=(20000, 500))
y_train = np.array([((x[0] > 0.55) or (x[1] > 0.55)) for x in X_train])
X_val = np.random.normal(size=(200, 500))
y_val = np.array([((x[0] > 0.55) or (x[1] > 0.55)) for x in X_val])

is_classification = True

ground_truth_groups = [np.array([0]), np.array([1])]

# This config should be changed to use your own data, and find specific
# hyperparameters for the problem.

compfs_config = {
    "model": CompFS,
    "model_config": {
        "lr": 0.003,
        "lr_decay": 0.99,
        "batchsize": 50,
        "num_epochs": 10,
        "loss_func": nn.CrossEntropyLoss(),
        "val_metric": accuracy,
        "in_dim": 500,
        "h_dim": 20,
        "out_dim": 2,
        "nlearners": 5,
        "threshold_func": make_lambda_threshold(0.7),
        "temp": 0.1,
        "beta_s": 4.5,
        "beta_s_decay": 0.99,
        "beta_d": 1.2,
        "beta_d_decay": 0.99,
    },
}

compfs_config["device"] = device


def test_sanity() -> None:
    train_data = NumpyDataset(X_train, y_train, classification=is_classification)
    val_data = NumpyDataset(X_val, y_val, classification=is_classification)
    model = TorchModel(compfs_config)
    model.train(train_data, val_data)

    # Get group similarity and group structure.
    tpr, fdr = tpr_fdr(ground_truth_groups, model.get_groups())
    group_sim, ntrue, npredicted = gsim(ground_truth_groups, model.get_groups())

    assert ntrue == npredicted

    # Give selected features and save the groups.
    print("\n\nSelected Features:")
    learnt_groups = model.get_groups()
    for i in range(len(learnt_groups)):
        print("Group: {}, Features: {}".format(i + 1, learnt_groups[i]))
