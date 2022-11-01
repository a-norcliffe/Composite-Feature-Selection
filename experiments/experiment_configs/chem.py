import copy

import torch.nn as nn

from compfs import metrics
from compfs.datasets import datasets
from compfs.models import (
    GBDT,
    CompFS,
    EnsembleSTG,
    Lasso,
    Oracle,
    OracleCluster,
    RandomForests,
    SKLearnModel,
    TorchModel,
)
from compfs.thresholding_functions import make_lambda_threshold

default_config = {
    "data_info": {
        "dataset": datasets.ChemistryBinding,
        "data_config": {"rule": 4, "train": True},
    },
    "model_info": {
        "compfs": {
            "base": TorchModel,
            "model": CompFS,
            "model_config": {
                "lr": 0.003,
                "lr_decay": 0.99,
                "batchsize": 20,
                "num_epochs": 35,
                "loss_func": nn.CrossEntropyLoss(),
                "val_metric": metrics.accuracy,
                "in_dim": 84,
                "h_dim": 20,
                "out_dim": 2,
                "nlearners": 5,
                "threshold_func": make_lambda_threshold(0.7),
                "temp": 0.1,
                "beta_s": 2.0,
                "beta_s_decay": 0.99,
                "beta_d": 1.2,
                "beta_d_decay": 0.99,
            },
        },
        "compfs1": {
            "base": TorchModel,
            "model": CompFS,
            "model_config": {
                "lr": 0.003,
                "lr_decay": 0.99,
                "batchsize": 20,
                "num_epochs": 35,
                "loss_func": nn.CrossEntropyLoss(),
                "val_metric": metrics.accuracy,
                "in_dim": 84,
                "h_dim": 30,
                "out_dim": 2,
                "nlearners": 1,
                "threshold_func": make_lambda_threshold(0.7),
                "temp": 0.1,
                "beta_s": 0.4,
                "beta_s_decay": 0.99,
                "beta_d": 1.2,
                "beta_d_decay": 0.99,
            },
        },
        "lasso": {
            "base": TorchModel,
            "model": Lasso,
            "model_config": {
                "lr": 0.003,
                "lr_decay": 0.99,
                "batchsize": 20,
                "num_epochs": 8,
                "loss_func": nn.MSELoss(),
                "val_metric": metrics.lasso_accuracy,
                "in_dim": 84,
                "out_dim": 1,
                "beta_s": 0.4,
                "threshold": 0.01,
            },
        },
        "ensemble_stg": {
            "base": TorchModel,
            "model": EnsembleSTG,
            "model_config": {
                "lr": 0.01,
                "lr_decay": 1.00,
                "batchsize": 500,
                "num_epochs": 500,
                "loss_func": nn.CrossEntropyLoss(),
                "val_metric": metrics.accuracy,
                "in_dim": 84,
                "h_dim": 50,
                "out_dim": 2,
                "sigma": 0.5,
                "num_stg": 4,
                "lam": 4.0,
            },
        },
        # STG was originally run in a notebook with the authors implementation, this alternative way to run
        # STG allows us to adapt our implementation of Ensemble STG, but they hyperparameters are correspondingly
        # edited as well to aquire the same (positive) results as reported in the paper.
        "stg": {
            "base": TorchModel,
            "model": EnsembleSTG,
            "model_config": {
                "lr": 0.01,
                "lr_decay": 1.00,
                "batchsize": 250,
                "num_epochs": 400,
                "loss_func": nn.CrossEntropyLoss(),
                "val_metric": metrics.accuracy,
                "in_dim": 84,
                "h_dim": 20,
                "out_dim": 2,
                "sigma": 0.5,
                "num_stg": 1,
                "lam": 15.0,
            },
        },
        "oracle": {"base": Oracle, "true_groups": datasets.chem_data_groups[4]},
        "oracle_cluster": {
            "base": OracleCluster,
            "true_groups": datasets.chem_data_groups[4],
        },
        "random_forests": {
            "base": SKLearnModel,
            "model": RandomForests,
            "model_config": {
                "val_metric": metrics.sklearn_accuracy,
                "n_estimators": 100,
                "max_depth": 5,
                "n_top_trees": 10,
                "max_idx": 10,
                "threshold": 0.1,
            },
        },
        "gbdt": {
            "base": SKLearnModel,
            "model": GBDT,
            "model_config": {
                "val_metric": metrics.sklearn_accuracy,
                "n_estimators": 15,
                "max_depth": 5,
                "n_top_trees": 10,
                "max_idx": 10,
                "threshold": 0.1,
            },
        },
    },
}


chem1_config = copy.deepcopy(default_config)

chem2_config = copy.deepcopy(default_config)
chem2_config["data_info"]["data_config"]["rule"] = 10
chem2_config["model_info"]["oracle"]["true_groups"] = datasets.chem_data_groups[10]
chem2_config["model_info"]["oracle_cluster"]["true_groups"] = datasets.chem_data_groups[
    10
]
chem2_config["model_info"]["compfs"]["model_config"]["beta_s"] = 3.4
chem2_config["model_info"]["lasso"]["model_config"]["beta_s"] = 0.2

chem3_config = copy.deepcopy(default_config)
chem3_config["data_info"]["data_config"]["rule"] = 13
chem3_config["model_info"]["oracle"]["true_groups"] = datasets.chem_data_groups[13]
chem3_config["model_info"]["oracle_cluster"]["true_groups"] = datasets.chem_data_groups[
    13
]
chem3_config["model_info"]["lasso"]["model_config"]["beta_s"] = 0.2
chem3_config["model_info"]["compfs1"]["model_config"]["beta_s"] = 0.7
chem3_config["model_info"]["stg"]["model_config"]["lam"] = 23.0
chem3_config["model_info"]["stg"]["model_config"]["batchsize"] = 500
chem3_config["model_info"]["stg"]["model_config"]["num_epochs"] = 150
