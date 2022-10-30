import copy
from datasets import datasets
import torch.nn as nn
from model.compfs import CompFS
from model.lasso import Lasso
from model.ensemble_stg import EnsembleSTG
from model.random_forests import RandomForests, GBDT
from model.base_model import TorchModel, SKLearnModel, Oracle, OracleCluster
from model import metrics
from model.thresholding_functions import make_lambda_threshold, make_top_k_threshold, make_std_threshold

metabric_config = {
    'data_info': {
        'dataset': datasets.Metabric, 
        'data_config': {
            'rule': 1,
            'train': True
            }
        },
    'model_info': {
        'compfs': {
            'base': TorchModel,
            'model': CompFS,
            'model_config': {
                'lr': 0.001,
                'lr_decay': 1.00,
                'batchsize': 50,
                'num_epochs': 250,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.auroc,
                'in_dim': 489,
                'h_dim': 75,
                'out_dim': 2,
                'nlearners': 5,
                'threshold_func': make_top_k_threshold(5),
                'temp': 0.1,
                'beta_s': 10.0,
                'beta_s_decay': 1.00,
                'beta_d': 10.0,
                'beta_d_decay': 1.00
                },
            },
        'stg': {
            'base': TorchModel,
            'model': EnsembleSTG,
            'model_config': {
                'lr': 0.001,
                'lr_decay': 1.00,
                'batchsize': 500,
                'num_epochs': 400,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.auroc,
                'in_dim': 489,
                'h_dim': 20,
                'out_dim': 2,
                'sigma': 0.5,
                'num_stg': 1,
                'lam': 0.1
                }
            },
        }
    }