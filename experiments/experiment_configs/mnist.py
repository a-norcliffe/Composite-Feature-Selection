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

mnist_config = {
    'data_info': {
        'dataset': datasets.MNIST, 
        'nclasses': 10,
        'data_config': {
            'train': True
            }
        },
    'model_info': {
        'compfs5': {
            'base': TorchModel,
            'model': CompFS,
            'model_config': {
                'lr': 0.001,
                'lr_decay': 0.99,
                'batchsize': 500,
                'num_epochs': 50,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.accuracy,
                'in_dim': 28*28,
                'h_dim': 256,
                'out_dim': 10,
                'nlearners': 4,
                'threshold_func': make_top_k_threshold(15),
                'temp': 0.1,
                'beta_s': 0.18,
                'beta_s_decay': 0.97,
                'beta_d': 0.18,
                'beta_d_decay': 0.97
                },
            },
        'stg': {
            'base': TorchModel,
            'model': EnsembleSTG,
            'model_config': {
                'lr': 0.001,
                'lr_decay': 1.00,
                'batchsize': 500,
                'num_epochs': 60,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.accuracy,
                'in_dim': 28*28,
                'h_dim': 256,
                'out_dim': 10,
                'sigma': 0.5,
                'num_stg': 1,
                'lam': 4.0
                }
            },
        }
    }