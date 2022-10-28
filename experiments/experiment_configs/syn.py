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

default_config = {
    'data_info': {
        'dataset': datasets.SyntheticGaussian, 
        'data_config': {
            'nfeatures': 500,
            'rule': 1,
            'train': True
            }
        },
    'model_info': {
        'compfs5': {
            'base': TorchModel,
            'model': CompFS,
            'model_config': {
                'lr': 0.003,
                'lr_decay': 0.99,
                'batchsize': 50,
                'num_epochs': 35,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.accuracy,
                'in_dim': 500,
                'h_dim': 20,
                'out_dim': 2,
                'nlearners': 5,
                'threshold_func': make_lambda_threshold(0.7),
                'temp': 0.1,
                'beta_s': 4.5,
                'beta_s_decay': 0.99,
                'beta_d': 1.2,
                'beta_d_decay': 0.99 
                },
            },
        'compfs1': {
            'base': TorchModel,
            'model': CompFS,
            'model_config': {
                'lr': 0.003,
                'lr_decay': 0.99,
                'batchsize': 100,
                'num_epochs': 35,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.accuracy,
                'in_dim': 500,
                'h_dim': 30,
                'out_dim': 2,
                'nlearners': 1,
                'threshold_func': make_lambda_threshold(0.7),
                'temp': 0.1,
                'beta_s': 0.35,
                'beta_s_decay': 0.99,
                'beta_d': 1.2,
                'beta_d_decay': 0.99 
                },
            },
        'lasso': {
            'base': TorchModel,
            'model': Lasso,
            'model_config': {
                'lr': 0.003,
                'lr_decay': 0.99,
                'batchsize': 50,
                'num_epochs': 8,
                'loss_func': nn.MSELoss(),
                'val_metric': metrics.lasso_accuracy,
                'in_dim': 500,
                'out_dim': 1,
                'beta_s': 0.4,
                'threshold': 0.01,
                }
            },
        'ensemble_stg': {
            'base': TorchModel,
            'model': EnsembleSTG,
            'model_config': {
                'lr': 0.001,
                'lr_decay': 1.00,
                'batchsize': 100,
                'num_epochs': 150,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.accuracy,
                'in_dim': 500,
                'h_dim': 50,
                'out_dim': 2,
                'sigma': 0.5,
                'num_stg': 4,
                'lam': 0.1
                }
            },
        'stg': {
            'base': TorchModel,
            'model': EnsembleSTG,
            'model_config': {
                'lr': 0.001,
                'lr_decay': 1.00,
                'batchsize': 250,
                'num_epochs': 250,
                'loss_func': nn.CrossEntropyLoss(),
                'val_metric': metrics.accuracy,
                'in_dim': 500,
                'h_dim': 20,
                'out_dim': 2,
                'sigma': 0.5,
                'num_stg': 1,
                'lam': 0.2
                }
            },
        'oracle': {
            'base': Oracle,
            'true_groups': datasets.gauss_groups[1]
            },
        'oracle_cluster': {
            'base': OracleCluster,
            'true_groups': datasets.gauss_groups[1]
            },
        'random_forests': {
            'base': SKLearnModel,
            'model': RandomForests,
            'model_config': {
                'val_metric': metrics.sklearn_accuracy,
                'n_estimators': 100,
                'max_depth': 5,
                'n_top_trees': 10,
                'max_idx': 10,
                'threshold': 0.1
                }
            },
        'gbdt': {
            'base': SKLearnModel,
            'model': GBDT,
            'model_config': {
                'val_metric': metrics.sklearn_accuracy,
                'n_estimators': 15,
                'max_depth': 5,
                'n_top_trees': 10,
                'max_idx': 10,
                'threshold': 0.1
                }
            },
        }
    }


syn1_config = copy.deepcopy(default_config)

syn2_config = copy.deepcopy(default_config)
syn2_config['data_info']['data_config']['rule'] = 2
syn2_config['model_info']['oracle']['true_groups'] = datasets.gauss_groups[2]
syn2_config['model_info']['oracle_cluster']['true_groups'] = datasets.gauss_groups[2]

syn3_config = copy.deepcopy(default_config)
syn3_config['data_info']['data_config']['rule'] = 3
syn3_config['model_info']['oracle']['true_groups'] = datasets.gauss_groups[3]
syn3_config['model_info']['oracle_cluster']['true_groups'] = datasets.gauss_groups[3]

syn4_config = copy.deepcopy(default_config)
syn4_config['data_info']['data_config']['rule'] = 4
syn4_config['model_info']['oracle']['true_groups'] = datasets.gauss_groups[4]
syn4_config['model_info']['oracle_cluster']['true_groups'] = datasets.gauss_groups[4]