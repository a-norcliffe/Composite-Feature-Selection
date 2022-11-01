"""Datasets for CompFS paper."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as visiondatasets
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset

from compfs.datasets import chem_featuriser
from compfs.datasets.network import download_if_needed


class NumpyDataset(Dataset):
    """
    Numpy dataset class, converts numpy data to torch dataset.

    This can be used to convert any numpy data into a CompFS
    dataset.

    Args:
        X_data: numpy array of X_data
        y_data: Numpy array of y_data
        classification: Bool tells the class whether to save y values as longs or floats
    """

    def __init__(self, X_data, y_data, classification=True):

        self.x_bar = torch.tensor(np.mean(X_data, axis=0)).float()
        self.num_data = X_data.shape[0]
        self.data = []
        for x_sample, y_sample in zip(X_data, y_data):
            x = torch.from_numpy(x_sample).float()
            if classification:
                y = torch.tensor(y_sample).long()
            else:
                y = torch.tensor(y_sample).float()
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_data

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0]
            self.x_bar = x_bar / self.num_data
            return self.x_bar


# Synthetic data.
class SyntheticGaussian(Dataset):
    """
    Synthetic Gaussian datasets.

    Have a decision rule whether we return 0 or 1,
    usually beating a threshold data drawn from standard normal.

    Args (as config_dict):
        nfeatures: number of features in the x vectors
        rule: the logit rule, numbers 1-4
        train: whether we are making a training set or not
    """

    def __init__(self, config_dict):
        nfeatures = config_dict["nfeatures"]
        rule = config_dict["rule"]
        train = config_dict["train"]
        num_data = 20000 if train else 200
        self.num_data = num_data
        self.data = []
        self.selection_rule = gauss_rules[rule]
        self.sampler = gauss_dists[rule]
        self.n0 = 0
        self.n1 = 0
        self.x_bar = 0
        self.groups = gauss_groups[rule]
        self.feature_names = list(np.arange(nfeatures))

        for _ in range(self.num_data):
            x = self.sampler(nfeatures)
            self.x_bar += x
            if self.selection_rule(x):
                y = torch.tensor(1).long()
                self.n1 += 1
            else:
                y = torch.tensor(0).long()
                self.n0 += 1

            self.data.append((x, y))
        self.x_bar /= self.num_data
        is_train_str = "\nTrain" if train else "Test"
        print(
            is_train_str
            + " Data Proportions:  0: {:.3f}, 1: {:.3f}".format(
                self.n0 / self.num_data,
                self.n1 / self.num_data,
            ),
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_data

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0]
            self.x_bar = x_bar / self.num_data
            return self.x_bar


def gauss_rule1(x):
    return (x[0] > 0.55) or (x[1] > 0.55)


def gauss_rule2(x):
    return (x[0] * x[1] > 0.30) or (x[2] * x[3] > 0.30)


def gauss_rule3(x):
    return (x[0] * x[1] > 0.30) or (x[0] * x[2] > 0.30)


def gauss_rule4(x):
    return (x[0] * x[3] > 0.30) or (x[6] * x[9] > 0.30)


def gauss_regular_sample(nfeatures):
    return torch.randn(nfeatures)


def gauss_correlated_sample(nfeatures):
    n_correlated = 3
    correlation_value = 0.99
    cov_correlated = (1 - correlation_value) * torch.eye(n_correlated) + torch.full(
        [n_correlated, n_correlated],
        correlation_value,
    )
    dist_correlated = MultivariateNormal(
        loc=torch.full([n_correlated], 0.0),
        covariance_matrix=cov_correlated,
    )
    noise = torch.randn((nfeatures - 4 * n_correlated))
    return torch.cat(
        [
            dist_correlated.sample(),
            dist_correlated.sample(),
            dist_correlated.sample(),
            dist_correlated.sample(),
            noise,
        ],
        dim=-1,
    )


gauss_rules = {1: gauss_rule1, 2: gauss_rule2, 3: gauss_rule3, 4: gauss_rule4}
gauss_dists = {
    1: gauss_regular_sample,
    2: gauss_regular_sample,
    3: gauss_regular_sample,
    4: gauss_correlated_sample,
}
gauss_groups = {
    1: [np.array([0]), np.array([1])],
    2: [np.array([0, 1]), np.array([2, 3])],
    3: [np.array([0, 1]), np.array([0, 2])],
    4: [np.array([0, 3]), np.array([6, 9])],
}


# Chemistry data.
class ChemistryBinding(Dataset):
    """
    Chemistry Dataset. Rules taken from:
    https://github.com/google-research/graph-attribution/raw/main/data/all_16_logics_train_and_test.zip

    The rules do not equate to the experiments from the paper, the mapping is:
    {
        experiment 1: rule 4,
        experiment 2: rule 10,
        experiment 3: rule 13
    }.

    Args (as config_dict):
        rule: the logit rule, numbers {4, 10, 13}
        train: whether we are making a training set or not
    """

    def __init__(self, config_dict):
        download_url = "https://github.com/google-research/graph-attribution/raw/main/data/all_16_logics_train_and_test.zip"

        rule = config_dict["rule"]
        train = config_dict["train"]
        folder = Path(__file__).parent / "chem_data"
        start_file_name = f"logic_{rule}_"
        end_file_name = "_train.npy" if train else "_test.npy"

        try:
            download_if_needed(
                download_path=folder / "data.zip",
                http_url=download_url,
                unarchive=True,
                unarchive_folder=folder,
            )
            x_data = np.load(folder / f"{start_file_name}X{end_file_name}")
            y_data = np.load(folder / f"{start_file_name}Y{end_file_name}")
        except FileNotFoundError:
            chem_featuriser.make_chem_data(rule)
            x_data = np.load(folder / f"{start_file_name}X{end_file_name}")
            y_data = np.load(folder / f"{start_file_name}Y{end_file_name}")

        self.n0 = 0
        self.n1 = 0
        self.num_data = x_data.shape[0]
        self.x_bar = torch.tensor(np.mean(x_data, axis=0)).float()
        self.groups = chem_data_groups[rule]
        self.feature_names = list(chem_featuriser.functional_groups_smarts.keys())

        self.data = []
        for x_sample, y_sample in zip(x_data, y_data):
            x = torch.from_numpy(x_sample).float()
            if y_sample:
                self.n1 += 1
                y = torch.tensor(1).long()
            else:
                y = torch.tensor(0).long()
                self.n0 += 1
            self.data.append((x, y))
        is_train_str = "\nTrain" if train else "Test"
        print(
            is_train_str
            + " Data Proportions:  0: {:.3f}, 1: {:.3f}".format(
                self.n0 / self.num_data,
                self.n1 / self.num_data,
            ),
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_data

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0]
            self.x_bar = x_bar / self.num_data
            return self.x_bar


chem_data_groups = {
    4: [np.array([40]), np.array([1])],  # logic_4 = ether OR NOT alkyne
    10: [
        np.array([56, 18]),
        np.array([40]),
    ],  # logic_10 = (primary amine AND NOT ether) OR (NOT benzene AND NOT ether)
    13: [np.array([18, 29]), np.array([1, 40])],
}  # logic_13 = (benzene AND NOT carbonyl) OR (alkyne AND NOT ether)


# Metabric Data.
class Metabric(Dataset):
    """
    Metabric Breast cancer data. Data taken from:
    https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric

    The rules are the type of y data we can use, we use PR status because it has even class balance.
    Choices are:
    {
        1: 'pr_status',
        2: 'er_status',
        3: 'er_status_measured_by_ihc',
        4: 'her2_status',
        5: 'her2_status_measured_by_snp6'
    }.

    Args (as config_dict):
        rule: the y type we want to carry out classification on, see names above
        train: whether we are making a training set or not
    """

    def __init__(self, config_dict):
        rule = config_dict["rule"]
        train = config_dict["train"]
        folder = Path(__file__).parent / "metabric_data/"

        try:
            raw_data = pd.read_csv(
                folder / "METABRIC_RNA_Mutation.csv",
                low_memory=False,
            )
        except FileNotFoundError:
            print(
                f"Data not found, download at: https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric and place in {folder.resolve()}",
            )
            raise

        self.num_data = 0
        self.n0 = 0
        self.n1 = 0
        self.data = []
        self.x_bar = 0
        self.groups = []
        self.feature_names = list(raw_data.columns[31:520])

        x_data = np.array(pd.DataFrame(raw_data.iloc[:, 31:520]))
        y_data = pd.DataFrame(raw_data[metabric_info[rule]["name"]])
        y_data = np.array(
            y_data[metabric_info[rule]["name"]].apply(
                metabric_info[rule]["preprocess"],
            ),
        )
        sample_ids = np.arange(len(x_data))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(sample_ids)
        train_ids = sample_ids[: int(0.8 * len(sample_ids))]
        test_ids = sample_ids[int(0.8 * len(sample_ids)) :]

        if train:
            x_data = x_data[train_ids]
            y_data = y_data[train_ids]
        else:
            x_data = x_data[test_ids]
            y_data = y_data[test_ids]

        for x_sample, y_sample in zip(x_data, y_data):
            if y_sample != -1:
                if y_sample:
                    self.n1 += 1
                    y = torch.tensor(1).long()
                else:
                    y = torch.tensor(0).long()
                    self.n0 += 1
                x = torch.tensor(x_sample).float()
                self.data.append((x, y))
                self.num_data += 1
                self.x_bar += x
        self.x_bar /= self.num_data
        is_train_str = "\nTrain" if train else "Test"
        print(
            is_train_str
            + " Data Proportions:  0: {:.3f}, 1: {:.3f}".format(
                self.n0 / self.num_data,
                self.n1 / self.num_data,
            ),
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_data

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0]
            self.x_bar = x_bar / self.num_data
            return self.x_bar


# These all have slightly different names in the csv file, so we preprocess with these rules.
def metabric_preprocess(positive_string, negative_string):
    def preprocess(x):
        x = str(x)
        if x == negative_string:
            return 0
        elif x == positive_string:
            return 1
        else:
            return -1

    return preprocess


metabric_info = {
    1: {"name": "pr_status", "preprocess": metabric_preprocess("Positive", "Negative")},
    2: {"name": "er_status", "preprocess": metabric_preprocess("Positive", "Negative")},
    3: {
        "name": "er_status_measured_by_ihc",
        "preprocess": metabric_preprocess("Positve", "Negative"),
    },
    4: {
        "name": "her2_status",
        "preprocess": metabric_preprocess("Positive", "Negative"),
    },
    5: {
        "name": "her2_status_measured_by_snp6",
        "preprocess": metabric_preprocess("GAIN", "NEUTRAL"),
    },
}


# MNIST data.
class MNIST(Dataset):
    """
    MNIST data with flattened vectors and x_bar included.

    Args (as config_dict):
        train: whether we are making a training set or not
    """

    def __init__(self, config_dict):
        data = visiondatasets.MNIST(
            root=Path(__file__).parent,
            train=config_dict["train"],
            transform=transforms.transforms.ToTensor(),
            download=True,
        )
        self.num_data = len(data)
        self.data = []
        self.x_bar = 0
        self.groups = []
        self.feature_names = list(np.arange(28 * 28))

        for x, y in data:
            x = x.flatten()
            self.x_bar += x

            self.data.append((x, y))
        self.x_bar /= self.num_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_data

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0] / self.num_data
            self.x_bar = x_bar
            return self.x_bar
