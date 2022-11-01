"""Implementation of CompFS."""

# stdlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# third party
from compfs.utils import is_array_in_list


class FullyConnected(nn.Module):
    """
    Two hidden layer ReLU MLP, goes to hidden representation ONLY.

    Args:
        in_dim: the number of features
        h_dim: hidden width
    """

    def __init__(self, in_dim, h_dim):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Gate(nn.Module):
    """
    Gate used in the CompFS individual feature selectors.

    Has weights w, and then apply a sigmoid to get p.
    When training, sample from Bernoulli with parameters p, using relaxed Bernoulli
    to get m.
    When testing we apply a thresholding function of choice to p and a step function
    to get m.
    During training and testing the output of the gate is given by:
    gate(x) = m*x + (1-m)*x_bar,
    where x_bar is the feature-wise mean of the input.

    Args:
        in_dim: number of features
        threshold_func: function which turns p into m
        temp: the "temperature"/sharpness of the reparametereised Bernoulli sampling
    """

    def __init__(self, in_dim, threshold_func, temp):
        super(Gate, self).__init__()
        self.w = nn.Parameter(torch.normal(torch.zeros(in_dim), torch.ones(in_dim)))
        self.threshold_func = threshold_func
        self.temp = temp

    def forward(self, x, x_bar=0, test=False):
        if test:
            m = self.make_m()
            m = m.repeat(
                len(x),
                1,
            ).float()  # Repeat to make it the same size given the batch.
        else:
            p = torch.sigmoid(self.w).repeat(
                len(x),
                1,
            )  # Repeat to make it the same size given the batch.
            u = torch.rand(p.shape).to(p.device)
            # Reparameterization trick for Bernoulli.
            m = torch.sigmoid(
                (torch.log(p) - torch.log(1 - p) + torch.log(u) - torch.log(1 - u))
                / self.temp,
            )
        return m * x + (1 - m) * x_bar

    def make_m(self):
        return self.threshold_func(torch.sigmoid(self.w))


class SingleFeatureSelector(nn.Module):
    """
    An feature selector based on stochastic gates, given by mlp and Bernoulli gate.
    https://arxiv.org/abs/1810.04247

    Args:
        in_dim: number of features
        h_dim: hidden width of learner
        out_dim: the dimenion of the output
        threshold: threshold for gate
        temp: temperature of Bernoulli reparameterisation
    """

    def __init__(self, in_dim, h_dim, out_dim, threshold_func, temp):
        super(SingleFeatureSelector, self).__init__()
        self.to_hidden = FullyConnected(in_dim, h_dim)
        self.gate = Gate(in_dim, threshold_func, temp)
        self.fc_individual = nn.Linear(h_dim, out_dim)
        self.fc_aggregate = nn.Linear(h_dim, out_dim)

    def forward(self, x, x_bar=0, test=False):
        return self.to_hidden(self.gate(x, x_bar, test))

    def predict(self, x, x_bar):
        return self.fc_individual(self.forward(x, x_bar, test=True))

    def count_features(self):
        # Count how many features there are in this learner.
        return torch.sum(self.gate.make_m()).item()

    def get_group(self):
        # Give the features that this learner uses.
        return torch.where(self.gate.make_m())[0]

    def get_importance(self):
        # Frobenius norm of final weight matrix, to compare to other learners.
        return torch.sqrt(torch.sum(self.fc_aggregate.weight**2)).item()


class CompFS(nn.Module):
    """
    The CompFS model.

    Has a set of weak learners, and given each p vector we punish them overlapping, i.e. p_i dot p_j
    and also having lots of features torch.sum(p)**2. We can control how much with beta_s (small groups)
    and beta_d (different groups).

    Args (in a config_dict):
        nlearners: how many groups we want
        in_dim: dimension of problem
        h_dim: hidden width of mlps
        out_dim: dimension of output
        threshold: function to determine a feature is included
        temp: temperature of the Bernoulli reparameterisation
    """

    def __init__(self, config_dict):
        super(CompFS, self).__init__()

        self.beta_s = config_dict["beta_s"]
        self.beta_s_decay = config_dict["beta_s_decay"]
        self.beta_d = config_dict["beta_d"]
        self.beta_d_decay = config_dict["beta_d_decay"]
        self.loss_func = config_dict["loss_func"]
        self.x_bar = 0
        self.nfeatures = config_dict["in_dim"]
        self.nlearners = config_dict["nlearners"]
        h_dim = config_dict["h_dim"]
        out_dim = config_dict["out_dim"]
        threshold_func = config_dict["threshold_func"]
        temp = config_dict["temp"]
        self.learners = nn.ModuleList(
            [
                SingleFeatureSelector(
                    self.nfeatures,
                    h_dim,
                    out_dim,
                    threshold_func,
                    temp,
                )
                for _ in range(self.nlearners)
            ],
        )

    def forward(self, x):
        x_b = self.x_bar.repeat(len(x), 1).to(x.device)
        total = 0
        individuals = torch.tensor([]).to(x.device)
        for learner in self.learners:
            hidden = learner(x, x_b).unsqueeze(0)
            total += learner.fc_aggregate(hidden)
            individuals = torch.cat(
                [individuals, learner.fc_individual(hidden.detach())],
                dim=0,
            )
        out = torch.cat(
            [total, individuals],
            dim=0,
        )  # We want to train the ensemble, and the individual learners together.
        return out

    def predict(self, x):
        # Test the ensemble.
        x_b = self.x_bar.repeat(len(x), 1).to(x.device)
        out = 0
        for learner in self.learners:
            out += learner.fc_aggregate(learner(x, x_b, test=True))
        return out

    def preprocess(self, data):
        return data

    def get_loss(self, x, y):
        output = self.forward(x)
        loss = self.loss_func(output[0], y)
        for i in range(self.nlearners):
            loss += self.loss_func(output[i + 1], y)
            pi_i = torch.sigmoid(self.learners[i].gate.w)
            # Multiply by square root of number of features. So we punish more features, but not as quickly as linearly.
            loss += (
                self.beta_s
                * (torch.mean(pi_i) ** 2)
                * (self.nfeatures**0.5)
                / (self.nlearners)
            )
            for j in range(i + 1, self.nlearners):
                pi_j = torch.sigmoid(self.learners[j].gate.w)
                loss += (
                    2
                    * self.beta_d
                    * torch.mean(pi_i * pi_j)
                    * (self.nfeatures**0.5)
                    / (self.nlearners * (self.nlearners - 1))
                )
        return loss

    def update_after_epoch(self):
        self.beta_d *= self.beta_d_decay
        self.beta_s *= self.beta_s_decay

    def count_features(self):
        # Return list of number of features in each group.
        out = []
        for learner in self.learners:
            out.append(learner.count_features())
        return out

    def get_overlap(self):
        # Count how many features overlap, and where they are.
        overlap = 0
        for learner in self.learners:
            overlap += learner.gate.make_m()
        overlap = overlap > 1
        noverlap = torch.sum(overlap).item()
        ids = torch.where(overlap)
        return noverlap, ids

    def get_groups(self):
        # Return a list of the groups as numpy arrays, which are not empty and unique.
        groups = []
        for learner in self.learners:
            g = learner.get_group().detach().cpu().numpy()
            if (len(g) != 0) and (not is_array_in_list(g, groups)):
                groups.append(g)
        return groups

    def set_threshold_func(self, new_func):
        # After training we can change how we threshold the scores of each learner. By giving
        # the ensemble a new thresholding function.
        for learner in self.learners:
            learner.gate.threshold_func = new_func

    def save_evaluation_info(self, x, y, folder, val_metric):
        output = self.predict(x)
        full_model_performance = val_metric(output, y)
        np.save(
            Path(folder) / "full_model_performance.npy",
            np.array([full_model_performance]),
        )
        print(
            "\n\nPerformance:\nFull Model Test Metric: {:.3f}".format(
                full_model_performance,
            ),
        )

        # print individual accuracies if using compfs
        for i in range(self.nlearners):
            output = self.learners[i].predict(
                x,
                self.x_bar.repeat(len(x), 1).to(x.device),
            )
            individual_performance = val_metric(output, y)
            np.save(
                Path(folder) / f"learner_{i + 1}_performance.npy",
                np.array([individual_performance]),
            )
            print(
                "Group: {}, Test Metric: {:.3f}".format(i + 1, individual_performance),
            )

        # print importances if using compfs
        print("\n\nImportances:")
        for i in range(self.nlearners):
            individual_importance = self.learners[i].get_importance()
            np.save(
                Path(folder) / f"learner_{i + 1}_importance.npy",
                np.array([individual_importance]),
            )
            print(f"Group: {i + 1}, Importance: {individual_importance}")
