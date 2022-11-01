"""Implementation of Ensemble STG."""

# stdlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# third party
from compfs.utils import is_array_in_list


class FeatureSelector(nn.Module):
    """
    Individual gate for STG.
    """

    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(
            0.1
            * torch.randn(
                input_dim,
            ),
            requires_grad=True,
        )
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x)
        z = self.mu + self.sigma * noise
        out = x * F.hardsigmoid(z + 0.5)
        return out

    def test(self, x):
        return x * F.hardsigmoid(self.mu + 0.5)

    def reg_loss(self):
        return 1 + torch.erf(self.mu + 0.5 / self.sigma)

    def get_gates(self):
        return F.hardsigmoid(self.mu + 0.5).detach().cpu().numpy()

    def get_features(self):
        return np.where(self.get_gates())[0]

    def count_features(self):
        return len(self.get_features())


class SingleSTG(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma):
        super(SingleSTG, self).__init__()
        self.gate = FeatureSelector(input_dim, sigma)
        self.to_latent = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.to_latent(self.gate(x))

    def test(self, x):
        return self.to_latent(self.gate.test(x))


class EnsembleSTG(nn.Module):
    """
    Ensemble STG implementation.

    An ensemble of STGs, which concatenate all their outputs to form one large latent
    representation. Can be used as single stg.

    Args (as dict):
        num_stg: number of stgs in ensemble
        lam: lambda parameter, makes number of features smaller
        loss_func: nn loss function
        in_dim: input dimension
        h_dim: hidden dim
        out_dim: output dimension
    """

    def __init__(self, config_dict):
        super(EnsembleSTG, self).__init__()
        self.num_stg = config_dict["num_stg"]
        self.lam = config_dict["lam"]
        self.loss_func = config_dict["loss_func"]
        self.stgs = nn.ModuleList(
            [
                SingleSTG(
                    config_dict["in_dim"],
                    config_dict["h_dim"],
                    config_dict["sigma"],
                )
                for i in range(self.num_stg)
            ],
        )
        self.fc1 = nn.Linear(self.num_stg * config_dict["h_dim"], config_dict["h_dim"])
        self.fc2 = nn.Linear(config_dict["h_dim"], config_dict["out_dim"])

    def forward(self, x):
        latent = torch.tensor([]).to(x.device)
        for s in self.stgs:
            latent = torch.cat([latent, s(x)], dim=-1)
        latent = F.relu(self.fc1(latent))
        return self.fc2(latent)

    def predict(self, x):
        latent = torch.tensor([]).to(x.device)
        for s in self.stgs:
            latent = torch.cat([latent, s.test(x)], dim=-1)
        latent = F.relu(self.fc1(latent))
        return self.fc2(latent)

    def preprocess(self, data):
        return data

    def get_loss(self, x, y):
        loss = 0
        for s in self.stgs:
            loss += s.gate.reg_loss()
        loss = self.lam * torch.sum(loss)
        output = self.forward(x)
        loss += self.loss_func(output, y)
        return loss

    def get_groups(self):
        # return a list of the groups as numpy arrays
        groups = []
        for s in self.stgs:
            g = s.gate.get_features()
            if (len(g) != 0) and (not is_array_in_list(g, groups)):
                groups.append(g)
        return groups

    def count_features(self):
        out = []
        for s in self.stgs:
            out.append(s.gate.count_features())
        return out

    def get_overlap(self):
        # Count how many features overlap
        overlap = 0
        for s in self.stgs:
            overlap += s.gate.get_gates() > 0.0
        overlap = overlap > 1
        noverlap = np.sum(overlap)
        return noverlap, np.nonzero(overlap)

    def update_after_epoch(self):
        pass

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
