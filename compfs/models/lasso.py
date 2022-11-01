"""Lasso for binary classification, adapted to work with the trainer and code for CompFS code."""

# stdlib
from pathlib import Path

# third party
import numpy as np
import torch
import torch.nn as nn


class Lasso(nn.Module):
    """
    Lasso implementation https://en.wikipedia.org/wiki/Lasso_(statistics).

    Standard lasso for binary classification, adapted to work with the trainer and code for compfs
    at test time, if w.x + b < 0 return -1, else 1

    Args (as config_dict):
        in_dim: the dimension of x
        out_dim: the target dimension, is 1 for standard lasso
        threshold: the value where if |w| > threshold we say the feature is selected
    """

    def __init__(self, config_dict):
        super(Lasso, self).__init__()
        self.beta_s = config_dict["beta_s"]
        self.nlearners = 1
        self.threshold = config_dict["threshold"]
        self.loss_func = config_dict["loss_func"]
        self.fc = nn.Linear(config_dict["in_dim"], config_dict["out_dim"])

    def forward(self, x):
        return self.fc(x)

    def predict(self, x):
        w = self.fc.weight
        w = (torch.abs(w) > self.threshold) * w
        x = torch.matmul(x, w.transpose(0, 1)) + self.fc.bias
        return 2 * (x >= 0) - 1

    def preprocess(self, data):
        new_data = []
        for sample in data.data:
            sample = list(sample)
            sample[1] = 2 * sample[1].float() - 1
            sample[1] = sample[1].unsqueeze(0)
            sample = tuple(sample)
            new_data.append(sample)
        data.data = new_data
        return data

    def get_loss(self, x, y):
        output = self.forward(x)
        loss = self.loss_func(output, y)
        loss += self.beta_s * torch.sum(torch.abs((self.fc.weight)))
        return loss

    def update_after_epoch(self):
        pass

    def count_features(self):
        # Return list of number of features in each group.
        w = self.fc.weight.reshape(-1)
        out = torch.sum(torch.abs(w) > self.threshold).item()
        return out

    def get_overlap(self):
        # Count how many features overlap, and where they are.
        return 0, 0

    def get_groups(self):
        # Return a list of the groups as numpy arrays.
        groups = []
        w = self.fc.weight.reshape(-1)
        group = torch.where(torch.abs(w) > self.threshold)[0]
        group = group.detach().cpu().numpy()
        if len(group) != 0:
            groups.append(group)
        return groups

    def set_threshold_func(self, new_func):
        # After training we can change how we threshold the scores of each learner. By giving
        # the ensemble a new thresholding function.
        print("Not implemented for LASSO")

    def save_evaluation_info(self, x, y, folder, val_metric):
        folder = Path(folder)
        output = self.predict(x)
        full_model_performance = val_metric(output, y)
        np.save(
            folder / "full_model_performance.npy",
            np.array([full_model_performance]),
        )
        print(
            "\n\nPerformance:\nFull Model Test Metric: {:.3f}".format(
                full_model_performance,
            ),
        )
