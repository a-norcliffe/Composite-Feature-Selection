"""Base models for CompFS paper, act as trainers for models."""

# stdlib
from functools import reduce
from pathlib import Path

# third party
import numpy as np
import torch
import torch.optim as optim
from sklearn import cluster
from torch.utils.data import DataLoader


class BaseModel:
    def __init__(self):
        pass

    def train(self, train_data, val_data):
        pass

    def save(self, folder):
        pass

    def load(self, folder):
        pass

    def save_training_stats(self, folder):
        pass

    def get_groups(self):
        pass

    def save_evaluation_info(self, val_data, folder):
        pass


class TorchModel(BaseModel):
    def __init__(self, model_config):
        self.model_config = model_config
        self.device = model_config["device"]
        self.model = model_config["model"](model_config["model_config"]).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_config["model_config"]["lr"],
        )
        self.lr_decay = model_config["model_config"]["lr_decay"]
        self.batchsize = model_config["model_config"]["batchsize"]
        self.val_metric = model_config["model_config"]["val_metric"]
        self.epoch_loss_history = []
        self.epoch_n_features_history = []
        self.epoch_overlap_history = []
        self.epoch_val_history = []
        super().__init__()

    def train(self, train_data, val_data):
        self.model.x_bar = train_data.get_x_bar()
        train_data = self.model.preprocess(train_data)
        val_data = self.model.preprocess(val_data)
        batch_size = len(train_data) if self.batchsize == 0 else self.batchsize
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        print(
            "\n\nTraining for {} Epochs:\n".format(
                self.model_config["model_config"]["num_epochs"],
            ),
        )

        for epoch in range(1, self.model_config["model_config"]["num_epochs"] + 1):
            # Train an epoch.
            epoch_loss = self.train_epoch(train_loader)

            # Evaluate the model and save values.
            val = self.calculate_val_metric(val_loader)
            nfeatures = self.model.count_features()
            overlap = self.model.get_overlap()[0]
            self.epoch_loss_history.append(epoch_loss)
            self.epoch_val_history.append(val)
            self.epoch_n_features_history.append(nfeatures)
            self.epoch_overlap_history.append(overlap)

            # Print information.
            print(
                "Epoch: {}, Average Loss: {:.3f}, Val Metric: {:.1f}, nfeatures: {}, Overlap: {}".format(
                    epoch,
                    epoch_loss,
                    val,
                    nfeatures,
                    overlap,
                ),
            )

            # Update learning rate.
            for g in self.optimizer.param_groups:
                g["lr"] *= self.lr_decay

    def train_epoch(self, train_loader):
        avg_loss = 0
        for x, y in train_loader:
            x = x.view(x.shape[0], -1)  # Flatten to vectors.
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.get_loss(x, y)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        self.model.update_after_epoch()
        return avg_loss / len(train_loader)

    def calculate_val_metric(self, val_loader):
        metric = 0
        for x, y in val_loader:
            x = x.view(x.shape[0], -1)  # Flatten to vectors.
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model.predict(x)
            metric += self.val_metric(out, y)
        return metric / len(val_loader)

    def save(self, folder):
        folder = Path(folder)
        print("\nSaving Model")
        torch.save(self.model.state_dict(), folder / "trained_model.pth")
        print("\nSaving Optimizer")
        torch.save(self.optimizer.state_dict(), folder / "optimizer.pth")

    def load(self, folder):
        folder = Path(folder)
        print("\nLoading Model")
        self.model.load_state_dict(torch.load(folder / "trained_model.pth"))
        print("\nLoading Optimizer")
        self.optimizer.load_state_dict(torch.load(folder / "optimizer.pth"))

    def save_training_stats(self, folder):
        folder = Path(folder)
        # Saves training stats from the trainer.
        np.save(
            folder / "epoch_loss_history.npy",
            np.array(self.epoch_loss_history),
        )
        np.save(
            folder / "epoch_val_history.npy",
            np.array(self.epoch_val_history),
        )
        np.save(
            folder / "epoch_overlap_history.npy",
            np.array(self.epoch_overlap_history),
        )
        np.save(
            folder / "epoch_n_features_history.npy",
            np.array(self.epoch_n_features_history),
        )

    def get_groups(self):
        return self.model.get_groups()

    def save_evaluation_info(self, val_data, folder):
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        for x, y in val_loader:
            x = x.view(x.shape[0], -1)  # flatten the vectors
            x = x.to(self.device)
            y = y.to(self.device)
        self.model.save_evaluation_info(x, y, folder, self.val_metric)


class SKLearnModel(BaseModel):
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = model_config["model"](model_config["model_config"])
        self.val_metric = model_config["model_config"]["val_metric"]

    def train(self, train_data, val_data):
        X_train, y_train = self.model.preprocess(train_data)
        X_val, y_val = self.model.preprocess(val_data)
        self.model.train(X_train, y_train, X_val, y_val)

    def get_groups(self):
        return self.model.groups

    def save_evaluation_info(self, val_data, folder):
        X_val, y_val = self.model.preprocess(val_data)
        self.model.save_evaluation_info(X_val, y_val, folder, self.val_metric)


class Oracle(BaseModel):
    def __init__(self, config_dict):
        true_groups = config_dict["true_groups"]
        self.features = np.unique(reduce(np.union1d, true_groups))

    def get_groups(self):
        return [self.features]

    def save_evaluation_info(self, val_data, folder):
        folder = Path(folder)
        full_model_performance = -1.0
        np.save(
            folder / "full_model_performance.npy",
            np.array([full_model_performance]),
        )
        print(
            "\n\nPerformance:\nFull Model Test Metric: {:.3f}".format(
                full_model_performance,
            ),
        )


class OracleCluster(BaseModel):
    def __init__(self, config_dict):
        true_groups = config_dict["true_groups"]
        self.features = np.unique(reduce(np.union1d, true_groups))
        self.num_clusters = len(true_groups)
        self.agglo = cluster.FeatureAgglomeration(n_clusters=self.num_clusters)

    def train(self, train_data, val_data):
        X_train = []
        for sample in train_data.data:
            X_train.append(sample[0].numpy())
        X_train = np.stack(X_train, axis=0)
        self.agglo.fit(X_train[:, self.features])

    def get_groups(self):
        groups = [[] for _ in range(self.num_clusters)]
        for idx, l in enumerate(self.agglo.labels_):
            groups[l].append(self.features[idx])
        return groups

    def save_evaluation_info(self, val_data, folder):
        folder = Path(folder)
        full_model_performance = -1.0
        np.save(
            folder / "full_model_performance.npy",
            np.array([full_model_performance]),
        )
        print(
            "\n\nPerformance:\nFull Model Test Metric: {:.3f}".format(
                full_model_performance,
            ),
        )
