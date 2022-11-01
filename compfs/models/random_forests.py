"""Implementations of Random Forests and GBDT."""

# stdlib
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

# third party
from compfs.utils import is_array_in_list


class RandomForests:
    """
    Implementation of group feature selection using random forests.

    Args (as dict):
        n_estimators: how many trees in the ensemble
        random_state: random seed
        max_depth: maximum depth of decision tree
        n_top_trees: number of trees to use for group feature selection
        max_idx: max number of features to consider
        threshold: feature importance threshold
    """

    def __init__(self, config_dict):
        super(RandomForests, self).__init__()
        self.rf = RandomForestClassifier(
            n_estimators=config_dict["n_estimators"],
            max_depth=config_dict["max_depth"],
        )
        self.n_top_trees = config_dict["n_top_trees"]
        self.max_idx = config_dict["max_idx"]
        self.threshold = config_dict["threshold"]

    def train(self, X_train, y_train, X_val, y_val):
        print("\nTraining Model")
        self.rf.fit(X_train, y_train)
        print("Training Complete")
        self.groups = []
        # Get performance of trees.
        perfs = self.test_trees(X_val, y_val)
        # Loop over best trees.
        for tree_idx in np.array(perfs).argsort()[::-1][: self.n_top_trees]:
            # Get feature importances for tree.
            tree = self.get_tree(self.rf.estimators_[tree_idx])
            sorted_idx = tree.feature_importances_.argsort()[::-1][: self.max_idx]
            tree_imp = tree.feature_importances_[sorted_idx]
            # Get set of key features.
            group = []
            for idx, imp in enumerate(tree_imp):
                if imp > self.threshold:
                    group.append(sorted_idx[idx])
            # Add group to groups.
            group = np.sort(np.array(group))
            if (not is_array_in_list(group, self.groups)) and (len(group) > 0):
                self.groups.append(group)

    def predict(self, x):
        return self.rf.predict_proba(x)[:, 1]

    def preprocess(self, train_data):
        X = []
        y = []
        for sample in train_data.data:
            X.append(sample[0].numpy())
            y.append(sample[1].numpy())
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        return X, y

    def get_tree(self, tree):
        return tree

    def get_tree_prediction(self, tree, x):
        return tree.predict_proba(x)[:, 1]

    def test_trees(self, X_val, y_val):
        perfs = []
        for tree in self.rf.estimators_:
            tree_preds = self.get_tree_prediction(self.get_tree(tree), X_val)
            perfs.append(roc_auc_score(y_val, tree_preds))
        return perfs

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


class GBDT(RandomForests):
    """
    Implementation of group feature selection using gradient boosted decision trees.

    Args (as dict):
        n_estimators: how many trees in the ensemble
        random_state: random seed
        max_depth: maximum depth of decision tree
        n_top_trees: number of trees to use for group feature selection
        max_idx: max number of features to consider
        threshold: feature importance threshold
    """

    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.rf = GradientBoostingClassifier(
            n_estimators=config_dict["n_estimators"],
            max_depth=config_dict["max_depth"],
        )

    def get_tree(self, tree):
        return tree[0]

    def get_tree_prediction(self, tree, x):
        return tree.predict(x)
