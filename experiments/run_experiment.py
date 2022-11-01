"""Run main experiments from command line."""

import argparse
import os
import os.path as osp

import numpy as np
import torch

from compfs.metrics import gsim, tpr_fdr
from experiments.experiment_configs import chem, metabric, mnist, syn

# Load in arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_no", type=int, default=1)
parser.add_argument(
    "--experiment",
    type=str,
    choices=[
        "syn1",
        "syn2",
        "syn3",
        "syn4",
        "chem1",
        "chem2",
        "chem3",
        "metabric",
        "mnist",
    ],
    default="syn1",
)
parser.add_argument(
    "--model",
    type=str,
    choices=[
        "compfs",
        "compfs1",
        "oracle",
        "oracle_cluster",
        "lasso",
        "stg",
        "ensemble_stg",
        "random_forests",
        "gbdt",
    ],
    default="compfs",
)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--load_model", type=bool, default=False)
args = parser.parse_args()


def set_seed(x):
    # Set a consistent seed, so we can run across different runs.
    x *= 10000
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


config_list = {
    "syn1": syn.syn1_config,
    "syn2": syn.syn2_config,
    "syn3": syn.syn3_config,
    "syn4": syn.syn4_config,
    "chem1": chem.chem1_config,
    "chem2": chem.chem2_config,
    "chem3": chem.chem3_config,
    "metabric": metabric.metabric_config,
    "mnist": mnist.mnist_config,
}


def set_device(x):
    # Sets the device.
    if x == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + x if torch.cuda.is_available() else "cpu")
    print("\nDevice: {}".format(device))
    return device


if __name__ == "__main__":

    # Get configs for this experiment.
    data_config = config_list[args.experiment]["data_info"]
    model_config = config_list[args.experiment]["model_info"][args.model]

    # Set the seed for consistency across runs.
    set_seed(args.experiment_no)

    # Set device.
    model_config["device"] = set_device(args.device)

    # Setup datasets.
    train_data = data_config["dataset"](data_config["data_config"])
    data_config["data_config"]["train"] = False
    val_data = data_config["dataset"](data_config["data_config"])

    # Make folder for results.
    folder = osp.join(
        "results",
        args.experiment,
        args.model,
        "run_" + str(args.experiment_no),
    )
    if not osp.exists(folder):
        os.makedirs(folder)

    # Setup model and optimizer, loading if necessary.
    model = model_config["base"](model_config)
    if args.load_model:
        model.load(folder)

    # Train model.
    model.train(train_data, val_data)

    # Save training stats and model if necessary.
    model.save_training_stats(folder)
    if args.save_model:
        model.save(folder)

    # Evaluate the model.
    model.save_evaluation_info(val_data, folder)

    # Get group similarity and group structure.
    tpr, fdr = tpr_fdr(train_data.groups, model.get_groups())
    group_sim, ntrue, npredicted = gsim(train_data.groups, model.get_groups())
    np.save(osp.join(folder, "gsim.npy"), np.array([group_sim]))
    np.save(osp.join(folder, "true_positive_rate.npy"), np.array([tpr]))
    np.save(osp.join(folder, "false_discovery_rate.npy"), np.array([fdr]))
    np.save(osp.join(folder, "true_group_count.npy"), np.array([ntrue]))
    np.save(osp.join(folder, "predicted_group_count.npy"), np.array([npredicted]))
    print("\n\nGroup Structure:")
    print(
        "Group Similarity: {:.3f}, True Positive Rate: {:.3f}%, False Discovery Rate: {:.3f}%".format(
            group_sim,
            tpr,
            fdr,
        ),
    )
    print(
        "Number of True Groups: {}, Number of Predicted Groups: {}".format(
            ntrue,
            npredicted,
        ),
    )

    # Give selected features and save the groups.
    print("\n\nSelected Features:")
    learnt_groups = model.get_groups()
    for i in range(len(learnt_groups)):
        np.save(
            osp.join(folder, "group_" + str(i + 1) + "_feature_indices.npy"),
            learnt_groups[i],
        )
        group = [train_data.feature_names[j] for j in learnt_groups[i]]
        print("Group: {}, Features: {}".format(i + 1, group))
