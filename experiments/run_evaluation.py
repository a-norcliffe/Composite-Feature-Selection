"""Run main evaluations from command line to make table."""

import argparse
import os
import os.path as osp

import numpy as np

# Load in arguments.
parser = argparse.ArgumentParser()
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
args = parser.parse_args()


if __name__ == "__main__":

    # specific to data choice
    folder = osp.join("results", args.experiment, args.model)
    folders = os.listdir(folder)

    gsim = []
    tpr = []
    fdr = []
    ngroups = []
    performance = []
    ntruegroups = []

    for f in folders:
        load_folder = osp.join(folder, f)
        gsim.append(np.load(osp.join(load_folder, "gsim.npy")))
        tpr.append(np.load(osp.join(load_folder, "true_positive_rate.npy")))
        fdr.append(np.load(osp.join(load_folder, "false_discovery_rate.npy")))
        ngroups.append(np.load(osp.join(load_folder, "predicted_group_count.npy")))
        performance.append(np.load(osp.join(load_folder, "full_model_performance.npy")))
        ntruegroups.append(np.load(osp.join(load_folder, "true_group_count.npy")))

    gsim = np.array(gsim)
    tpr = np.array(tpr)
    fdr = np.array(fdr)
    ngroups = np.array(ngroups)
    performance = np.array(performance)
    ntruegroups = np.array(ntruegroups)

    print("{} performance on {}:".format(args.model, args.experiment))
    print("Gim: {:.5f} +- {:.5f}".format(np.mean(gsim), np.std(gsim)))
    print("TPR: {:.5f} +- {:.5f}".format(np.mean(tpr), np.std(tpr)))
    print("FDR: {:.5f} +- {:.5f}".format(np.mean(fdr), np.std(fdr)))
    print("ngroups: {:.5f} +- {:.5f}".format(np.mean(ngroups), np.std(ngroups)))
    print(
        "Performance: {:.5f} +- {:.5f}".format(
            np.mean(performance),
            np.std(performance),
        ),
    )
