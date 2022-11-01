import shutil
import subprocess
from pathlib import Path

import pytest

working_dir = Path(__file__).parent.parent


def experiment_helper(
    experiment_name: str,
    model_name: str,
    experiment_no: str,
) -> None:
    try:
        shutil.rmtree(working_dir / "results")
    except BaseException:
        pass

    subprocess.check_output(
        [
            "python",
            "-m",
            "experiments.run_experiment",
            "--experiment_no",
            experiment_no,
            "--experiment",
            experiment_name,
            "--model",
            model_name,
        ],
        cwd=working_dir,
    )
    subprocess.check_output(
        [
            "python",
            "-m",
            "experiments.run_evaluation",
            "--experiment",
            experiment_name,
            "--model",
            model_name,
        ],
        cwd=working_dir,
    )

    assert (working_dir / "results" / experiment_name).exists()
    assert (working_dir / "results" / experiment_name / model_name).exists()
    assert (
        working_dir / "results" / experiment_name / model_name / f"run_{experiment_no}"
    ).exists()

    for resfile in ["gsim.npy", "false_discovery_rate.npy", "true_positive_rate.npy"]:
        assert (
            working_dir
            / "results"
            / experiment_name
            / model_name
            / f"run_{experiment_no}"
            / resfile
        ).exists()


@pytest.mark.parametrize("experiment_name", ["syn1", "syn2", "syn3", "syn4"])
@pytest.mark.parametrize("model_name", ["compfs1", "lasso"])
def test_syn(experiment_name: str, model_name: str) -> None:
    experiment_no = "1"

    experiment_helper(
        experiment_name=experiment_name,
        model_name=model_name,
        experiment_no=experiment_no,
    )


@pytest.mark.parametrize("experiment_name", ["chem1", "chem2", "chem3"])
@pytest.mark.parametrize("model_name", ["compfs1", "lasso"])
def test_chem(experiment_name: str, model_name: str) -> None:
    experiment_no = "1"

    experiment_helper(
        experiment_name=experiment_name,
        model_name=model_name,
        experiment_no=experiment_no,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["compfs", "stg"])
def test_mnist(model_name: str) -> None:
    experiment_no = "1"
    experiment_name = "mnist"

    experiment_helper(
        experiment_name=experiment_name,
        model_name=model_name,
        experiment_no=experiment_no,
    )
