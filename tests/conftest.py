import random

import numpy as np
import torch


def enable_reproducible_results(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


enable_reproducible_results()
