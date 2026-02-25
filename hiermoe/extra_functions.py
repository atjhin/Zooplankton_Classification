import numpy as np
import torch
import random

def set_seed(seed: int = 666):

    """
    Sets the random seed across Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int, optional): The seed value to use. Defaults to 666.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False