import numpy as np


def zero_mask_noise(x, zero_prob=0.25, dtype=None):
    """
    Randomly 'erase' the data by setting them to 0
    with bernoulli probability `zero_prob`.
    """
    if dtype is None:
        dtype = x.dtype
    zero_mask = np.random.choice([0, 1], size=x.shape, p=[zero_prob, 1 - zero_prob])
    return (x * zero_mask).astype(dtype)
