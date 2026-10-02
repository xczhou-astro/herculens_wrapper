"""Recognition of explicitly named distribution specifications."""
from collections.abc import Mapping
import numpy as np


def is_distribution_prior(value):
    return isinstance(value, Mapping) and "distribution" in value


def is_sampled_prior(value):
    return isinstance(value, (list, tuple)) or is_distribution_prior(value)


def sampled_prior_size(value):
    if is_distribution_prior(value):
        shape = np.broadcast_shapes(*(np.shape(v) for k, v in value.items() if k != "distribution"))
        return int(np.prod(shape)) if shape else 1
    return 1
