"""Gaussian image metrics shared by posterior predictions and saved products."""
from __future__ import annotations

import numpy as np


def gaussian_image_metrics(data, prediction, noise, *, mask=None, likelihood_scale=1.0):
    """Evaluate an image using the same Gaussian normalization as the sampler.

    ``chi2`` is unscaled. ``log_likelihood`` includes ``likelihood_scale``.
    A mask selects likelihood pixels, not the source reconstruction region.
    """
    data, prediction, noise = (np.asarray(x) for x in (data, prediction, noise))
    if data.shape != prediction.shape or data.shape != noise.shape:
        raise ValueError("Data, prediction, and noise must have the same shape.")
    valid = np.isfinite(data) & np.isfinite(noise) & (noise > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.all(np.isfinite(prediction[valid])):
        raise ValueError("Model prediction is non-finite on likelihood pixels.")
    n_pixels = int(np.sum(valid))
    if not n_pixels:
        raise ValueError("No valid likelihood pixels.")
    residual = (data[valid] - prediction[valid]) / noise[valid]
    chi2 = float(np.sum(residual ** 2))
    normalization = float(np.sum(np.log(2 * np.pi * noise[valid] ** 2)))
    return {
        "chi2": chi2,
        "n_data_pixels": n_pixels,
        "log_likelihood": float(-0.5 * likelihood_scale * (chi2 + normalization)),
    }
