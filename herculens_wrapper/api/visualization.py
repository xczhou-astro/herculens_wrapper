"""Notebook-friendly visualizations for the public API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np

if TYPE_CHECKING:
    from .data import SingleBandData


PlotScale = Literal["linear", "log"]


def _in_jupyter_notebook() -> bool:
    """Return whether this code is executing in a Jupyter kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _extent(shape: tuple[int, int], pixel_scale: float) -> list[float]:
    ny, nx = shape
    return [
        -(nx // 2) * pixel_scale,
        (nx - nx // 2 - 1) * pixel_scale,
        -(ny // 2) * pixel_scale,
        (ny - ny // 2 - 1) * pixel_scale,
    ]


def _normalization(values: np.ndarray, scale: PlotScale, *, signed: bool):
    if scale == "linear":
        return None
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return None
    if signed and np.any(finite < 0):
        magnitude = np.abs(finite)
        nonzero = magnitude[magnitude > 0]
        if nonzero.size == 0:
            return None
        linthresh = max(float(np.percentile(nonzero, 5)), float(np.max(nonzero)) * 1e-6)
        return SymLogNorm(linthresh=linthresh, vmin=-float(np.max(nonzero)), vmax=float(np.max(nonzero)))
    positive = finite[finite > 0]
    if positive.size == 0:
        return None
    vmin = max(float(np.percentile(positive, 1)), float(np.max(positive)) * 1e-6)
    return LogNorm(vmin=vmin, vmax=float(np.max(positive)))


def plot_single_band_data(
    data: "SingleBandData",
    *,
    scale: PlotScale = "linear",
    residual_vis_max: float = 0.0,
    save_path: str | Path | None = None,
):
    """Plot image, noise, signal-to-noise, and PSF, returning ``(figure, axes)``.

    ``scale='log'`` uses logarithmic normalization for positive arrays and a
    symmetric-log normalization when an image contains negative values.
    """
    if scale not in ("linear", "log"):
        raise ValueError("scale must be either 'linear' or 'log'.")
    if residual_vis_max < 0:
        raise ValueError("residual_vis_max must be non-negative.")

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    image_extent = _extent(data.image.shape, data.pixel_scale)
    snr = np.divide(
        data.image, data.noise, out=np.zeros_like(data.image, dtype=float),
        where=np.isfinite(data.noise) & (data.noise > 0),
    )
    finite_snr = np.abs(snr[np.isfinite(snr)])
    snr_limit = max(float(np.percentile(finite_snr, 99.5)) if finite_snr.size else 1.0, 1.0)
    panels = (
        (axes[0, 0], data.image, "Image data", "twilight", True, image_extent, "Pixel flux"),
        (axes[0, 1], data.noise, "Noise map", "twilight", False, image_extent, "Pixel flux uncertainty"),
        (axes[1, 0], snr, "Signal-to-noise", "bwr", True, image_extent, "Signal-to-noise"),
        (axes[1, 1], data.psf, "PSF kernel", "twilight", False, _extent(data.psf.shape, data.pixel_scale), "PSF value"),
    )
    for axis, image, title, cmap, signed, extent, colorbar_label in panels:
        norm = _normalization(image, scale, signed=signed)
        kwargs = {"norm": norm}
        if title == "Signal-to-noise" and scale == "linear":
            kwargs.update(vmin=-snr_limit, vmax=snr_limit)
        rendered = axis.imshow(
            image, origin="lower", cmap=cmap, extent=extent, **kwargs,
        )
        if title in ("Image data", "Signal-to-noise"):
            if data.source_arc_mask is not None:
                axis.contour(data.source_arc_mask, levels=[0.5], colors="lime", linewidths=1.0, extent=extent)
            if data.contaminate_mask is not None:
                axis.contour(data.contaminate_mask, levels=[0.5], colors="orange", linewidths=1.2, linestyles="--", extent=extent)
        axis.set(title=title, xlabel="arcsec", ylabel="arcsec")
        if scale == "log" and title not in ("Signal-to-noise",):
            colorbar_label += " (log scale)"
        figure.colorbar(rendered, ax=axis, shrink=0.85, label=colorbar_label)

    if save_path is not None:
        output = Path(save_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200, bbox_inches="tight")
    # A notebook needs an explicit display for interactive backends.  Batch
    # scripts only create the requested output file and never open a figure.
    if _in_jupyter_notebook():
        plt.show()
    return figure, axes
