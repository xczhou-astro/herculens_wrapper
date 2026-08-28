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
    """Return the legacy-wrapper normalization for an image-like panel.

    ``signed`` remains part of the public internal call signature, but log
    plots intentionally follow ``visualizations._norm_from_plot_scale``:
    negative values are masked by ``LogNorm`` and do not turn the panel into
    a symmetric-log plot.
    """
    if scale == "linear":
        return None
    finite = np.asarray(values, dtype=float)
    positive = finite[np.isfinite(finite) & (finite > 0)]
    if positive.size == 0:
        return None
    vmin, vmax = float(np.percentile(positive, 1.0)), float(np.percentile(positive, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    vmax = max(vmax, vmin * 10.0)
    vmin = max(vmin, vmax * 1e-3)
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_single_band_data(
    data: "SingleBandData",
    *,
    scale: PlotScale = "linear",
    residual_vis_max: float = 0.0,
    save_path: str | Path | None = None,
):
    """Plot image, noise, signal-to-noise, and PSF, returning ``(figure, axes)``.

    ``scale='log'`` uses the same percentile normalization as the legacy
    wrapper, including its symmetric-log S/N panel.
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
        if title == "Signal-to-noise" and scale == "log":
            norm = SymLogNorm(
                linthresh=1.0, linscale=1.0, vmin=-snr_limit,
                vmax=snr_limit, base=10,
            )
        else:
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


def plot_multiband_data(
    bands: dict[str, "SingleBandData"],
    *,
    scale: PlotScale = "linear",
    residual_vis_max: float = 0.0,
    save_path: str | Path | None = None,
):
    """Plot one ``data / noise / SNR / PSF`` row for every observation band."""
    if scale not in ("linear", "log"):
        raise ValueError("scale must be either 'linear' or 'log'.")
    if residual_vis_max < 0:
        raise ValueError("residual_vis_max must be non-negative.")

    figure, axes = plt.subplots(
        len(bands), 4, figsize=(22, 5 * len(bands)), squeeze=False,
        constrained_layout=True,
    )
    titles = ("Image data", "Noise map", "Signal-to-noise", "PSF kernel")
    for row, (band_name, data) in enumerate(bands.items()):
        image_extent = _extent(data.image.shape, data.pixel_scale)
        snr = np.divide(
            data.image, data.noise, out=np.zeros_like(data.image, dtype=float),
            where=np.isfinite(data.noise) & (data.noise > 0),
        )
        finite_snr = np.abs(snr[np.isfinite(snr)])
        snr_limit = max(
            float(np.percentile(finite_snr, 99.5)) if finite_snr.size else 1.0,
            1.0,
        )
        panels = (
            (data.image, "twilight", True, image_extent, "Pixel flux"),
            (data.noise, "twilight", False, image_extent, "Pixel flux uncertainty"),
            (snr, "bwr", True, image_extent, "Signal-to-noise"),
            (data.psf, "twilight", False, _extent(data.psf.shape, data.pixel_scale), "PSF value"),
        )
        for column, (image, cmap, signed, extent, colorbar_label) in enumerate(panels):
            axis = axes[row, column]
            if column == 2 and scale == "log":
                norm = SymLogNorm(
                    linthresh=1.0, linscale=1.0, vmin=-snr_limit,
                    vmax=snr_limit, base=10,
                )
            else:
                norm = _normalization(image, scale, signed=signed)
            kwargs = {"norm": norm}
            if column == 2 and scale == "linear":
                kwargs.update(vmin=-snr_limit, vmax=snr_limit)
            rendered = axis.imshow(image, origin="lower", cmap=cmap, extent=extent, **kwargs)
            if column in (0, 2):
                if data.source_arc_mask is not None:
                    axis.contour(data.source_arc_mask, levels=[0.5], colors="lime", linewidths=1.0, extent=extent)
                if data.contaminate_mask is not None:
                    axis.contour(data.contaminate_mask, levels=[0.5], colors="orange", linewidths=1.2, linestyles="--", extent=extent)
            axis.set_xlabel("arcsec")
            axis.set_ylabel(f"{band_name}\narcsec" if column == 0 else "arcsec")
            if row == 0:
                axis.set_title(titles[column])
            if scale == "log" and column != 2:
                colorbar_label += " (log scale)"
            figure.colorbar(rendered, ax=axis, shrink=0.85, label=colorbar_label)

    if save_path is not None:
        output = Path(save_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200, bbox_inches="tight")
    if _in_jupyter_notebook():
        plt.show()
    return figure, axes
