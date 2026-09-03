"""Post-processing utilities for API modelling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_PROFILE_PARAMETERS = {
    "SERSIC_ELLIPSE": ("amp", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"),
    "GAUSSIAN": ("amp", "sigma", "center_x", "center_y"),
}


def _read_truth(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Truth JSON does not exist: {path}")
    with path.open() as stream:
        return json.load(stream)


def _source_geometry(source_path: Path, source_grid_scale: float | None):
    """Return source-cell centres and areas from uniform or RTU source FITS."""
    from astropy.io import fits

    with fits.open(source_path, memmap=False) as hdul:
        pixels = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
        if "X_CORNERS" in hdul and "Y_CORNERS" in hdul:
            xcorn, ycorn = np.asarray(hdul["X_CORNERS"].data), np.asarray(hdul["Y_CORNERS"].data)
            x = 0.25 * (xcorn[:-1, :-1] + xcorn[1:, :-1] + xcorn[:-1, 1:] + xcorn[1:, 1:])
            y = 0.25 * (ycorn[:-1, :-1] + ycorn[1:, :-1] + ycorn[:-1, 1:] + ycorn[1:, 1:])
            # Shoelace area of each quadrilateral RTU cell.
            area = 0.5 * np.abs(
                xcorn[:-1, :-1] * ycorn[1:, :-1] + xcorn[1:, :-1] * ycorn[1:, 1:]
                + xcorn[1:, 1:] * ycorn[:-1, 1:] + xcorn[:-1, 1:] * ycorn[:-1, :-1]
                - ycorn[:-1, :-1] * xcorn[1:, :-1] - ycorn[1:, :-1] * xcorn[1:, 1:]
                - ycorn[1:, 1:] * xcorn[:-1, 1:] - ycorn[:-1, 1:] * xcorn[:-1, :-1]
            )
            return pixels, x, y, area, "ray_transformed_uniform"
    if source_grid_scale is None or source_grid_scale <= 0:
        raise ValueError(
            "Uniform source FITS has no physical coordinates; provide positive source_grid_scale (arcsec half-width)."
        )
    ny, nx = pixels.shape
    xx = np.linspace(-source_grid_scale, source_grid_scale, nx, endpoint=False) + source_grid_scale / nx
    yy = np.linspace(-source_grid_scale, source_grid_scale, ny, endpoint=False) + source_grid_scale / ny
    x, y = np.meshgrid(xx, yy)
    area = np.full_like(pixels, (2 * source_grid_scale / nx) * (2 * source_grid_scale / ny))
    return pixels, x, y, area, "uniform"


def _circular_crop_mask(
    x: np.ndarray,
    y: np.ndarray,
    radius: float | None,
    center: tuple[float, float],
) -> np.ndarray:
    """Select a physical source-plane circle, or the whole grid if unset."""
    if radius is None:
        return np.ones_like(x, dtype=bool)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("crop_radius must be a positive finite value in arcsec or None.")
    return np.hypot(x - center[0], y - center[1]) <= radius


def _light_model(profile: str, values: Mapping[str, float], x: np.ndarray, y: np.ndarray, image_pixel_scale: float):
    from herculens.LightModel.light_model import LightModel

    return np.asarray(LightModel([profile]).surface_brightness(x, y, [dict(values)]), dtype=float) * image_pixel_scale**2


def _initial_and_bounds(profile: str, pixels: np.ndarray, x: np.ndarray, y: np.ndarray, area: np.ndarray, image_pixel_scale: float):
    params = _PROFILE_PARAMETERS[profile]
    brightness = np.clip(pixels / image_pixel_scale**2, 0, None)
    total = float(np.sum(brightness * area))
    norm = float(brightness.sum())
    cx = float((brightness * x).sum() / norm) if norm > 0 else float(np.median(x))
    cy = float((brightness * y).sum() / norm) if norm > 0 else float(np.median(y))
    extent = max(float(np.ptp(x)), float(np.ptp(y)), 1e-4)
    resolution = max(float(np.sqrt(np.nanmedian(area))), extent / max(pixels.shape))
    if profile == "SERSIC_ELLIPSE":
        initial = [max(float(np.nanmax(brightness)), 1e-12), 0.2 * extent, 1.0, 0.0, 0.0, cx, cy]
        lower = [0.0, resolution * 0.15, 0.3, -0.7, -0.7, float(x.min()), float(y.min())]
        upper = [np.inf, 2.0 * extent, 8.0, 0.7, 0.7, float(x.max()), float(y.max())]
    else:
        initial = [max(total, 1e-12), 0.2 * extent, cx, cy]
        lower = [0.0, resolution * 0.15, float(x.min()), float(y.min())]
        upper = [np.inf, 2.0 * extent, float(x.max()), float(y.max())]
    return params, np.asarray(initial), (np.asarray(lower), np.asarray(upper))


def _flux_weighted_center(
    pixels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray | None,
    image_pixel_scale: float,
) -> tuple[float, float]:
    """Return the positive-flux centroid of one reconstructed source plane."""
    brightness = np.clip(np.asarray(pixels, dtype=float) / image_pixel_scale**2, 0.0, None)
    weights = brightness if area is None else brightness * np.asarray(area, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    normalisation = float(weights.sum())
    if normalisation <= 0:
        return float(0.5 * (np.nanmin(x) + np.nanmax(x))), float(0.5 * (np.nanmin(y) + np.nanmax(y)))
    return float((weights * x).sum() / normalisation), float((weights * y).sum() / normalisation)


def _fit_analytic_source_image(
    profile: str,
    pixels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    image_pixel_scale: float,
    crop_radius: float | None,
    crop_center: tuple[float, float] | None = None,
) -> tuple[dict[str, float], float, tuple[float, float]]:
    """Fit one source image in physical coordinates, including its centre."""
    from scipy.optimize import least_squares

    if crop_center is None:
        crop_center = _flux_weighted_center(pixels, x, y, area, image_pixel_scale)
    crop_center = (float(crop_center[0]), float(crop_center[1]))
    fit_mask = _circular_crop_mask(x, y, crop_radius, crop_center)
    if not np.any(fit_mask):
        raise ValueError("crop_radius does not include any source pixels.")
    local_x, local_y, local_area, local_pixels = (
        value[fit_mask] for value in (x, y, area, pixels)
    )
    parameter_names, initial, bounds = _initial_and_bounds(
        profile, local_pixels, local_x, local_y, local_area, image_pixel_scale,
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        values = dict(zip(parameter_names, vector))
        return _light_model(
            profile, values, local_x.ravel(), local_y.ravel(), image_pixel_scale,
        ) - np.asarray(local_pixels, dtype=float).ravel()

    answer = least_squares(residual, initial, bounds=bounds, method="trf")
    return (
        {name: float(value) for name, value in zip(parameter_names, answer.x)},
        float(np.sqrt(np.mean(answer.fun ** 2))),
        crop_center,
    )


def _reconstruct_source_draws(samples: Mapping[str, Any], indices: np.ndarray, fallback: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct physical source pixels for selected HMC draws."""
    from herculens_wrapper.models import PowerSpectrum

    required = "pixels_wn_source_grid"
    if required not in samples:
        raise ValueError("HMC archive has no pixelated source latent site 'pixels_wn_source_grid'.")
    pwn = np.asarray(samples[required])
    ny, nx = pwn.shape[1:]
    k_values = PowerSpectrum.K_grid((ny, nx)).k

    def value_at(name: str, index: int):
        if name in samples:
            return np.asarray(samples[name])[index]
        if name in fallback and fallback[name] is not None:
            return fallback[name]
        raise ValueError(f"Cannot reconstruct source samples: missing {name!r} in HMC archive and kwargs_result.")

    nonlinear = "pow_lam_source_grid" in samples or "scale_lam_source_grid" in samples
    reconstructed = []
    for index in indices:
        params = {
            "pixels_wn_source_grid": value_at("pixels_wn_source_grid", index),
            "n_source_grid": value_at("n_source_grid", index),
            "sigma_source_grid": value_at("sigma_source_grid", index),
            "rho_source_grid": value_at("rho_source_grid", index),
        }
        if nonlinear:
            params["pow_lam_source_grid"] = value_at("pow_lam_source_grid", index)
            params["scale_lam_source_grid"] = value_at("scale_lam_source_grid", index)
        reconstructed.append(np.asarray(PowerSpectrum.pixels_from_params(
            params, "source_grid", k_values, positive=True, nonlinear_brightness=nonlinear,
        )))
    return np.asarray(reconstructed)


def _output_plot_path(save_path: str | Path | None, default_name: str) -> Path:
    """Resolve a plot file path, accepting either a directory or PNG path."""
    if save_path is None:
        output = Path(default_name)
    else:
        candidate = Path(save_path).expanduser()
        output = candidate if candidate.suffix else candidate / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _truth_map(
    truth: Mapping[str, Any] | str | Path | None,
    x: np.ndarray,
    y: np.ndarray,
    image_pixel_scale: float,
    coordinate_center: tuple[float, float] | None = None,
) -> tuple[np.ndarray | None, str | None, dict[str, float] | None]:
    """Evaluate a supported analytic truth model in an optional local frame."""
    truth_data = _read_truth(truth)
    if truth_data is None:
        return None, None, None
    source_truth = (truth_data.get("kwargs_source") or [{}])[0]
    for profile, names in _PROFILE_PARAMETERS.items():
        if all(name in source_truth for name in names):
            values = dict(source_truth)
            if coordinate_center is not None:
                values["center_x"] = float(values["center_x"]) - coordinate_center[0]
                values["center_y"] = float(values["center_y"]) - coordinate_center[1]
            return _light_model(profile, values, x, y, image_pixel_scale), profile, values
    raise ValueError("truth must contain a supported analytic kwargs_source component.")


def plot_pixelated_source_reconstruction(
    pixels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    crop_radius: float | None,
    truth: Mapping[str, Any] | str | Path | None,
    image_pixel_scale: float,
    fitted_profile: str | None = None,
    fitted_parameters: Mapping[str, float] | None = None,
    coordinate_center: tuple[float, float] | None = None,
    crop_center: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    """Plot the pixelated source, truth, and optional analytic construction.

    When analytic fitted parameters are supplied, the six panels are the
    pixelated median, truth, pixelated-minus-truth, fitted construction,
    fitted-minus-pixelated, and fitted-minus-truth.  This deliberately keeps
    the pre-existing pixelated-minus-truth diagnostic as the sixth comparison
    is added, rather than replacing it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    pixels = np.asarray(pixels, dtype=float)
    if coordinate_center is None:
        coordinate_center = _flux_weighted_center(pixels, x, y, None, image_pixel_scale)
    coordinate_center = (float(coordinate_center[0]), float(coordinate_center[1]))
    x_local, y_local = x - coordinate_center[0], y - coordinate_center[1]
    if crop_center is None:
        crop_center = coordinate_center
    local_crop_center = (
        float(crop_center[0]) - coordinate_center[0],
        float(crop_center[1]) - coordinate_center[1],
    )
    mask = _circular_crop_mask(x_local, y_local, crop_radius, local_crop_center)
    truth_pixels, _, truth_values = _truth_map(
        truth, x_local, y_local, image_pixel_scale, coordinate_center=coordinate_center,
    )
    output = _output_plot_path(save_path, "pixelated_source_reconstruction.png")
    extent = [float(x_local.min()), float(x_local.max()), float(y_local.min()), float(y_local.max())]

    def mark_reference_points(axis):
        """Draw the native source-plane origin and, when present, truth centre."""
        axis.axvline(0.0, color="0.7", lw=0.9, ls="--", zorder=4)
        axis.axhline(0.0, color="0.7", lw=0.9, ls="--", zorder=4)
        if truth_values is not None:
            axis.plot(
                float(truth_values["center_x"]), float(truth_values["center_y"]),
                marker="x", markersize=9, markeredgewidth=2.0, color="lime",
                zorder=5,
            )

    fitted_pixels = None
    if fitted_parameters is not None:
        if fitted_profile is None:
            raise ValueError("fitted_profile is required with fitted_parameters.")
        fitted_profile = str(fitted_profile).upper()
        if fitted_profile not in _PROFILE_PARAMETERS:
            raise ValueError(f"Unsupported fitted_profile {fitted_profile!r}.")
        missing = [name for name in _PROFILE_PARAMETERS[fitted_profile] if name not in fitted_parameters]
        if missing:
            raise ValueError(f"fitted_parameters is missing {missing} for {fitted_profile}.")
        # The median source is fitted in physical coordinates; translate the
        # fitted centre along with the grid for this source-centred display.
        local_fitted_parameters = dict(fitted_parameters)
        local_fitted_parameters["center_x"] = float(local_fitted_parameters["center_x"]) - coordinate_center[0]
        local_fitted_parameters["center_y"] = float(local_fitted_parameters["center_y"]) - coordinate_center[1]
        fitted_pixels = _light_model(
            fitted_profile, local_fitted_parameters, x_local, y_local, image_pixel_scale,
        )

    if fitted_pixels is None and truth_pixels is None:
        figure, axis = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        handle = axis.imshow(np.where(mask, pixels, np.nan), origin="lower", extent=extent, cmap="twilight")
        axis.set(title="Pixelated source (median)", xlabel="source-plane arcsec", ylabel="source-plane arcsec")
        mark_reference_points(axis)
        figure.colorbar(handle, ax=axis, label="pixel flux")
    elif fitted_pixels is None:
        residual = pixels - truth_pixels
        vmax = float(np.nanpercentile(np.abs(np.concatenate([pixels[mask], truth_pixels[mask]])), 99.5))
        rmax = max(float(np.nanpercentile(np.abs(residual[mask]), 99)), 1e-12)
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
        for axis, image, title in zip(
            axes[:2], (pixels, truth_pixels), ("Pixelated source (median)", "Truth")
        ):
            handle = axis.imshow(np.where(mask, image, np.nan), origin="lower", extent=extent, cmap="twilight", vmin=0, vmax=vmax)
            axis.set(title=title, xlabel="source-plane arcsec", ylabel="source-plane arcsec")
            mark_reference_points(axis)
            figure.colorbar(handle, ax=axis, label="pixel flux")
        handle = axes[2].imshow(np.where(mask, residual, np.nan), origin="lower", extent=extent, cmap="bwr", norm=TwoSlopeNorm(vcenter=0, vmin=-rmax, vmax=rmax))
        rmse = float(np.sqrt(np.mean(residual[mask] ** 2)))
        axes[2].set(title=f"Pixelated − Truth\nRMSE = {rmse:.4g}", xlabel="source-plane arcsec", ylabel="source-plane arcsec")
        mark_reference_points(axes[2])
        figure.colorbar(handle, ax=axes[2], label="pixel flux")
    else:
        brightness_images = [pixels, fitted_pixels]
        if truth_pixels is not None:
            brightness_images.append(truth_pixels)
        vmax = max(float(np.nanpercentile(np.abs(image[mask]), 99.5)) for image in brightness_images)
        vmax = max(vmax, 1e-30)
        pixelated_minus_truth = None if truth_pixels is None else pixels - truth_pixels
        fitted_minus_pixelated = fitted_pixels - pixels
        fitted_minus_truth = None if truth_pixels is None else fitted_pixels - truth_pixels
        residual_images = [fitted_minus_pixelated]
        if pixelated_minus_truth is not None:
            residual_images.extend((pixelated_minus_truth, fitted_minus_truth))
        rmax = max(float(np.nanpercentile(np.abs(image[mask]), 99)) for image in residual_images)
        rmax = max(rmax, 1e-12)
        figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

        def draw_brightness(axis, image, title):
            handle = axis.imshow(np.where(mask, image, np.nan), origin="lower", extent=extent, cmap="twilight", vmin=0, vmax=vmax)
            axis.set(title=title, xlabel="source-plane arcsec", ylabel="source-plane arcsec")
            mark_reference_points(axis)
            figure.colorbar(handle, ax=axis, label="pixel flux")

        def draw_residual(axis, image, title):
            if image is None:
                axis.set_axis_off()
                axis.set_title(f"{title}\n(no truth supplied)")
                return
            handle = axis.imshow(np.where(mask, image, np.nan), origin="lower", extent=extent, cmap="bwr", norm=TwoSlopeNorm(vcenter=0, vmin=-rmax, vmax=rmax))
            rmse = float(np.sqrt(np.mean(image[mask] ** 2)))
            axis.set(title=f"{title}\nRMSE = {rmse:.4g}", xlabel="source-plane arcsec", ylabel="source-plane arcsec")
            mark_reference_points(axis)
            figure.colorbar(handle, ax=axis, label="pixel flux")

        draw_brightness(axes[0, 0], pixels, "Pixelated source (median)")
        if truth_pixels is None:
            axes[0, 1].set_axis_off()
            axes[0, 1].set_title("Truth (not supplied)")
        else:
            draw_brightness(axes[0, 1], truth_pixels, "Truth")
        draw_residual(axes[0, 2], pixelated_minus_truth, "Pixelated − Truth")
        draw_brightness(axes[1, 0], fitted_pixels, f"Fitted source ({fitted_profile})")
        draw_residual(axes[1, 1], fitted_minus_pixelated, "Fitted − Pixelated median")
        draw_residual(axes[1, 2], fitted_minus_truth, "Fitted − Truth")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_pixelated_source_radial_profile(
    pixels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    crop_radius: float | None,
    truth: Mapping[str, Any] | str | Path | None,
    image_pixel_scale: float,
    coordinate_center: tuple[float, float] | None = None,
    fitted_profile: str | None = None,
    fitted_parameters: Mapping[str, float] | None = None,
    save_path: str | Path | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
) -> Path:
    """Plot radial surface brightness and its truth-relative residual."""
    import matplotlib.pyplot as plt

    pixels = np.asarray(pixels, dtype=float)
    if coordinate_center is None:
        coordinate_center = _flux_weighted_center(pixels, x, y, None, image_pixel_scale)
    coordinate_center = (float(coordinate_center[0]), float(coordinate_center[1]))
    x_local, y_local = x - coordinate_center[0], y - coordinate_center[1]
    truth_pixels, profile, _ = _truth_map(
        truth, x_local, y_local, image_pixel_scale, coordinate_center=coordinate_center,
    )
    fitted_pixels = None
    if fitted_parameters is not None:
        if fitted_profile is None:
            raise ValueError("fitted_profile is required with fitted_parameters.")
        fitted_profile = str(fitted_profile).upper()
        if fitted_profile not in _PROFILE_PARAMETERS:
            raise ValueError(f"Unsupported fitted_profile {fitted_profile!r}.")
        missing = [name for name in _PROFILE_PARAMETERS[fitted_profile] if name not in fitted_parameters]
        if missing:
            raise ValueError(f"fitted_parameters is missing {missing} for {fitted_profile}.")
        local_fitted_parameters = dict(fitted_parameters)
        local_fitted_parameters["center_x"] = float(local_fitted_parameters["center_x"]) - coordinate_center[0]
        local_fitted_parameters["center_y"] = float(local_fitted_parameters["center_y"]) - coordinate_center[1]
        fitted_pixels = _light_model(
            fitted_profile, local_fitted_parameters, x_local, y_local, image_pixel_scale,
        )
    radius = np.hypot(x_local, y_local)
    radial_limit = float(crop_radius) if crop_radius is not None else float(radius.max())
    if not np.isfinite(radial_limit) or radial_limit <= 0:
        raise ValueError("crop_radius must be a positive finite value in arcsec or None.")
    mask = radius <= radial_limit
    if not np.any(mask):
        raise ValueError("crop_radius does not include any source pixels.")
    n_bins = min(24, max(4, int(np.sqrt(mask.sum()))))
    edges = np.linspace(0, radial_limit, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    brightness = pixels / image_pixel_scale**2
    def annular_mean(image: np.ndarray, lo: float, hi: float) -> float:
        selected = image[(radius >= lo) & (radius < hi) & mask]
        return float(np.mean(selected)) if selected.size else np.nan

    radial_brightness = np.asarray([
        annular_mean(brightness, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
    ])
    output = _output_plot_path(save_path, "source_radial_profiles.png")
    figure = plt.figure(figsize=(7.5, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(3.0, 1.0))
    brightness_axis = figure.add_subplot(grid[0, 0])
    residual_axis = figure.add_subplot(grid[1, 0], sharex=brightness_axis)
    brightness_axis.plot(centres, radial_brightness, "o-", label="pixelated median")
    radial_fitted = None
    if fitted_pixels is not None:
        fitted_brightness = fitted_pixels / image_pixel_scale**2
        radial_fitted = np.asarray([
            annular_mean(fitted_brightness, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
        ])
        brightness_axis.plot(centres, radial_fitted, "-", label=f"fitted {fitted_profile.lower()}")
    if truth_pixels is not None:
        truth_brightness = truth_pixels / image_pixel_scale**2
        radial_truth = np.asarray([
            annular_mean(truth_brightness, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
        ])
        brightness_axis.plot(centres, radial_truth, "-", label=f"truth {profile.lower()}")
        valid = np.isfinite(radial_brightness) & np.isfinite(radial_truth) & (np.abs(radial_truth) > 0)
        relative_residual = np.full_like(radial_brightness, np.nan, dtype=float)
        np.divide(
            radial_brightness - radial_truth, radial_truth,
            out=relative_residual, where=valid,
        )
        residual_axis.axhline(0.0, color="black", lw=1.0, ls="--")
        residual_axis.plot(
            centres[valid], relative_residual[valid],
            "o-", color="tab:blue",
            label="pixelated − truth",
        )
        if radial_fitted is not None:
            fitted_valid = np.isfinite(radial_fitted) & np.isfinite(radial_truth) & (np.abs(radial_truth) > 0)
            fitted_relative_residual = np.full_like(radial_fitted, np.nan, dtype=float)
            np.divide(
                radial_fitted - radial_truth, radial_truth,
                out=fitted_relative_residual, where=fitted_valid,
            )
            residual_axis.plot(
                centres[fitted_valid], fitted_relative_residual[fitted_valid],
                "o-", color="tab:orange", label="fitted − truth",
            )
        residual_axis.set(
            xlabel="radius from source centre [arcsec]",
            ylabel="(brightness − truth) / truth",
            title="Relative brightness residual",
            xscale=xscale,
            xlim=(float(edges[1] * 0.5) if xscale == "log" else 0.0, radial_limit),
        )
        residual_axis.grid(alpha=0.25)
        residual_axis.legend(fontsize=8)
    else:
        residual_axis.set_axis_off()
    x_lower = float(edges[1] * 0.5) if xscale == "log" else 0.0
    brightness_axis.set(
        xlabel="radius from source centre [arcsec]" if truth_pixels is None else None,
        ylabel="surface brightness",
        title="Radial brightness profile",
        xscale=xscale,
        yscale=yscale,
        xlim=(x_lower, radial_limit),
    )
    brightness_axis.legend()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def fit_analytic_pixelated_source(
    hmc_run: str | Path,
    *,
    profile: str = "SERSIC_ELLIPSE",
    n_samples: int = 200,
    crop_radius: float | None = None,
    image_pixel_scale: float,
    source_grid_scale: float | None = None,
    truth: Mapping[str, Any] | str | Path | None = None,
    save_path: str | Path | None = None,
    random_seed: int = 42,
    round_to: int = 3,
    model: Any | None = None,
    median_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit an analytic profile to selected pixelated-HMC source reconstructions.

    ``hmc_run`` must contain ``hmc_samples.h5``, ``kwargs_result.json``, and
    ``kwargs_source_pixels.fits``. Each reconstructed draw is fit in its
    physical source-plane coordinates, including free ``center_x`` and
    ``center_y``. ``crop_radius`` selects a circular source-plane region in
    arcsec around a flux-centroid seed, while the profile centre remains free.
    For uniform source grids,
    ``source_grid_scale`` is required to express fitted radii in arcsec.
    ``image_pixel_scale`` converts the stored source pixel fluxes to the
    surface-brightness convention used by Herculens analytic profiles.
    ``round_to`` controls only the decimal places in corner-plot annotations.
    Supplying ``model`` and its HMC median parameters activates strict
    per-draw geometry: every draw ray-traces the source-arc mask through its
    own lens mass before fitting.  This is the recommended API path.
    """
    from herculens_wrapper.samplers import _load_hmc_samples_hdf5

    profile = str(profile).upper()
    if profile not in _PROFILE_PARAMETERS:
        raise ValueError("profile must be 'SERSIC_ELLIPSE' or 'GAUSSIAN'.")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(round_to, int) or isinstance(round_to, bool) or round_to < 0:
        raise ValueError("round_to must be a non-negative integer.")
    if image_pixel_scale <= 0:
        raise ValueError("image_pixel_scale must be positive.")
    run = Path(hmc_run).expanduser()
    if run.name == "hmc_samples.h5":
        run = run.parent
    archive, kwargs_path, source_path = run / "hmc_samples.h5", run / "kwargs_result.json", run / "kwargs_source_pixels.fits"
    if not archive.is_file() or not kwargs_path.is_file() or not source_path.is_file():
        raise FileNotFoundError("hmc_run must contain hmc_samples.h5, kwargs_result.json, and kwargs_source_pixels.fits.")
    with kwargs_path.open() as stream:
        result_kwargs = json.load(stream)
    source_kwargs = (result_kwargs.get("kwargs_source") or [{}])[0]
    if model is not None:
        if getattr(model.lens_image, "_rtu_grid_source", False):
            raise NotImplementedError("Strict analytic source fitting currently supports uniform grids only.")
        if median_parameters is None:
            raise ValueError("Strict geometry requires median_parameters from the attached HMC FitResult.")
        from astropy.io import fits
        median_pixels = np.asarray(fits.getdata(source_path), dtype=float)
        median_kwargs = model.prob_model.params2kwargs(median_parameters)
        mx, my, _ = model.lens_image.get_source_coordinates(
            median_kwargs.get("kwargs_lens"), force=True, npix_src=median_pixels.shape[0],
            source_grid_scale=model.source_grid_scale,
        )
        x, y = np.meshgrid(np.asarray(mx), np.asarray(my))
        dx, dy = float(np.mean(np.diff(mx))), float(np.mean(np.diff(my)))
        area, grid_kind = np.full_like(median_pixels, abs(dx * dy)), "uniform_per_draw_mask"
    else:
        median_pixels, x, y, area, grid_kind = _source_geometry(source_path, source_grid_scale)
    samples, _ = _load_hmc_samples_hdf5(str(archive))
    count = np.asarray(samples["pixels_wn_source_grid"]).shape[0]
    selected = np.random.default_rng(random_seed).choice(count, size=min(n_samples, count), replace=False)
    if model is None:
        source_draws = _reconstruct_source_draws(samples, selected, source_kwargs)
        draw_coordinates = [(x, y, area)] * len(source_draws)
    else:
        source_draws, draw_coordinates = [], []
        for index in selected:
            draw = {name: np.asarray(values)[index] for name, values in samples.items()}
            kwargs = model.prob_model.params2kwargs(draw)
            pixels = np.asarray(kwargs["kwargs_source"][0]["pixels"])
            sx, sy, _ = model.lens_image.get_source_coordinates(
                kwargs.get("kwargs_lens"), force=True, npix_src=pixels.shape[0],
                source_grid_scale=model.source_grid_scale,
            )
            sx, sy = np.meshgrid(np.asarray(sx), np.asarray(sy))
            cell_area = abs(float(np.mean(np.diff(sx[0]))) * float(np.mean(np.diff(sy[:, 0]))))
            source_draws.append(pixels)
            draw_coordinates.append((sx, sy, np.full_like(pixels, cell_area)))
        source_draws = np.asarray(source_draws)
    parameter_names = _PROFILE_PARAMETERS[profile]

    median_source_fit_parameters, median_source_fit_rmse, median_crop_center = _fit_analytic_source_image(
        profile, median_pixels, x, y, area, image_pixel_scale, crop_radius,
    )
    median_coordinate_center = (
        median_source_fit_parameters["center_x"],
        median_source_fit_parameters["center_y"],
    )
    draw_fit_results = [
        _fit_analytic_source_image(
            profile, draw, draw_x, draw_y, draw_area, image_pixel_scale, crop_radius,
        )
        for draw, (draw_x, draw_y, draw_area) in zip(source_draws, draw_coordinates)
    ]
    fitted = np.asarray([
        [values[name] for name in parameter_names]
        for values, _, _ in draw_fit_results
    ])
    rmses = np.asarray([rmse for _, rmse, _ in draw_fit_results])
    draw_centers = np.asarray([center for _, _, center in draw_fit_results], dtype=float)
    parameter_median = np.median(fitted, axis=0)
    p16, p84 = np.percentile(fitted, [16, 84], axis=0)
    output = Path(save_path).expanduser() if save_path is not None else run / "analytic_source_fit"
    output.mkdir(parents=True, exist_ok=True)

    truth_data = _read_truth(truth)
    truth_values = None
    if truth_data is not None:
        source_truth = (truth_data.get("kwargs_source") or [{}])[0]
        if all(name in source_truth for name in parameter_names):
            truth_values = np.asarray([source_truth[name] for name in parameter_names], dtype=float)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(int(np.ceil(len(parameter_names) / 3)), min(3, len(parameter_names)), figsize=(14, 3.6 * int(np.ceil(len(parameter_names) / 3))), squeeze=False)
    summary = {}
    for i, name in enumerate(parameter_names):
        axis = axes.flat[i]; median, low, high = parameter_median[i], p16[i], p84[i]
        summary[name] = {"median": float(median), "lower_1sigma": float(median-low), "upper_1sigma": float(high-median)}
        axis.hist(fitted[:, i], bins="auto", density=True, color="tab:blue", alpha=.55)
        axis.axvspan(low, high, color="tab:blue", alpha=.18); axis.axvline(median, color="tab:blue")
        if truth_values is not None: axis.axvline(truth_values[i], color="tab:red", ls="--")
        axis.set(title=name, xlabel="parameter value", ylabel="posterior density")
    for axis in axes.flat[len(parameter_names):]: axis.remove()
    one_d_plot = output / "analytic_source_parameters_1d.png"
    figure.tight_layout(); figure.savefig(one_d_plot, dpi=180, bbox_inches="tight"); plt.close(figure)
    corner_plot = None
    if len(parameter_names) >= 2 and len(fitted) >= len(parameter_names):
        import corner
        figure = corner.corner(
            fitted, labels=list(parameter_names), truths=truth_values,
            truth_color="tab:red", quantiles=[.16, .5, .84],
            levels=[.393, .865, .989], show_titles=True,
            title_fmt=f".{round_to}f",
        )
        corner_plot = output / "analytic_source_parameters_corner.png"; figure.savefig(corner_plot, dpi=180, bbox_inches="tight"); plt.close(figure)
    summary.update({
        "profile": profile,
        "median_parameters": {
            name: float(value) for name, value in zip(parameter_names, parameter_median)
        },
        "median_fit_rmse": float(np.median(rmses)),
        "median_source_fit_parameters": median_source_fit_parameters,
        "median_source_fit_rmse": float(median_source_fit_rmse),
        "crop_radius": crop_radius,
        "grid_kind": grid_kind,
        "n_source_samples": int(len(selected)),
        "round_to": round_to,
        "median_coordinate_center": [
            float(median_coordinate_center[0]), float(median_coordinate_center[1]),
        ],
        "median_source_crop_center": [
            float(median_crop_center[0]), float(median_crop_center[1]),
        ],
        "draw_crop_center_median": [
            float(np.median(draw_centers[:, 0])), float(np.median(draw_centers[:, 1])),
        ],
    })
    summary_path = output / "analytic_source_fit_summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "parameters": summary,
        "profile": profile,
        "median_parameters": summary["median_parameters"],
        "samples": fitted,
        "source_draw_indices": selected,
        "crop_centers": draw_centers,
        "median_coordinate_center": tuple(median_coordinate_center),
        "median_source_fit_parameters": median_source_fit_parameters,
        "one_dimensional": one_d_plot,
        "corner": corner_plot,
        "summary_file": summary_path,
    }
