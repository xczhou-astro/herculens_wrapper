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


def _central_crop(shape: tuple[int, int], crop_size: int | tuple[int, int] | None) -> tuple[slice, slice]:
    if crop_size is None:
        return slice(0, shape[0]), slice(0, shape[1])
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if len(crop_size) != 2 or min(crop_size) < 1:
        raise ValueError("crop_size must be a positive int or (ny, nx) tuple.")
    ny, nx = min(int(crop_size[0]), shape[0]), min(int(crop_size[1]), shape[1])
    y0, x0 = (shape[0] - ny) // 2, (shape[1] - nx) // 2
    return slice(y0, y0 + ny), slice(x0, x0 + nx)


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


def fit_analytic_pixelated_source(
    hmc_run: str | Path,
    *,
    profile: str = "SERSIC_ELLIPSE",
    n_samples: int = 200,
    crop_size: int | tuple[int, int] | None = None,
    image_pixel_scale: float,
    source_grid_scale: float | None = None,
    truth: Mapping[str, Any] | str | Path | None = None,
    save_path: str | Path | None = None,
    random_seed: int = 42,
    model: Any | None = None,
    median_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit an analytic profile to selected pixelated-HMC source reconstructions.

    ``hmc_run`` must contain ``hmc_samples.h5``, ``kwargs_result.json``, and
    ``kwargs_source_pixels.fits``.  For uniform source grids,
    ``source_grid_scale`` is required to express fitted radii in arcsec.
    ``image_pixel_scale`` converts the stored source pixel fluxes to the
    surface-brightness convention used by Herculens analytic profiles.
    Supplying ``model`` and its HMC median parameters activates strict
    per-draw geometry: every draw ray-traces the source-arc mask through its
    own lens mass before fitting.  This is the recommended API path.
    """
    from scipy.optimize import least_squares
    from herculens_wrapper.samplers import _load_hmc_samples_hdf5

    profile = str(profile).upper()
    if profile not in _PROFILE_PARAMETERS:
        raise ValueError("profile must be 'SERSIC_ELLIPSE' or 'GAUSSIAN'.")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
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
    crop = _central_crop(median_pixels.shape, crop_size)
    median_crop, x_crop, y_crop, area_crop = (value[crop] for value in (median_pixels, x, y, area))
    samples, _ = _load_hmc_samples_hdf5(str(archive))
    count = np.asarray(samples["pixels_wn_source_grid"]).shape[0]
    selected = np.random.default_rng(random_seed).choice(count, size=min(n_samples, count), replace=False)
    if model is None:
        source_draws = _reconstruct_source_draws(samples, selected, source_kwargs)[:, crop[0], crop[1]]
        draw_coordinates = [(x_crop, y_crop, area_crop)] * len(source_draws)
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
            source_draws.append(pixels[crop]); draw_coordinates.append((sx[crop], sy[crop], np.full_like(pixels[crop], cell_area)))
        source_draws = np.asarray(source_draws)
    parameter_names, initial, bounds = _initial_and_bounds(profile, median_crop, x_crop, y_crop, area_crop, image_pixel_scale)
    x_flat, y_flat, data_flat = x_crop.ravel(), y_crop.ravel(), source_draws.reshape(len(source_draws), -1)

    def fit_one(data: np.ndarray, coords):
        local_x, local_y, local_area = coords
        local_names, local_initial, local_bounds = _initial_and_bounds(profile, data, local_x, local_y, local_area, image_pixel_scale)
        def residual(vector):
            values = dict(zip(local_names, vector))
            return (_light_model(profile, values, local_x.ravel(), local_y.ravel(), image_pixel_scale) - data).ravel()
        answer = least_squares(residual, local_initial, bounds=local_bounds, method="trf")
        return answer.x, float(np.sqrt(np.mean(answer.fun ** 2)))

    fitted, rmses = zip(*(fit_one(draw, coords) for draw, coords in zip(source_draws, draw_coordinates)))
    fitted = np.asarray(fitted)
    parameter_median = np.median(fitted, axis=0)
    p16, p84 = np.percentile(fitted, [16, 84], axis=0)
    analytic_median = _light_model(profile, dict(zip(parameter_names, parameter_median)), x_crop, y_crop, image_pixel_scale)
    output = Path(save_path).expanduser() if save_path is not None else run / "analytic_source_fit"
    output.mkdir(parents=True, exist_ok=True)

    truth_data = _read_truth(truth)
    truth_map = None
    truth_values = None
    if truth_data is not None:
        source_truth = (truth_data.get("kwargs_source") or [{}])[0]
        if all(name in source_truth for name in parameter_names):
            truth_values = np.asarray([source_truth[name] for name in parameter_names], dtype=float)
            truth_map = _light_model(profile, source_truth, x_crop, y_crop, image_pixel_scale)

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    residual = median_crop - truth_map if truth_map is not None else median_crop - analytic_median
    reference_name = "Truth" if truth_map is not None else "Analytic median fit"
    rmse = float(np.sqrt(np.mean(residual**2)))
    vmax = float(np.nanpercentile(np.abs(np.concatenate([median_crop.ravel(), (truth_map if truth_map is not None else analytic_median).ravel()])), 99.5))
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    panels = [(median_crop, "Pixelated source (median)"), (truth_map if truth_map is not None else analytic_median, reference_name), (analytic_median, "Analytic fit")]
    extent = [float(x_crop.min()), float(x_crop.max()), float(y_crop.min()), float(y_crop.max())]
    for axis, (image, title) in zip(axes[:3], panels):
        handle = axis.imshow(image, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
        axis.set(title=title, xlabel="arcsec", ylabel="arcsec"); figure.colorbar(handle, ax=axis, label="pixel flux")
    rmax = max(float(np.nanpercentile(np.abs(residual), 99)), 1e-12)
    handle = axes[3].imshow(residual, origin="lower", extent=extent, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=-rmax, vmax=rmax))
    axes[3].set(title=f"Pixelated − {reference_name}\nRMSE = {rmse:.4g}", xlabel="arcsec", ylabel="arcsec")
    figure.colorbar(handle, ax=axes[3], label="pixel flux")
    source_plot = output / "pixelated_source_truth_residual.png"
    figure.savefig(source_plot, dpi=180, bbox_inches="tight"); plt.close(figure)

    # Radial surface-brightness and cumulative-flux profiles around the fitted centre.
    radius = np.hypot(x_crop - parameter_median[parameter_names.index("center_x")], y_crop - parameter_median[parameter_names.index("center_y")])
    n_radial_bins = min(24, max(4, int(np.sqrt(radius.size))))
    edges = np.linspace(0, float(radius.max()), n_radial_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    brightness = median_crop / image_pixel_scale**2
    analytic_brightness = analytic_median / image_pixel_scale**2
    def annular_mean(image, lo, hi):
        selected = image[(radius >= lo) & (radius < hi)]
        return float(np.mean(selected)) if selected.size else np.nan
    radial = [annular_mean(brightness, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    radial_fit = [annular_mean(analytic_brightness, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    flux = [np.sum(brightness[radius < edge] * area_crop[radius < edge]) for edge in edges[1:]]
    flux_fit = [np.sum(analytic_brightness[radius < edge] * area_crop[radius < edge]) for edge in edges[1:]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(centres, radial, "o-", label="pixelated median"); axes[0].plot(centres, radial_fit, "-", label="analytic median")
    axes[0].set(xlabel="radius [arcsec]", ylabel="surface brightness", title="Radial brightness profile"); axes[0].legend()
    axes[1].plot(edges[1:], flux, "o-", label="pixelated median"); axes[1].plot(edges[1:], flux_fit, "-", label="analytic median")
    axes[1].set(xlabel="radius [arcsec]", ylabel="enclosed flux", title="Cumulative flux profile"); axes[1].legend()
    radial_plot = output / "source_radial_profiles.png"
    figure.savefig(radial_plot, dpi=180, bbox_inches="tight"); plt.close(figure)

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
        figure = corner.corner(fitted, labels=list(parameter_names), truths=truth_values, truth_color="tab:red", quantiles=[.16,.5,.84], levels=[.393,.865,.989], show_titles=True)
        corner_plot = output / "analytic_source_parameters_corner.png"; figure.savefig(corner_plot, dpi=180, bbox_inches="tight"); plt.close(figure)
    summary.update({"rmse_median_source": rmse, "median_fit_rmse": float(np.median(rmses)), "grid_kind": grid_kind, "n_source_samples": int(len(selected))})
    summary_path = output / "analytic_source_fit_summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {"parameters": summary, "samples": fitted, "source_draw_indices": selected, "one_dimensional": one_d_plot, "corner": corner_plot, "source_comparison": source_plot, "radial_profiles": radial_plot, "summary_file": summary_path}
