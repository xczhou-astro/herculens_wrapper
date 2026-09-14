#!/usr/bin/env python3
"""Overlay a fitted lens-mass model on an observed image-plane FITS image.

The output is intended for mass--light alignment checks, not a comparison of
surface-brightness and convergence amplitudes: those quantities have different
units.  It produces two panels:

* observed image as an asinh grayscale image with total-kappa contours;
* total kappa as a color image with observed-image isophotes.

Examples
--------
Use saved public-API outputs (mass types are inferred):

    python utils/plot_mass_light_overlay.py \
        --image data.fits --kwargs-result run/kwargs_result.json \
        --model-configuration run/model_configuration.json \
        --pixel-scale 0.03

Or provide the ordered Herculens profile names explicitly:

    python utils/plot_mass_light_overlay.py \
        --image data.fits --kwargs-result run/kwargs_result.json \
        --mass-profiles EPL MPPL_OFFSET SHEAR --pixel-scale 0.03

The image must have the same centre and crop as the image used for modelling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D


# Allow direct ``python utils/plot_mass_light_overlay.py ...`` execution.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _load_json(path: Path) -> dict:
    with path.open() as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _mass_types(configuration: dict) -> list[str]:
    """Read mass profile names from public-API or legacy saved configuration."""
    candidates = (
        configuration.get("backend_definition", {}).get("types", {}),
        configuration.get("type_list", {}),
        configuration,
    )
    for candidate in candidates:
        types = candidate.get("lens_mass_type_list")
        if types:
            return [str(item) for item in types]
    raise ValueError(
        "Could not find lens_mass_type_list in the configuration. "
        "Pass --mass-profiles explicitly."
    )


def _center_crop(image: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None:
        return image
    if crop_size <= 0 or crop_size > min(image.shape):
        raise ValueError("--crop-size must be positive and no larger than both image dimensions.")
    y0 = (image.shape[0] - crop_size) // 2
    x0 = (image.shape[1] - crop_size) // 2
    return image[y0:y0 + crop_size, x0:x0 + crop_size]


def _image_grid(shape: tuple[int, int], pixel_scale: float, center_x: float, center_y: float):
    """Return image-plane coordinates using the wrapper's centred-grid convention."""
    ny, nx = shape
    x_axis = (np.arange(nx) - (nx - 1) / 2.0) * pixel_scale + center_x
    y_axis = (np.arange(ny) - (ny - 1) / 2.0) * pixel_scale + center_y
    return np.meshgrid(x_axis, y_axis), [x_axis[0] - pixel_scale / 2, x_axis[-1] + pixel_scale / 2,
                                         y_axis[0] - pixel_scale / 2, y_axis[-1] + pixel_scale / 2]


def _asinh_image(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels.")
    median = float(np.median(finite))
    scale = float(np.percentile(np.abs(finite - median), 68.0))
    scale = max(scale, np.finfo(float).eps)
    return np.arcsinh((image - median) / scale)


def _light_contour_levels(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size < 10:
        return np.empty(0)
    levels = np.percentile(finite, [70.0, 85.0, 94.0, 98.0])
    return np.unique(levels[np.isfinite(levels)])


def _kappa_norm(kappa: np.ndarray):
    finite = kappa[np.isfinite(kappa)]
    if finite.size == 0:
        return None
    if np.nanmin(finite) >= 0:
        positive = finite[finite > 0]
        if positive.size:
            lower, upper = np.percentile(positive, [1.0, 99.0])
            return LogNorm(vmin=max(float(lower), float(upper) * 1e-4), vmax=float(upper))
    maximum = float(np.percentile(np.abs(finite), 99.0))
    return SymLogNorm(linthresh=max(maximum * 1e-3, 1e-12), vmin=-maximum, vmax=maximum)


def _usable_levels(levels: list[float], values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    lower, upper = float(np.min(finite)), float(np.max(finite))
    return [level for level in levels if lower < level < upper]


def _evaluate_on_image_grid(mass_model, x_grid: np.ndarray, y_grid: np.ndarray,
                            kwargs_lens: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate lensing quantities point-by-point, then restore image layout.

    A Herculens ``PixelGrid`` supplies a flattened list of image coordinates
    internally.  Most mass profiles also accept 2-D arrays, but this is not
    true consistently for JAX-compiled composite profiles.  In particular,
    evaluating a joint elliptical-multipole profile on a 2-D broadcast grid
    can silently yield a sparse-looking result.  Use the backend's native
    point-list convention here and make the required reshape explicit.
    """
    shape = x_grid.shape
    x_points = np.asarray(x_grid, dtype=float).reshape(-1)
    y_points = np.asarray(y_grid, dtype=float).reshape(-1)
    kappa = np.asarray(mass_model.kappa(x_points, y_points, kwargs_lens), dtype=float)
    inverse_magnification = np.asarray(
        mass_model.inverse_magnification(x_points, y_points, kwargs_lens), dtype=float,
    )
    expected_size = int(np.prod(shape))
    if kappa.size != expected_size or inverse_magnification.size != expected_size:
        raise ValueError(
            "Mass model evaluation did not return one value per image pixel: "
            f"expected {expected_size}, got kappa={kappa.size}, "
            f"inverse_magnification={inverse_magnification.size}."
        )
    kappa = kappa.reshape(shape)
    inverse_magnification = inverse_magnification.reshape(shape)
    kappa_fraction = float(np.isfinite(kappa).mean())
    inverse_fraction = float(np.isfinite(inverse_magnification).mean())
    if kappa_fraction < 0.99 or inverse_fraction < 0.99:
        raise ValueError(
            "Mass model returned non-finite values on the image grid "
            f"(finite fraction: kappa={kappa_fraction:.1%}, "
            f"inverse_magnification={inverse_fraction:.1%}). This usually means "
            "that the supplied mass-profile order or kwargs_lens do not match the "
            "fit, or that a fitted mass parameter is outside its physical domain."
        )
    return kappa, inverse_magnification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Observed image-plane FITS file.")
    parser.add_argument("--kwargs-result", type=Path, required=True,
                        help="kwargs_result.json containing kwargs_lens.")
    parser.add_argument("--model-configuration", type=Path,
                        help="model_configuration.json; used to infer mass profile names.")
    parser.add_argument("--mass-profiles", nargs="+",
                        help="Ordered mass profile names; overrides --model-configuration.")
    parser.add_argument("--pixel-scale", type=float, required=True, help="Arcsec per image pixel.")
    parser.add_argument("--output", type=Path, help="Output PNG path (default: next to kwargs JSON).")
    parser.add_argument("--crop-size", type=int,
                        help="Optional centred crop; must match the crop used for fitting.")
    parser.add_argument("--grid-center-x", type=float, default=0.0,
                        help="Model coordinate of the image centre in arcsec (default: 0).")
    parser.add_argument("--grid-center-y", type=float, default=0.0,
                        help="Model coordinate of the image centre in arcsec (default: 0).")
    parser.add_argument("--kappa-levels", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0],
                        help="Convergence contour levels (default: 0.2 0.5 1.0 2.0).")
    parser.add_argument("--no-critical-curves", action="store_true",
                        help="Do not overlay det(A)=0 critical curves.")
    args = parser.parse_args()

    if args.pixel_scale <= 0:
        raise ValueError("--pixel-scale must be positive.")
    for path in (args.image, args.kwargs_result):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.model_configuration is not None and not args.model_configuration.is_file():
        raise FileNotFoundError(args.model_configuration)

    result = _load_json(args.kwargs_result)
    kwargs_lens = result.get("kwargs_lens")
    if not isinstance(kwargs_lens, list) or not kwargs_lens:
        raise ValueError("kwargs_result.json must contain a non-empty 'kwargs_lens' list.")

    if args.mass_profiles:
        mass_profiles = list(args.mass_profiles)
    elif args.model_configuration:
        mass_profiles = _mass_types(_load_json(args.model_configuration))
    else:
        raise ValueError("Provide either --model-configuration or --mass-profiles.")
    if len(mass_profiles) != len(kwargs_lens):
        raise ValueError(
            f"Received {len(mass_profiles)} mass profile names but kwargs_lens has "
            f"{len(kwargs_lens)} entries."
        )

    # SIE/NIE evaluates its Hessian through a 1e-10 finite difference.  With
    # JAX's float32 default that displacement is rounded away, producing a
    # nearly all-zero (and visually sparse) kappa map.  This must run before
    # importing Herculens, which imports JAX itself.
    import jax

    jax.config.update("jax_enable_x64", True)

    # Register wrapper-local MPPL profiles before constructing MassModel.
    from herculens.MassModel.mass_model import MassModel
    from herculens_wrapper.profiles import register_mass_profiles

    register_mass_profiles()
    image = _center_crop(np.asarray(fits.getdata(args.image), dtype=float), args.crop_size)
    if image.ndim != 2:
        raise ValueError("--image must contain a 2-D FITS image.")
    (x_grid, y_grid), extent = _image_grid(
        image.shape, args.pixel_scale, args.grid_center_x, args.grid_center_y,
    )
    mass_model = MassModel(mass_profiles)
    kappa, inverse_magnification = _evaluate_on_image_grid(
        mass_model, x_grid, y_grid, kwargs_lens,
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    transformed = _asinh_image(image)
    axes[0].imshow(transformed, origin="lower", extent=extent, cmap="gray")
    axes[0].set_title("Observed image with mass convergence contours")
    axes[0].set_xlabel("arcsec")
    axes[0].set_ylabel("arcsec")
    levels = _usable_levels(args.kappa_levels, kappa)
    if levels:
        axes[0].contour(x_grid, y_grid, kappa, levels=levels, colors="cyan", linewidths=1.2)

    kappa_norm = _kappa_norm(kappa)
    cmap = "magma" if np.nanmin(kappa) >= 0 else "coolwarm"
    mass_image = axes[1].imshow(kappa, origin="lower", extent=extent, cmap=cmap, norm=kappa_norm)
    axes[1].set_title(r"Total convergence $\kappa$ with image isophotes")
    axes[1].set_xlabel("arcsec")
    axes[1].set_ylabel("arcsec")
    light_levels = _light_contour_levels(image)
    if light_levels.size:
        axes[1].contour(x_grid, y_grid, image, levels=light_levels, colors="white", linewidths=1.0)
    figure.colorbar(mass_image, ax=axes[1], label=r"Convergence $\kappa$")

    if not args.no_critical_curves:
        critical_level = _usable_levels([0.0], inverse_magnification)
        if critical_level:
            for axis in axes:
                axis.contour(x_grid, y_grid, inverse_magnification, levels=[0.0],
                             colors="orange", linewidths=1.25)

    primary = kwargs_lens[0] if isinstance(kwargs_lens[0], dict) else {}
    center = (float(primary.get("center_x", 0.0)), float(primary.get("center_y", 0.0)))
    for axis in axes:
        axis.plot(*center, marker="+", color="lime", markersize=10, markeredgewidth=1.8)
        axis.set_aspect("equal")
    handles = [
        Line2D([], [], color="cyan", label=r"Mass $\kappa$ contours"),
        Line2D([], [], color="white", label="Image isophotes"),
        Line2D([], [], color="orange", label="Critical curve"),
        Line2D([], [], color="lime", marker="+", linestyle="None", label="Primary mass centre"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    figure.suptitle("Mass--light alignment diagnostic", fontsize=14)

    output = args.output or args.kwargs_result.parent / "mass_light_overlay.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"[mass-light] Saved {output}")
    print(f"[mass-light] Profiles: {mass_profiles}")


if __name__ == "__main__":
    main()
