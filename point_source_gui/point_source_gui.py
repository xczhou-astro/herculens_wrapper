#!/usr/bin/env python3
"""Interactive ray-tracing viewer for an image and a fitted Herculens mass model.

The viewer reads an observed FITS image, ``kwargs_result.json``, and the saved
source-plane FITS when available. The mass profile order is discovered from a
neighbouring configuration or entered in the browser.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from astropy.io import fits
from flask import Flask, jsonify, render_template, request


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
ROOT = Path(__file__).resolve().parent
# ``start_gui.py`` is commonly invoked with an absolute path from a notebook
# directory.  Make the sibling ``herculens_wrapper`` package importable in
# that case instead of relying on the caller's current working directory.
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@dataclass
class ViewerModel:
    image: np.ndarray  # Display orientation: y increases upward.
    pixel_scale: float
    center_x: float
    center_y: float
    mass_types: list[str]
    kwargs_lens: list[dict[str, Any]]
    mass_model: Any
    x_grid: np.ndarray
    y_grid: np.ndarray
    beta_x: np.ndarray
    beta_y: np.ndarray
    inverse_magnification: np.ndarray
    image_extent: list[float]
    source_extent: list[float]
    magnification_display: list[int]
    critical_curves: list[list[list[float]]]
    caustics: list[list[list[float]]]
    source: dict[str, Any] | None


ACTIVE_MODEL: ViewerModel | None = None
app = Flask(__name__, template_folder=str(ROOT / "templates"))


def _resolve_existing_file(value: str | None, *, suffixes: tuple[str, ...] = ()) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file() and (not suffixes or candidate.suffix.lower() in suffixes):
        return candidate.resolve()
    return None


def _find_result_file(value: str | None, image_file: Path | None) -> Path:
    """Find kwargs_result.json from a supplied path or next to the image."""
    candidates: list[Path] = []
    if value:
        supplied = Path(value).expanduser()
        if supplied.is_file():
            candidates.append(supplied)
        elif supplied.is_dir():
            candidates.append(supplied / "kwargs_result.json")
            candidates.extend(sorted(supplied.glob("run_*/kwargs_result.json")))
    if image_file is not None:
        candidates.append(image_file.parent / "kwargs_result.json")
        candidates.append(image_file.parent / image_file.stem / "kwargs_result.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find kwargs_result.json. Choose the result JSON or its run directory "
        "(or put it next to the image)."
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _find_source_file(value: str | None, result: dict[str, Any], result_file: Path) -> Path | None:
    """Prefer an explicit source FITS, then the file referenced by the fit."""
    if value:
        supplied = Path(value).expanduser()
        candidate = supplied / "kwargs_source_pixels.fits" if supplied.is_dir() else supplied
        if not candidate.is_file() or candidate.suffix.lower() not in (".fits", ".fit", ".fts"):
            raise FileNotFoundError(f"Choose a valid source-plane FITS file: {candidate}")
        return candidate.resolve()
    for entry in result.get("kwargs_source") or []:
        pixels = entry.get("pixels") if isinstance(entry, dict) else None
        if isinstance(pixels, dict) and isinstance(pixels.get("file"), str):
            candidate = result_file.parent / pixels["file"]
            if candidate.is_file() and candidate.suffix.lower() in (".fits", ".fit", ".fts"):
                return candidate.resolve()
    candidate = result_file.parent / "kwargs_source_pixels.fits"
    return candidate.resolve() if candidate.is_file() else None


def _run_configuration(result_file: Path) -> dict[str, Any]:
    for directory in (result_file.parent, result_file.parent.parent):
        for name in ("model_configuration.json", "config.json", "configuration.json"):
            candidate = directory / name
            if candidate.is_file():
                try:
                    return _load_json(candidate)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    LOGGER.warning("Unable to read model configuration %s: %s", candidate, error)
    return {}


def _mass_types_from_configuration(result_file: Path) -> list[str]:
    """Locate profile types in standard public-API and legacy run outputs."""
    directories = [result_file.parent, result_file.parent.parent]
    names = ("model_configuration.json", "config.json", "configuration.json")
    for directory in directories:
        for name in names:
            candidate = directory / name
            if not candidate.is_file():
                continue
            try:
                config = _load_json(candidate)
                containers = (
                    config.get("backend_definition", {}).get("types", {}),
                    config.get("type_list", {}),
                    config,
                )
                for container in containers:
                    profiles = container.get("lens_mass_type_list") if isinstance(container, dict) else None
                    if isinstance(profiles, list) and profiles:
                        return [str(profile).strip() for profile in profiles]
            except (OSError, ValueError, json.JSONDecodeError) as error:
                LOGGER.warning("Unable to read model configuration %s: %s", candidate, error)
    return []


def _parse_mass_types(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace("\n", ",").replace(" ", ",").split(",") if part.strip()]


def _read_fits_image(path: Path, hdu: str | int | None) -> np.ndarray:
    key: str | int | None = hdu
    if isinstance(key, str) and key.strip().isdigit():
        key = int(key.strip())
    with fits.open(path) as hdul:
        source = hdul[key].data if key not in (None, "") else None
        if source is None:
            for candidate in hdul:
                if candidate.data is not None and candidate.data.ndim == 2:
                    source = candidate.data
                    break
    if source is None:
        raise ValueError("No two-dimensional image HDU was found.")
    image = np.asarray(source, dtype=float)
    if image.ndim != 2:
        raise ValueError("The selected FITS HDU is not two-dimensional.")
    # FITS row 0 is visually at the top; canvas and model coordinates use +y up.
    return np.flipud(image)


def _image_grid(shape: tuple[int, int], pixel_scale: float, center_x: float, center_y: float):
    ny, nx = shape
    x_axis = (np.arange(nx, dtype=float) - (nx - 1) / 2.0) * pixel_scale + center_x
    y_axis = (np.arange(ny, dtype=float) - (ny - 1) / 2.0) * pixel_scale + center_y
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    return x_grid, y_grid, [
        float(x_axis[0] - pixel_scale / 2.0), float(x_axis[-1] + pixel_scale / 2.0),
        float(y_axis[0] - pixel_scale / 2.0), float(y_axis[-1] + pixel_scale / 2.0),
    ]


def _contours(x: np.ndarray, y: np.ndarray, field: np.ndarray, levels: list[float]) -> list[list[list[float]]]:
    """Extract contour paths as JSON-safe x/y vertex pairs."""
    finite = np.isfinite(field)
    if not np.any(finite) or min(levels) > np.nanmax(field[finite]) or max(levels) < np.nanmin(field[finite]):
        return []
    figure = Figure(figsize=(1, 1))
    axis = figure.subplots()
    try:
        contour_set = axis.contour(x, y, field, levels=levels)
        paths: list[list[list[float]]] = []
        # ``get_paths()`` can represent several disconnected pieces in one
        # Path separated by MOVETO codes.  Sending only its vertices to the
        # canvas joins those pieces with a spurious straight line.  allsegs
        # preserves each physical contour as its own segment.
        for level_segments in contour_set.allsegs:
            for segment in level_segments:
                vertices = np.asarray(segment, dtype=float)
                if len(vertices) >= 3 and np.all(np.isfinite(vertices)):
                    paths.append(vertices.tolist())
        return paths
    finally:
        figure.clear()


def _display_image(image: np.ndarray) -> list[int]:
    """A robust asinh stretch encoded as an 8-bit grayscale raster."""
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0] * values.size
    low, high = np.percentile(finite, [1.0, 99.7])
    scale = max((high - low) / 5.0, np.finfo(float).eps)
    stretched = np.arcsinh((np.nan_to_num(values, nan=low) - low) / scale)
    minimum, maximum = np.nanmin(stretched), np.nanmax(stretched)
    if maximum <= minimum:
        return [0] * values.size
    return np.clip((stretched - minimum) / (maximum - minimum) * 255, 0, 255).astype(np.uint8).ravel().tolist()


def _display_scalar_map(
    values: np.ndarray, *, logarithmic: bool = False, unit_interval: bool = False,
    flip_vertical: bool = True,
) -> list[int]:
    """Encode a model map for the canvas, with its y axis matching the FITS view."""
    data = np.asarray(values, dtype=float)
    if unit_interval:
        normalized = np.clip(np.nan_to_num(data, nan=0.0), 0.0, 1.0) * 255
        return (np.flipud(normalized) if flip_vertical else normalized).astype(np.uint8).ravel().tolist()
    if logarithmic:
        data = np.log10(np.maximum(data, 1e-8))
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return [0] * data.size
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        normalized = np.zeros_like(data)
    else:
        normalized = np.clip((np.nan_to_num(data, nan=low, posinf=high, neginf=low) - low) / (high - low), 0, 1)
    normalized *= 255
    return (np.flipud(normalized) if flip_vertical else normalized).astype(np.uint8).ravel().tolist()


def _source_mask(
    result_file: Path, image_path: Path, image_shape: tuple[int, int],
    config: dict[str, Any],
) -> np.ndarray:
    """Use the saved arc mask for the adaptive source grid when available."""
    observation = config.get("observation", {})
    mask_name = (
        observation.get("input_paths", {}).get("source_arc_mask")
        or observation.get("source_arc_mask_path")
    ) if isinstance(observation, dict) else None
    names = ["source_arc_mask.fits", "mask_1.fits", "mask.fits"]
    candidates = []
    if mask_name:
        configured = Path(mask_name).expanduser()
        candidates.append(configured)
        names.insert(0, configured.name)
    directories = (result_file.parent / "data", result_file.parent.parent / "data", image_path.parent)
    for directory in directories:
        for name in names:
            candidates.append(directory / name)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with fits.open(candidate) as hdul:
            mask = np.asarray(hdul[0].data, dtype=bool)
        if mask.shape == image_shape and np.any(mask):
            return mask
    return np.ones(image_shape, dtype=bool)


def _uniform_source_extent(
    mask: np.ndarray, pixel_scale: float,
    center_x: float, center_y: float, mass_model: Any, kwargs_lens: list[dict[str, Any]],
    grid_scale: float, supersampling: int,
) -> list[float]:
    """Recreate Herculens' square adaptive source grid from its mask outline."""
    from scipy.ndimage import binary_erosion

    if not np.isfinite(grid_scale) or grid_scale <= 0:
        raise ValueError("source_grid_scale must be positive.")
    if supersampling < 1:
        raise ValueError("supersampling_factor must be positive.")
    support = np.repeat(np.repeat(mask, supersampling, axis=0), supersampling, axis=1)
    outline = support & ~binary_erosion(support)
    rows, columns = np.nonzero(outline)
    ny, nx = mask.shape
    step = pixel_scale / supersampling
    x = (columns + 0.5 - nx * supersampling / 2) * step + center_x
    y = (rows + 0.5 - ny * supersampling / 2) * step + center_y
    bx, by = mass_model.ray_shooting(x, y, kwargs_lens)
    bx, by = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
    finite = np.isfinite(bx) & np.isfinite(by)
    if not np.any(finite):
        raise ValueError("Cannot locate the source grid: ray tracing returned no finite mask points.")
    x0, x1 = float(bx[finite].min()), float(bx[finite].max())
    y0, y1 = float(by[finite].min()), float(by[finite].max())
    half_size = grid_scale * max(x1 - x0, y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [cx - half_size, cx + half_size, cy - half_size, cy + half_size]


def _read_source_plane(
    path: Path, result_file: Path, image_path: Path, image_shape: tuple[int, int],
    pixel_scale: float, center_x: float, center_y: float,
    mass_model: Any, kwargs_lens: list[dict[str, Any]],
) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        pixels = np.asarray(hdul[0].data, dtype=float)
        if pixels.ndim != 2 or not np.any(np.isfinite(pixels)):
            raise ValueError(f"Source-plane FITS must contain a finite 2-D primary image: {path}")
        ny, nx = pixels.shape
        source: dict[str, Any] = {"width": nx, "height": ny, "path": str(path)}
        if ("X_CORNERS" in hdul) != ("Y_CORNERS" in hdul):
            raise ValueError(f"Source-plane FITS must contain both X_CORNERS and Y_CORNERS: {path}")
        if "X_CORNERS" in hdul:
            x_corners = np.asarray(hdul["X_CORNERS"].data, dtype=float)
            y_corners = np.asarray(hdul["Y_CORNERS"].data, dtype=float)
            if x_corners.shape != (ny + 1, nx + 1) or y_corners.shape != x_corners.shape:
                raise ValueError(f"Source-plane corner arrays do not match source pixels: {path}")
            if not (np.all(np.isfinite(x_corners)) and np.all(np.isfinite(y_corners))):
                raise ValueError(f"Source-plane corners contain non-finite coordinates: {path}")
            source["x_corners"] = x_corners.ravel().tolist()
            source["y_corners"] = y_corners.ravel().tolist()
            source["extent"] = [float(x_corners.min()), float(x_corners.max()),
                                float(y_corners.min()), float(y_corners.max())]
            source["pixels"] = _display_scalar_map(pixels, flip_vertical=False)
        else:
            header = hdul[0].header
            if all(key in header for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2")):
                x_edges = (
                    (np.array([0.5, nx + 0.5]) - float(header["CRPIX1"]))
                    * float(header["CDELT1"]) + float(header["CRVAL1"])
                )
                y_edges = (
                    (np.array([0.5, ny + 0.5]) - float(header["CRPIX2"]))
                    * float(header["CDELT2"]) + float(header["CRVAL2"])
                )
                if not (np.all(np.isfinite(x_edges)) and np.all(np.isfinite(y_edges))):
                    raise ValueError("Source-plane FITS has non-finite physical coordinates.")
                if x_edges[0] > x_edges[1]:
                    pixels = np.fliplr(pixels)
                if y_edges[0] > y_edges[1]:
                    pixels = np.flipud(pixels)
                source["extent"] = [float(min(x_edges)), float(max(x_edges)),
                                    float(min(y_edges)), float(max(y_edges))]
            else:
                config = _run_configuration(result_file)
                grid_scale = float(config.get("source_grid_scale", 1.0))
                supersampling = int(config.get("supersampling_factor", config.get("numerics", {}).get("supersampling_factor", 1)))
                mask = _source_mask(result_file, image_path, image_shape, config)
                source["extent"] = _uniform_source_extent(
                    mask, pixel_scale, center_x, center_y,
                    mass_model, kwargs_lens, grid_scale, supersampling,
                )
            x0, x1, y0, y1 = source["extent"]
            if not (x0 < x1 and y0 < y1):
                raise ValueError(f"Source-plane FITS has zero physical extent: {path}")
            source["pixels"] = _display_scalar_map(pixels)
    return source


def _fixed_source_extent(caustics: list[list[list[float]]], beta_x: np.ndarray, beta_y: np.ndarray,
                         source: dict[str, Any] | None = None) -> list[float]:
    """Choose one stationary source-plane field centered on the caustic."""
    curves = [np.asarray(curve, dtype=float) for curve in caustics if len(curve)]
    if curves or source is not None:
        points = np.concatenate(curves, axis=0) if curves else np.empty((0, 2))
        if source is not None:
            sx0, sx1, sy0, sy1 = source["extent"]
            points = np.vstack((points, [[sx0, sy0], [sx1, sy1]]))
        x_values, y_values = points[:, 0], points[:, 1]
    else:
        x_values, y_values = beta_x[np.isfinite(beta_x)], beta_y[np.isfinite(beta_y)]
    if not len(x_values) or not len(y_values):
        return [-1.0, 1.0, -1.0, 1.0]
    x0, x1 = float(np.min(x_values)), float(np.max(x_values))
    y0, y1 = float(np.min(y_values)), float(np.max(y_values))
    half_size = max((x1 - x0) / 2.0, (y1 - y0) / 2.0, 0.05)
    half_size *= 1.25
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [center_x - half_size, center_x + half_size, center_y - half_size, center_y + half_size]


def _enable_legacy_jax_clip_keywords() -> None:
    """Bridge Herculens profiles using ``a_min``/``a_max`` to new JAX.

    Recent JAX releases renamed ``jax.numpy.clip`` keyword arguments to
    ``min``/``max``.  Some released Herculens and jaxtronomy profiles still
    call it with the old NumPy spelling.  Patch only this viewer process, and
    only when the installed JAX no longer accepts the legacy spelling.
    """
    import jax.numpy as jnp

    if "a_max" in inspect.signature(jnp.clip).parameters:
        return
    native_clip = jnp.clip

    def compatible_clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None):
        if min is not None and a_min is not None:
            raise TypeError("clip() received both 'min' and 'a_min'.")
        if max is not None and a_max is not None:
            raise TypeError("clip() received both 'max' and 'a_max'.")
        lower = min if min is not None else a_min
        upper = max if max is not None else a_max
        if out is not None:
            # JAX does not support an output buffer; preserve its normal error
            # rather than silently returning a result in the wrong object.
            return native_clip(a, lower, upper, out=out)
        return native_clip(a, lower, upper)

    jnp.clip = compatible_clip


def _build_model(payload: dict[str, Any]) -> tuple[ViewerModel, Path, Path]:
    image_path = _resolve_existing_file(payload.get("image_path"), suffixes=(".fits", ".fit", ".fts"))
    if image_path is None:
        raise FileNotFoundError("Choose a valid FITS image.")
    result_path = _find_result_file(payload.get("result_path"), image_path)
    result = _load_json(result_path)
    kwargs_lens = result.get("kwargs_lens")
    if not isinstance(kwargs_lens, list) or not kwargs_lens or not all(isinstance(item, dict) for item in kwargs_lens):
        raise ValueError("kwargs_result.json must contain a non-empty 'kwargs_lens' list.")

    mass_types = _parse_mass_types(payload.get("mass_types")) or _mass_types_from_configuration(result_path)
    if not mass_types:
        raise ValueError(
            "Mass profile types could not be inferred. Enter them in order, e.g. 'EPL, SHEAR'."
        )
    if len(mass_types) != len(kwargs_lens):
        raise ValueError(
            f"There are {len(kwargs_lens)} kwargs_lens components but {len(mass_types)} mass profile types."
        )

    pixel_scale = float(payload.get("pixel_scale", 0.05))
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("Pixel scale must be a positive number in arcsec/pixel.")
    center_x = float(payload.get("grid_center_x", 0.0))
    center_y = float(payload.get("grid_center_y", 0.0))
    image = _read_fits_image(image_path, payload.get("image_hdu"))
    x_grid, y_grid, extent = _image_grid(image.shape, pixel_scale, center_x, center_y)

    # Import here so the page remains useful for selecting files even when the
    # optional modelling environment has not been activated yet.
    import jax

    jax.config.update("jax_enable_x64", True)
    _enable_legacy_jax_clip_keywords()
    from herculens.MassModel.mass_model import MassModel
    from herculens_wrapper.profiles import register_mass_profiles

    register_mass_profiles()
    mass_model = MassModel(mass_types)
    points_x, points_y = x_grid.ravel(), y_grid.ravel()
    beta_x = np.asarray(mass_model.ray_shooting(points_x, points_y, kwargs_lens)[0], dtype=float).reshape(image.shape)
    beta_y = np.asarray(mass_model.ray_shooting(points_x, points_y, kwargs_lens)[1], dtype=float).reshape(image.shape)
    determinant = np.asarray(
        mass_model.inverse_magnification(points_x, points_y, kwargs_lens), dtype=float,
    ).reshape(image.shape)
    if float(np.isfinite(determinant).mean()) < 0.99:
        raise ValueError("The mass model returned non-finite values on the image grid.")

    critical_curves = _contours(x_grid, y_grid, determinant, [0.0])
    caustics: list[list[list[float]]] = []
    for curve in critical_curves:
        vertices = np.asarray(curve, dtype=float)
        mapped_x, mapped_y = mass_model.ray_shooting(vertices[:, 0], vertices[:, 1], kwargs_lens)
        caustics.append(np.column_stack([np.asarray(mapped_x), np.asarray(mapped_y)]).astype(float).tolist())
    source_path = _find_source_file(payload.get("source_path"), result, result_path)
    source = (
        _read_source_plane(
            source_path, result_path, image_path, image.shape, pixel_scale,
            center_x, center_y, mass_model, kwargs_lens,
        ) if source_path is not None else None
    )
    source_extent = _fixed_source_extent(caustics, beta_x, beta_y, source)
    with np.errstate(divide="ignore", invalid="ignore"):
        absolute_magnification = 1.0 / np.abs(determinant)

    return ViewerModel(
        image=image, pixel_scale=pixel_scale, center_x=center_x, center_y=center_y,
        mass_types=mass_types, kwargs_lens=kwargs_lens, mass_model=mass_model,
        x_grid=x_grid, y_grid=y_grid, beta_x=beta_x, beta_y=beta_y,
        inverse_magnification=determinant, image_extent=extent,
        source_extent=source_extent,
        magnification_display=_display_scalar_map(absolute_magnification, logarithmic=True),
        critical_curves=critical_curves, caustics=caustics, source=source,
    ), image_path, result_path


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/select_path")
def select_path():
    """Open a local macOS/Tk picker; the text fields remain available remotely."""
    kind = request.args.get("kind", "file")
    try:
        if sys.platform == "darwin":
            if kind == "folder":
                script = 'POSIX path of (choose folder with prompt "Select result directory")'
            elif kind == "source":
                script = 'POSIX path of (choose file with prompt "Select source-plane FITS" of type {"fits", "fit", "fts"})'
            else:
                script = 'POSIX path of (choose file with prompt "Select FITS image" of type {"fits", "fit", "fts"})'
            response = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
            if response.returncode == 0 and response.stdout.strip():
                return jsonify({"success": True, "path": response.stdout.strip()})
    except Exception as error:
        LOGGER.warning("Native picker failed: %s", error)
    return jsonify({"success": False, "message": "No path selected; paste the path into the field."})


@app.post("/api/load_model")
def load_model():
    global ACTIVE_MODEL
    try:
        model, image_path, result_path = _build_model(request.get_json(force=True) or {})
        ACTIVE_MODEL = model
        return jsonify({
            "image": {"width": int(model.image.shape[1]), "height": int(model.image.shape[0]),
                      "pixels": _display_image(model.image), "extent": model.image_extent},
            "mass_types": model.mass_types,
            "image_path": str(image_path), "result_path": str(result_path),
            "critical_curves": model.critical_curves, "caustics": model.caustics,
            "source_extent": model.source_extent, "source": model.source,
            "magnification": {"pixels": model.magnification_display, "scale": "log10_abs_mu"},
        })
    except Exception as error:
        LOGGER.exception("Could not load lens model")
        return jsonify({"message": str(error)}), 400


@app.post("/api/ray_trace")
def ray_trace():
    if ACTIVE_MODEL is None:
        return jsonify({"message": "Load an image and mass model first."}), 400
    try:
        payload = request.get_json(force=True) or {}
        pixel_x, pixel_y = float(payload["pixel_x"]), float(payload["pixel_y"])
        sigma = float(payload.get("sigma", 0.03))
        if not (np.isfinite(pixel_x) and np.isfinite(pixel_y) and np.isfinite(sigma)) or sigma <= 0:
            raise ValueError("Click coordinates and Gaussian sigma must be finite; sigma must be positive.")
        model = ACTIVE_MODEL
        height, width = model.image.shape
        if not (0 <= pixel_x < width and 0 <= pixel_y < height):
            raise ValueError("The selected point is outside the image.")
        theta_x = (pixel_x - (width - 1) / 2.0) * model.pixel_scale + model.center_x
        theta_y = (pixel_y - (height - 1) / 2.0) * model.pixel_scale + model.center_y
        beta_x, beta_y = model.mass_model.ray_shooting(np.asarray([theta_x]), np.asarray([theta_y]), model.kwargs_lens)
        beta_x, beta_y = float(np.asarray(beta_x)[0]), float(np.asarray(beta_y)[0])
        radius_in_source = np.hypot(model.beta_x - beta_x, model.beta_y - beta_y)
        lensed_brightness = np.exp(-0.5 * (radius_in_source / sigma) ** 2)
        image_rings = {
            str(level): _contours(model.x_grid, model.y_grid, radius_in_source, [level * sigma])
            for level in (1, 2, 3)
        }
        return jsonify({
            "theta": [theta_x, theta_y], "beta": [beta_x, beta_y], "sigma": sigma,
            "source_extent": model.source_extent, "caustics": model.caustics,
            "image_rings": image_rings,
            # Gaussian is normalized to I_0=1, so preserve that physical
            # display scale rather than contrast-stretching each click.
            "lensed_brightness": _display_scalar_map(lensed_brightness, unit_interval=True),
        })
    except Exception as error:
        LOGGER.exception("Ray trace failed")
        return jsonify({"message": str(error)}), 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5050, help="Local server port (default: 5050).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    args = parser.parse_args()
    if not args.no_browser:
        import webbrowser
        from threading import Timer

        Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
