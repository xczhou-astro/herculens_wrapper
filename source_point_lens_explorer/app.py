#!/usr/bin/env python3
"""Local interactive source-plane point-source explorer for a fitted SIE lens.

The application intentionally reads the saved fit products directly.  It does
not need a running Herculens/JAX installation: the SIE (NIE backend) and
external-shear lens equations used by this result are evaluated with NumPy and
SciPy, which keeps the viewer easy to start on a laptop.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from flask import Flask, jsonify, render_template, request
from matplotlib import colormaps
from scipy.ndimage import binary_erosion
from scipy.optimize import root


DEFAULT_RESULT_DIR = Path(
    "/Users/xczhou/Library/CloudStorage/GoogleDrive-xczhou95@gmail.com/My Drive/"
    "modelling/cowls_data/modelling_F150W_SIE/pixelated_hmc"
)
APP_ROOT = Path(__file__).resolve().parent
SAVED_SOURCES_FILENAME = "point_source_lens_explorer_results.json"
app = Flask(__name__, template_folder=str(APP_ROOT / "templates"))


def _display(values: np.ndarray, *, logarithmic: bool = False) -> list[int]:
    """Robustly scale a 2-D array to an 8-bit browser raster."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0] * values.size
    if logarithmic:
        floor = max(float(np.quantile(finite, 0.01)), 1e-12)
        values = np.log10(np.maximum(values, floor))
        finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [0.01, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0, 1)
    return np.nan_to_num(scaled * 255, nan=0).astype(np.uint8).ravel().tolist()


@dataclass
class SIELens:
    """Herculens SIE (NIE limit) plus external shear, in arcseconds."""

    sie: dict
    shear: dict

    def __post_init__(self) -> None:
        e1, e2 = float(self.sie["e1"]), float(self.sie["e2"])
        ellipticity = min(float(np.hypot(e1, e2)), 0.9999)
        self.phi = float(np.arctan2(e2, e1) / 2)
        self.q = (1 - ellipticity) / (1 + ellipticity)
        theta_e = float(self.sie["theta_E"])
        theta_major = theta_e / np.sqrt((1 + self.q**2) / (2 * self.q))
        self.b = theta_major * np.sqrt((1 + self.q**2) / 2)
        self.core = 1e-10 / np.sqrt(self.q)
        self.cos_phi, self.sin_phi = np.cos(self.phi), np.sin(self.phi)

    def deflection(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x - float(self.sie.get("center_x", 0)), y - float(self.sie.get("center_y", 0))
        major = dx * self.cos_phi + dy * self.sin_phi
        minor = -dx * self.sin_phi + dy * self.cos_phi
        omq = np.sqrt(max(1 - self.q**2, 1e-14))
        psi = np.sqrt(self.q**2 * (self.core**2 + major**2) + minor**2)
        alpha_major = self.b / omq * np.arctan(omq * major / (psi + self.core))
        atanh_arg = np.clip(omq * minor / (psi + self.q**2 * self.core), -0.999999999, 0.999999999)
        alpha_minor = self.b / omq * np.arctanh(atanh_arg)
        alpha_x = alpha_major * self.cos_phi - alpha_minor * self.sin_phi
        alpha_y = alpha_major * self.sin_phi + alpha_minor * self.cos_phi
        sx, sy = x - float(self.shear.get("ra_0", 0)), y - float(self.shear.get("dec_0", 0))
        gamma1, gamma2 = float(self.shear["gamma1"]), float(self.shear["gamma2"])
        return alpha_x + gamma1 * sx + gamma2 * sy, alpha_y + gamma2 * sx - gamma1 * sy

    def ray_shoot(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        ax, ay = self.deflection(x, y)
        return np.asarray(x, dtype=float) - ax, np.asarray(y, dtype=float) - ay

    def inverse_magnification(self, x: float, y: float, step: float = 1e-5) -> float:
        bx_p, by_p = self.ray_shoot(x + step, y)
        bx_m, by_m = self.ray_shoot(x - step, y)
        cx_p, cy_p = self.ray_shoot(x, y + step)
        cx_m, cy_m = self.ray_shoot(x, y - step)
        jacobian = np.array([
            [(bx_p - bx_m) / (2 * step), (cx_p - cx_m) / (2 * step)],
            [(by_p - by_m) / (2 * step), (cy_p - cy_m) / (2 * step)],
        ])
        return float(np.linalg.det(jacobian))


@dataclass
class ViewerData:
    result_dir: Path
    lens: SIELens
    image: np.ndarray
    source: np.ndarray
    ring: np.ndarray
    psf: np.ndarray
    pixel_scale: float
    image_extent: list[float]
    source_extent: list[float]


ACTIVE: ViewerData | None = None


TWILIGHT_PALETTE = (colormaps["twilight"](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8).tolist()


def _source_extent(lens: SIELens, mask: np.ndarray, pixel_scale: float, source_scale: float) -> list[float]:
    """Recreate the uniform adaptive grid extent saved by the modelling run."""
    height, width = mask.shape
    x_axis = (np.arange(width) - (width - 1) / 2) * pixel_scale
    y_axis = (np.arange(height) - (height - 1) / 2) * pixel_scale
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    outline = mask.astype(bool) & ~binary_erosion(mask.astype(bool))
    support = outline if np.any(outline) else mask.astype(bool)
    beta_x, beta_y = lens.ray_shoot(x_grid[support], y_grid[support])
    x0, x1, y0, y1 = map(float, (beta_x.min(), beta_x.max(), beta_y.min(), beta_y.max()))
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    half_width = 0.5 * source_scale * max(x1 - x0, y1 - y0)
    return [center_x - half_width, center_x + half_width, center_y - half_width, center_y + half_width]


def _saved_source_pixel_scale(result_dir: Path) -> float | None:
    """Use the grid scale printed alongside the saved source-plane product."""
    log_path = result_dir / "log.txt"
    if not log_path.is_file():
        return None
    matches = re.findall(r"Source pixel scale:\s*([0-9.eE+-]+)\s*arcsec/pixel", log_path.read_text())
    return float(matches[-1]) if matches else None


def _lens_light_image(kwargs_lens_light: list[dict], shape: tuple[int, int], pixel_scale: float) -> np.ndarray:
    """Evaluate the fitted GAUSSIAN_ELLIPSE light model in image pixels."""
    height, width = shape
    x_axis = (np.arange(width) - (width - 1) / 2) * pixel_scale
    y_axis = (np.arange(height) - (height - 1) / 2) * pixel_scale
    x, y = np.meshgrid(x_axis, y_axis)
    light = np.zeros_like(x)
    for component in kwargs_lens_light:
        e1, e2 = float(component["e1"]), float(component["e2"])
        ellipticity = min(float(np.hypot(e1, e2)), 0.9999)
        q = (1 - ellipticity) / (1 + ellipticity)
        phi = np.arctan2(e2, e1) / 2
        dx, dy = x - float(component["center_x"]), y - float(component["center_y"])
        major = (np.cos(phi) * dx + np.sin(phi) * dy) * np.sqrt(q)
        minor = (-np.sin(phi) * dx + np.cos(phi) * dy) / np.sqrt(q)
        sigma = float(component["sigma"])
        brightness = float(component["amp"]) / (2 * np.pi * sigma**2)
        light += brightness * np.exp(-0.5 * (major**2 + minor**2) / sigma**2)
    return light * pixel_scale**2


def _display_signed(values: np.ndarray) -> list[int]:
    limit = max(float(np.quantile(np.abs(values), 0.995)), 1e-12)
    return np.clip(127.5 + 127.5 * values / limit, 0, 255).astype(np.uint8).ravel().tolist()


def load_result(result_dir: Path) -> ViewerData:
    result_dir = result_dir.expanduser().resolve()
    with (result_dir / "kwargs_result.json").open() as handle:
        result = json.load(handle)
    with (result_dir / "config.json").open() as handle:
        config = json.load(handle)
    lens_kwargs = result["kwargs_lens"]
    if len(lens_kwargs) < 2 or "theta_E" not in lens_kwargs[0] or "gamma1" not in lens_kwargs[1]:
        raise ValueError("This viewer currently expects kwargs_lens = [SIE, SHEAR].")
    image = np.asarray(fits.getdata(result_dir / "data" / "Data_cutout.fits"), dtype=float)
    source = np.asarray(fits.getdata(result_dir / "kwargs_source_pixels.fits"), dtype=float)
    psf = np.asarray(fits.getdata(result_dir / "data" / "psf.fits"), dtype=float)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.maximum(psf, 0.0)
    if psf.sum() <= 0:
        raise ValueError("The saved PSF has no positive flux.")
    psf /= psf.sum()
    mask = np.asarray(fits.getdata(result_dir / "data" / "mask_1.fits"), dtype=bool)
    pixel_scale = float(config["pixel_scale"])
    lens = SIELens(lens_kwargs[0], lens_kwargs[1])
    lens_light = _lens_light_image(result.get("kwargs_lens_light", []), image.shape, pixel_scale)
    height, width = image.shape
    half_x, half_y = width * pixel_scale / 2, height * pixel_scale / 2
    source_extent = _source_extent(lens, mask, pixel_scale, float(config.get("source_grid_scale", 1.0)))
    saved_scale = _saved_source_pixel_scale(result_dir)
    if saved_scale is not None:
        # The adaptive uniform grid is frozen while sampling.  The output log
        # records its exact spacing, whereas kwargs_result contains the later
        # posterior lens parameters.  Keep the grid centre from ray tracing
        # and restore that saved spacing for click-to-coordinate fidelity.
        source_center_x = (source_extent[0] + source_extent[1]) / 2
        source_center_y = (source_extent[2] + source_extent[3]) / 2
        source_half_x, source_half_y = source.shape[1] * saved_scale / 2, source.shape[0] * saved_scale / 2
        source_extent = [source_center_x - source_half_x, source_center_x + source_half_x,
                         source_center_y - source_half_y, source_center_y + source_half_y]
    return ViewerData(
        result_dir=result_dir,
        lens=lens,
        image=image,
        source=source,
        ring=image - lens_light,
        psf=psf,
        pixel_scale=pixel_scale,
        image_extent=[-half_x, half_x, -half_y, half_y],
        source_extent=source_extent,
    )


def _all_images(model: ViewerData, beta_x: float, beta_y: float) -> list[dict[str, float]]:
    """Find all in-frame solutions of beta = theta - alpha(theta)."""
    x0, x1, y0, y1 = model.image_extent
    roots: list[np.ndarray] = []

    def equation(theta: np.ndarray) -> np.ndarray:
        bx, by = model.lens.ray_shoot(theta[0], theta[1])
        return np.array([bx - beta_x, by - beta_y])

    # Pick solver seeds from the nearest points in a forward-ray-shooting
    # grid.  This is substantially more reliable than starting everywhere in
    # the full cutout: each seed already maps close to the chosen beta.
    seed_axis = np.linspace(x0 * 0.97, x1 * 0.97, 51)
    seed_x, seed_y = np.meshgrid(seed_axis, seed_axis)
    trial_beta_x, trial_beta_y = model.lens.ray_shoot(seed_x, seed_y)
    distance = (trial_beta_x - beta_x) ** 2 + (trial_beta_y - beta_y) ** 2
    nearest = np.argpartition(distance.ravel(), 48)[:48]
    for index in nearest:
            sx, sy = seed_x.ravel()[index], seed_y.ravel()[index]
            solution = root(equation, [sx, sy], method="hybr", options={"xtol": 1e-10, "maxfev": 100})
            theta = np.asarray(solution.x, dtype=float)
            if not solution.success or not np.all(np.isfinite(theta)) or np.linalg.norm(equation(theta)) > 2e-7:
                continue
            if not (x0 <= theta[0] <= x1 and y0 <= theta[1] <= y1):
                continue
            if all(np.linalg.norm(theta - existing) > 2e-4 for existing in roots):
                roots.append(theta)
    images = []
    for theta in sorted(roots, key=lambda p: (p[0], p[1])):
        determinant = model.lens.inverse_magnification(float(theta[0]), float(theta[1]))
        if np.isfinite(determinant) and abs(determinant) > 1e-8:
            images.append({"x": float(theta[0]), "y": float(theta[1]), "mu": float(1 / determinant)})
    return images


def _saved_sources_path(result_dir: Path) -> Path:
    return result_dir / SAVED_SOURCES_FILENAME


def _read_saved_sources(result_dir: Path) -> tuple[dict | None, str | None]:
    """Read saved explorer points without making a stale/corrupt file fatal."""
    path = _saved_sources_path(result_dir)
    if not path.is_file():
        return None, None
    try:
        saved = json.loads(path.read_text())
        if not isinstance(saved.get("point_sources"), list):
            raise ValueError("missing point_sources list")
        return saved, None
    except Exception as error:
        return None, f"Could not read {path.name}: {error}"


def _validated_sources(model: ViewerData, sources: object) -> list[dict[str, float | str]]:
    if not isinstance(sources, list) or not sources:
        raise ValueError("At least one point source is required.")
    x0, x1, y0, y1 = model.source_extent
    output = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            raise ValueError(f"Source {index} is not an object.")
        x, y = float(source["x"]), float(source["y"])
        amplitude = max(0.0, float(source.get("amplitude", 1.0)))
        if not np.isfinite([x, y, amplitude]).all():
            raise ValueError(f"Source {index} contains a non-finite value.")
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError(f"Source {index} is outside the saved source-plane extent.")
        output.append({"id": str(source.get("id") or f"ps-{index}"), "x": x, "y": y, "amplitude": amplitude})
    return output


def _saved_source_payload(model: ViewerData, sources: list[dict[str, float | str]]) -> dict:
    x0, x1, y0, y1 = model.source_extent
    point_sources = []
    for index, source in enumerate(sources, start=1):
        x, y, amplitude = float(source["x"]), float(source["y"]), float(source["amplitude"])
        images = _all_images(model, x, y)
        point_sources.append({
            "id": source["id"],
            "label": f"Point source {index}",
            "source_plane_position_arcsec": {"beta_x": x, "beta_y": y},
            "intrinsic_flux": amplitude,
            "image_plane_images": [
                {"theta_x": image["x"], "theta_y": image["y"], "magnification": image["mu"],
                 "parity": "negative" if image["mu"] < 0 else "positive",
                 "lensed_relative_flux": abs(image["mu"]) * amplitude}
                for image in images
            ],
        })
    return {
        "schema_version": 1,
        "model_result_dir": str(model.result_dir),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source_plane": {
            "extent_arcsec": [x0, x1, y0, y1],
            "pixel_scale_arcsec": {"x": (x1 - x0) / model.source.shape[1], "y": (y1 - y0) / model.source.shape[0]},
        },
        "point_sources": point_sources,
    }


@app.get("/")
def index():
    return render_template("index.html", default_result_dir=str(DEFAULT_RESULT_DIR))


@app.post("/api/load")
def api_load():
    global ACTIVE
    try:
        payload = request.get_json(silent=True) or {}
        ACTIVE = load_result(Path(payload.get("result_dir") or DEFAULT_RESULT_DIR))
        saved_sources, saved_sources_error = _read_saved_sources(ACTIVE.result_dir)
        response = {
            "result_dir": str(ACTIVE.result_dir),
            "image": {"width": int(ACTIVE.image.shape[1]), "height": int(ACTIVE.image.shape[0]),
                      "pixels": _display(np.flipud(ACTIVE.image), logarithmic=False), "extent": ACTIVE.image_extent,
                      "palette": TWILIGHT_PALETTE},
            "source": {"width": int(ACTIVE.source.shape[1]), "height": int(ACTIVE.source.shape[0]),
                       "pixels": _display(np.flipud(ACTIVE.source), logarithmic=False), "extent": ACTIVE.source_extent,
                       "palette": TWILIGHT_PALETTE},
            "ring": {"width": int(ACTIVE.ring.shape[1]), "height": int(ACTIVE.ring.shape[0]),
                     "pixels": _display_signed(np.flipud(ACTIVE.ring)), "extent": ACTIVE.image_extent},
            "psf": {"width": int(ACTIVE.psf.shape[1]), "height": int(ACTIVE.psf.shape[0]),
                    "pixels": ACTIVE.psf.ravel().tolist()},
            "pixel_scale": ACTIVE.pixel_scale,
            "lens": {"theta_E": ACTIVE.lens.sie["theta_E"], "q": ACTIVE.lens.q},
            "saved_sources": saved_sources,
            "saved_sources_file": SAVED_SOURCES_FILENAME if saved_sources else None,
        }
        if saved_sources_error:
            response["saved_sources_error"] = saved_sources_error
        return jsonify(response)
    except Exception as error:
        return jsonify({"message": str(error)}), 400


@app.post("/api/trace")
def api_trace():
    if ACTIVE is None:
        return jsonify({"message": "Please load a model first."}), 400
    try:
        payload = request.get_json(force=True)
        sources = payload.get("sources", [])
        response = []
        for source in sources:
            x, y = float(source["x"]), float(source["y"])
            amplitude = max(0.0, float(source.get("amplitude", 1.0)))
            response.append({
                "id": str(source["id"]), "images": _all_images(ACTIVE, x, y), "amplitude": amplitude,
            })
        return jsonify({"traces": response})
    except Exception as error:
        return jsonify({"message": str(error)}), 400


@app.post("/api/save-sources")
def api_save_sources():
    if ACTIVE is None:
        return jsonify({"message": "Please load a model first."}), 400
    try:
        payload = request.get_json(force=True)
        requested_dir = Path(payload.get("result_dir") or ACTIVE.result_dir).resolve()
        if requested_dir != ACTIVE.result_dir.resolve():
            raise ValueError("Save location must be the currently loaded model-result directory.")
        sources = _validated_sources(ACTIVE, payload.get("sources"))
        saved = _saved_source_payload(ACTIVE, sources)
        target = _saved_sources_path(ACTIVE.result_dir)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(saved, indent=2) + "\n")
        temporary.replace(target)
        return jsonify({"saved_file": str(target), "source_count": len(sources)})
    except Exception as error:
        return jsonify({"message": str(error)}), 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--port", type=int, default=5052)
    args = parser.parse_args()
    # Load before serving, so an invalid path is reported in the terminal.
    global ACTIVE
    ACTIVE = load_result(args.result_dir)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
