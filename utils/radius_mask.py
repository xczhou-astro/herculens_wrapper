#!/usr/bin/env python
"""
Build circular radius masks for detected point sources.

Creates a combined mask and one HDU per source in {band}/radius_mask/.
Per-source mask radii can override a universal default radius.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from matplotlib.patches import Circle


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


SCRIPT_DIR = Path(__file__).resolve().parent
NIRCAM_BANDS = ("F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W")
CROP_SIZE = 61
PIXEL_SCALE = 0.03
SOURCE_RADIUS_PATTERN = re.compile(r"^\s*(\d+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")


def parse_band(value: str) -> str:
    """Normalize a NIRCam filter name."""
    band = value.upper()
    if band not in NIRCAM_BANDS:
        bands = ", ".join(NIRCAM_BANDS)
        raise argparse.ArgumentTypeError(
            f"Unknown band {value!r}. Expected one of: {bands}."
        )
    return band


def parse_source_radius(value: str) -> tuple[int, float]:
    """Parse a per-source radius override like '3:0.12' (arcsec)."""
    match = SOURCE_RADIUS_PATTERN.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"Invalid source radius {value!r}. Expected format ID:RADIUS, e.g. 1:0.12."
        )
    return int(match.group(1)), float(match.group(2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create circular radius masks for point sources."
    )
    parser.add_argument(
        "--band",
        type=parse_band,
        default="F150W",
        help="Band to process (default: F150W).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Root directory containing {band}/ data and photometry files.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Detection catalog path. Defaults to {band}/photometry/point_source_catalog.ecsv.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.09,
        help="Default circular mask radius in arcsec (default: 0.09).",
    )
    parser.add_argument(
        "--source-radius",
        type=parse_source_radius,
        action="append",
        default=[],
        help=(
            "Per-source radius override in arcsec, formatted as ID:RADIUS. "
            "Repeat for multiple sources, e.g. --source-radius 1:0.12 --source-radius 2:0.15."
        ),
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=PIXEL_SCALE,
        help="Plate scale in arcsec/pixel for overlays.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="point_source_radius_masks.fits",
        help="Output FITS filename under {band}/radius_mask/.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=np.nan,
        help="Value used in masked_image.fits for masked pixels (default: NaN).",
    )
    return parser.parse_args()


def crop_center(array: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Return a square crop centered on the array."""
    half = size // 2
    cy, cx = array.shape[0] // 2, array.shape[1] // 2
    return array[cy - half : cy + half + 1, cx - half : cx + half + 1].copy()


def arcsec_to_pixel(
    ra_arcsec: float,
    dec_arcsec: float,
    center: float,
    pixel_scale: float,
) -> tuple[float, float]:
    """Convert arcsecond offsets to pixel coordinates."""
    x = ra_arcsec / pixel_scale + center
    y = dec_arcsec / pixel_scale + center
    return x, y


def ra_dec_extent_arcsec(pixel_scale: float, size: int = CROP_SIZE) -> list[float]:
    """Matplotlib extent [ra_min, ra_max, dec_min, dec_max] in arcseconds."""
    half = (size - 1) / 2.0
    span_arcsec = half * pixel_scale
    return [-span_arcsec, span_arcsec, -span_arcsec, span_arcsec]


def resolve_catalog_path(input_root: Path, band: str, catalog: Path | None) -> Path:
    """Return the detection catalog path."""
    if catalog is not None:
        return catalog
    return input_root / band / "photometry" / "point_source_catalog.ecsv"


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    """Load the detection catalog."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing detection catalog: {catalog_path}")
    table = Table.read(catalog_path, format="ascii.ecsv")
    required = {"id", "ra_arcsec", "dec_arcsec"}
    missing = required - set(table.colnames)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{catalog_path} is missing required columns: {missing_cols}")
    return table.to_pandas().sort_values("id").reset_index(drop=True)


def load_science_image(input_root: Path, band: str) -> tuple[np.ndarray, float]:
    """Load the background-subtracted cropped science image."""
    path = input_root / band / "lens_light_subtracted.fits"
    if not path.exists():
        raise FileNotFoundError(f"Missing science image for {band}: {path}")
    data = crop_center(fits.getdata(path).astype(np.float64))
    masked = np.where(data != 0, data, np.nan)
    _, median, _ = sigma_clipped_stats(masked, sigma=3.0)
    valid = data != 0
    return np.where(valid, data - median, 0.0), float(median)


def build_radius_lookup(
    default_radius_arcsec: float,
    source_radii: list[tuple[int, float]],
) -> dict[int, float]:
    """Build a lookup table of mask radius in arcsec."""
    radii = {source_id: radius for source_id, radius in source_radii}
    if len(radii) != len(source_radii):
        raise ValueError("Duplicate source IDs in --source-radius overrides.")
    return radii


def radius_arcsec_for_source(
    source_id: int,
    default_radius_arcsec: float,
    source_radii: dict[int, float],
) -> float:
    """Return the mask radius in arcsec for one source."""
    return source_radii.get(source_id, default_radius_arcsec)


def radius_arcsec_to_pixels(radius_arcsec: float, pixel_scale: float) -> float:
    """Convert a mask radius from arcsec to pixels."""
    return radius_arcsec / pixel_scale


def circular_mask(
    shape: tuple[int, int],
    x: float,
    y: float,
    radius_px: float,
) -> np.ndarray:
    """Return a boolean mask for a circle centered at (x, y)."""
    yy, xx = np.indices(shape)
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius_px**2


def build_source_masks(
    catalog: pd.DataFrame,
    default_radius_arcsec: float,
    source_radii: dict[int, float],
    pixel_scale: float,
    shape: tuple[int, int] = (CROP_SIZE, CROP_SIZE),
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, pd.Series, float]]]:
    """Build combined and per-source circular masks."""
    center = (CROP_SIZE - 1) / 2.0
    combined = np.zeros(shape, dtype=bool)
    source_masks: list[tuple[int, np.ndarray, pd.Series, float]] = []

    for row in catalog.itertuples(index=False):
        source_id = int(row.id)
        radius_arcsec = radius_arcsec_for_source(
            source_id, default_radius_arcsec, source_radii
        )
        radius_px = radius_arcsec_to_pixels(radius_arcsec, pixel_scale)
        x, y = arcsec_to_pixel(row.ra_arcsec, row.dec_arcsec, center, pixel_scale)
        mask = circular_mask(shape, x, y, radius_px)
        combined |= mask
        source_masks.append((source_id, mask, pd.Series(row._asdict()), radius_arcsec))

    return combined, source_masks


def apply_mask(
    science: np.ndarray,
    combined_mask: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Apply the combined mask to the science image."""
    masked = science.astype(np.float64).copy()
    masked[combined_mask] = fill_value
    return masked


def mask_to_fits_array(mask: np.ndarray) -> np.ndarray:
    """Convert internal mask (True=masked) to FITS values (0=masked, 1=unmasked)."""
    return np.where(mask, 0, 1).astype(np.uint8)


def write_mask_fits(
    output_path: Path,
    band: str,
    default_radius_arcsec: float,
    combined_mask: np.ndarray,
    source_masks: list[tuple[int, np.ndarray, pd.Series, float]],
    pixel_scale: float,
) -> None:
    """Write combined and per-source radius masks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary = fits.PrimaryHDU(mask_to_fits_array(combined_mask))
    primary.header["BAND"] = band
    primary.header["MASKTYPE"] = "Circular radius"
    primary.header["DEFRASC"] = (default_radius_arcsec, "Default mask radius in arcsec")
    primary.header["PIXSCALE"] = (pixel_scale, "arcsec/pixel")
    primary.header["NSRC"] = (len(source_masks), "Number of source HDUs")
    primary.header["COMMENT"] = "Combined mask: 0=masked, 1=unmasked"

    hdus = [primary]
    for source_id, mask, row, radius_arcsec in source_masks:
        hdu = fits.ImageHDU(mask_to_fits_array(mask), name=f"SRC{source_id:03d}")
        hdu.header["SRCID"] = source_id
        hdu.header["RADASC"] = (radius_arcsec, "Mask radius in arcsec")
        hdu.header["RAASEC"] = (float(row["ra_arcsec"]), "Detection RA offset arcsec")
        hdu.header["DECSEC"] = (float(row["dec_arcsec"]), "Detection Dec offset arcsec")
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(output_path, overwrite=True)


def plot_masked_image(
    science: np.ndarray,
    masked: np.ndarray,
    catalog: pd.DataFrame,
    default_radius_arcsec: float,
    source_radii: dict[int, float],
    band: str,
    pixel_scale: float,
    output_path: Path,
    plot_scale: str = "linear",
) -> None:
    """Plot original and radius-masked images side by side."""
    from matplotlib.colors import LogNorm
    extent = ra_dec_extent_arcsec(pixel_scale)
    valid = science != 0
    cmap = plt.get_cmap("tab20")

    if plot_scale == "log":
        pos_science = science[science > 0]
        if pos_science.size > 0:
            vmin = float(np.nanpercentile(pos_science, 1.0))
            vmax = float(np.nanpercentile(pos_science, 99.0))
            vmax = max(vmax, vmin * 10.0)
            vmin = max(vmin, vmax * 1e-12)
        else:
            vmin, vmax = 1e-3, 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmax = np.nanpercentile(science[valid], 99.5) if np.any(valid) else None
        vmin = 0
        norm = None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax_orig, ax_masked = axes
    ax_orig.imshow(
        science,
        origin="lower",
        cmap="viridis",
        extent=extent,
        norm=norm,
        vmin=None if plot_scale == "log" else vmin,
        vmax=None if plot_scale == "log" else vmax,
    )
    ax_orig.set_title(f"{band} science image ({plot_scale})")
    ax_orig.set_xlabel("RA (arcsec)")
    ax_orig.set_ylabel("Dec (arcsec)")

    masked_cmap = plt.get_cmap("viridis").copy()
    masked_cmap.set_bad(color="lightgray")
    ax_masked.imshow(
        np.ma.masked_invalid(masked),
        origin="lower",
        cmap=masked_cmap,
        extent=extent,
        norm=norm,
        vmin=None if plot_scale == "log" else vmin,
        vmax=None if plot_scale == "log" else vmax,
    )
    ax_masked.set_title(f"{band} after radius masking ({plot_scale})")
    ax_masked.set_xlabel("RA (arcsec)")
    ax_masked.set_ylabel("Dec (arcsec)")

    for index, row in enumerate(catalog.itertuples(index=False)):
        color = cmap(index % 20)
        source_id = int(row.id)
        radius_arcsec = radius_arcsec_for_source(
            source_id, default_radius_arcsec, source_radii
        )
        circle = Circle(
            (row.ra_arcsec, row.dec_arcsec),
            radius=radius_arcsec,
            fill=False,
            ec=color,
            lw=1.2,
        )
        ax_orig.add_patch(circle)
        ax_orig.annotate(
            str(source_id),
            (row.ra_arcsec, row.dec_arcsec),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
        )
        ax_masked.add_patch(Circle(
            (row.ra_arcsec, row.dec_arcsec),
            radius=radius_arcsec,
            fill=False,
            ec=color,
            lw=1.2,
        ))

    for ax in axes:
        ax.axhline(0.0, color="white", ls="--", lw=0.6, alpha=0.5)
        ax.axvline(0.0, color="white", ls="--", lw=0.6, alpha=0.5)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_masked_image_fits(
    output_path: Path,
    band: str,
    masked: np.ndarray,
    background_median: float,
    default_radius_arcsec: float,
    fill_value: float,
) -> None:
    """Write the masked science image."""
    hdu = fits.PrimaryHDU(masked.astype(np.float32))
    hdu.header["BAND"] = band
    hdu.header["BKGSUB"] = (True, "Median background removed from data.fits")
    hdu.header["BKGMED"] = (background_median, "Median subtracted from data.fits")
    hdu.header["DEFRASC"] = (default_radius_arcsec, "Default mask radius in arcsec")
    if np.isnan(fill_value):
        hdu.header["FILLVAL"] = ("NaN", "Value assigned to masked pixels")
    else:
        hdu.header["FILLVAL"] = (fill_value, "Value assigned to masked pixels")
    hdu.writeto(output_path, overwrite=True)


def main() -> None:
    args = parse_args()
    catalog_path = resolve_catalog_path(args.input_root, args.band, args.catalog)
    catalog = load_catalog(catalog_path)
    source_radii = build_radius_lookup(args.radius, args.source_radius)

    unknown_ids = sorted(set(source_radii) - set(catalog["id"].astype(int)))
    if unknown_ids:
        ids = ", ".join(str(source_id) for source_id in unknown_ids)
        raise ValueError(f"Unknown source IDs in --source-radius: {ids}")

    combined_mask, source_masks = build_source_masks(
        catalog,
        args.radius,
        source_radii,
        args.pixel_scale,
    )
    science, background_median = load_science_image(args.input_root, args.band)
    masked = apply_mask(science, combined_mask, args.fill_value)

    output_dir = args.input_root / args.band / "radius_mask"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.txt"
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Invoked: {shlex.join([sys.executable, *sys.argv])}")

    mask_path = output_dir / args.output_name
    masked_image_path = output_dir / "masked_image.fits"
    figure_path_linear = output_dir / "radius_masked_image_linear.png"
    figure_path_log = output_dir / "radius_masked_image_log.png"

    write_mask_fits(
        mask_path,
        args.band,
        args.radius,
        combined_mask,
        source_masks,
        args.pixel_scale,
    )
    write_masked_image_fits(
        masked_image_path,
        args.band,
        masked,
        background_median,
        args.radius,
        args.fill_value,
    )
    plot_masked_image(
        science,
        masked,
        catalog,
        args.radius,
        source_radii,
        args.band,
        args.pixel_scale,
        figure_path_linear,
        plot_scale="linear",
    )
    plot_masked_image(
        science,
        masked,
        catalog,
        args.radius,
        source_radii,
        args.band,
        args.pixel_scale,
        figure_path_log,
        plot_scale="log",
    )

    print(f"Band: {args.band}")
    print(f"Default radius: {args.radius} arcsec")
    if source_radii:
        overrides = ", ".join(
            f"{source_id}:{radius} arcsec"
            for source_id, radius in sorted(source_radii.items())
        )
        print(f"Per-source radii: {overrides}")
    print(f"Sources masked: {len(source_masks)}")
    print(f"Catalog: {catalog_path}")
    print(f"Mask FITS written to: {mask_path}")
    print(f"Masked image FITS written to: {masked_image_path}")
    print(f"Linear figure written to: {figure_path_linear}")
    print(f"Log figure written to: {figure_path_log}")


if __name__ == "__main__":
    main()
