#!/usr/bin/env python3
"""Crop and renormalize per-band PSF FITS images without touching originals.

The default mode finds the smallest *odd* square, centred on the PSF peak,
that retains at least 99.5 percent of the original PSF integral.  The cropped
kernel is then normalized to a sum of one, as required for image convolution.

Examples
--------
# Scientifically conservative production kernels (99.5% retained flux)
python utils/crop_psfs.py \
  --input-root "/path/to/primer" \
  --output-root "/path/to/primer_psf_99p5"

# A faster, deliberately fixed 61 x 61-pixel test set
python utils/crop_psfs.py \
  --input-root "/path/to/primer" \
  --output-root "/path/to/primer_psf_61" \
  --size 61

The fixed-size option is useful for convergence tests, but it may discard a
non-negligible fraction of a broad PSF wing.  Always compare the inferred lens
light and mass parameters with the default retained-flux result before using a
smaller kernel for a final analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing F* band directories with psf.fits files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New root directory for cropped PSFs; original files are never changed.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.995,
        help="Minimum original PSF integral retained by automatic cropping (default: 0.995).",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Use this fixed odd square size instead of selecting a retained-flux size.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        help="Optional band names, e.g. F277W F356W F444W. Default: every F* directory.",
    )
    parser.add_argument(
        "--psf-name",
        default="psf.fits",
        help="Input PSF filename within each band directory (default: psf.fits).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output FITS file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the proposed crops without writing FITS files.",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in the interval (0, 1].")
    if args.size is not None and (args.size < 3 or args.size % 2 == 0):
        parser.error("--size must be an odd integer of at least 3.")
    return args


def _crop_bounds(shape: tuple[int, int], center_y: int, center_x: int, size: int) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive bounds for an odd square crop, or raise."""
    half_size = size // 2
    y0, y1 = center_y - half_size, center_y + half_size + 1
    x0, x1 = center_x - half_size, center_x + half_size + 1
    if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
        raise ValueError(
            f"{size}x{size} crop around ({center_x}, {center_y}) does not fit within {shape}."
        )
    return y0, y1, x0, x1


def _smallest_size_for_fraction(
    psf: np.ndarray, center_y: int, center_x: int, fraction: float
) -> tuple[int, float]:
    """Find the smallest valid odd square that retains the requested integral."""
    total = float(psf.sum())
    max_half_size = min(center_y, center_x, psf.shape[0] - center_y - 1, psf.shape[1] - center_x - 1)
    for half_size in range(max_half_size + 1):
        size = 2 * half_size + 1
        y0, y1, x0, x1 = _crop_bounds(psf.shape, center_y, center_x, size)
        retained = float(psf[y0:y1, x0:x1].sum() / total)
        if retained >= fraction:
            return size, retained
    raise RuntimeError("The requested retained fraction cannot be reached within the input PSF.")


def _update_header(header: fits.Header, *, source: Path, center_x: int, center_y: int,
                   bounds: tuple[int, int, int, int], original_sum: float,
                   retained_fraction: float, requested_fraction: float | None) -> fits.Header:
    """Return a copy of the FITS header documenting the crop and normalization."""
    y0, y1, x0, x1 = bounds
    result = header.copy()
    # If a PSF happens to carry a WCS, shift its reference pixel to the crop frame.
    if "CRPIX1" in result:
        result["CRPIX1"] = (result["CRPIX1"] - x0, "Shifted for PSF crop")
    if "CRPIX2" in result:
        result["CRPIX2"] = (result["CRPIX2"] - y0, "Shifted for PSF crop")
    result["PSFNORM"] = (True, "Normalized to unit sum")
    result["PSFRAWSM"] = (original_sum, "Input PSF integral")
    result["PSFKEEP"] = (retained_fraction, "Fraction retained pre-normalization")
    result["PSFCENX"] = (center_x, "Peak x; zero based")
    result["PSFCENY"] = (center_y, "Peak y; zero based")
    result["PSFX0"] = (x0, "Crop x start; inclusive")
    result["PSFY0"] = (y0, "Crop y start; inclusive")
    result["PSFX1"] = (x1, "Crop x stop; exclusive")
    result["PSFY1"] = (y1, "Crop y stop; exclusive")
    result["PSFMETH"] = ("PEAKSQ", "Peak-centred square")
    if requested_fraction is not None:
        result["PSFTARG"] = (requested_fraction, "Requested retained PSF fraction")
    result.add_history(f"Cropped from {source.name} about its maximum pixel.")
    result.add_history("PSF renormalized so that sum(PSF) = 1.")
    return result


def crop_one(input_path: Path, output_path: Path, args: argparse.Namespace) -> tuple[int, float]:
    """Crop a PSF FITS file and return (size, retained_fraction)."""
    with fits.open(input_path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"{input_path} has no primary-image data.")
        psf = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header

    if psf.ndim != 2:
        raise ValueError(f"{input_path} must contain a 2-D PSF; found shape {psf.shape}.")
    if not np.all(np.isfinite(psf)):
        raise ValueError(f"{input_path} contains non-finite PSF values; repair it before cropping.")
    original_sum = float(psf.sum())
    if original_sum <= 0.0:
        raise ValueError(f"{input_path} has non-positive total PSF integral ({original_sum}).")

    center_y, center_x = np.unravel_index(np.argmax(psf), psf.shape)
    if args.size is None:
        size, retained = _smallest_size_for_fraction(psf, center_y, center_x, args.fraction)
        requested_fraction = args.fraction
    else:
        size = args.size
        y0, y1, x0, x1 = _crop_bounds(psf.shape, center_y, center_x, size)
        retained = float(psf[y0:y1, x0:x1].sum() / original_sum)
        requested_fraction = None

    bounds = _crop_bounds(psf.shape, center_y, center_x, size)
    y0, y1, x0, x1 = bounds
    cropped = psf[y0:y1, x0:x1].copy()
    crop_sum = float(cropped.sum())
    if crop_sum <= 0.0:
        raise ValueError(f"{input_path}: cropped PSF has non-positive integral ({crop_sum}).")
    cropped /= crop_sum

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_header = _update_header(
            header,
            source=input_path,
            center_x=center_x,
            center_y=center_y,
            bounds=bounds,
            original_sum=original_sum,
            retained_fraction=retained,
            requested_fraction=requested_fraction,
        )
        fits.PrimaryHDU(data=cropped, header=out_header).writeto(output_path, overwrite=args.overwrite)
    return size, retained


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if output_root == input_root:
        raise ValueError("--output-root must differ from --input-root to protect the original PSFs.")

    if args.bands:
        band_dirs = [input_root / band for band in args.bands]
    else:
        band_dirs = sorted(path for path in input_root.glob("F*") if path.is_dir())
    if not band_dirs:
        raise ValueError(f"No band directories found below {input_root}.")

    mode = f"fixed {args.size}x{args.size}" if args.size else f"at least {args.fraction:.2%} retained"
    print(f"Cropping PSFs from {input_root} ({mode})")
    for band_dir in band_dirs:
        input_path = band_dir / args.psf_name
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing PSF: {input_path}")
        output_path = output_root / band_dir.name / args.psf_name
        if output_path.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"Output exists (use --overwrite): {output_path}")
        size, retained = crop_one(input_path, output_path, args)
        action = "would write" if args.dry_run else "wrote"
        print(f"{band_dir.name:6s} {size:3d}x{size:<3d}  keeps {retained:8.5%}  {action} {output_path}")


if __name__ == "__main__":
    main()
