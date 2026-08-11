#!/usr/bin/env python3
"""Measure relative sky orientation of FITS exposures from their SCI WCS.

The reported position angle is the direction of increasing image x pixel,
measured east of north on the sky.  ``orientation_difference_deg`` is the
orientation of an exposure minus that of the reference exposure; the opposite
quantity, ``rotation_to_align_x_axis_deg``, aligns its x axis with the
reference x axis in sky coordinates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def _wrap_degrees(angle):
    """Return an angle in the interval [-180, 180)."""
    return float((angle + 180.0) % 360.0 - 180.0)


def _science_wcs(path):
    """Return the first image extension with a usable celestial WCS."""
    with fits.open(path) as hdus:
        preferred = [hdu for hdu in hdus if hdu.header.get('EXTNAME') == 'SCI']
        candidates = [*preferred, *[hdu for hdu in hdus if hdu not in preferred]]
        for hdu in candidates:
            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue
            wcs = WCS(hdu.header).celestial
            if wcs.has_celestial:
                return wcs, tuple(hdu.data.shape), hdu.header.get('EXTNAME', 'PRIMARY')
    raise ValueError(f'No 2D celestial WCS image extension was found in {path}.')


def _position_angle_of_x_axis(wcs, shape):
    """Measure local PA of increasing x at the image centre, east of north."""
    ny, nx = shape
    x_center = (nx - 1) / 2.0
    y_center = (ny - 1) / 2.0
    centre, one_x_pixel = wcs.pixel_to_world(x_center, y_center), wcs.pixel_to_world(
        x_center + 1.0, y_center,
    )
    if not isinstance(centre, SkyCoord) or not isinstance(one_x_pixel, SkyCoord):
        raise TypeError('The celestial WCS did not return SkyCoord positions.')

    delta_ra = (one_x_pixel.ra - centre.ra).wrap_at(180.0 * u.deg)
    east = delta_ra.to_value(u.deg) * np.cos(centre.dec.to_value(u.rad))
    north = (one_x_pixel.dec - centre.dec).to_value(u.deg)
    position_angle = np.degrees(np.arctan2(east, north)) % 360.0
    pixel_scale_arcsec = centre.separation(one_x_pixel).to_value(u.arcsec)
    return float(position_angle), float(pixel_scale_arcsec)


def measure_rotations(input_dir, pattern='*cutout*.fits', reference_index=0):
    """Return WCS orientations and relative rotations for matching FITS files."""
    paths = sorted(Path(input_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No FITS files matching {pattern!r} in {input_dir!r}.')
    if not 0 <= reference_index < len(paths):
        raise IndexError(
            f'reference_index={reference_index} is outside the {len(paths)} matched exposures.'
        )

    measurements = []
    for path in paths:
        wcs, shape, extension = _science_wcs(path)
        orientation, pixel_scale = _position_angle_of_x_axis(wcs, shape)
        measurements.append({
            'file': path.name,
            'extension': extension,
            'shape': list(shape),
            'x_axis_position_angle_east_of_north_deg': orientation,
            'x_pixel_scale_arcsec': pixel_scale,
        })

    reference_orientation = measurements[reference_index]['x_axis_position_angle_east_of_north_deg']
    for measurement in measurements:
        difference = _wrap_degrees(
            measurement['x_axis_position_angle_east_of_north_deg'] - reference_orientation,
        )
        measurement['orientation_difference_deg'] = difference
        measurement['rotation_to_align_x_axis_deg'] = -difference

    return {
        'reference_file': measurements[reference_index]['file'],
        'reference_index': reference_index,
        'sign_convention': (
            'orientation_difference_deg = exposure PA minus reference PA; '
            'PA is measured east of north for increasing image x. '
            'rotation_to_align_x_axis_deg is its negative.'
        ),
        'exposures': measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_dir', nargs='?',
        default='/Users/xczhou/Desktop/modelling/single_exposure_data/dataset/F277W',
        help='Directory containing the exposure FITS files.',
    )
    parser.add_argument('--pattern', default='*cutout*.fits', help='FITS filename glob.')
    parser.add_argument('--reference-index', type=int, default=0, help='Sorted exposure index used as reference.')
    parser.add_argument('--output', default=None, help='Optional JSON output path.')
    args = parser.parse_args()

    result = measure_rotations(args.input_dir, args.pattern, args.reference_index)
    print(f"Reference exposure: {result['reference_file']}")
    for index, exposure in enumerate(result['exposures']):
        print(
            f"{index}: {exposure['file']}: "
            f"PA_x={exposure['x_axis_position_angle_east_of_north_deg']:.6f} deg, "
            f"delta={exposure['orientation_difference_deg']:+.6f} deg, "
            f"align rotation={exposure['rotation_to_align_x_axis_deg']:+.6f} deg"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as handle:
            json.dump(result, handle, indent=2)
        print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
