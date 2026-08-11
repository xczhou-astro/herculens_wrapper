#!/usr/bin/env python3
"""Reproject same-band FITS exposures onto one anchor exposure's WCS grid.

This is WCS reprojection, not a centre-only image rotation: it handles the
relative rotation, shift, and local scale encoded by each exposure WCS.  It
creates new files and never alters the original exposures.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


def _science_hdu_index(hdus, path):
    for index, hdu in enumerate(hdus):
        if hdu.header.get('EXTNAME') == 'SCI' and hdu.data is not None:
            return index
    raise ValueError(f'No SCI image extension found in {path}.')


def _celestial_wcs(hdu, path):
    wcs = WCS(hdu.header).celestial
    if not wcs.has_celestial:
        raise ValueError(f'SCI extension in {path} has no celestial WCS.')
    return wcs


def _reference_to_source_coordinates(reference_wcs, reference_shape, source_wcs):
    """Map every anchor-grid pixel centre into a source exposure pixel."""
    y_pixels, x_pixels = np.indices(reference_shape, dtype=np.float64)
    ra, dec = reference_wcs.pixel_to_world_values(x_pixels, y_pixels)
    source_x, source_y = source_wcs.world_to_pixel_values(ra, dec)
    return np.asarray(source_y), np.asarray(source_x)


def _resample_continuous(data, coordinates, *, cval, order=1):
    return map_coordinates(
        np.asarray(data, dtype=np.float64), coordinates, order=order,
        mode='constant', cval=cval, prefilter=False,
    )


def _resample_extension(data, extension_name, coordinates):
    """Resample science, uncertainty, variance, and quality planes safely."""
    name = str(extension_name or '').upper()
    original_dtype = np.asarray(data).dtype
    if name == 'DQ':
        invalid_value = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 1
        return map_coordinates(
            np.asarray(data), coordinates, order=0, mode='constant',
            cval=invalid_value, prefilter=False,
        ).astype(original_dtype)
    if name == 'ERR':
        variance = np.square(np.asarray(data, dtype=np.float64))
        return np.sqrt(_resample_continuous(variance, coordinates, cval=np.inf))
    if name.startswith('VAR_'):
        return _resample_continuous(data, coordinates, cval=np.inf)
    if name == 'AREA':
        return _resample_continuous(data, coordinates, cval=0.0)
    return _resample_continuous(data, coordinates, cval=np.nan)


def align_exposure(input_path, anchor_path, output_path):
    """Resample every 2D image extension in ``input_path`` onto ``anchor_path``."""
    with fits.open(anchor_path) as anchor_hdus, fits.open(input_path) as input_hdus:
        anchor_sci_index = _science_hdu_index(anchor_hdus, anchor_path)
        input_sci_index = _science_hdu_index(input_hdus, input_path)
        reference_hdu = anchor_hdus[anchor_sci_index]
        source_hdu = input_hdus[input_sci_index]
        reference_shape = tuple(reference_hdu.data.shape)
        reference_wcs = _celestial_wcs(reference_hdu, anchor_path)
        source_wcs = _celestial_wcs(source_hdu, input_path)
        coordinates = _reference_to_source_coordinates(
            reference_wcs, reference_shape, source_wcs,
        )

        primary_header = input_hdus[0].header.copy()
        primary_header['ALNANCHR'] = (Path(anchor_path).name, 'WCS anchor exposure')
        primary_header['ALNMETH'] = ('WCS_LINEAR', 'Linear WCS resampling')
        output_hdus = [fits.PrimaryHDU(header=primary_header)]

        for index, input_hdu in enumerate(input_hdus[1:], start=1):
            if input_hdu.data is None or np.ndim(input_hdu.data) != 2:
                output_hdus.append(input_hdu.copy())
                continue
            if index >= len(anchor_hdus) or anchor_hdus[index].data is None:
                raise ValueError(
                    f'Anchor {anchor_path} has no corresponding image extension {index} '
                    f'for {input_path}.\n'
                )
            extension_name = input_hdu.header.get('EXTNAME', f'EXT{index}')
            output_data = _resample_extension(input_hdu.data, extension_name, coordinates)
            header = anchor_hdus[index].header.copy()
            header['ALNINPUT'] = (Path(input_path).name, 'Original exposure before WCS alignment')
            header['ALNANCHR'] = (Path(anchor_path).name, 'WCS anchor exposure')
            output_hdus.append(fits.ImageHDU(data=output_data, header=header, name=extension_name))

        output_hdus.append(fits.ImageHDU(
            data=(_resample_continuous(
                np.ones_like(source_hdu.data, dtype=np.float64), coordinates, cval=0.0,
            ) > 0.5).astype(np.uint8),
            name='ALIGNMENT_MASK',
        ))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(output_hdus).writeto(output_path, overwrite=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_dir', nargs='?',
        default='/Users/xczhou/Desktop/modelling/single_exposure_data/dataset/F277W',
        help='Directory containing the original exposure FITS files.',
    )
    parser.add_argument('--pattern', default='*cutout*.fits', help='Input FITS filename glob.')
    parser.add_argument(
        '--anchor', default='jw01837003013_02201_00001_nrcalong_match_cutout_r30as.fits',
        help='Anchor filename or absolute path.',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for aligned FITS files (default: aligned_to_anchor below input_dir).',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    anchor_path = Path(args.anchor)
    if not anchor_path.is_absolute():
        anchor_path = input_dir / anchor_path
    anchor_path = anchor_path.resolve()
    if not anchor_path.is_file():
        raise FileNotFoundError(f'Anchor FITS file not found: {anchor_path}')

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else input_dir / 'aligned_to_anchor'
    )
    input_paths = sorted(input_dir.glob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(f'No FITS files matching {args.pattern!r} in {input_dir}.')
    print(f'Anchor: {anchor_path.name}')
    for input_path in input_paths:
        output_path = output_dir / input_path.name
        if input_path.resolve() == anchor_path:
            if output_path.exists():
                with fits.open(output_path, mode='update') as hdus:
                    if 'ALIGNMENT_MASK' not in hdus:
                        science = hdus[_science_hdu_index(hdus, output_path)].data
                        hdus.append(fits.ImageHDU(
                            data=np.ones_like(science, dtype=np.uint8),
                            name='ALIGNMENT_MASK',
                        ))
                        hdus.flush()
                print(f'Updated anchor coverage mask in {output_path}')
                continue
            with fits.open(anchor_path) as hdus:
                copied = fits.HDUList([hdu.copy() for hdu in hdus])
                copied[0].header['ALNANCHR'] = (anchor_path.name, 'WCS anchor exposure')
                copied[0].header['ALNMETH'] = ('IDENTITY', 'Anchor copied without resampling')
                science = copied[_science_hdu_index(copied, anchor_path)].data
                copied.append(fits.ImageHDU(
                    data=np.ones_like(science, dtype=np.uint8), name='ALIGNMENT_MASK',
                ))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                copied.writeto(output_path, overwrite=False)
            print(f'Copied anchor to {output_path}')
            continue
        if output_path.exists():
            print(f'Skipping existing {output_path}')
            continue
        align_exposure(input_path, anchor_path, output_path)
        print(f'Aligned {input_path.name} -> {output_path}')


if __name__ == '__main__':
    main()
