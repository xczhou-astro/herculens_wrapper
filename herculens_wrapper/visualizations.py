"""Diagnostic plots for Herculens wrapper runs."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

try:
    import corner
except ImportError:
    corner = None

from herculens.Util import model_util

from herculens_wrapper.utils import (
    fit_dof_and_reduced_chi2,
    json_serializer,
    kwargs_best_to_json_pixelated_npy,
    pytree_flat_param_labels,
)


def _point_source_colors(n):
    n = int(max(n, 1))
    cmap = plt.get_cmap('tab10')
    return [cmap(i % 10) for i in range(n)]


def _norm_from_plot_scale(plot_scale, arr):
    ps = (plot_scale or 'linear').strip().lower()
    if ps in ('linear', 'lin'):
        return None, 'linear'
    if ps in ('log', 'log10'):
        a = np.asarray(arr, dtype=float)
        pos = a[np.isfinite(a) & (a > 0)]
        if pos.size == 0:
            return None, 'linear'
        vmin = float(np.percentile(pos, 1.0))
        vmax = float(np.percentile(pos, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None, 'linear'
        vmax = max(vmax, vmin * 10.0)
        vmin = max(vmin, vmax * 1e-3)
        return LogNorm(vmin=vmin, vmax=vmax), 'log'
    return None, 'linear'


def _image_extent(ny, nx, pixel_scale):
    x_center = nx // 2
    y_center = ny // 2
    return [
        -x_center * pixel_scale, (nx - x_center - 1) * pixel_scale,
        -y_center * pixel_scale, (ny - y_center - 1) * pixel_scale,
    ]


def display(plot_data, titles, pixel_scale, savefilename=None, plot_scale='linear', contour_mask=None, residual_vis_max=0.0):
    num = len(plot_data)
    fig, axes = plt.subplots(1, num, figsize=(4 * num + 2 * num, 5))
    if num == 1:
        axes = [axes]
    for i in range(num):
        ny, nx = plot_data[i].shape
        extent = _image_extent(ny, nx, pixel_scale)
        if plot_scale == 'log' and i < 2:
            norm, cbar_label = _norm_from_plot_scale('log', plot_data[i])
        else:
            norm, cbar_label = None, 'linear'
        c_map = 'bwr' if (i == 2 or 'residual' in titles[i].lower() or 'chi' in titles[i].lower()) else 'twilight'
        if c_map == 'bwr':
            if residual_vis_max > 0.0:
                vmax = float(residual_vis_max)
                vmin = -vmax
            else:
                vmax = float(np.max(np.abs(plot_data[i])))
                vmin = -vmax
        else:
            vmin, vmax = None, None
        im = axes[i].imshow(plot_data[i], origin='lower', cmap=c_map, extent=extent, norm=norm, vmin=vmin, vmax=vmax)
        if contour_mask is not None:
            axes[i].contour(np.asarray(contour_mask), levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
        axes[i].set_xlabel('arcsec')
        axes[i].set_ylabel('arcsec')
        axes[i].set_title(titles[i])
        is_residual = i == 2 or 'residual' in titles[i].lower() or 'chi' in titles[i].lower()
        value_label = 'Standardized residual' if is_residual else 'Pixel flux'
        if not is_residual and cbar_label == 'log':
            value_label += ' (log scale)'
        plt.colorbar(im, ax=axes[i], label=value_label)
    plt.tight_layout()
    if savefilename is not None:
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_input_data(
    image_data,
    noise_map,
    psf_data,
    pixel_scale,
    save_path=None,
    point_source_type_list=None,
    point_source_params_list=None,
    source_arc_mask=None,
    # background_subtract_corner=0,
    # background_subtract_which_corner='bottom_left',
    background_offset=0.0,
):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    title_suffix = f" (bkg offset: {background_offset:.4f})" if background_offset != 0.0 else ""

    # 1. Linear Scale Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(image_data, origin='lower', cmap='twilight', extent=extent)
    if source_arc_mask is not None:
        axes[0].contour(np.asarray(source_arc_mask), levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    axes[0].set_title(f'Image data{title_suffix}')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=axes[0], label='Pixel flux')

    if (
        point_source_type_list is not None
        and point_source_params_list is not None
        and any(t == 'IMAGE_POSITIONS' for t in point_source_type_list)
    ):
        n_ps = sum(1 for t in point_source_type_list if t == 'IMAGE_POSITIONS')
        colors = _point_source_colors(n_ps)
        k = 0
        for t, ps in zip(point_source_type_list, point_source_params_list):
            if t != 'IMAGE_POSITIONS':
                continue
            ras = np.atleast_1d(np.asarray(ps.get('ra', []), dtype=float))
            decs = np.atleast_1d(np.asarray(ps.get('dec', []), dtype=float))
            if ras.size and decs.size:
                axes[0].scatter(
                    ras, decs, s=40, marker='o', facecolors='none',
                    edgecolors=colors[k], linewidths=1.5, label=f'PS {k + 1}',
                )
                k += 1
        axes[0].legend(loc='best', fontsize=8)

    im1 = axes[1].imshow(noise_map, origin='lower', cmap='twilight', extent=extent)
    axes[1].set_title('Noise map')
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=axes[1], label='Pixel-flux uncertainty')

    im2 = axes[2].imshow(psf_data, origin='lower', cmap='twilight')
    axes[2].set_title('PSF kernel')
    axes[2].set_xlabel('pixel')
    axes[2].set_ylabel('pixel')
    plt.colorbar(im2, ax=axes[2], label='Normalized PSF pixel value')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'input_data_linear.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Log Scale Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    norm_img, _ = _norm_from_plot_scale('log', image_data)
    im0 = axes[0].imshow(image_data, origin='lower', cmap='twilight', extent=extent, norm=norm_img)
    if source_arc_mask is not None:
        axes[0].contour(np.asarray(source_arc_mask), levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    axes[0].set_title(f'Image data (log){title_suffix}')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=axes[0], label='Pixel flux (log scale)')

    if (
        point_source_type_list is not None
        and point_source_params_list is not None
        and any(t == 'IMAGE_POSITIONS' for t in point_source_type_list)
    ):
        k = 0
        for t, ps in zip(point_source_type_list, point_source_params_list):
            if t != 'IMAGE_POSITIONS':
                continue
            ras = np.atleast_1d(np.asarray(ps.get('ra', []), dtype=float))
            decs = np.atleast_1d(np.asarray(ps.get('dec', []), dtype=float))
            if ras.size and decs.size:
                axes[0].scatter(
                    ras, decs, s=40, marker='o', facecolors='none',
                    edgecolors=colors[k], linewidths=1.5, label=f'PS {k + 1}',
                )
                k += 1
        axes[0].legend(loc='best', fontsize=8)

    norm_noise, _ = _norm_from_plot_scale('log', noise_map)
    im1 = axes[1].imshow(noise_map, origin='lower', cmap='twilight', extent=extent, norm=norm_noise)
    axes[1].set_title('Noise map (log)')
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=axes[1], label='Pixel-flux uncertainty (log scale)')

    norm_psf, _ = _norm_from_plot_scale('log', psf_data)
    im2 = axes[2].imshow(psf_data, origin='lower', cmap='twilight', norm=norm_psf)
    axes[2].set_title('PSF kernel (log)')
    axes[2].set_xlabel('pixel')
    axes[2].set_ylabel('pixel')
    plt.colorbar(im2, ax=axes[2], label='Normalized PSF pixel value (log scale)')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'input_data_log.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_image_plane(
    lens_image, kwargs_result, pixel_scale, image_data, noise_map, save_path,
    residual_vis_max=0.0, output_filename='image_plane.png',
    model_extended_override=None,
    model_lens_light_override=None,
    model_composite_override=None,
    model_point_sources_override=None,
):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    mask = getattr(lens_image, 'source_arc_mask', None)
    if mask is not None:
        mask = np.asarray(mask)

    if model_extended_override is not None:
        model_extended = model_extended_override
    else:
        model_extended = lens_image.model(
            **kwargs_result, source_add=True, lens_light_add=False, point_source_add=False,
        )

    if model_lens_light_override is not None:
        model_lens_light = model_lens_light_override
    elif 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result, lens_light_add=True, source_add=False, point_source_add=False,
        )
    else:
        model_lens_light = np.zeros((ny, nx))

    model_point_sources = np.zeros((ny, nx))
    ra_image_list = []
    dec_image_list = []
    if model_point_sources_override is not None:
        model_point_sources = model_point_sources_override
    elif 'kwargs_point_source' in kwargs_result:
        model_point_sources = lens_image.model(
            **kwargs_result, source_add=False, lens_light_add=False, point_source_add=True,
        )
    if 'kwargs_point_source' in kwargs_result:
        theta_x, theta_y, amps = lens_image.PointSourceModel.get_multiple_images(
            kwargs_result['kwargs_point_source'],
            kwargs_lens=kwargs_result['kwargs_lens'],
            kwargs_solver=lens_image.kwargs_lens_equation_solver,
            with_amplitude=True,
        )
        for i in range(len(theta_x)):
            ra_image_list.append(np.asarray(theta_x[i]))
            dec_image_list.append(np.asarray(theta_y[i]))
            print(f'RA for lensed point source {i}: {ra_image_list[-1]}')
            print(f'Dec for lensed point source {i}: {dec_image_list[-1]}')
            print(f'Amplitudes for lensed point source {i}: {amps[i]}')

    if model_composite_override is not None:
        model_composite = model_composite_override
    else:
        model_composite = lens_image.model(**kwargs_result, source_add=True, point_source_add=True)
    residuals = (model_composite - image_data) / noise_map


    n_ps = len(ra_image_list)
    ps_colors = _point_source_colors(n_ps) if n_ps else []

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    im0 = ax[0, 0].imshow(model_extended, origin='lower', cmap='twilight', extent=extent)
    if mask is not None:
        ax[0, 0].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
        ax[0, 0].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
    ax[0, 0].set_title('Extended Source (Lensed)')
    plt.colorbar(im0, ax=ax[0, 0], label='Pixel flux')

    im1 = ax[0, 1].imshow(model_lens_light, origin='lower', cmap='twilight', extent=extent)
    ax[0, 1].set_title('Lens Light')
    plt.colorbar(im1, ax=ax[0, 1], label='Pixel flux')

    im2 = ax[0, 2].imshow(model_point_sources, origin='lower', cmap='twilight', extent=extent)
    ax[0, 2].set_title('Point Sources')
    plt.colorbar(im2, ax=ax[0, 2], label='Pixel flux')

    im3 = ax[1, 0].imshow(model_composite, origin='lower', cmap='twilight', extent=extent)
    if mask is not None:
        ax[1, 0].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
        ax[1, 0].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
    ax[1, 0].set_title('Composite')
    plt.colorbar(im3, ax=ax[1, 0], label='Pixel flux')

    im4 = ax[1, 1].imshow(image_data, origin='lower', cmap='twilight', extent=extent)
    if mask is not None:
        ax[1, 1].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    ax[1, 1].set_title('Image Data')
    plt.colorbar(im4, ax=ax[1, 1], label='Pixel flux')

    if residual_vis_max > 0.0:
        vmax_res = float(residual_vis_max)
    else:
        vmax_res = float(np.max(np.abs(residuals)))
    im5 = ax[1, 2].imshow(residuals, origin='lower', cmap='bwr', extent=extent, vmin=-vmax_res, vmax=vmax_res)
    if mask is not None:
        ax[1, 2].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    ax[1, 2].set_title('Residuals (model - data) / noise')
    plt.colorbar(im5, ax=ax[1, 2], label='Standardized residual')

    for a in ax.ravel():
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_source_plane(
    lens_image,
    kwargs_result,
    save_path,
    source_pixel_scale=0.01,
    num_pixel=200,
    plot_caustics=True,
    plot_scale='linear',
    output_filename='source_plane.png',
    source_arc_mask=None,
):
    """Plot source-plane values in Herculens image-data-pixel flux units."""
    is_pixelated = (
        'kwargs_source' in kwargs_result
        and len(kwargs_result['kwargs_source']) > 0
        and isinstance(kwargs_result['kwargs_source'][0], dict)
        and 'pixels' in kwargs_result['kwargs_source'][0]
    )

    is_adaptive = False
    if is_pixelated:
        source_for_plot = np.asarray(kwargs_result['kwargs_source'][0]['pixels'])
        if getattr(lens_image, '_src_adaptive_grid', False) and hasattr(lens_image, 'get_source_coordinates'):
            is_adaptive = True
            kwargs_lens = kwargs_result.get('kwargs_lens', None)
            npix_src = source_for_plot.shape[0]
            xx, yy, extent = lens_image.get_source_coordinates(
                kwargs_lens,
                npix_src=npix_src,
                source_grid_scale=getattr(lens_image, '_source_grid_scale', 1.0)
            )
            xx = np.asarray(xx)
            yy = np.asarray(yy)
            extent = list(np.asarray(extent))
            if xx.ndim == 1 and yy.ndim == 1:
                xx, yy = np.meshgrid(xx, yy)
        else:
            extent = list(lens_image.SourceModel.pixel_grid.extent)
            xx, yy = lens_image.SourceModel.pixel_grid.pixel_coordinates
        
        # Compute and print pixel scale
        grid_width = extent[1] - extent[0]
        npix = source_for_plot.shape[0]
        adapted_pixel_scale = grid_width / npix
        print(f"[plot_source_plane] Source pixel scale: {adapted_pixel_scale:.6f} arcsec/pixel")
    else:
        try:
            ny, nx = lens_image.Grid.num_pixel_axes
            p_scale = float(lens_image.Grid.pixel_width)
        except Exception:
            ny, nx = num_pixel, num_pixel
            p_scale = source_pixel_scale

        extent = _image_extent(ny, nx, p_scale)
        x = np.linspace(extent[0], extent[1], nx)
        y = np.linspace(extent[2], extent[3], ny)
        xx, yy = np.meshgrid(x, y)

        source_for_plot = np.asarray(
            lens_image.SourceModel.surface_brightness(xx, yy, kwargs_result['kwargs_source'])
        ) * float(getattr(lens_image.Grid, 'pixel_area', p_scale**2))

    # Initialize adaptive limits defaulting to the full grid extent
    xmin_sq, xmax_sq = extent[0], extent[1]
    ymin_sq, ymax_sq = extent[2], extent[3]

    norm, cbar_label = _norm_from_plot_scale(plot_scale, source_for_plot)

    ra_source_list = []
    dec_source_list = []
    if 'kwargs_point_source' in kwargs_result:
        beta_x, beta_y = lens_image.PointSourceModel.get_source_plane_points(
            kwargs_result['kwargs_point_source'],
            kwargs_lens=kwargs_result['kwargs_lens'],
            with_amplitude=False,
        )
        ra_source_list = [np.atleast_1d(np.asarray(b)) for b in beta_x]
        dec_source_list = [np.atleast_1d(np.asarray(d)) for d in beta_y]

    caustics = []
    if plot_caustics:
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image, kwargs_result['kwargs_lens'], supersampling=5,
            )
        except Exception as e:
            print(f'[plot_source_plane] Could not compute caustics: {e}')

    # Map lensed ring / source_arc_mask boundary back to source plane
    ring_mask = source_arc_mask
    if ring_mask is None:
        ring_mask = getattr(lens_image, 'source_arc_mask', None)

    mapped_ring_contours = []
    if ring_mask is not None:
        try:
            mask_arr = np.asarray(ring_mask).astype(bool)
            if np.any(mask_arr) and not np.all(mask_arr):
                if hasattr(lens_image, 'Grid') and hasattr(lens_image.Grid, 'pixel_coordinates'):
                    img_x, img_y = lens_image.Grid.pixel_coordinates
                    img_x = np.asarray(img_x)
                    img_y = np.asarray(img_y)
                    if img_x.ndim == 1 and img_y.ndim == 1:
                        img_x, img_y = np.meshgrid(img_x, img_y)

                    fig_dummy, ax_dummy = plt.subplots()
                    cs = ax_dummy.contour(img_x, img_y, mask_arr.astype(float), levels=[0.5])
                    segments = cs.allsegs[0] if (hasattr(cs, 'allsegs') and len(cs.allsegs) > 0) else []
                    plt.close(fig_dummy)

                    if len(segments) > 0:
                        def _seg_area(seg):
                            if len(seg) < 3:
                                return 0.0
                            x_s, y_s = seg[:, 0], seg[:, 1]
                            return 0.5 * np.abs(np.dot(x_s, np.roll(y_s, 1)) - np.dot(y_s, np.roll(x_s, 1)))
                        outer_seg = max(segments, key=_seg_area)
                        segments = [outer_seg]

                    kwargs_lens = kwargs_result.get('kwargs_lens', None)
                    for seg in segments:
                        if len(seg) > 0:
                            x_b_img, y_b_img = seg[:, 0], seg[:, 1]
                            beta_x_b, beta_y_b = lens_image.MassModel.ray_shooting(
                                x_b_img, y_b_img, kwargs_lens
                            )
                            mapped_ring_contours.append((np.asarray(beta_x_b), np.asarray(beta_y_b)))
        except Exception as e:
            print(f'[plot_source_plane] Could not compute mapped ring boundary: {e}')

    colors = _point_source_colors(len(ra_source_list)) if ra_source_list else []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scale_suffix = f" ({adapted_pixel_scale:.5f}\"/pix)" if is_pixelated else ""

    im0 = axes[0].imshow(source_for_plot, origin='lower', extent=extent, cmap='twilight', norm=norm)
    axes[0].set_title(f'Extended Source{scale_suffix}')
    source_flux_label = 'Pixel flux'
    if cbar_label == 'log':
        source_flux_label += ' (log scale)'
    plt.colorbar(im0, ax=axes[0], label=source_flux_label)

    im1 = axes[1].imshow(source_for_plot, origin='lower', extent=extent, cmap='twilight', norm=norm)
    for i, (ras, decs) in enumerate(zip(ra_source_list, dec_source_list)):
        axes[1].scatter(ras, decs, s=30, marker='*', color=colors[i], label=f'PS {i + 1}')
    for caust_x, caust_y in caustics:
        axes[1].plot(caust_x, caust_y, color='lime', lw=1.0)
    axes[1].set_title(f'Source Plane Reconstruction{scale_suffix}')
    plt.colorbar(im1, ax=axes[1], label=source_flux_label)

    if mapped_ring_contours:
        for ax in axes:
            for beta_x_b, beta_y_b in mapped_ring_contours:
                ax.plot(beta_x_b, beta_y_b, color='orange', lw=1.5, ls='--', alpha=0.95)

    support_bounds = locals().get('source_support_bounds', None)
    if support_bounds is None:
        support_bounds = getattr(lens_image, 'source_support_bounds', None)
    if support_bounds is not None:
        xmin_b, xmax_b, ymin_b, ymax_b = [float(v) for v in support_bounds]
        boundary_lines = [
            ([xmin_b, xmin_b], [ymin_b, ymax_b]),
            ([xmax_b, xmax_b], [ymin_b, ymax_b]),
            ([xmin_b, xmax_b], [ymin_b, ymin_b]),
            ([xmin_b, xmax_b], [ymax_b, ymax_b]),
        ]
        for ax in axes:
            for xs, ys in boundary_lines:
                ax.plot(xs, ys, color='cyan', lw=1.2, ls='--', alpha=0.9)

    for a in axes:
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')
        a.set_xlim(xmin_sq, xmax_sq)
        a.set_ylim(ymin_sq, ymax_sq)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_composite_2x3_panel(
    lens_image,
    kwargs_result,
    pixel_scale,
    image_data,
    noise_map,
    save_path,
    residual_vis_max=0.0,
    output_filename='composite.png',
    model_extended_override=None,
    model_lens_light_override=None,
    model_composite_override=None,
    source_arc_mask=None,
):
    ny, nx = image_data.shape
    extent_img = _image_extent(ny, nx, pixel_scale)

    mask = source_arc_mask
    if mask is None:
        mask = getattr(lens_image, 'source_arc_mask', None)

    # Clean kwargs_source for model evaluation if needed
    clean_kwargs = dict(kwargs_result)
    if 'kwargs_source' in kwargs_result:
        clean_kwargs['kwargs_source'] = [
            {'pixels': k['pixels']} if (isinstance(k, dict) and 'pixels' in k) else k
            for k in kwargs_result['kwargs_source']
        ]

    # 1. Evaluate image plane components
    if model_lens_light_override is not None:
        model_lens_light = model_lens_light_override
    elif 'kwargs_lens_light' in kwargs_result and len(kwargs_result.get('kwargs_lens_light', [])) > 0:
        model_lens_light = lens_image.model(
            **clean_kwargs, lens_light_add=True, source_add=False, point_source_add=False,
        )
    else:
        model_lens_light = np.zeros((ny, nx))

    if model_extended_override is not None:
        model_extended = model_extended_override
    else:
        model_extended = lens_image.model(
            **clean_kwargs, source_add=True, lens_light_add=False, point_source_add=False,
        )

    if model_composite_override is not None:
        model_composite = model_composite_override
    else:
        model_composite = lens_image.model(**clean_kwargs, source_add=True, point_source_add=True, lens_light_add=True)

    residuals = (model_composite - image_data) / noise_map
    chi2 = float(np.sum(residuals ** 2))
    subtracted = image_data - model_lens_light

    # 2. Evaluate source plane reconstruction
    is_pixelated = (
        'kwargs_source' in kwargs_result
        and len(kwargs_result['kwargs_source']) > 0
        and isinstance(kwargs_result['kwargs_source'][0], dict)
        and 'pixels' in kwargs_result['kwargs_source'][0]
    )

    if is_pixelated:
        source_for_plot = np.asarray(kwargs_result['kwargs_source'][0]['pixels'])
        if getattr(lens_image, '_src_adaptive_grid', False) and hasattr(lens_image, 'get_source_coordinates'):
            kwargs_lens = kwargs_result.get('kwargs_lens', None)
            npix_src = source_for_plot.shape[0]
            _, _, extent_src = lens_image.get_source_coordinates(
                kwargs_lens,
                npix_src=npix_src,
                source_grid_scale=getattr(lens_image, '_source_grid_scale', 1.0)
            )
            extent_src = list(np.asarray(extent_src))
        else:
            extent_src = list(lens_image.SourceModel.pixel_grid.extent)
    else:
        p_scale = float(getattr(lens_image.Grid, 'pixel_width', pixel_scale))
        extent_src = _image_extent(ny, nx, p_scale)
        x_src = np.linspace(extent_src[0], extent_src[1], nx)
        y_src = np.linspace(extent_src[2], extent_src[3], ny)
        xx_src, yy_src = np.meshgrid(x_src, y_src)
        source_for_plot = np.asarray(
            lens_image.SourceModel.surface_brightness(xx_src, yy_src, kwargs_result['kwargs_source'])
        ) * float(getattr(lens_image.Grid, 'pixel_area', p_scale**2))

    # Caustics
    caustics = []
    try:
        _, caustics = model_util.critical_lines_caustics(
            lens_image, kwargs_result['kwargs_lens'], supersampling=5,
        )
    except Exception:
        pass

    # Ray-trace outer ring boundary to source plane
    mapped_ring_contours = []
    if mask is not None:
        try:
            mask_arr = np.asarray(mask).astype(bool)
            if np.any(mask_arr) and not np.all(mask_arr):
                if hasattr(lens_image, 'Grid') and hasattr(lens_image.Grid, 'pixel_coordinates'):
                    img_x, img_y = lens_image.Grid.pixel_coordinates
                    img_x = np.asarray(img_x)
                    img_y = np.asarray(img_y)
                    if img_x.ndim == 1 and img_y.ndim == 1:
                        img_x, img_y = np.meshgrid(img_x, img_y)

                    fig_dummy, ax_dummy = plt.subplots()
                    cs = ax_dummy.contour(img_x, img_y, mask_arr.astype(float), levels=[0.5])
                    segments = cs.allsegs[0] if (hasattr(cs, 'allsegs') and len(cs.allsegs) > 0) else []
                    plt.close(fig_dummy)

                    if len(segments) > 0:
                        def _seg_area(seg):
                            if len(seg) < 3:
                                return 0.0
                            x_s, y_s = seg[:, 0], seg[:, 1]
                            return 0.5 * np.abs(np.dot(x_s, np.roll(y_s, 1)) - np.dot(y_s, np.roll(x_s, 1)))
                        outer_seg = max(segments, key=_seg_area)
                        segments = [outer_seg]

                    kwargs_lens = kwargs_result.get('kwargs_lens', None)
                    for seg in segments:
                        if len(seg) > 0:
                            x_b_img, y_b_img = seg[:, 0], seg[:, 1]
                            beta_x_b, beta_y_b = lens_image.MassModel.ray_shooting(
                                x_b_img, y_b_img, kwargs_lens
                            )
                            mapped_ring_contours.append((np.asarray(beta_x_b), np.asarray(beta_y_b)))
        except Exception:
            pass

    # Render 2x3 panel plot with mixed scales: Data & Model in log scale; rest in linear scale. Colorbar ONLY on Residual.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    norm_data, _ = _norm_from_plot_scale('log', image_data)
    norm_model, _ = _norm_from_plot_scale('log', model_composite)
    norm_sub, _ = _norm_from_plot_scale('linear', subtracted)
    norm_src_lensed, _ = _norm_from_plot_scale('linear', model_extended)
    norm_src_plane, _ = _norm_from_plot_scale('linear', source_for_plot)

    # Panel (0, 0): Data (Log scale, no colorbar)
    axes[0, 0].imshow(image_data, origin='lower', extent=extent_img, cmap='twilight', norm=norm_data)
    if mask is not None:
        axes[0, 0].contour(np.asarray(mask), levels=[0.5], colors='lime', extent=extent_img, linewidths=1.0)
    axes[0, 0].set_title('Data')

    # Panel (0, 1): Model (Log scale, no colorbar)
    axes[0, 1].imshow(model_composite, origin='lower', extent=extent_img, cmap='twilight', norm=norm_model)
    if mask is not None:
        axes[0, 1].contour(np.asarray(mask), levels=[0.5], colors='lime', extent=extent_img, linewidths=1.0)
    axes[0, 1].set_title('Model')

    # Panel (0, 2): Residual (Linear scale, WITH COLORBAR)
    if residual_vis_max > 0.0:
        vmax_res = float(residual_vis_max)
    else:
        vmax_res = float(np.max(np.abs(residuals)))
    im2 = axes[0, 2].imshow(residuals, origin='lower', extent=extent_img, cmap='bwr', vmin=-vmax_res, vmax=vmax_res)
    if mask is not None:
        axes[0, 2].contour(np.asarray(mask), levels=[0.5], colors='lime', extent=extent_img, linewidths=1.0)
    axes[0, 2].set_title(f'Residual (chi^2 = {chi2:.2f})')
    plt.colorbar(im2, ax=axes[0, 2], label='(model - data) / noise')

    # Panel (1, 0): Data - Lens Light (Linear scale, no colorbar)
    axes[1, 0].imshow(subtracted, origin='lower', extent=extent_img, cmap='twilight', norm=norm_sub)
    if mask is not None:
        axes[1, 0].contour(np.asarray(mask), levels=[0.5], colors='lime', extent=extent_img, linewidths=1.0)
    axes[1, 0].set_title('Data - Lens Light')

    # Panel (1, 1): Lensed Source (Linear scale, no colorbar)
    axes[1, 1].imshow(model_extended, origin='lower', extent=extent_img, cmap='twilight', norm=norm_src_lensed)
    if mask is not None:
        axes[1, 1].contour(np.asarray(mask), levels=[0.5], colors='lime', extent=extent_img, linewidths=1.0)
    axes[1, 1].set_title('Lensed Source')

    # Panel (1, 2): Source (Source Plane, Linear scale, no colorbar)
    axes[1, 2].imshow(source_for_plot, origin='lower', extent=extent_src, cmap='twilight', norm=norm_src_plane)
    for caust_x, caust_y in caustics:
        axes[1, 2].plot(caust_x, caust_y, color='lime', lw=1.0)
    if mapped_ring_contours:
        for beta_x_b, beta_y_b in mapped_ring_contours:
            axes[1, 2].plot(beta_x_b, beta_y_b, color='orange', lw=1.5, ls='--', alpha=0.95)
    axes[1, 2].set_title('Source')

    for a in axes.ravel():
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiband_composite(
    band_results,
    save_path,
    residual_vis_max=0.0,
    output_filename='multiband_composite.png',
):
    """Save one six-panel diagnostic row for every band in a joint fit."""
    if not band_results:
        return

    panel_titles = (
        'Data', 'Model', 'Residual', 'Data - Lens Light',
        'Lensed Source', 'Source Reconstruction',
    )
    figure, axes = plt.subplots(
        len(band_results), len(panel_titles), figsize=(30, 5 * len(band_results)),
        squeeze=False,
    )

    for row, band in enumerate(band_results):
        lens_image = band['lens_image']
        kwargs_result = band['kwargs_result']
        image_data = np.asarray(band['image_data'])
        noise_map = np.asarray(band['noise_map'])
        pixel_scale = float(band['pixel_scale'])
        extent_img = _image_extent(*image_data.shape, pixel_scale)
        image_mask = getattr(lens_image, 'source_arc_mask', None)
        if image_mask is not None:
            image_mask = np.asarray(image_mask)

        model_lens_light = band.get('model_lens_light')
        if model_lens_light is None:
            if kwargs_result.get('kwargs_lens_light'):
                model_lens_light = lens_image.model(
                    **kwargs_result, lens_light_add=True, source_add=False, point_source_add=False,
                )
            else:
                model_lens_light = np.zeros_like(image_data)
        model_lens_light = np.asarray(model_lens_light)

        model_lensed_source = band.get('model_lensed_source')
        if model_lensed_source is None:
            model_lensed_source = lens_image.model(
                **kwargs_result, lens_light_add=False, source_add=True, point_source_add=False,
            )
        model_lensed_source = np.asarray(model_lensed_source)

        model_total = band.get('model_total')
        if model_total is None:
            model_total = lens_image.model(
                **kwargs_result, lens_light_add=True, source_add=True, point_source_add=True,
            )
        model_total = np.asarray(model_total)
        residual = (model_total - image_data) / noise_map
        chi2 = float(np.sum(residual ** 2))
        data_minus_lens = image_data - model_lens_light

        source = kwargs_result.get('kwargs_source', [{}])[0]
        source_pixels = source.get('pixels')
        if source_pixels is not None and getattr(lens_image, '_src_adaptive_grid', False) and hasattr(lens_image, 'get_source_coordinates'):
            source_pixels = np.asarray(source_pixels)
            _, _, extent_src = lens_image.get_source_coordinates(
                kwargs_result.get('kwargs_lens'), npix_src=source_pixels.shape[0],
                source_grid_scale=getattr(lens_image, '_source_grid_scale', 1.0),
            )
            extent_src = list(np.asarray(extent_src))
        elif source_pixels is not None:
            source_pixels = np.asarray(source_pixels)
            extent_src = list(lens_image.SourceModel.pixel_grid.extent)
        else:
            extent_src = _image_extent(*image_data.shape, pixel_scale)
            x_src = np.linspace(extent_src[0], extent_src[1], image_data.shape[1])
            y_src = np.linspace(extent_src[2], extent_src[3], image_data.shape[0])
            xx_src, yy_src = np.meshgrid(x_src, y_src)
            source_pixels = np.asarray(
                lens_image.SourceModel.surface_brightness(
                    xx_src, yy_src, kwargs_result.get('kwargs_source', []),
                )
            ) * float(getattr(lens_image.Grid, 'pixel_area', pixel_scale**2))

        image_panels = (image_data, model_total, residual, data_minus_lens, model_lensed_source)
        for column, values in enumerate(image_panels):
            axis = axes[row, column]
            if column < 2:
                norm, _ = _norm_from_plot_scale('log', values)
                image = axis.imshow(values, origin='lower', extent=extent_img, cmap='twilight', norm=norm)
            elif column == 2:
                finite = values[np.isfinite(values)]
                vmax = float(residual_vis_max) if residual_vis_max > 0 else (
                    float(np.max(np.abs(finite))) if finite.size else 1.0
                )
                image = axis.imshow(
                    values, origin='lower', extent=extent_img, cmap='bwr', vmin=-vmax, vmax=vmax,
                )
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label='(model - data) / noise')
            else:
                image = axis.imshow(values, origin='lower', extent=extent_img, cmap='twilight')
            if image_mask is not None:
                axis.contour(image_mask, levels=[0.5], colors='lime', extent=extent_img, linewidths=1.0)

        source_axis = axes[row, 5]
        source_axis.imshow(source_pixels, origin='lower', extent=extent_src, cmap='twilight')
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image, kwargs_result['kwargs_lens'], supersampling=5,
            )
            for caustic_x, caustic_y in caustics:
                source_axis.plot(caustic_x, caustic_y, color='lime', lw=1.0)
        except Exception:
            pass
        support_bounds = getattr(lens_image, 'source_support_bounds', None)
        if support_bounds is not None:
            xmin, xmax, ymin, ymax = [float(value) for value in support_bounds]
            source_axis.plot(
                [xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
                color='cyan', lw=1.2, ls='--', alpha=0.9,
            )

        for column, axis in enumerate(axes[row]):
            if column == 2:
                axis.set_title(f'Residual (chi^2 = {chi2:.2f})')
            elif row == 0:
                axis.set_title(panel_titles[column])
            axis.set_xlabel('arcsec')
            axis.set_ylabel('arcsec')
        axes[row, 0].set_ylabel(f"{band['name']}\narcsec")

    figure.tight_layout()
    output_path = os.path.join(save_path, output_filename)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    print(f'[plots] {output_path}')


def plot_multiband_source_reconstructions(band_results, save_path, output_filename):
    """Plot one initial or final pixelated source reconstruction per band."""
    def has_pixels(band):
        source = band['kwargs_result'].get('kwargs_source', [])
        return bool(source) and isinstance(source[0], dict) and source[0].get('pixels') is not None

    pixelated_bands = [band for band in band_results if has_pixels(band)]
    if not pixelated_bands:
        return

    figure, axes = plt.subplots(len(pixelated_bands), 1, figsize=(6, 5 * len(pixelated_bands)), squeeze=False)
    for row, band in enumerate(pixelated_bands):
        lens_image = band['lens_image']
        kwargs_result = band['kwargs_result']
        source_pixels = np.asarray(kwargs_result['kwargs_source'][0]['pixels'])
        if getattr(lens_image, '_src_adaptive_grid', False) and hasattr(lens_image, 'get_source_coordinates'):
            _, _, extent = lens_image.get_source_coordinates(
                kwargs_result.get('kwargs_lens'), npix_src=source_pixels.shape[0],
                source_grid_scale=getattr(lens_image, '_source_grid_scale', 1.0),
            )
            extent = list(np.asarray(extent))
        else:
            extent = list(lens_image.SourceModel.pixel_grid.extent)

        axis = axes[row, 0]
        image = axis.imshow(source_pixels, origin='lower', extent=extent, cmap='twilight')
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image, kwargs_result['kwargs_lens'], supersampling=5,
            )
            for caustic_x, caustic_y in caustics:
                axis.plot(caustic_x, caustic_y, color='lime', lw=1.0)
        except Exception:
            pass
        axis.set_title(f"{band['name']} Initial Source Reconstruction")
        axis.set_xlabel('arcsec')
        axis.set_ylabel('arcsec')
        figure.colorbar(image, ax=axis, label='Pixel flux')

    figure.tight_layout()
    output_path = os.path.join(save_path, output_filename)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    print(f'[plots] {output_path}')


def plot_lens_light_subtracted_image(
    lens_image, kwargs_result, pixel_scale, image_data, noise_map=None, save_path=None,
    plot_scale='linear', residual_vis_max=0.0,
    model_lens_light_override=None,
):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    mask = getattr(lens_image, 'source_arc_mask', None)
    if mask is not None:
        mask = np.asarray(mask)

    if model_lens_light_override is not None:
        model_lens_light = model_lens_light_override
    elif 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result, lens_light_add=True, source_add=False, point_source_add=False,
        )
    else:
        model_lens_light = np.zeros((ny, nx))

    subtracted = image_data - model_lens_light

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    norm_0, label_0 = _norm_from_plot_scale(plot_scale, image_data)
    im0 = ax[0].imshow(image_data, origin='lower', cmap='twilight', extent=extent, norm=norm_0)
    if mask is not None:
        ax[0].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
    ax[0].set_title('Image data')
    label_0 = 'Pixel flux (log scale)' if label_0 == 'log' else 'Pixel flux'
    plt.colorbar(im0, ax=ax[0], label=label_0)

    norm_1, label_1 = _norm_from_plot_scale(plot_scale, model_lens_light)
    im1 = ax[1].imshow(model_lens_light, origin='lower', cmap='twilight', extent=extent, norm=norm_1)
    
    # Overlay 1-sigma contours for Gaussian/MGE components (if present)
    if 'kwargs_lens_light' in kwargs_result:
        from matplotlib.patches import Ellipse
        for kw in kwargs_result['kwargs_lens_light']:
            if isinstance(kw, dict) and 'sigma' in kw:
                sigma = float(kw['sigma'])
                center_x = float(kw.get('center_x', 0.0))
                center_y = float(kw.get('center_y', 0.0))
                e1 = float(kw.get('e1', 0.0))
                e2 = float(kw.get('e2', 0.0))
                
                eps = np.sqrt(e1**2 + e2**2)
                if eps > 0.0:
                    eps = min(eps, 0.999)
                    q = (1.0 - eps) / (1.0 + eps)
                    phi = 0.5 * np.arctan2(e2, e1)
                    angle_deg = phi * (180.0 / np.pi)
                else:
                    q = 1.0
                    angle_deg = 0.0
                
                ellipse = Ellipse(
                    xy=(center_x, center_y),
                    width=2 * sigma,
                    height=2 * q * sigma,
                    angle=angle_deg,
                    edgecolor='black',
                    facecolor='none',
                    linestyle='--',
                    linewidth=1.2,
                    alpha=0.5
                )
                ax[1].add_patch(ellipse)
                ax[1].plot(center_x, center_y, '+', color='black', markersize=4, alpha=0.8)
                
    ax[1].set_title('Lens light model')
    label_1 = 'Pixel flux (log scale)' if label_1 == 'log' else 'Pixel flux'
    plt.colorbar(im1, ax=ax[1], label=label_1)

    if noise_map is not None:
        res_data = subtracted / noise_map
        vmax_res = float(np.max(np.abs(res_data)))
        im2 = ax[2].imshow(res_data, origin='lower', cmap='bwr', extent=extent, vmin=-vmax_res, vmax=vmax_res)
        if mask is not None:
            ax[2].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
        ax[2].set_title('Data - Lens light (S/N)')
        plt.colorbar(im2, ax=ax[2], label='Signal-to-noise')
    else:
        vmax_res = float(np.max(np.abs(subtracted)))
        im2 = ax[2].imshow(subtracted, origin='lower', cmap='bwr', extent=extent, vmin=-vmax_res, vmax=vmax_res)
        if mask is not None:
            ax[2].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
        ax[2].set_title('Data - Lens light')
        plt.colorbar(im2, ax=ax[2], label='Pixel flux')

    for a in ax:
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')

    plt.tight_layout()
    suffix = '_log' if plot_scale == 'log' else ''
    filename = f'lens_light_subtracted_image{suffix}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ring_model_comparison(
    lens_image,
    kwargs_result,
    pixel_scale,
    image_data,
    noise_map,
    save_path,
    plot_scale='linear',
    residual_vis_max=0.0,
    output_filename=None,
    model_no_lens_light_override=None,
    model_lens_light_override=None,
):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    mask = getattr(lens_image, 'source_arc_mask', None)
    if mask is not None:
        mask = np.asarray(mask)

    if model_lens_light_override is not None:
        model_lens_light = model_lens_light_override
    elif 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result,
            lens_light_add=True,
            source_add=False,
            point_source_add=False,
        )
    else:
        model_lens_light = np.zeros((ny, nx))

    if model_no_lens_light_override is not None:
        model_no_lens_light = model_no_lens_light_override
    else:
        model_no_lens_light = lens_image.model(
            **kwargs_result,
            lens_light_add=False,
            source_add=True,
            point_source_add=True,
        )

    image_minus_lens = np.asarray(image_data) - model_lens_light
    residual = (model_no_lens_light - image_minus_lens) / noise_map

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (model_no_lens_light, 'Model without lens light', 'twilight'),
        (image_minus_lens, 'Image - lens light', 'twilight'),
        (residual, 'Residuals', 'bwr'),
    ]

    for idx, (panel, title, cmap) in enumerate(panels):
        if idx < 2:
            norm, cbar_label = _norm_from_plot_scale(plot_scale, panel)
            cbar_label = 'Pixel flux (log scale)' if cbar_label == 'log' else 'Pixel flux'
            vmin, vmax = None, None
        else:
            norm, cbar_label = None, 'Standardized residual'
            if residual_vis_max > 0.0:
                vmax = float(residual_vis_max)
            else:
                finite = np.asarray(panel)[np.isfinite(panel)]
                vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
            vmin = -vmax

        im = axes[idx].imshow(
            panel,
            origin='lower',
            cmap=cmap,
            extent=extent,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        if mask is not None:
            axes[idx].contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
        axes[idx].set_xlabel('arcsec')
        axes[idx].set_ylabel('arcsec')
        axes[idx].set_title(title)
        plt.colorbar(im, ax=axes[idx], label=cbar_label)

    plt.tight_layout()
    if output_filename is None:
        suffix = '_log' if plot_scale == 'log' else '_linear'
        output_filename = f'ring_model_comparison{suffix}.png'
    plt.savefig(os.path.join(save_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_weights(weights_list, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights_list[0][0], origin='lower', cmap='twilight')
    plt.colorbar()
    plt.title('Regularization weights')
    plt.savefig(os.path.join(save_path, 'weights_list.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curve(loss_curve, save_path, lr_curve=None):
    n_total = len(loss_curve)
    if n_total == 0:
        return

    loss_curve = np.asarray(loss_curve)
    loss_curve_2d = loss_curve[:, None] if loss_curve.ndim == 1 else loss_curve
    x_all = np.arange(1, n_total + 1)

    begin_idx = min(int(n_total * 0.8), n_total - 1)
    tail = loss_curve_2d[begin_idx:]
    window = 100
    n_bins = tail.shape[0] // window
    if n_bins >= 1:
        trimmed_tail = tail[: n_bins * window]
        y_tail = trimmed_tail.reshape(n_bins, window, tail.shape[1]).mean(axis=1)
        x_tail = (begin_idx + 1) + (np.arange(n_bins) * window) + (window / 2.0)
    else:
        x_tail = np.arange(begin_idx + 1, n_total + 1)
        y_tail = tail

    fig, (ax_full, ax_tail) = plt.subplots(2, 1, figsize=(10, 8))
    loss_lines = ax_full.plot(x_all, loss_curve_2d, color='tab:blue')
    ax_full.set_xlabel('Iteration')
    ax_full.set_ylabel('Loss')
    ax_full.set_title('Loss Curve (Full)')
    ax_full.grid(True, alpha=0.3)

    if lr_curve is not None:
        lr_arr = np.asarray(lr_curve).reshape(-1)
        if lr_arr.shape[0] == n_total:
            ax_lr = ax_full.twinx()
            (lr_line,) = ax_lr.plot(x_all, lr_arr, color='tab:orange', alpha=0.85, label='Step size')
            ax_lr.set_ylabel('Step size')
            ax_lr.set_yscale('log')
            if loss_lines:
                ax_full.legend([loss_lines[0], lr_line], ['Loss', 'Step size'], loc='upper right')

    primary_loss = loss_curve_2d[:, 0]
    best_loss = float(np.nanmin(primary_loss))
    y_tail_smoothed = y_tail[:, 0] if y_tail.ndim == 2 else y_tail
    ax_tail.plot(x_tail, y_tail_smoothed, color='tab:red', linewidth=2.0, label='Tail mean')
    ax_tail.set_xlabel('Iteration')
    ax_tail.set_ylabel('Loss')
    ax_tail.set_title(f'Loss Tail (last 20%, best={best_loss:.6f})')
    ax_tail.grid(True, alpha=0.3)
    ax_tail.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _get_mge_exclude_list(all_names, threshold=3):
    """
    Identify parameter prefixes that have at least `threshold` components,
    which indicates they are part of a Multi-Gaussian Expansion (MGE),
    and return all parameter names matching those prefixes.
    """
    import re
    groups = {}
    for name in all_names:
        match = re.match(r'^(.*?)(?:_(\d+)|\[(\d+)\])$', name)
        if match:
            prefix = match.group(1)
            idx = match.group(2) or match.group(3)
            groups.setdefault(prefix, set()).add(idx)
            
    exclude = set()
    for prefix, indices in groups.items():
        if len(indices) >= threshold:
            for name in all_names:
                if name.startswith(prefix):
                    exclude.add(name)
    return exclude


def _get_param_order(param_list):
    order = []
    if not param_list:
        return order
    
    # 1. Lens Mass
    for i, model in enumerate(param_list.get('lens_mass_params_list', [])):
        if isinstance(model, dict):
            for key in model.keys():
                order.append(f'lens_{key}_{i}')
            
    # 2. Lens Light
    for i, model in enumerate(param_list.get('lens_light_params_list', [])):
        if isinstance(model, dict):
            for key in model.keys():
                order.append(f'lens_light_{key}_{i}')
            
    # 3. Source Light
    for i, model in enumerate(param_list.get('source_light_params_list', [])):
        if isinstance(model, dict):
            for key in model.keys():
                order.append(f'source_{key}_{i}')
            
    # 4. Point Source
    for i, model in enumerate(param_list.get('point_source_params_list', [])):
        if isinstance(model, dict):
            for key in model.keys():
                order.append(f'ps_{key}_{i}')
            
    return order


def plot_corner_traced_params(samples, save_path, max_samples=15_000, exclude=None, filename='corner_traced_params.png', param_list=None):
    if corner is None:
        print(f'[plots] corner package not installed; skipping {filename}')
        return

    exclude = set(exclude) if exclude is not None else {'source_pixels', 'source_scales', 'source_coarse'}
    mge_excludes = _get_mge_exclude_list(list(samples.keys()))
    if mge_excludes:
        print(f"[plots] MGE detected, excluding from corner plot: {sorted(list(mge_excludes))}")
        exclude.update(mge_excludes)

    desired_order = _get_param_order(param_list)
    order_map = {name: idx for idx, name in enumerate(desired_order)}

    def key_fn(name):
        base_name = name.split('[')[0]
        if base_name in order_map:
            return (0, order_map[base_name])
        return (1, name)

    sorted_keys = sorted(samples.keys(), key=key_fn)

    cols = []
    labels = []
    for name in sorted_keys:
        if name in exclude or name.startswith('ps_'):
            continue
        arr = np.asarray(samples[name])
        if arr.ndim == 1:
            cols.append(arr)
            labels.append(name)
        elif arr.ndim == 2 and arr.shape[1] <= 32:
            for j in range(arr.shape[1]):
                cols.append(arr[:, j])
                labels.append(f'{name}[{j}]')

    if len(cols) < 2:
        print(f'[plots] Corner plot skipped: need >= 2 traced scalars (got {len(cols)}).')
        return

    data = np.column_stack(cols)
    n = data.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        data = data[rng.choice(n, size=max_samples, replace=False)]

    fig = corner.corner(
        data, labels=labels, show_titles=True, title_fmt='.3f', quantiles=[0.16, 0.5, 0.84],
    )
    out = os.path.join(save_path, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] Saved {out}')


def display_init(
    prob_model,
    init_params,
    lens_image,
    image_data,
    noise_map,
    pixel_scale,
    save_path,
    num_params,
    type_list=None,
):
    """Plot the initial guess model before inference."""
    kwargs_init = prob_model.params2kwargs(init_params)
    if save_path is not None and type_list is not None:
        kwargs_init_json = kwargs_best_to_json_pixelated_npy(
            kwargs_init, save_path, type_list,
            pixels_filename='kwargs_source_pixels_init.npy',
            pixels_wn_filename='kwargs_source_pixels_wn_init.npy'
        )
        with open(os.path.join(save_path, 'kwargs_init.json'), 'w') as f:
            json.dump(kwargs_init_json, f, indent=4, default=json_serializer)

    initial_model = lens_image.model(**kwargs_init)
    mask = getattr(lens_image, 'source_arc_mask', None)
    if mask is not None:
        mask = np.asarray(mask)

    init_chi2 = float(np.sum(((initial_model - image_data) / noise_map) ** 2))
    init_reduced, _, _, dof_init = fit_dof_and_reduced_chi2(init_chi2, image_data, num_params)
    print(
        f'Initial chi^2: {init_chi2:.2f} (reduced: {init_reduced:.4f}, dof={dof_init})'
    )

    display(
        [initial_model, image_data, (initial_model - image_data) / noise_map],
        titles=[
            'Initial guess model',
            'Image data',
            f'Residuals (chi^2 = {init_chi2:.2f})',
        ],
        pixel_scale=pixel_scale,
        savefilename=os.path.join(save_path, 'initial_guess_model.png'),
        contour_mask=mask,
    )

    is_pixelated = (
        'kwargs_source' in kwargs_init
        and len(kwargs_init['kwargs_source']) > 0
        and isinstance(kwargs_init['kwargs_source'][0], dict)
        and 'pixels' in kwargs_init['kwargs_source'][0]
    )
    if is_pixelated:
        try:
            plot_source_plane(
                lens_image=lens_image,
                kwargs_result=kwargs_init,
                save_path=save_path,
                plot_caustics=True,
                output_filename='initial_source_plane.png',
            )
            print('[plots] initial_source_plane.png')
        except Exception as e:
            print(f'[plots] initial_source_plane.png skipped: {e}')


def plot_corner_emcee(
    flat_samples,
    prob_model,
    init_params,
    save_path,
    max_samples=15_000,
    exclude_sites=('source_pixels',),
    param_list=None,
):
    if corner is None:
        print('[plots] corner package not installed; skipping corner_emcee.png')
        return

    from herculens_wrapper.samplers import to_unconstrained
    from jax.flatten_util import ravel_pytree

    init_u = to_unconstrained(prob_model, init_params)
    flat_ref, _ = ravel_pytree(init_u)
    labels = pytree_flat_param_labels(init_u)

    exclude_sites = set(exclude_sites)
    mge_excludes = _get_mge_exclude_list(labels)
    if mge_excludes:
        print(f"[plots] MGE detected, excluding from emcee corner plot: {sorted(list(mge_excludes))}")
        exclude_sites.update(mge_excludes)

    desired_order = _get_param_order(param_list)
    if desired_order:
        label_to_index = {lab: idx for idx, lab in enumerate(labels)}
        ordered_indices = []
        for name in desired_order:
            if name in label_to_index:
                ordered_indices.append(label_to_index[name])
            else:
                for lab, idx in label_to_index.items():
                    if lab.startswith(f'{name}['):
                        ordered_indices.append(idx)
        seen_indices = set(ordered_indices)
        for idx in range(len(labels)):
            if idx not in seen_indices:
                ordered_indices.append(idx)
        labels = [labels[idx] for idx in ordered_indices]
        X_orig = np.asarray(flat_samples, dtype=np.float64)
        cols_all = []
        for idx in ordered_indices:
            if idx < X_orig.shape[1]:
                cols_all.append(X_orig[:, idx])
        X = np.column_stack(cols_all) if cols_all else X_orig
    else:
        X = np.asarray(flat_samples, dtype=np.float64)

    cols = []
    sel_labels = []
    for i, lab in enumerate(labels):
        if any(lab == ex or lab.startswith(f'{ex}[') for ex in exclude_sites):
            continue
        if i >= X.shape[1]:
            break
        cols.append(X[:, i])
        sel_labels.append(lab)

    if len(cols) < 2:
        print(f'[plots] emcee corner skipped: need >= 2 scalar params (got {len(cols)}).')
        return

    data = np.column_stack(cols)
    n = data.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        data = data[rng.choice(n, size=max_samples, replace=False)]

    fig = corner.corner(
        data, labels=sel_labels, show_titles=True, title_fmt='.3f', quantiles=[0.16, 0.5, 0.84],
    )
    out = os.path.join(save_path, 'corner_emcee.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plots] Saved {out}')


def lens_mass_ellipticity_summary(lens_image, kwargs_result):
    """Return human-readable ellipticity and PA values for mass components."""
    profile_types = list(getattr(lens_image.MassModel, 'profile_type_list', []))
    components = []
    for index, kwargs_mass in enumerate(kwargs_result.get('kwargs_lens', [])):
        if not isinstance(kwargs_mass, dict) or not {'e1', 'e2'}.issubset(kwargs_mass):
            continue

        e1 = float(np.asarray(kwargs_mass['e1']))
        e2 = float(np.asarray(kwargs_mass['e2']))
        ellipticity = float(np.hypot(e1, e2))
        pa_deg = float(np.degrees(0.5 * np.arctan2(e2, e1)))
        # The ellipse has a 180-degree symmetry, so this is the natural PA range.
        if pa_deg >= 90.0:
            pa_deg -= 180.0
        axis_ratio = None
        if ellipticity < 1.0:
            axis_ratio = float((1.0 - ellipticity) / (1.0 + ellipticity))

        component = {
            'index': index,
            'e1': e1,
            'e2': e2,
            'ellipticity': ellipticity,
            'axis_ratio': axis_ratio,
            'PA_deg': pa_deg,
            'PA_defined': bool(ellipticity > 0.0),
        }
        if index < len(profile_types):
            component['profile_type'] = str(profile_types[index])
        components.append(component)

    return {
        'ellipticity_definition': 'ellipticity = sqrt(e1^2 + e2^2)',
        'axis_ratio_definition': 'q = (1 - ellipticity) / (1 + ellipticity)',
        'position_angle_definition': (
            'PA_deg = 0.5 * atan2(e2, e1), in degrees, measured counter-clockwise '
            'from the positive x axis and normalized to [-90, 90).'
        ),
        'components': components,
    }


def save_lens_mass_ellipticity_summary(lens_image, kwargs_result, save_path):
    """Save and print final mass-profile ellipticity and position angles."""
    summary = lens_mass_ellipticity_summary(lens_image, kwargs_result)
    output_path = os.path.join(save_path, 'lens_mass_parameters.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)

    components = summary['components']
    if not components:
        print('[lens_mass_parameters] No lens-mass components with e1/e2; no ellipticity or PA to report.')
        return summary

    for component in components:
        name = component.get('profile_type', 'lens_mass')
        label = f"{name}[{component['index']}]"
        q_text = 'undefined' if component['axis_ratio'] is None else f"{component['axis_ratio']:.5f}"
        pa_text = 'undefined (circular)' if not component['PA_defined'] else f"{component['PA_deg']:.3f} deg"
        print(
            f"[lens_mass_parameters] {label}: e={component['ellipticity']:.5f} "
            f"(e1={component['e1']:.5f}, e2={component['e2']:.5f}), q={q_text}, PA={pa_text}"
        )
    print(f'[lens_mass_parameters] Saved {output_path}')
    return summary


def _mass_ellipticity_annotation(summary):
    components = summary.get('components', [])
    if not components:
        return None

    lines = ['Lens-mass shape:']
    for component in components:
        name = component.get('profile_type', 'mass')
        label = f"{name}[{component['index']}]"
        pa_text = 'PA undefined (circular)' if not component['PA_defined'] else f"PA={component['PA_deg']:.2f} deg"
        q_text = 'q undefined' if component['axis_ratio'] is None else f"q={component['axis_ratio']:.3f}"
        lines.append(f"{label}: e={component['ellipticity']:.3f}, {q_text}, {pa_text}")
    return '\n'.join(lines)


def plot_mass_and_convergence(lens_image, kwargs_result, pixel_scale, save_path, lens_mass_summary=None):
    """Plot 2D convergence, magnification, and radial convergence profiles."""
    # 1. Evaluate 2D convergence and magnification on image grid
    nx, ny = lens_image.Grid.num_pixel_axes
    x_grid_img, y_grid_img = lens_image.Grid.pixel_coordinates
    kwargs_lens = kwargs_result.get('kwargs_lens', [])
    
    kappa_map = np.asarray(lens_image.MassModel.kappa(x_grid_img, y_grid_img, kwargs_lens))
    mag_map = np.asarray(lens_image.MassModel.magnification(x_grid_img, y_grid_img, kwargs_lens))
    abs_mag_map = np.abs(mag_map)
    
    # 2. Compute critical lines
    crit_lines = []
    try:
        crit_lines, _ = model_util.critical_lines_caustics(
            lens_image, kwargs_lens, supersampling=5
        )
    except Exception as e:
        print(f"[plot_mass_and_convergence] Could not compute critical lines: {e}")

    # 3. Azimuthally averaged convergence about the primary mass centre.
    primary_mass = kwargs_lens[0] if kwargs_lens else {}
    center_x = float(primary_mass.get('center_x', 0.0))
    center_y = float(primary_mass.get('center_y', 0.0))
    radius_map = np.hypot(np.asarray(x_grid_img) - center_x, np.asarray(y_grid_img) - center_y)
    valid_kappa = np.isfinite(radius_map) & np.isfinite(kappa_map)
    radial_bins = np.linspace(0.0, float(np.nanmax(radius_map[valid_kappa])), 61)
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_index = np.digitize(radius_map[valid_kappa], radial_bins) - 1
    radial_mean = np.full(radial_centers.shape, np.nan)
    radial_p16 = np.full(radial_centers.shape, np.nan)
    radial_p84 = np.full(radial_centers.shape, np.nan)
    kappa_values = kappa_map[valid_kappa]
    for index in range(radial_centers.size):
        values = kappa_values[radial_index == index]
        if values.size:
            radial_mean[index] = np.mean(values)
            radial_p16[index], radial_p84[index] = np.percentile(values, [16.0, 84.0])

    # 4. Plotting (1x3 grid)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = _image_extent(ny, nx, pixel_scale)
    
    # --- Panel 0: 2D Convergence Map ---
    norm_kappa, cbar_label_kappa = _norm_from_plot_scale('log', kappa_map)
    im_kappa = axes[0].imshow(kappa_map, origin='lower', extent=extent, cmap='twilight', norm=norm_kappa)
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    axes[0].set_title(r'2D Convergence ($\kappa$) Map')
    
    # Overlay critical lines on 2D Convergence
    for i, (cline_x, cline_y) in enumerate(crit_lines):
        label = 'Critical Lines' if i == 0 else ""
        axes[0].plot(cline_x, cline_y, color='cyan', lw=1.5, ls='-', label=label)
    if crit_lines:
        axes[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im_kappa, ax=axes[0], label=cbar_label_kappa)
    
    # --- Panel 1: 2D Magnification Map ---
    # Robust LogNorm limit selection for absolute magnification
    valid_mag = abs_mag_map[np.isfinite(abs_mag_map) & (abs_mag_map > 0)]
    if len(valid_mag) > 0:
        vmin_mag = max(0.1, float(np.percentile(valid_mag, 10.0)))
        vmax_mag = min(100.0, float(np.percentile(valid_mag, 99.0)))
        if vmax_mag <= vmin_mag:
            vmax_mag = vmin_mag * 10.0
        norm_mag = LogNorm(vmin=vmin_mag, vmax=vmax_mag)
    else:
        norm_mag = LogNorm(vmin=0.1, vmax=100.0)
        
    im_mag = axes[1].imshow(abs_mag_map, origin='lower', extent=extent, cmap='twilight', norm=norm_mag)
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    axes[1].set_title(r'2D Magnification ($|\mu|$) Map')
    
    # Overlay critical lines on 2D Magnification
    for i, (cline_x, cline_y) in enumerate(crit_lines):
        label = 'Critical Lines' if i == 0 else ""
        axes[1].plot(cline_x, cline_y, color='red', lw=1.5, ls='-', label=label)
    if crit_lines:
        axes[1].legend(loc='upper right', fontsize=8)
    plt.colorbar(im_mag, ax=axes[1], label=r'log10($|\mu|$)')

    # --- Panel 2: Radial convergence profile ---
    finite_profile = np.isfinite(radial_mean)
    axes[2].plot(
        radial_centers[finite_profile], radial_mean[finite_profile],
        color='black', lw=1.8, label=r'Azimuthal mean $\kappa$',
    )
    axes[2].fill_between(
        radial_centers[finite_profile], radial_p16[finite_profile], radial_p84[finite_profile],
        color='tab:blue', alpha=0.25, label='16th-84th percentile',
    )
    axes[2].axhline(0.0, color='0.5', lw=0.8, ls='--')
    axes[2].set_xlabel('Radius from primary mass centre (arcsec)')
    axes[2].set_ylabel(r'Convergence $\kappa$')
    axes[2].set_title(r'Radial Convergence Profile')
    axes[2].legend(loc='best', fontsize=8)
    
    annotation = _mass_ellipticity_annotation(lens_mass_summary or {})
    if annotation is not None:
        fig.text(
            0.01, 0.01, annotation, ha='left', va='bottom', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': '0.6', 'alpha': 0.85, 'pad': 3},
        )
        plt.tight_layout(rect=(0, 0.13, 1, 1))
    else:
        plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mass_profile_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_run_plots(
    *,
    lens_image,
    kwargs_best,
    image_data,
    noise_map,
    psf_data,
    pixel_scale,
    save_path,
    sampler,
    best_fit_model,
    chi2=None,
    reduced_chi2=None,
    extra=None,
    mcmc_samples=None,
    flat_samples=None,
    prob_model=None,
    init_params=None,
    point_source_type_list=None,
    point_source_params_list=None,
    regul_model=None,
    param_list=None,
    residual_vis_max=0.0,
    mcmc_component_medians=None,
):
    lens_mass_summary = save_lens_mass_ellipticity_summary(
        lens_image, kwargs_best, save_path,
    )
    mask = getattr(lens_image, 'source_arc_mask', None)
    if mask is not None:
        mask = np.asarray(mask)

    if mcmc_component_medians is None and mcmc_samples is not None and prob_model is not None:
        try:
            from herculens_wrapper.samplers import evaluate_mcmc_component_medians
            print("[plots] Evaluating pixel-by-pixel median model component images across MCMC samples...")
            mcmc_component_medians = evaluate_mcmc_component_medians(prob_model, mcmc_samples)
        except Exception as e:
            print(f"[plots] Could not compute MCMC component medians: {e}")

    comp_src = mcmc_component_medians.get('source') if mcmc_component_medians else None
    comp_lens_light = mcmc_component_medians.get('lens_light') if mcmc_component_medians else None
    comp_total = mcmc_component_medians.get('total') if mcmc_component_medians else None
    comp_no_lens = mcmc_component_medians.get('no_lens_light') if mcmc_component_medians else None
    comp_ps = mcmc_component_medians.get('point_source') if mcmc_component_medians else None

    if comp_total is not None:
        best_fit_model = comp_total

    if best_fit_model is not None and image_data is not None and noise_map is not None:
        chi2 = float(np.sum(((best_fit_model - image_data) / noise_map) ** 2))

    def _try(name, fn):
        try:
            fn()
            print(f'[plots] {name}')
        except Exception as e:
            print(f'[plots] {name} skipped: {e}')

    _try('best_fit_model_linear.png', lambda: display(
        [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
        titles=[
            'Best fit model',
            'Image data',
            f'Residuals (chi^2 = {chi2:.2f})' if chi2 is not None else 'Residuals',
        ],
        pixel_scale=pixel_scale,
        savefilename=os.path.join(save_path, 'best_fit_model_linear.png'),
        plot_scale='linear',
        contour_mask=mask,
        residual_vis_max=residual_vis_max,
    ))

    _try('best_fit_model_log.png', lambda: display(
        [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
        titles=[
            'Best fit model',
            'Image data',
            f'Residuals (chi^2 = {chi2:.2f})' if chi2 is not None else 'Residuals',
        ],
        pixel_scale=pixel_scale,
        savefilename=os.path.join(save_path, 'best_fit_model_log.png'),
        plot_scale='log',
        contour_mask=mask,
        residual_vis_max=residual_vis_max,
    ))

    _try('composite.png', lambda: plot_composite_2x3_panel(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map, save_path,
        residual_vis_max=residual_vis_max,
        output_filename='composite.png',
        model_extended_override=comp_src,
        model_lens_light_override=comp_lens_light,
        model_composite_override=comp_total,
    ))

    _try('image_plane.png', lambda: plot_image_plane(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map, save_path,
        residual_vis_max=residual_vis_max,
        model_extended_override=comp_src,
        model_lens_light_override=comp_lens_light,
        model_composite_override=comp_total,
        model_point_sources_override=comp_ps,
    ))

    _try('source_plane_linear.png', lambda: plot_source_plane(
        lens_image, kwargs_best, save_path,
        plot_scale='linear', output_filename='source_plane_linear.png',
    ))
    _try('source_plane_log.png', lambda: plot_source_plane(
        lens_image, kwargs_best, save_path,
        plot_scale='log', output_filename='source_plane_log.png',
    ))

    _try('lens_light_subtracted_image.png', lambda: plot_lens_light_subtracted_image(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map=noise_map, save_path=save_path,
        plot_scale='linear', residual_vis_max=residual_vis_max,
        model_lens_light_override=comp_lens_light,
    ))
    
    _try('lens_light_subtracted_image_log.png', lambda: plot_lens_light_subtracted_image(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map=noise_map, save_path=save_path,
        plot_scale='log', residual_vis_max=residual_vis_max,
        model_lens_light_override=comp_lens_light,
    ))

    _try('ring_model_comparison_linear.png', lambda: plot_ring_model_comparison(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map, save_path,
        plot_scale='linear',
        residual_vis_max=residual_vis_max,
        output_filename='ring_model_comparison_linear.png',
        model_no_lens_light_override=comp_no_lens,
        model_lens_light_override=comp_lens_light,
    ))
    _try('ring_model_comparison_log.png', lambda: plot_ring_model_comparison(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map, save_path,
        plot_scale='log',
        residual_vis_max=residual_vis_max,
        output_filename='ring_model_comparison_log.png',
        model_no_lens_light_override=comp_no_lens,
        model_lens_light_override=comp_lens_light,
    ))


    _try('mass_profile_convergence.png', lambda: plot_mass_and_convergence(
        lens_image, kwargs_best, pixel_scale, save_path, lens_mass_summary,
    ))

    if extra and 'loss_history' in extra:
        _try('loss_curve.png', lambda: plot_loss_curve(
            np.asarray(extra['loss_history']), save_path,
        ))

    if regul_model is not None:
        _try('weights_list.png', lambda: plot_weights(regul_model.get_weights(), save_path))

    if sampler == 'svi' and extra is not None and 'guide' in extra and 'result' in extra:
        def _svi_corner():
            import jax
            import numpy as np
            rng_key = jax.random.PRNGKey(42)
            
            # Run the guide sampling on CPU to avoid GPU Out of Memory (OOM)
            # especially when pixelated source or large MGE profiles are used.
            cpu_device = jax.devices('cpu')[0]
            params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), extra['result'].params)
            
            # Reduce sample shape to 5000 to save memory/time while keeping corner plots clean
            try:
                with jax.default_device(cpu_device):
                    guide_samples = extra['guide'].sample_posterior(
                        rng_key, params_cpu, sample_shape=(5000,)
                    )
            except AttributeError:
                # Fallback for older JAX versions without jax.default_device
                guide_samples = extra['guide'].sample_posterior(
                    rng_key, params_cpu, sample_shape=(5000,)
                )
                
            guide_samples_np = {k: np.asarray(v) for k, v in guide_samples.items()}
            plot_corner_traced_params(guide_samples_np, save_path, filename='corner_svi.png', param_list=param_list)
        _try('corner_svi.png', _svi_corner)

    if mcmc_samples is not None:
        _try('corner_traced_params.png', lambda: plot_corner_traced_params(mcmc_samples, save_path, param_list=param_list))




def recreate_best_fit_plots_for_run(run_dir):
    """Recreate best_fit_model_linear.png and best_fit_model_log.png from an existing run directory."""
    import os
    import json
    npz_path = os.path.join(run_dir, 'modeling_result.npz')
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} does not exist.")
        return False
        
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return False
        
    if 'best_fit_model' not in data or 'image_data' not in data or 'noise_map' not in data:
        print(f"Error: {npz_path} does not contain required arrays.")
        return False
        
    best_fit_model = data['best_fit_model']
    image_data = data['image_data']
    noise_map = data['noise_map']

    source_arc_mask = None
    if 'source_arc_mask' in data and data['source_arc_mask'] is not None:
        try:
            mask_arr = data['source_arc_mask']
            if mask_arr.ndim > 0 and mask_arr.dtype != object:
                source_arc_mask = mask_arr
        except Exception:
            pass

    # Try to load pixel_scale from args.json
    pixel_scale = 0.08  # default fallback
    args_json_path = os.path.join(run_dir, 'args.json')
    residual_vis_max = 0.0
    if os.path.exists(args_json_path):
        try:
            with open(args_json_path, 'r') as f:
                args_dict = json.load(f)
                pixel_scale = args_dict.get('pixel_scale', pixel_scale)
                residual_vis_max = float(args_dict.get('residual_vis_max', 0.0))
        except Exception:
            pass
            
    # Try to load metrics.json for chi2
    chi2 = None
    metrics_json_path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, 'r') as f:
                metrics_dict = json.load(f)
                chi2 = metrics_dict.get('CHI2', None)
        except Exception:
            pass
            
    if chi2 is None:
        chi2 = float(np.sum(((best_fit_model - image_data) / noise_map) ** 2))
        
    # Recreate the two plots
    try:
        display(
            [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
            titles=[
                'Best fit model',
                'Image data',
                f'Residuals (chi^2 = {chi2:.2f})' if chi2 is not None else 'Residuals',
            ],
            pixel_scale=pixel_scale,
            savefilename=os.path.join(run_dir, 'best_fit_model_linear.png'),
            plot_scale='linear',
            contour_mask=source_arc_mask,
            residual_vis_max=residual_vis_max,
        )
        print(f"[plots] Saved {os.path.join(run_dir, 'best_fit_model_linear.png')}")
    except Exception as e:
        print(f"Failed to create best_fit_model_linear.png: {e}")
        
    try:
        display(
            [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
            titles=[
                'Best fit model',
                'Image data',
                f'Residuals (chi^2 = {chi2:.2f})' if chi2 is not None else 'Residuals',
            ],
            pixel_scale=pixel_scale,
            savefilename=os.path.join(run_dir, 'best_fit_model_log.png'),
            plot_scale='log',
            contour_mask=source_arc_mask,
            residual_vis_max=residual_vis_max,
        )
        print(f"[plots] Saved {os.path.join(run_dir, 'best_fit_model_log.png')}")
    except Exception as e:
        print(f"Failed to create best_fit_model_log.png: {e}")
        
    return True
