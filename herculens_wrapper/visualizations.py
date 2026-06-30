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
        vmin = max(vmin, vmax * 1e-12)
        return LogNorm(vmin=vmin, vmax=vmax), 'log'
    return None, 'linear'


def _image_extent(ny, nx, pixel_scale):
    x_center = nx // 2
    y_center = ny // 2
    return [
        -x_center * pixel_scale, (nx - x_center - 1) * pixel_scale,
        -y_center * pixel_scale, (ny - y_center - 1) * pixel_scale,
    ]


def display(plot_data, titles, pixel_scale, savefilename=None, plot_scale='linear'):
    num = len(plot_data)
    fig, axes = plt.subplots(1, num, figsize=(4 * num + 2, 4))
    if num == 1:
        axes = [axes]
    for i in range(num):
        ny, nx = plot_data[i].shape
        extent = _image_extent(ny, nx, pixel_scale)
        if plot_scale == 'log' and i < 2:
            norm, cbar_label = _norm_from_plot_scale('log', plot_data[i])
        else:
            norm, cbar_label = None, 'linear'
        im = axes[i].imshow(plot_data[i], origin='lower', cmap='magma', extent=extent, norm=norm)
        axes[i].set_xlabel('arcsec')
        axes[i].set_ylabel('arcsec')
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i], label=cbar_label)
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
):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    # 1. Linear Scale Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(image_data, origin='lower', cmap='magma', extent=extent)
    axes[0].set_title('Image data')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=axes[0], label='linear')

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

    im1 = axes[1].imshow(noise_map, origin='lower', cmap='viridis', extent=extent)
    axes[1].set_title('Noise map')
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=axes[1], label='linear')

    im2 = axes[2].imshow(psf_data, origin='lower', cmap='magma')
    axes[2].set_title('PSF kernel')
    axes[2].set_xlabel('pixel')
    axes[2].set_ylabel('pixel')
    plt.colorbar(im2, ax=axes[2], label='linear')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'input_data_linear.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Log Scale Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    norm_img, label_img = _norm_from_plot_scale('log', image_data)
    im0 = axes[0].imshow(image_data, origin='lower', cmap='magma', extent=extent, norm=norm_img)
    axes[0].set_title('Image data (log)')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=axes[0], label=label_img)

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

    norm_noise, label_noise = _norm_from_plot_scale('log', noise_map)
    im1 = axes[1].imshow(noise_map, origin='lower', cmap='viridis', extent=extent, norm=norm_noise)
    axes[1].set_title('Noise map (log)')
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=axes[1], label=label_noise)

    norm_psf, label_psf = _norm_from_plot_scale('log', psf_data)
    im2 = axes[2].imshow(psf_data, origin='lower', cmap='magma', norm=norm_psf)
    axes[2].set_title('PSF kernel (log)')
    axes[2].set_xlabel('pixel')
    axes[2].set_ylabel('pixel')
    plt.colorbar(im2, ax=axes[2], label=label_psf)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'input_data_log.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_image_plane(lens_image, kwargs_result, pixel_scale, image_data, noise_map, save_path):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    model_extended = lens_image.model(
        **kwargs_result, source_add=True, lens_light_add=False, point_source_add=False,
    )

    model_lens_light = np.zeros((ny, nx))
    if 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result, lens_light_add=True, source_add=False, point_source_add=False,
        )

    model_point_sources = np.zeros((ny, nx))
    ra_image_list = []
    dec_image_list = []
    if 'kwargs_point_source' in kwargs_result:
        model_point_sources = lens_image.model(
            **kwargs_result, source_add=False, lens_light_add=False, point_source_add=True,
        )
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

    model_composite = lens_image.model(**kwargs_result, source_add=True, point_source_add=True)
    residuals = (model_composite - image_data) / noise_map

    n_ps = len(ra_image_list)
    ps_colors = _point_source_colors(n_ps) if n_ps else []

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    im0 = ax[0, 0].imshow(model_extended, origin='lower', cmap='magma', extent=extent)
    for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
        ax[0, 0].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
    ax[0, 0].set_title('Extended Source (Lensed)')
    plt.colorbar(im0, ax=ax[0, 0], label='linear')

    im1 = ax[0, 1].imshow(model_lens_light, origin='lower', cmap='magma', extent=extent)
    ax[0, 1].set_title('Lens Light')
    plt.colorbar(im1, ax=ax[0, 1], label='linear')

    im2 = ax[0, 2].imshow(model_point_sources, origin='lower', cmap='magma', extent=extent)
    ax[0, 2].set_title('Point Sources')
    plt.colorbar(im2, ax=ax[0, 2], label='linear')

    im3 = ax[1, 0].imshow(model_composite, origin='lower', cmap='magma', extent=extent)
    for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
        ax[1, 0].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
    ax[1, 0].set_title('Composite')
    plt.colorbar(im3, ax=ax[1, 0], label='linear')

    im4 = ax[1, 1].imshow(image_data, origin='lower', cmap='magma', extent=extent)
    ax[1, 1].set_title('Image Data')
    plt.colorbar(im4, ax=ax[1, 1], label='linear')

    im5 = ax[1, 2].imshow(residuals, origin='lower', cmap='RdBu_r', extent=extent)
    ax[1, 2].set_title('Residuals (model - data) / noise')
    plt.colorbar(im5, ax=ax[1, 2])

    for a in ax.ravel():
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'image_plane.png'), dpi=300, bbox_inches='tight')
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
):
    is_pixelated = (
        'kwargs_source' in kwargs_result
        and len(kwargs_result['kwargs_source']) > 0
        and isinstance(kwargs_result['kwargs_source'][0], dict)
        and 'pixels' in kwargs_result['kwargs_source'][0]
    )

    if is_pixelated:
        source_for_plot = np.asarray(kwargs_result['kwargs_source'][0]['pixels']) / float(lens_image.Grid.pixel_area)
        extent = list(lens_image.SourceModel.pixel_grid.extent)
    else:
        fov = num_pixel * source_pixel_scale
        x = np.linspace(-fov / 2, fov / 2, num_pixel)
        y = np.linspace(-fov / 2, fov / 2, num_pixel)
        xx, yy = np.meshgrid(x, y)
        source_for_plot = np.asarray(
            lens_image.SourceModel.surface_brightness(xx, yy, kwargs_result['kwargs_source'])
        )
        extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]

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

    colors = _point_source_colors(len(ra_source_list)) if ra_source_list else []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(source_for_plot, origin='lower', extent=extent, cmap='magma', norm=norm)
    axes[0].set_title('Extended Source')
    plt.colorbar(im0, ax=axes[0], label=cbar_label)

    for i, (ras, decs) in enumerate(zip(ra_source_list, dec_source_list)):
        axes[1].scatter(ras, decs, s=30, marker='*', color=colors[i], label=f'PS {i + 1}')
    for caust_x, caust_y in caustics:
        axes[1].plot(caust_x, caust_y, color='lime', lw=1.0)
    axes[1].set_title('Point Sources + Caustics')
    # axes[1].legend(fontsize=8)
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])

    im2 = axes[2].imshow(source_for_plot, origin='lower', extent=extent, cmap='magma', norm=norm)
    for i, (ras, decs) in enumerate(zip(ra_source_list, dec_source_list)):
        axes[2].scatter(ras, decs, s=30, marker='*', color=colors[i])
    for caust_x, caust_y in caustics:
        axes[2].plot(caust_x, caust_y, color='lime', lw=1.0)
    axes[2].set_title('Source Plane Reconstruction')
    plt.colorbar(im2, ax=axes[2], label=cbar_label)

    for a in axes:
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, output_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_lens_light_subtracted_image(
    lens_image, kwargs_result, pixel_scale, image_data, noise_map=None, save_path=None,
):
    ny, nx = image_data.shape
    extent = _image_extent(ny, nx, pixel_scale)

    model_lens_light = np.zeros((ny, nx))
    if 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result, lens_light_add=True, source_add=False, point_source_add=False,
        )
    subtracted = image_data - model_lens_light

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    im0 = ax[0].imshow(image_data, origin='lower', cmap='magma', extent=extent)
    ax[0].set_title('Image data')
    plt.colorbar(im0, ax=ax[0], label='linear')

    im1 = ax[1].imshow(model_lens_light, origin='lower', cmap='magma', extent=extent)
    ax[1].set_title('Lens light model')
    plt.colorbar(im1, ax=ax[1], label='linear')

    if noise_map is not None:
        im2 = ax[2].imshow(subtracted / noise_map, origin='lower', cmap='RdBu_r', extent=extent)
        ax[2].set_title('Data - Lens light (S/N)')
    else:
        im2 = ax[2].imshow(subtracted, origin='lower', cmap='magma', extent=extent)
        ax[2].set_title('Data - Lens light')
        plt.colorbar(im2, ax=ax[2], label='linear')
    if noise_map is not None:
        plt.colorbar(im2, ax=ax[2])

    for a in ax:
        a.set_xlabel('arcsec')
        a.set_ylabel('arcsec')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lens_light_subtracted_image.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_weights(weights_list, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights_list[0][0], origin='lower', cmap='gist_stern')
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
    y_tail_delta = (y_tail[:, 0] if y_tail.ndim == 2 else y_tail) - best_loss
    ax_tail.plot(x_tail, y_tail_delta, color='tab:red', linewidth=2.0, label='Tail mean (delta)')
    ax_tail.axhline(0.0, color='k', linestyle='--', linewidth=1.0)
    ax_tail.set_xlabel('Iteration')
    ax_tail.set_ylabel('Delta loss to best')
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
        kwargs_init_json = kwargs_best_to_json_pixelated_npy(kwargs_init, save_path, type_list)
        with open(os.path.join(save_path, 'kwargs_init.json'), 'w') as f:
            json.dump(kwargs_init_json, f, indent=4, default=json_serializer)

    initial_model = lens_image.model(**kwargs_init)
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


def plot_mass_and_convergence(lens_image, kwargs_result, pixel_scale, save_path):
    """Plot 2D convergence and magnification maps with critical lines."""
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

    # 3. Plotting (1x2 grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = _image_extent(ny, nx, pixel_scale)
    
    # --- Panel 0: 2D Convergence Map ---
    norm_kappa, cbar_label_kappa = _norm_from_plot_scale('log', kappa_map)
    im_kappa = axes[0].imshow(kappa_map, origin='lower', extent=extent, cmap='magma', norm=norm_kappa)
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
        
    im_mag = axes[1].imshow(abs_mag_map, origin='lower', extent=extent, cmap='viridis', norm=norm_mag)
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
):
    """Best-effort diagnostic figures; failures are logged and skipped."""
    if chi2 is None and best_fit_model is not None and image_data is not None and noise_map is not None:
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
    ))

    _try('image_plane.png', lambda: plot_image_plane(
        lens_image, kwargs_best, pixel_scale, image_data, noise_map, save_path,
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
    ))

    _try('mass_profile_convergence.png', lambda: plot_mass_and_convergence(
        lens_image, kwargs_best, pixel_scale, save_path,
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

    if flat_samples is not None and prob_model is not None and init_params is not None:
        _try('corner_emcee.png', lambda: plot_corner_emcee(
            flat_samples, prob_model, init_params, save_path, param_list=param_list,
        ))


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
    
    # Try to load pixel_scale from args.json
    pixel_scale = 0.08  # default fallback
    args_json_path = os.path.join(run_dir, 'args.json')
    if os.path.exists(args_json_path):
        try:
            with open(args_json_path, 'r') as f:
                args_dict = json.load(f)
                pixel_scale = args_dict.get('pixel_scale', pixel_scale)
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
        )
        print(f"[plots] Saved {os.path.join(run_dir, 'best_fit_model_log.png')}")
    except Exception as e:
        print(f"Failed to create best_fit_model_log.png: {e}")
        
    return True

