#!/usr/bin/env python3
import os
import sys
import json
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.segmentation import detect_sources, deblend_sources

# Ensure wrapper can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from herculens_wrapper.models import create_lens_image


def deblend_and_ray_trace(run_dir, threshold_frac=0.05, plot_scale='log', n_pixels=5, contrast=0.001, n_levels=32):
    print(f"Loading outputs from run directory: {run_dir}")
    
    # Load JSON settings
    args_path = os.path.join(run_dir, 'args.json')
    config_path = os.path.join(run_dir, 'config.json')
    result_path = os.path.join(run_dir, 'kwargs_result.json')
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Missing args.json in {run_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing kwargs_result.json in {run_dir}")
        
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    with open(result_path, 'r') as f:
        kwargs_result = json.load(f)
        
    # Load npy pixelated source
    source_pixels_path = os.path.join(run_dir, 'kwargs_source_pixels.npy')
    if not os.path.exists(source_pixels_path):
        raise FileNotFoundError(f"Missing kwargs_source_pixels.npy in {run_dir}")
    source_pixels = np.load(source_pixels_path)
    
    # Load FITS data cutout, noise map, PSF, mask
    print("Loading data FITS files...")
    image_data = fits.getdata(args_dict['data_path']).astype(np.float64)
    background_offset = float(args_dict.get('background_offset', 0.0))
    if background_offset != 0.0:
        image_data = image_data - background_offset
        print(f"[bkg] Applied stored global background offset: {background_offset:.6f}")
    noise_map = fits.getdata(args_dict['noise_path']).astype(np.float64)
    psf_data = fits.getdata(args_dict['psf_path']).astype(np.float64)
    psf_data = psf_data / np.sum(psf_data) # normalize PSF
    
    source_arc_mask = None
    if args_dict.get('source_arc_mask_path'):
        source_arc_mask = fits.getdata(args_dict['source_arc_mask_path']).astype(bool)
        
    # Initialize LensImage Extension
    print("Reconstructing LensImage model...")
    kwargs_numerics = {
        'supersampling_factor': args_dict.get('supersampling_factor', 2)
    }
    kwargs_lens_equation_solver = {
        'nsolutions': args_dict.get('ps_nsolutions', 5),
        'niter': args_dict.get('ps_niter', 10),
        'scale_factor': args_dict.get('ps_scale_factor', 2),
        'nsubdivisions': args_dict.get('ps_nsubdivisions', 3),
    }
    
    lens_image = create_lens_image(
        param_list=config_dict['param_list'],
        type_list=config_dict['type_list'],
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=args_dict['pixel_scale'],
        kwargs_numerics=kwargs_numerics,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver,
        source_arc_mask=source_arc_mask,
        source_grid_scale=args_dict.get('source_grid_scale', 1.0),
    )
    
    # Watershed-based deblending using photutils
    print("Deblending source plane components using photutils...")
    peak_flux = np.max(source_pixels)
    threshold = threshold_frac * peak_flux
    
    segm = detect_sources(source_pixels, threshold, n_pixels=n_pixels)
    if segm is not None:
        try:
            deblended = deblend_sources(source_pixels, segm, n_pixels=n_pixels, contrast=contrast, n_levels=n_levels)
            if deblended is not None:
                labeled_array = deblended.data
                labels = deblended.labels
            else:
                labeled_array = segm.data
                labels = segm.labels
        except Exception as e:
            print(f"Warning: photutils deblend failed: {e}. Falling back to initial detection.")
            labeled_array = segm.data
            labels = segm.labels
    else:
        labeled_array = np.zeros_like(source_pixels, dtype=int)
        labels = []
        
    num_features = len(labels)
    print(f"Photutils found {num_features} deblended source features above threshold of {threshold:.4e} ({threshold_frac*100}% of peak flux {peak_flux:.4e})")
    
    # Compile components sorted by total flux
    components = []
    for lbl in labels:
        mask = (labeled_array == lbl)
        flux = np.sum(source_pixels[mask])
        components.append({
            'id': lbl,
            'flux': flux,
            'mask': mask,
        })
    components.sort(key=lambda x: x['flux'], reverse=True)
    
    num_to_show = len(components)
    if num_to_show == 0:
        print("Warning: No components detected above the threshold.")
        num_to_show = 1
        # Fallback to single dummy component of entire source
        components = [{
            'id': 1,
            'flux': np.sum(source_pixels),
            'mask': np.ones_like(source_pixels, dtype=bool)
        }]
        
    print(f"Evaluating lensed contributions for all {num_to_show} components...")
    
    # Pre-render standard combined model and lens light
    kwargs_all = copy.deepcopy(kwargs_result)
    kwargs_all['kwargs_source'][0]['pixels'] = source_pixels
    
    model_combined = lens_image.model(**kwargs_all, source_add=True, lens_light_add=True, point_source_add=True)
    model_lens_light = lens_image.model(**kwargs_all, source_add=False, lens_light_add=True, point_source_add=False)
    
    # Render each component
    lensed_components = []
    for k in range(num_to_show):
        comp = components[k]
        masked_source = np.zeros_like(source_pixels)
        masked_source[comp['mask']] = source_pixels[comp['mask']]
        
        kwargs_comp = copy.deepcopy(kwargs_result)
        kwargs_comp['kwargs_source'][0]['pixels'] = masked_source
        
        comp_lensed = lens_image.model(
            **kwargs_comp, source_add=True, lens_light_add=False, point_source_add=False
        )
        lensed_components.append(comp_lensed)
        
    # Extents for plot axes in arcseconds
    ny_img, nx_img = image_data.shape
    pixel_scale = args_dict['pixel_scale']
    img_half_w_x = nx_img * pixel_scale / 2.0
    img_half_w_y = ny_img * pixel_scale / 2.0
    img_extent = [-img_half_w_x, img_half_w_x, -img_half_w_y, img_half_w_y]
    
    # Calculate coordinate limits of the zoomed source plane footprint
    src_x, src_y = lens_image.SourceModel.pixel_grid.pixel_coordinates
    img_x, img_y = lens_image.Grid.pixel_coordinates
    if source_arc_mask is not None:
        # Find ray-traced coordinate range of the mask footprint
        x_mapped, y_mapped = lens_image.MassModel.ray_shooting(
            img_x[source_arc_mask], img_y[source_arc_mask], kwargs_result['kwargs_lens']
        )
        center_x = 0.5 * (np.min(x_mapped) + np.max(x_mapped))
        center_y = 0.5 * (np.min(y_mapped) + np.max(y_mapped))
        half_range = 0.5 * max(np.max(x_mapped) - np.min(x_mapped), np.max(y_mapped) - np.min(y_mapped))
        src_extent = [center_x - half_range, center_x + half_range, center_y - half_range, center_y + half_range]
    else:
        src_extent = [np.min(src_x), np.max(src_x), np.min(src_y), np.max(src_y)]
        
    # Set up matplotlib figure
    # Row 0 has 4 panels: Cutout, Combined model, Lens light, Segmented source plane.
    # Row 1 has N panels: Lensed Component 1, 2, ..., N.
    n_cols = max(5, num_to_show)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols + 1 * n_cols, 10))
    
    # Helpers for rendering in linear/log
    pixel_area = float(lens_image.Grid.pixel_area)
    
    def render_im(ax, img_raw, extent, title, colormap='twilight', is_log=False):
        img = img_raw / pixel_area
        if is_log:
            vmin = np.percentile(img[img > 0], 10) if np.any(img > 0) else 1e-4
            log_img = np.log10(np.maximum(img, vmin))
            im = ax.imshow(log_img, origin='lower', cmap=colormap, extent=extent)
            plt.colorbar(im, ax=ax, label='log10(flux / arcsec$^2$)')
        else:
            im = ax.imshow(img, origin='lower', cmap=colormap, extent=extent)
            plt.colorbar(im, ax=ax, label='flux / arcsec$^2$')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        
    is_log = (plot_scale == 'log')
    
    # Panel 1: Image Data cutout (log scale)
    render_im(axes[0, 0], image_data, img_extent, "Original Data Cutout", is_log=is_log)
    if source_arc_mask is not None:
        axes[0, 0].contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)
        
    # Panel 2: Whole Combined Model (log scale)
    render_im(axes[0, 1], model_combined, img_extent, "Combined Model Fit", is_log=is_log)
    if source_arc_mask is not None:
        axes[0, 1].contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)
        
    # Panel 3: Lens Light Model (log scale)
    render_im(axes[0, 2], model_lens_light, img_extent, "Lens Light Model", is_log=is_log)
    
    # Source plane extents for rendering source
    ny_src, nx_src = source_pixels.shape
    src_pixel_scale = (src_extent[1] - src_extent[0]) / nx_src
    src_plot_extent = [src_extent[0], src_extent[1], src_extent[2], src_extent[3]]
    
    # Panel 4: Model - Lens Light (log scale)
    render_im(axes[0, 3], model_combined - model_lens_light, img_extent, "Model - Lens Light", is_log=is_log)
    if source_arc_mask is not None:
        axes[0, 3].contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)

    # Panel 5: Segmented Source Plane (always linear scale!)
    ax_src = axes[0, 4]
    im_src = ax_src.imshow(source_pixels / pixel_area, origin='lower', cmap='twilight', extent=src_plot_extent)
    plt.colorbar(im_src, ax=ax_src, label='flux / arcsec$^2$')
        
    # Overlay colors/contours for each component on source plane
    color_cycle = ['cyan', 'magenta', 'orange', 'yellow', 'lime', 'pink', 'purple']
    from matplotlib.patches import Patch
    legend_elements = []
    for idx in range(num_to_show):
        comp = components[idx]
        color = color_cycle[idx % len(color_cycle)]
        # Outline contour around component mask
        ax_src.contour(comp['mask'], levels=[0.5], colors=[color], extent=src_plot_extent, linewidths=2.0)
        legend_elements.append(Patch(facecolor='none', edgecolor=color, linewidth=2, label=f"C{idx+1}"))
        
    ax_src.legend(handles=legend_elements, loc='upper right', framealpha=0.8)
    ax_src.set_title("Segmented Source Plane", fontsize=12, pad=10)
    ax_src.set_xlabel('arcsec')
    ax_src.set_ylabel('arcsec')
    
    # Hide any unused subplots in row 0
    for col_idx in range(5, n_cols):
        axes[0, col_idx].axis('off')
    
    # Row 2: Lensed components (log scale)
    for idx in range(num_to_show):
        color = color_cycle[idx % len(color_cycle)]
        ax_comp = axes[1, idx]
        comp_img = lensed_components[idx]
        
        render_im(ax_comp, comp_img, img_extent, f"Lensed Component {idx+1}", is_log=is_log)
        if source_arc_mask is not None:
            ax_comp.contour(source_arc_mask, levels=[0.5], colors='lime', extent=img_extent, linewidths=1.0)
            
        for spine in ax_comp.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.5)
            
    # Hide any unused subplots in row 1
    for col_idx in range(num_to_show, n_cols):
        axes[1, col_idx].axis('off')
        
    plt.tight_layout()
    output_plot_path = os.path.join(run_dir, 'deblended_contributions.png')
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Success] Generated deblended component ray-tracing diagnostic plot!")
    print(f"Saved to: {output_plot_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deblend pixelated source reconstructions and ray-trace individual component contributions.")
    parser.add_argument(
        '--run_dir', type=str, default='/Users/xczhou/Desktop/modelling/modeling_F277W/pixelated',
        help='Path to the pixelated run directory containing kwargs_result.json'
    )
    parser.add_argument(
        '--threshold_frac', type=float, default=0.05,
        help='Threshold fraction of the peak source intensity to define components'
    )
    parser.add_argument(
        '--plot_scale', type=str, default='log', choices=['linear', 'log'],
        help='Scale to plot flux profiles (linear or log)'
    )
    parser.add_argument(
        '--n_pixels', type=int, default=5,
        help='Minimum number of connected pixels to detect a component'
    )
    parser.add_argument(
        '--contrast', type=float, default=0.001,
        help='Fraction of total flux a local peak must have to be deblended (contrast threshold)'
    )
    parser.add_argument(
        '--n_levels', type=int, default=32,
        help='Number of multi-thresholding levels for watershed deblending'
    )
    
    args = parser.parse_args()
    deblend_and_ray_trace(
        args.run_dir, args.threshold_frac, args.plot_scale,
        n_pixels=args.n_pixels, contrast=args.contrast, n_levels=args.n_levels
    )
