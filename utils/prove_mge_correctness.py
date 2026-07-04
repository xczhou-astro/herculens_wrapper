#!/usr/bin/env python3
import os
import sys
import json
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits

# Ensure wrapper can be imported
sys.path.insert(0, '/Users/xczhou/Desktop/modelling/herculens_wrapper')
from herculens_wrapper.models import create_lens_image


def prove_mge_correctness(run_dir, order='ascending', num_mges=3):
    print(f"Loading outputs from run directory: {run_dir}")
    
    # Load JSON files
    args_path = os.path.join(run_dir, 'args.json')
    config_path = os.path.join(run_dir, 'config.json')
    result_path = os.path.join(run_dir, 'kwargs_result.json')
    
    if not os.path.exists(args_path) or not os.path.exists(config_path) or not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing run output files in {run_dir}")
        
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    with open(result_path, 'r') as f:
        kwargs_result = json.load(f)
        
    # Reconstruct LensImage
    print("Reconstructing LensImage model...")
    image_data = fits.getdata(args_dict['data_path']).astype(np.float64)
    noise_map = fits.getdata(args_dict['noise_path']).astype(np.float64)
    psf_data = fits.getdata(args_dict['psf_path']).astype(np.float64)
    psf_data = psf_data / np.sum(psf_data)
    
    lens_image = create_lens_image(
        param_list=config_dict['param_list'],
        type_list=config_dict['type_list'],
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=args_dict['pixel_scale'],
    )
    
    # Sort MGE components by sigma
    kwargs_lens_light = kwargs_result.get('kwargs_lens_light', [])
    if not kwargs_lens_light:
        print("Error: No lens light parameters found in kwargs_result.json")
        return
        
    # Zip index with sigma to sort
    mge_components = []
    for idx, kw in enumerate(kwargs_lens_light):
        if isinstance(kw, dict) and 'sigma' in kw:
            mge_components.append((idx, float(kw['sigma'])))
            
    # Sort by sigma ascending
    mge_components.sort(key=lambda x: x[1])
    total_mges = len(mge_components)
    print(f"Found {total_mges} MGE Gaussian components.")
    
    # Select components based on order and count
    num_mges = min(num_mges, total_mges)
    if order == 'ascending':
        # Smallest sigmas (former components)
        selected_mges = mge_components[:num_mges]
        print(f"Selected first {num_mges} components (ascending sigmas): {[x[1] for x in selected_mges]}")
    else:
        # Largest sigmas (later components)
        selected_mges = mge_components[-num_mges:]
        print(f"Selected last {num_mges} components (descending sigmas): {[x[1] for x in selected_mges]}")
        
    selected_indices = set(x[0] for x in selected_mges)
    
    # Load and replace the source pixels stub with the actual numpy array to avoid JAX type errors
    source_pixels_path = os.path.join(run_dir, 'kwargs_source_pixels.npy')
    if not os.path.exists(source_pixels_path):
        raise FileNotFoundError(f"Missing kwargs_source_pixels.npy in {run_dir}")
    source_pixels = np.load(source_pixels_path)
    
    # Zero out unselected components
    kwargs_custom = copy.deepcopy(kwargs_result)
    kwargs_custom['kwargs_source'][0]['pixels'] = source_pixels
    for idx, kw in enumerate(kwargs_custom['kwargs_lens_light']):
        if idx not in selected_indices:
            kw['amp'] = 0.0
            
    # Compute combined surface brightness of selected components
    model_custom = lens_image.model(
        **kwargs_custom, lens_light_add=True, source_add=False, point_source_add=False
    )
    pixel_area = float(lens_image.Grid.pixel_area)
    model_custom_sb = model_custom / pixel_area
    
    # Generate verification plot
    ny, nx = image_data.shape
    pixel_scale = args_dict['pixel_scale']
    img_half_w_x = nx * pixel_scale / 2.0
    img_half_w_y = ny * pixel_scale / 2.0
    extent = [-img_half_w_x, img_half_w_x, -img_half_w_y, img_half_w_y]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot custom MGE image (linear scale to see core/halo details clearly)
    im = ax.imshow(model_custom_sb, origin='lower', cmap='bwr', extent=extent)
    plt.colorbar(im, ax=ax, label='Surface Brightness (flux / arcsec$^2$)')
    
    # Overlay selected ellipses
    color_cycle = ['black', 'darkgreen', 'indigo', 'darkred', 'purple']
    for color_idx, (idx, sigma) in enumerate(selected_mges):
        kw = kwargs_lens_light[idx]
        color = color_cycle[color_idx % len(color_cycle)]
        
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
            edgecolor=color,
            facecolor='none',
            linestyle='--',
            linewidth=1.8,
            label=f"C{idx} ($\\sigma$={sigma:.4f})"
        )
        ax.add_patch(ellipse)
        ax.plot(center_x, center_y, 'x', color=color, markersize=6)
        
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title(f"MGE Verification ({order.capitalize()}, N={num_mges})")
    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    
    output_plot_path = os.path.join(run_dir, f"mge_verification_{order}_{num_mges}.png")
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Success] Generated MGE verification plot!")
    print(f"Saved to: {output_plot_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify MGE components correctness.")
    parser.add_argument(
        '--run_dir', type=str, default='/Users/xczhou/Desktop/modelling/modeling_F277W/pixelated',
        help='Path to the pixelated run directory containing kwargs_result.json'
    )
    parser.add_argument(
        '--order', type=str, default='ascending', choices=['ascending', 'descending'],
        help='Select MGEs starting from smallest (ascending) or largest (descending) sigmas'
    )
    parser.add_argument(
        '--num_mges', type=int, default=3,
        help='Number of MGE components to select and sum'
    )
    
    args = parser.parse_args()
    prove_mge_correctness(args.run_dir, args.order, args.num_mges)
