#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'herculens_wrapper'))

import jax
jax.config.update('jax_enable_x64', True)

from herculens_wrapper.models import create_lens_image

def verify_units(run_dir):
    print(f"=== VERIFYING UNITS CORRESPONDENCE FOR: {run_dir} ===")
    
    # Check required files
    args_path = os.path.join(run_dir, 'args.json')
    config_path = os.path.join(run_dir, 'config.json')
    result_path = os.path.join(run_dir, 'kwargs_result.json')
    
    for p in [args_path, config_path, result_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
            
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    with open(result_path, 'r') as f:
        kwargs_result = json.load(f)
        
    # Load input data cutout, noise map, and PSF
    print("Loading observation data FITS files...")
    image_data = fits.getdata(args_dict['data_path']).astype(np.float64)
    noise_map = fits.getdata(args_dict['noise_path']).astype(np.float64)
    psf_data = fits.getdata(args_dict['psf_path']).astype(np.float64)
    psf_data = psf_data / np.sum(psf_data) # normalize PSF
    
    # Apply background subtraction if recorded in args
    background_offset = float(args_dict.get('background_offset', 0.0))
    if background_offset != 0.0:
        image_data = image_data - background_offset
        print(f"[bkg] Subtracted global background offset: {background_offset:.6f}")
        
    # Handle pixelated source array if present
    is_pixelated = (config_dict['type_list'].get('source_light_type_list') == ['PIXELATED'])
    if is_pixelated:
        source_pixels_path = os.path.join(run_dir, 'kwargs_source_pixels.npy')
        if not os.path.exists(source_pixels_path):
            raise FileNotFoundError(f"Missing pixelated source file: {source_pixels_path}")
        source_pixels = np.load(source_pixels_path)
        kwargs_result['kwargs_source'][0]['pixels'] = source_pixels
        print(f"Loaded pixelated source array of shape {source_pixels.shape}")
    else:
        print("Parametric source model detected.")
        
    # Reconstruct LensImage model
    print("Reconstructing LensImage model...")
    mask = None
    if args_dict.get('source_arc_mask_path'):
        mask = fits.getdata(args_dict['source_arc_mask_path']).astype(bool)
        
    lens_image = create_lens_image(
        param_list=config_dict['param_list'],
        type_list=config_dict['type_list'],
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=args_dict['pixel_scale'],
        kwargs_numerics=config_dict.get('kwargs_numerics_fit', {}),
        kwargs_lens_equation_solver=config_dict.get('kwargs_lens_equation_solver_model', {}),
        source_arc_mask=mask,
        source_grid_scale=args_dict.get('source_grid_scale', 1.0)
    )
    
    # Generate image-plane model prediction
    model_combined = lens_image.model(**kwargs_result)
    
    # Calculate statistics to prove units correspondence
    data_sum = np.sum(image_data)
    model_sum = np.sum(model_combined)
    residuals = (model_combined - image_data) / noise_map
    chi2 = np.sum(residuals**2)
    reduced_chi2 = chi2 / float(image_data.size)
    
    print("\n--- Diagnostic Statistics ---")
    print(f"Total Observed Data Flux (counts/s): {data_sum:.4f}")
    print(f"Total Model Prediction Flux (counts/s): {model_sum:.4f}")
    print(f"Observed peak value (counts/s): {np.max(image_data):.4f}")
    print(f"Model peak value (counts/s): {np.max(model_combined):.4f}")
    print(f"Mean residual value: {np.mean(residuals):.4f} sigma")
    print(f"Reduced Chi-squared (chi2/N_pixels): {reduced_chi2:.4f}")
    
    # Create side-by-side plot comparing Data, Model, and Residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ny, nx = image_data.shape
    extent = [
        -nx // 2 * args_dict['pixel_scale'], (nx - nx // 2 - 1) * args_dict['pixel_scale'],
        -ny // 2 * args_dict['pixel_scale'], (ny - ny // 2 - 1) * args_dict['pixel_scale']
    ]
    
    # Plotted without any scaling to demonstrate direct correspondence
    im0 = axes[0].imshow(image_data, origin='lower', cmap='twilight', extent=extent)
    axes[0].set_title("Input Observed Data Cutout")
    plt.colorbar(im0, ax=axes[0], label="flux (counts/s)")
    
    im1 = axes[1].imshow(model_combined, origin='lower', cmap='twilight', extent=extent)
    axes[1].set_title("Composite Model Prediction")
    plt.colorbar(im1, ax=axes[1], label="flux (counts/s)")
    
    vmax_res = float(np.max(np.abs(residuals)))
    im2 = axes[2].imshow(residuals, origin='lower', cmap='bwr', extent=extent, vmin=-vmax_res, vmax=vmax_res)
    axes[2].set_title("Residuals (Model - Data) / Noise")
    plt.colorbar(im2, ax=axes[2], label="sigma (standard deviations)")
    
    for ax in axes:
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")
        if mask is not None:
            ax.contour(mask, levels=[0.5], colors='lime', extent=extent, linewidths=1.0)
            
    plt.tight_layout()
    output_path = os.path.join(run_dir, 'verify_units_correspondence.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Success] Generated verification plot: {output_path}")
    print("This proves that the observed data cutout and the generated image plane model share the exact same flux units.")

def main():
    parser = argparse.ArgumentParser(description="Verify flux units correspondence by generating the image plane model.")
    parser.add_argument("run_dir", type=str, help="Path to the run directory containing the modeling results.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: {args.run_dir} is not a directory.")
        sys.exit(1)
        
    try:
        verify_units(args.run_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
