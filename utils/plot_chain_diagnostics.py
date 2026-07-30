#!/usr/bin/env python3
"""
Script to load HMC batch NPZ samples (e.g. hmc_samples_batch_4.npz), split the samples
by chain, and generate all diagnostic plots (image plane, best fit, ring comparison,
source plane, source pixel summary, ArviZ trace & summary) for each individual chain.

Usage:
    python herculens_wrapper/utils/plot_chain_diagnostics.py --run_dir modeling_F277W/hmc_new_exp_2 --batch 4
"""

import os
import sys
import json
import argparse
import numpy as np
import astropy.io.fits as fits

# Dynamic sys.path insertions to handle project structure
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
sys.path.append(os.path.abspath('herculens_wrapper'))

from herculens_wrapper.models import create_lens_image, create_prob_model
from herculens_wrapper.samplers import (
    get_active_sample_sites,
    median_deterministics_from_samples,
    kwargs_with_deterministics,
    model_image_from_deterministics,
    evaluate_mcmc_component_medians,
    save_hmc_diagnostics,
    _save_hmc_pixels_wn_summary,
    evaluate_mcmc_source_pixels_summary,
)
from herculens_wrapper.visualizations import (
    plot_image_plane,
    plot_ring_model_comparison,
    plot_source_plane,
    plot_composite_2x3_panel,
    display,
)
from herculens_wrapper.utils import kwargs_best_to_json_pixelated_npy, json_serializer


def generate_chain_diagnostics(run_dir, batch_idx=4, output_subdir='chain_diagnostics'):
    run_dir = os.path.abspath(run_dir)
    
    # Locate NPZ file
    npz_name = f"hmc_samples_batch_{batch_idx}.npz"
    npz_path = os.path.join(run_dir, npz_name)
    if not os.path.exists(npz_path):
        npz_path = os.path.join(run_dir, 'diagnostics', npz_name)
        
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find {npz_name} in {run_dir} or {run_dir}/diagnostics")

    print(f"Loading samples from {npz_path}...")
    npz_data = np.load(npz_path)
    samples_dict = {k: np.asarray(npz_data[k]) for k in npz_data.files}

    # Load args and config
    args_path = os.path.join(run_dir, 'args.json')
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(args_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing args.json or config.json in {run_dir}")

    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Determine num_chains and samples per chain
    num_chains = int(args_dict.get('num_chains_hmc_numpyro', 4))
    first_key = list(samples_dict.keys())[0]
    total_samples = samples_dict[first_key].shape[0]
    samples_per_chain = total_samples // num_chains

    print(f"Total samples: {total_samples}, num_chains: {num_chains}, samples_per_chain: {samples_per_chain}")

    # Load FITS data
    image_data = fits.getdata(args_dict['data_path']).astype(np.float64)
    background_offset = float(args_dict.get('background_offset', 0.0))
    if background_offset != 0.0:
        image_data = image_data - background_offset
    noise_map = fits.getdata(args_dict['noise_path']).astype(np.float64)
    psf_data = fits.getdata(args_dict['psf_path']).astype(np.float64)
    psf_data = psf_data / np.sum(psf_data)

    source_arc_mask = None
    if args_dict.get('source_arc_mask_path'):
        source_arc_mask = fits.getdata(args_dict['source_arc_mask_path']).astype(bool)

    pixel_scale = float(args_dict.get('pixel_scale', 0.08))

    kwargs_numerics = {
        'supersampling_factor': args_dict.get('supersampling_factor', 2)
    }
    kwargs_lens_equation_solver = {
        'nsolutions': args_dict.get('ps_nsolutions', 5),
        'niter': args_dict.get('ps_niter', 10),
        'scale_factor': args_dict.get('ps_scale_factor', 2),
        'nsubdivisions': args_dict.get('ps_nsubdivisions', 3),
    }

    # Reconstruct LensImage and ProbModel
    print("Reconstructing LensImage and ProbModel...")
    lens_image = create_lens_image(
        param_list=config_dict['param_list'],
        type_list=config_dict['type_list'],
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=pixel_scale,
        kwargs_numerics=kwargs_numerics,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver,
        source_arc_mask=source_arc_mask,
        source_grid_scale=args_dict.get('source_grid_scale', 1.0),
    )

    prob_model = create_prob_model(
        param_list=config_dict['param_list'],
        type_list=config_dict['type_list'],
        lens_image=lens_image,
        image_data=image_data,
        noise_map=noise_map,
    )

    active_sites = get_active_sample_sites(prob_model)
    type_list = getattr(prob_model, 'type_list', {})
    residual_vis_max = float(args_dict.get('residual_vis_max', 0.0))

    main_out_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(main_out_dir, exist_ok=True)

    # Iterate through each chain
    for c in range(num_chains):
        print(f"\n==========================================")
        print(f"   Processing Chain {c + 1}/{num_chains}")
        print(f"==========================================")
        chain_dir = os.path.join(main_out_dir, f"chain_{c}")
        os.makedirs(chain_dir, exist_ok=True)

        # Slice samples for chain c
        chain_samples = {
            k: val[c * samples_per_chain : (c + 1) * samples_per_chain]
            for k, val in samples_dict.items()
        }

        # 1. Compute medians for chain c
        chain_medians = {
            k: np.median(np.asarray(v), axis=0)
            for k, v in chain_samples.items()
            if k in active_sites
        }
        chain_deterministics = median_deterministics_from_samples(chain_samples, active_sites=active_sites)
        chain_kwargs, chain_deterministics = kwargs_with_deterministics(
            prob_model,
            chain_medians,
            deterministics=chain_deterministics,
            rng_seed=0,
            active_sites=active_sites,
        )

        # Save kwargs_result.json for chain c
        try:
            chain_kwargs_json = kwargs_best_to_json_pixelated_npy(
                chain_kwargs, chain_dir, type_list, save_pixel_arrays=False
            )
            with open(os.path.join(chain_dir, f"kwargs_result_chain_{c}.json"), 'w') as f:
                json.dump(chain_kwargs_json, f, indent=4, default=json_serializer)
            print(f"[chain {c}] Saved kwargs_result_chain_{c}.json")
        except Exception as e:
            print(f"[chain {c}] Warning: could not save kwargs_result: {e}")

        # 2. Source pixels summary for chain c
        try:
            _save_hmc_pixels_wn_summary(
                chain_samples,
                chain_dir,
                plot_filename=f"source_pixels_wn_median_uncertainties_chain_{c}.png",
            )
            evaluate_mcmc_source_pixels_summary(prob_model, chain_samples, chain_dir, save_npy=False)
            print(f"[chain {c}] Saved source_pixels_wn_median_uncertainties_chain_{c}.png")
        except Exception as e:
            print(f"[chain {c}] Warning: could not compute source pixels summary: {e}")

        # 3. Component medians for chain c
        try:
            chain_comp_medians = evaluate_mcmc_component_medians(prob_model, chain_samples)
        except Exception as e:
            print(f"[chain {c}] Warning: could not evaluate component medians: {e}")
            chain_comp_medians = None

        chain_best_fit = chain_comp_medians['total'] if chain_comp_medians else model_image_from_deterministics(
            prob_model, chain_kwargs, chain_deterministics
        )
        chain_src = chain_comp_medians['source'] if chain_comp_medians else None
        chain_lens_light = chain_comp_medians['lens_light'] if chain_comp_medians else None
        chain_no_lens = chain_comp_medians['no_lens_light'] if chain_comp_medians else None
        chain_ps = chain_comp_medians.get('point_source') if chain_comp_medians else None

        # 4. Plot Image Plane
        try:
            plot_image_plane(
                lens_image,
                chain_kwargs,
                pixel_scale,
                image_data,
                noise_map,
                chain_dir,
                output_filename=f"image_plane_chain_{c}.png",
                model_extended_override=chain_src,
                model_lens_light_override=chain_lens_light,
                model_composite_override=chain_best_fit,
                model_point_sources_override=chain_ps,
            )
            print(f"[chain {c}] Saved image_plane_chain_{c}.png")
        except Exception as e:
            print(f"[chain {c}] Warning: could not plot image plane: {e}")

        # 5. Plot Best Fit Models (Linear & Log)
        try:
            chi2 = float(np.sum(((chain_best_fit - image_data) / noise_map) ** 2))
            mask = getattr(lens_image, 'source_arc_mask', None)
            if mask is not None:
                mask = np.asarray(mask)

            display(
                [chain_best_fit, image_data, (chain_best_fit - image_data) / noise_map],
                titles=['Best fit model', 'Image data', f'Residuals (chi^2 = {chi2:.2f})'],
                pixel_scale=pixel_scale,
                savefilename=os.path.join(chain_dir, f"best_fit_model_linear_chain_{c}.png"),
                plot_scale='linear',
                contour_mask=mask,
                residual_vis_max=residual_vis_max,
            )

            display(
                [chain_best_fit, image_data, (chain_best_fit - image_data) / noise_map],
                titles=['Best fit model', 'Image data', f'Residuals (chi^2 = {chi2:.2f})'],
                pixel_scale=pixel_scale,
                savefilename=os.path.join(chain_dir, f"best_fit_model_log_chain_{c}.png"),
                plot_scale='log',
                contour_mask=mask,
                residual_vis_max=residual_vis_max,
            )
            print(f"[chain {c}] Saved best_fit_model (linear & log) for chain_{c}")
        except Exception as e:
            print(f"[chain {c}] Warning: could not plot best fit model: {e}")

        # 5.5 Plot Composite 2x3 Panel (Mixed scale)
        try:
            plot_composite_2x3_panel(
                lens_image,
                chain_kwargs,
                pixel_scale,
                image_data,
                noise_map,
                chain_dir,
                residual_vis_max=residual_vis_max,
                output_filename=f"composite_2x3_chain_{c}.png",
                model_extended_override=chain_src,
                model_lens_light_override=chain_lens_light,
                model_composite_override=chain_best_fit,
            )
            print(f"[chain {c}] Saved composite_2x3_panel for chain_{c}")
        except Exception as e:
            print(f"[chain {c}] Warning: could not plot composite 2x3 panel: {e}")

        # 6. Plot Ring Model Comparison (Linear & Log)
        try:
            plot_ring_model_comparison(
                lens_image,
                chain_kwargs,
                pixel_scale,
                image_data,
                noise_map,
                chain_dir,
                plot_scale='linear',
                residual_vis_max=residual_vis_max,
                output_filename=f"ring_model_comparison_linear_chain_{c}.png",
                model_no_lens_light_override=chain_no_lens,
                model_lens_light_override=chain_lens_light,
            )

            plot_ring_model_comparison(
                lens_image,
                chain_kwargs,
                pixel_scale,
                image_data,
                noise_map,
                chain_dir,
                plot_scale='log',
                residual_vis_max=residual_vis_max,
                output_filename=f"ring_model_comparison_log_chain_{c}.png",
                model_no_lens_light_override=chain_no_lens,
                model_lens_light_override=chain_lens_light,
            )
            print(f"[chain {c}] Saved ring_model_comparison (linear & log) for chain_{c}")
        except Exception as e:
            print(f"[chain {c}] Warning: could not plot ring model comparison: {e}")

        # 7. Plot Source Plane (Linear & Log)
        try:
            plot_source_plane(
                lens_image,
                chain_kwargs,
                chain_dir,
                plot_scale='linear',
                output_filename=f"source_plane_linear_chain_{c}.png",
            )
            plot_source_plane(
                lens_image,
                chain_kwargs,
                chain_dir,
                plot_scale='log',
                output_filename=f"source_plane_log_chain_{c}.png",
            )
            print(f"[chain {c}] Saved source_plane (linear & log) for chain_{c}")
        except Exception as e:
            print(f"[chain {c}] Warning: could not plot source plane: {e}")

        # 8. Save ArviZ Diagnostics & Trace plots for chain c
        try:
            save_hmc_diagnostics(chain_samples, num_chains=1, target_dir=chain_dir, suffix=f"chain_{c}", prob_model=prob_model)
        except Exception as e:
            print(f"[chain {c}] Warning: could not save arviz diagnostics: {e}")

    print(f"\n==========================================")
    print(f"[Done] All per-chain diagnostic plots successfully saved in:\n  {main_out_dir}")
    print(f"==========================================")


def main():
    parser = argparse.ArgumentParser(description="Generate per-chain diagnostic plots from HMC NPZ samples.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory (e.g. modeling_F277W/hmc_new_exp_2)")
    parser.add_argument("--batch", type=int, default=4, help="Batch index to load (default: 4 for hmc_samples_batch_4.npz)")
    parser.add_argument("--output_dir", type=str, default="chain_diagnostics", help="Output subdirectory name (default: chain_diagnostics)")
    args = parser.parse_args()

    generate_chain_diagnostics(args.run_dir, batch_idx=args.batch, output_subdir=args.output_dir)


if __name__ == "__main__":
    main()
