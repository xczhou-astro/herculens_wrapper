#!/usr/bin/env python3
"""
Utility script to load back an SVI guide, draw posterior samples, and plot a corner plot.

Usage:
    python herculens_wrapper/utils/plot_corner_from_svi.py -d <run_dir_path> -n 5000 -o <output_corner_plot_path>
"""

import sys
import os
import argparse
import json
import pickle
import numpy as np

# Dynamically adjust import path to include project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_WRAPPER_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

if _WRAPPER_ROOT not in sys.path:
    sys.path.insert(0, _WRAPPER_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from herculens_wrapper.utils import (
    configure_import_paths,
    get_fits_data,
    center_crop,
)
configure_import_paths()

import jax
import numpyro.infer.autoguide as autoguide

from herculens_wrapper.models import (
    create_lens_image,
    create_prob_model,
)
from herculens_wrapper.visualizations import (
    plot_corner_traced_params,
)

def main():
    parser = argparse.ArgumentParser(description="Recreate corner plot from saved SVI guide parameters.")
    parser.add_argument("-d", "--run_dir", type=str, required=True, help="Path to the SVI run directory containing config/args and svi_guide_params.pkl")
    parser.add_argument("-n", "--num_samples", type=int, default=5000, help="Number of posterior samples to draw from the guide")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for posterior sampling")
    parser.add_argument("-o", "--output", type=str, default="corner_svi_recreated.png", help="Path to save the generated corner plot")
    args_cli = parser.parse_args()

    run_dir = os.path.abspath(args_cli.run_dir)
    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} is not a directory.")
        sys.exit(1)

    args_json_path = os.path.join(run_dir, 'args.json')
    if not os.path.exists(args_json_path):
        print(f"Error: args.json not found in {run_dir}.")
        sys.exit(1)

    with open(args_json_path, 'r') as f:
        args_dict = json.load(f)

    # Reconstruct simple namespace for backward-compatibility with wrapper functions
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = SimpleNamespace(**args_dict)

    # Find the config file copy in the run directory or fall back to the original path
    config_filename = os.path.basename(args_dict.get('config_file', 'config.py'))
    config_path = os.path.join(run_dir, config_filename)
    if not os.path.exists(config_path):
        config_path = args_dict.get('config_file')

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading configuration from {config_path}...")
    import importlib.util
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config_module
    spec.loader.exec_module(config_module)

    # Load observation data and apply modifications (crop and background offsets)
    print("Loading datasets...")
    image_data = get_fits_data(args.data_path)
    noise_map = get_fits_data(args.noise_path)
    psf_data = get_fits_data(args.psf_path)
    image_size = image_data.shape[0]

    if getattr(args, 'crop_size', None) is not None:
        image_data = center_crop(image_data, args.crop_size)
        noise_map = center_crop(noise_map, args.crop_size)

    if getattr(args, 'background_offset', 0.0) > 0.0:
        image_data = image_data - args.background_offset
        print(f"Subtracted saved background offset of {args.background_offset:.6f}")

    # Apply masks if path is specified
    if getattr(args, 'ps_mask_path', None) is not None:
        from astropy.io import fits
        mask_file = fits.open(args.ps_mask_path)
        all_mask = mask_file[0].data
        if getattr(args, 'relieve_mask_indices', None) is not None:
            for i in np.array(args.relieve_mask_indices, dtype=int):
                mask_comp = mask_file[i].data
                mask_comp = np.where(mask_comp > 0.5, 0.0, 1.0)
                all_mask = all_mask + mask_comp
        mask_bool = all_mask > 0.5
        image_data = image_data * mask_bool
        noise_map = np.where(mask_bool, noise_map, 1e10)

    # Parse configs to obtain profile lists
    lens_mass_config = getattr(config_module, 'lens_mass_config')
    lens_light_config = getattr(config_module, 'lens_light_config')
    source_light_config = getattr(config_module, 'source_light_config')
    point_source_config = getattr(config_module, 'point_source_config', None)

    lens_mass_type_list, lens_mass_params_list = lens_mass_config(
        image_size=image_size, pixel_scale=args.pixel_scale, args=args,
    )
    lens_light_type_list, lens_light_params_list = lens_light_config(
        image_size=image_size, pixel_scale=args.pixel_scale, args=args,
    )
    source_light_type_list, source_light_params_list = source_light_config(
        image_size=image_size, pixel_scale=args.pixel_scale, args=args,
    )
    if not getattr(args, 'exclude_ps', False) and point_source_config is not None:
        point_source_type_list, point_source_params_list = point_source_config(
            image_size=image_size, pixel_scale=args.pixel_scale, args=args,
        )
    else:
        point_source_type_list, point_source_params_list = [], []

    param_list = {
        'lens_mass_params_list': lens_mass_params_list,
        'lens_light_params_list': lens_light_params_list,
        'source_light_params_list': source_light_params_list,
        'point_source_params_list': point_source_params_list,
    }
    type_list = {
        'lens_mass_type_list': lens_mass_type_list,
        'lens_light_type_list': lens_light_type_list,
        'source_light_type_list': source_light_type_list,
        'point_source_type_list': point_source_type_list,
    }

    # Initialize lens_image and prob_model objects
    print("Reconstructing modeling objects...")
    lens_image = create_lens_image(
        param_list,
        type_list,
        image_data,
        noise_map,
        psf_data,
        args.pixel_scale,
        kwargs_numerics={'supersampling_factor': args.supersampling_factor},
    )

    prob_model = create_prob_model(
        param_list,
        type_list,
        lens_image,
        image_data,
        noise_map,
        fix_lens_light=args.fix_component and 'lens_light' in args.fix_component,
        fix_lens_mass=args.fix_component and 'lens_mass' in args.fix_component,
        fix_source_light=args.fix_component and 'source_light' in args.fix_component,
        args=args,
    )

    # Load SVI guide parameters
    params_pkl_path = os.path.join(run_dir, 'svi_guide_params.pkl')
    if not os.path.exists(params_pkl_path):
        print(f"Error: svi_guide_params.pkl not found in {run_dir}.")
        sys.exit(1)

    print(f"Loading SVI guide parameters from {params_pkl_path}...")
    with open(params_pkl_path, 'rb') as f:
        guide_params = pickle.load(f)

    # Set up JAX on CPU to prevent GPU OOM issues
    cpu_device = jax.devices('cpu')[0]
    params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), guide_params)

    # Recreate the AutoLowRankMultivariateNormal guide instance
    guide = autoguide.AutoLowRankMultivariateNormal(prob_model.model)

    print(f"Drawing {args_cli.num_samples} posterior samples from SVI guide...")
    rng_key = jax.random.PRNGKey(args_cli.seed)
    with jax.default_device(cpu_device):
        guide_samples = guide.sample_posterior(
            rng_key, params_cpu, sample_shape=(args_cli.num_samples,)
        )

    guide_samples_np = {k: np.asarray(v) for k, v in guide_samples.items()}

    # Resolve output directory and filename
    output_path = os.path.abspath(args_cli.output)
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Plotting corner plot and saving to {output_path}...")
    plot_corner_traced_params(
        guide_samples_np,
        output_dir,
        filename=output_filename,
        param_list=param_list,
    )
    print("Done!")

if __name__ == "__main__":
    main()
