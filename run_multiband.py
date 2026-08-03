"""Run a joint strong-lensing fit across several imaging bands."""

import importlib.util
import json
import os
import shutil
import sys

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from herculens_wrapper.utils import (
    _configure_cuda_from_args,
    _resolve_single_config_spec,
    center_crop,
    configure_import_paths,
    empty_config,
    get_fits_data,
    json_serializer,
    kwargs_best_to_json_pixelated_npy,
    normalize_run_args_paths,
    resolve_project_path,
    run_arguments_namespace,
)

configure_import_paths()

_BAND_FILES = {
    'data_path': 'Data_cutout.fits',
    'noise_path': 'noise.fits',
    'psf_path': 'psf_modelled.fits',
    'source_arc_mask_path': 'mask_1.fits',
}


def _load_config(config_path):
    name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(name, config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    args = run_arguments_namespace(module, config_path)
    return module, normalize_run_args_paths(args, config_dir=os.path.dirname(config_path))


def _call_config(function, image_size, pixel_scale, args, band_name):
    try:
        return function(image_size=image_size, pixel_scale=pixel_scale, args=args, band_name=band_name)
    except TypeError as error:
        if 'band_name' not in str(error):
            raise
        return function(image_size=image_size, pixel_scale=pixel_scale, args=args)


def _band_paths(root, band_name):
    directory = os.path.join(root, band_name)
    paths = {key: os.path.join(directory, filename) for key, filename in _BAND_FILES.items()}
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing multiband files for {band_name!r}: {missing}")
    return paths


def _materialize_pixel_arrays(kwargs, run_dir):
    source = kwargs.get('kwargs_source', [])
    if not source or not isinstance(source[0], dict):
        return kwargs
    result = dict(kwargs)
    source_entry = dict(source[0])
    for key in ('pixels', 'pixels_wn'):
        value = source_entry.get(key)
        if isinstance(value, dict) and value.get('_format') == 'pixelated_pixels_npy':
            source_entry[key] = np.load(os.path.join(run_dir, value['file']))
    result['kwargs_source'] = [source_entry, *source[1:]]
    return result


def _load_joint_initialization(prob_model, bands, init_path, random_seed):
    """Load a prior joint run into constrained multiband sampling sites."""
    from herculens_wrapper.models import kwargs2params

    init_params = prob_model.get_sample(jax.random.PRNGKey(random_seed))
    init_root = os.path.abspath(init_path)
    shared_path = os.path.join(init_root, 'kwargs_lens_shared.json')
    if not os.path.isfile(shared_path):
        raise FileNotFoundError(f'Multiband HMC warm start is missing {shared_path!r}.')
    with open(shared_path) as handle:
        shared_kwargs = json.load(handle)
    mass_params = kwargs2params(
        {'lens_mass_params_list': prob_model.lens_mass_params_list}, shared_kwargs,
        type_list={'lens_mass_type_list': prob_model.lens_mass_type_list},
    )
    for key, value in mass_params.items():
        if key in init_params:
            init_params[key] = value

    for band, band_model in zip(bands, prob_model.band_models):
        run_dir = os.path.join(init_root, band['name'])
        result_path = os.path.join(run_dir, 'kwargs_result.json')
        if not os.path.isfile(result_path):
            raise FileNotFoundError(f'Missing warm-start result for band {band["name"]!r}: {result_path!r}')
        with open(result_path) as handle:
            kwargs = _materialize_pixel_arrays(json.load(handle), run_dir)
        band_params = kwargs2params(band['param_list'], kwargs, type_list=band['type_list'])
        if (
            band['type_list'].get('source_light_type_list') == ['PIXELATED']
            and getattr(band_model, 'prior_type', 'matern') == 'matern'
        ):
            source = kwargs.get('kwargs_source', [{}])[0]
            required = ('pixels_wn', 'n_source_grid', 'rho_source_grid', 'sigma_source_grid')
            if all(key in source for key in required):
                band_params.update({
                    'pixels_wn_source_grid': jnp.asarray(source['pixels_wn']),
                    'n_source_grid': jnp.asarray(source['n_source_grid']),
                    'rho_source_grid': jnp.asarray(source['rho_source_grid']),
                    'sigma_source_grid': jnp.asarray(source['sigma_source_grid']),
                })
        prefix = f"{band['site_prefix']}/"
        for key, value in band_params.items():
            if prefix + key in init_params:
                init_params[prefix + key] = value
    print(f'[multiband] Warm-started joint model from {init_root}')
    return init_params


def _band_hmc_samples(samples, band):
    """Expose shared mass and one band's sites under the usual single-band names."""
    prefix = f"{band['site_prefix']}/"
    result = {
        key: value for key, value in samples.items()
        if key.startswith('lens_') and not key.startswith('lens_light_')
    }
    result.update({
        key[len(prefix):]: value for key, value in samples.items() if key.startswith(prefix)
    })
    return result


def build_and_run_multiband(config_path=None):
    if config_path is None:
        config_path = _resolve_single_config_spec(sys.argv[1] if len(sys.argv) > 1 else 'config.py')
    config_path = os.path.abspath(config_path)
    config_module, args = _load_config(config_path)
    _configure_cuda_from_args(args)

    if not bool(getattr(args, 'use_multiband', False)):
        raise ValueError('run_multiband.py requires use_multiband=True.')
    band_names = list(getattr(args, 'band_names', []))
    if len(band_names) < 2:
        raise ValueError('band_names must contain at least two bands.')
    data_root = resolve_project_path(args.multiband_data_path, config_dir=os.path.dirname(config_path))
    save_path = resolve_project_path(args.save_path, config_dir=os.path.dirname(config_path))
    os.makedirs(save_path, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_path, os.path.basename(config_path)))
    with open(os.path.join(save_path, 'args.json'), 'w') as handle:
        json.dump(vars(args), handle, indent=4, default=json_serializer)

    from herculens_wrapper.models import create_lens_image
    from herculens_wrapper.multiband import band_site_prefix, create_multiband_prob_model
    from herculens_wrapper.samplers import (
        evaluate_mcmc_source_pixels_summary,
        run_hmc,
        run_optax,
        run_svi,
        save_hmc_diagnostics,
    )
    from herculens_wrapper.visualizations import generate_run_plots, plot_input_data

    lens_mass_config = getattr(config_module, 'lens_mass_config', empty_config)
    lens_light_config = getattr(config_module, 'lens_light_config', empty_config)
    source_light_config = getattr(config_module, 'source_light_config', empty_config)
    point_source_config = getattr(config_module, 'point_source_config', empty_config)

    bands = []
    shared_mass_params = None
    shared_mass_types = None
    for index, band_name in enumerate(band_names):
        paths = _band_paths(data_root, band_name)
        image_data = get_fits_data(paths['data_path'])
        noise_map = get_fits_data(paths['noise_path'])
        psf_data = get_fits_data(paths['psf_path'])
        source_arc_mask = get_fits_data(paths['source_arc_mask_path']).astype(bool)
        if args.crop_size is not None:
            image_data = center_crop(image_data, args.crop_size)
            noise_map = center_crop(noise_map, args.crop_size)
            source_arc_mask = center_crop(source_arc_mask, args.crop_size)
        if image_data.shape != noise_map.shape or image_data.shape != source_arc_mask.shape:
            raise ValueError(f'Data, noise, and mask shapes must agree for {band_name!r}.')

        image_size = image_data.shape[0]
        mass_types, mass_params = _call_config(lens_mass_config, image_size, args.pixel_scale, args, band_name)
        if shared_mass_params is None:
            shared_mass_params, shared_mass_types = mass_params, mass_types
        elif mass_types != shared_mass_types or mass_params != shared_mass_params:
            raise ValueError('lens_mass_config must return identical types and priors for every band.')
        lens_light_types, lens_light_params = _call_config(lens_light_config, image_size, args.pixel_scale, args, band_name)
        source_types, source_params = _call_config(source_light_config, image_size, args.pixel_scale, args, band_name)
        point_types, point_params = ([], []) if getattr(args, 'exclude_ps', True) else _call_config(
            point_source_config, image_size, args.pixel_scale, args, band_name
        )
        type_list = {
            'lens_mass_type_list': mass_types,
            'lens_light_type_list': lens_light_types,
            'source_light_type_list': source_types,
            'point_source_type_list': point_types,
        }
        param_list = {
            'lens_mass_params_list': mass_params,
            'lens_light_params_list': lens_light_params,
            'source_light_params_list': source_params,
            'point_source_params_list': point_params,
        }
        lens_image = create_lens_image(
            param_list, type_list, image_data, noise_map, psf_data, args.pixel_scale,
            kwargs_numerics={'supersampling_factor': args.supersampling_factor},
            source_arc_mask=source_arc_mask,
            source_grid_scale=float(getattr(args, 'source_grid_scale', 1.0)),
            conjugate_points=getattr(args, 'conjugate_points', None),
        )
        band_dir = os.path.join(save_path, band_name)
        os.makedirs(band_dir, exist_ok=True)
        plot_input_data(image_data, noise_map, psf_data, args.pixel_scale, band_dir, point_types, point_params, source_arc_mask)
        bands.append({
            'name': band_name,
            'site_prefix': band_site_prefix(index, band_name),
            'lens_image': lens_image,
            'image_data': image_data,
            'noise_map': noise_map,
            'psf_data': psf_data,
            'type_list': type_list,
            'param_list': param_list,
            'save_path': band_dir,
        })

    prob_model = create_multiband_prob_model(bands, shared_mass_params, shared_mass_types, args)
    for band, band_model in zip(bands, prob_model.band_models):
        band['prob_model'] = band_model
    sampler = args.sampler
    mcmc_samples = None
    init_path = getattr(args, 'init_params_path', None)
    init_params = (
        _load_joint_initialization(prob_model, bands, init_path, args.random_seed)
        if init_path else prob_model.get_sample(jax.random.PRNGKey(args.random_seed))
    )
    if sampler == 'svi':
        best_params, extra = run_svi(prob_model, None, args, init_params)
    elif sampler == 'optax':
        best_params, extra = run_optax(prob_model, args, init_params)
    elif sampler == 'hmc':
        if not init_path:
            raise ValueError('Joint multiband HMC requires init_params_path from a joint SVI/Optax run.')
        mcmc_samples, best_params, extra = run_hmc(prob_model, args, init_params, init_path)
        np.savez_compressed(os.path.join(save_path, 'hmc_samples.npz'), **{
            key: np.asarray(value) for key, value in mcmc_samples.items()
        })
    else:
        raise ValueError(f'Unsupported multiband sampler {sampler!r}.')

    kwargs_by_band = prob_model.params2kwargs_by_band(best_params)
    shared_lens = kwargs_by_band[band_names[0]]['kwargs_lens']
    with open(os.path.join(save_path, 'kwargs_lens_shared.json'), 'w') as handle:
        json.dump({'kwargs_lens': shared_lens}, handle, indent=4, default=json_serializer)
    for band in bands:
        kwargs_best = kwargs_by_band[band['name']]
        band_samples = _band_hmc_samples(mcmc_samples, band) if mcmc_samples is not None else None
        if band_samples is not None:
            source_summary = evaluate_mcmc_source_pixels_summary(
                band['prob_model'], band_samples, band['save_path'], save_npy=True,
            )
            if source_summary is not None and kwargs_best.get('kwargs_source'):
                kwargs_best['kwargs_source'][0]['pixels'] = source_summary[0]
        kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_best, band['save_path'], band['type_list'])
        with open(os.path.join(band['save_path'], 'kwargs_result.json'), 'w') as handle:
            json.dump(kwargs_json, handle, indent=4, default=json_serializer)
        with open(os.path.join(band['save_path'], 'kwargs_lens_shared.json'), 'w') as handle:
            json.dump({'kwargs_lens': shared_lens}, handle, indent=4, default=json_serializer)
        if band_samples is not None:
            save_hmc_diagnostics(band_samples, int(args.num_chains_hmc_numpyro), band['save_path'], 'final', band['prob_model'])
        best_fit_model = band['lens_image'].model(**kwargs_best)
        chi2 = float(np.sum(((best_fit_model - band['image_data']) / band['noise_map']) ** 2))
        n_params = sum(np.asarray(value).size for value in best_params.values())
        reduced_chi2 = chi2 / max(int(band['image_data'].size) - n_params, 1)
        generate_run_plots(
            lens_image=band['lens_image'], kwargs_best=kwargs_best,
            image_data=band['image_data'], noise_map=band['noise_map'], psf_data=band['psf_data'],
            pixel_scale=args.pixel_scale, save_path=band['save_path'], sampler=sampler,
            best_fit_model=best_fit_model, chi2=chi2, reduced_chi2=reduced_chi2,
            extra=extra, mcmc_samples=band_samples, flat_samples=None, prob_model=band['prob_model'],
            init_params=None, point_source_type_list=band['type_list']['point_source_type_list'],
            point_source_params_list=band['param_list']['point_source_params_list'],
            regul_model=None, param_list=band['param_list'],
            residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
        )
    print(f'[multiband] Run complete. Outputs in {save_path}')
    return save_path


if __name__ == '__main__':
    build_and_run_multiband()
