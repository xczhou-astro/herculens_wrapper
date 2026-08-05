"""Run a joint strong-lensing fit across several imaging bands."""

import importlib.util
import json
import os
import pickle
import shutil
import shlex
import sys
import copy
from datetime import datetime
from types import SimpleNamespace

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
    log_jax_device_layout,
    normalize_run_args_paths,
    resolve_project_path,
    resolve_init_run_dir,
    run_arguments_namespace,
    Tee,
)

configure_import_paths()

_BAND_FILE_NAMES = {
    'data_path': ('data_name', 'Data_cutout.fits'),
    'noise_path': ('noise_name', 'noise.fits'),
    'psf_path': ('psf_name', 'psf_modelled.fits'),
    'source_arc_mask_path': ('source_arc_mask_name', 'mask_1.fits'),
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


def _band_paths(args, band_name):
    """Resolve ``{base_path}/{band}/{filename}`` inputs for one band."""
    paths = {
        path_key: os.path.join(
            getattr(args, path_key),
            band_name,
            getattr(args, name_key, default_name),
        )
        for path_key, (name_key, default_name) in _BAND_FILE_NAMES.items()
    }
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


def _validate_pixelated_svi_initialization(init_root, bands):
    """Require a completed pixelated-SVI run before starting joint HMC."""
    config_path = os.path.join(init_root, 'config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f'Multiband HMC warm start is missing {config_path!r}; '
            'select a completed pixelated-SVI run directory.'
        )
    with open(config_path) as handle:
        init_config = json.load(handle)
    if str(init_config.get('sampler', '')).lower() != 'svi':
        raise ValueError(
            'Joint multiband HMC must be initialized from a pixelated SVI run, '
            f'but {config_path!r} records sampler={init_config.get("sampler")!r}.'
        )
    config_bands = init_config.get('bands', {})
    for band in bands:
        source_types = config_bands.get(band['name'], {}).get('type_list', {}).get(
            'source_light_type_list'
        )
        if source_types != ['PIXELATED']:
            raise ValueError(
                f'Multiband HMC requires a pixelated-SVI warm start for {band["name"]!r}; '
                f'found source_light_type_list={source_types!r}.'
            )


def _load_joint_initialization(
    prob_model, bands, init_path, random_seed, require_pixelated_svi=False,
):
    """Load a prior joint run into constrained multiband sampling sites."""
    from herculens_wrapper.models import kwargs2params

    init_params = prob_model.get_sample(jax.random.PRNGKey(random_seed))
    # ``get_sample`` includes the Matérn deterministic pixel image.  It is a
    # prior draw and must not override restored pixels_wn/n/rho/sigma when
    # params2kwargs builds the initial diagnostic source/model images.
    init_params = {
        key: value for key, value in init_params.items()
        if key.rsplit('/', 1)[-1] != 'pixels_source_grid'
    }
    init_root = os.path.abspath(init_path)
    if require_pixelated_svi:
        _validate_pixelated_svi_initialization(init_root, bands)
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
            value_array = jnp.asarray(value)
            expected_array = jnp.asarray(init_params[key])
            if value_array.shape != expected_array.shape:
                if value_array.size == expected_array.size == 1:
                    value_array = jnp.reshape(value_array, expected_array.shape)
                elif require_pixelated_svi:
                    raise ValueError(
                        f'Multiband HMC warm-start shape mismatch for {key}: '
                        f'saved={value_array.shape}, expected={expected_array.shape}.'
                    )
                else:
                    continue
            init_params[key] = value_array

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
            missing = [key for key in required if key not in source]
            if missing:
                if require_pixelated_svi:
                    raise ValueError(
                        f'Pixelated SVI warm start for {band["name"]!r} is missing {missing}. '
                        'Expected saved pixels_wn and Matérn parameters.'
                    )
                print(
                    f'[multiband:svi] {band["name"]} has no saved pixelated source; '
                    'initializing its pixels_wn and Matérn parameters from their priors.'
                )
            else:
                band_params.update({
                    'pixels_wn_source_grid': jnp.asarray(source['pixels_wn']),
                    'n_source_grid': jnp.asarray(source['n_source_grid']),
                    'rho_source_grid': jnp.asarray(source['rho_source_grid']),
                    'sigma_source_grid': jnp.asarray(source['sigma_source_grid']),
                })
        prefix = f"{band['site_prefix']}/"
        for key, value in band_params.items():
            joint_key = prefix + key
            if joint_key not in init_params:
                continue
            value_array = jnp.asarray(value)
            expected_array = jnp.asarray(init_params[joint_key])
            if value_array.shape != expected_array.shape:
                if value_array.size == expected_array.size == 1:
                    value_array = jnp.reshape(value_array, expected_array.shape)
                elif require_pixelated_svi:
                    raise ValueError(
                        f'Multiband HMC warm-start shape mismatch for {band["name"]!r} '
                        f'site {key}: saved={value_array.shape}, expected={expected_array.shape}.'
                    )
                else:
                    continue
            init_params[joint_key] = value_array
    print(f'[multiband] Warm-started joint model from {init_root}')
    return init_params


def _hmc_run_finished(log_path):
    """Whether a prior HMC invocation reached its normal completion marker."""
    if not os.path.isfile(log_path):
        return False
    with open(log_path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return bool(lines and lines[-1].lower().startswith('end at'))


def _hmc_checkpoint_samples_per_chain(run_path, args):
    """Return completed draws per chain from a compatible HMC checkpoint."""
    checkpoint_path = os.path.join(run_path, 'hmc_checkpoint.pkl')
    if not os.path.isfile(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, 'rb') as handle:
            checkpoint = pickle.load(handle)
        configured_chains = int(args.num_chains_hmc_numpyro)
        saved_chains = checkpoint.get('num_chains')
        if saved_chains is not None and int(saved_chains) != configured_chains:
            raise ValueError(
                f'checkpoint uses num_chains={saved_chains}, but configuration requests '
                f'num_chains_hmc_numpyro={configured_chains}'
            )
        completed = 0
        for batch in checkpoint.get('all_samples', []):
            if not batch:
                continue
            first_value = next(iter(batch.values()))
            batch_total = int(np.asarray(first_value).shape[0])
            if batch_total % configured_chains:
                raise ValueError(
                    'checkpoint sample count is incompatible with the configured number of chains'
                )
            completed += batch_total // configured_chains
        return completed
    except Exception as error:
        raise ValueError(f'Could not inspect HMC checkpoint {checkpoint_path!r}: {error}') from error


def _report_hmc_warm_start_reproduction(init_root, bands, initial_kwargs_by_band):
    """Numerically compare restored HMC inputs with the selected SVI outputs."""
    for band in bands:
        name = band['name']
        restored_kwargs = initial_kwargs_by_band[name]
        restored_model = np.asarray(band['lens_image'].model(**restored_kwargs))
        prior_model_path = os.path.join(init_root, 'modeling_result.npz')
        model_key = f'{name}_best_fit_model'
        if os.path.isfile(prior_model_path):
            with np.load(prior_model_path) as prior_result:
                if model_key in prior_result:
                    saved_model = np.asarray(prior_result[model_key])
                    if saved_model.shape == restored_model.shape:
                        delta = restored_model - saved_model
                        rms = float(np.sqrt(np.mean(delta**2)))
                        max_abs = float(np.max(np.abs(delta)))
                        print(
                            f'[hmc:init] {name} restored-model check: '
                            f'rms_difference={rms:.3e}, max_abs_difference={max_abs:.3e}'
                        )
                    else:
                        print(
                            f'[hmc:init] {name} restored-model check skipped: '
                            f'saved shape={saved_model.shape}, current shape={restored_model.shape}'
                        )

        source_path = os.path.join(init_root, name, 'kwargs_source_pixels.npy')
        restored_source = restored_kwargs.get('kwargs_source', [{}])[0].get('pixels')
        if os.path.isfile(source_path) and restored_source is not None:
            saved_source = np.load(source_path)
            restored_source = np.asarray(restored_source)
            if saved_source.shape == restored_source.shape:
                delta = restored_source - saved_source
                print(
                    f'[hmc:init] {name} restored-source check: '
                    f'rms_difference={float(np.sqrt(np.mean(delta**2))):.3e}, '
                    f'max_abs_difference={float(np.max(np.abs(delta))):.3e}'
                )


def _load_fixed_light_kwargs(init_path, bands):
    """Load the shared mass and each band's lens light for source-only warmup."""
    init_root = os.path.abspath(init_path)
    with open(os.path.join(init_root, 'kwargs_lens_shared.json')) as handle:
        kwargs_lens = json.load(handle)['kwargs_lens']
    kwargs_lens_light = {}
    for band in bands:
        run_dir = os.path.join(init_root, band['name'])
        with open(os.path.join(run_dir, 'kwargs_result.json')) as handle:
            kwargs = _materialize_pixel_arrays(
                json.load(handle), run_dir,
            )
        kwargs_lens_light[band['name']] = kwargs.get('kwargs_lens_light', [])
    return kwargs_lens, kwargs_lens_light


def _run_pixelated_svi_warmup(prob_model, bands, args, init_params, init_path):
    """Optimize joint source parameters with the prior mass and lens light fixed."""
    if (
        args.sampler != 'svi'
        or getattr(args, 'pixelated_init_match', 'image') != 'image'
        or not init_path
        or not any(band['type_list'].get('source_light_type_list') == ['PIXELATED'] for band in bands)
    ):
        return init_params
    max_iterations = int(getattr(args, 'num_iterations_warmup', 0))
    if max_iterations <= 0:
        return init_params

    from herculens_wrapper.multiband import create_multiband_prob_model
    from herculens_wrapper.samplers import get_active_sample_sites, run_svi

    kwargs_lens, kwargs_lens_light = _load_fixed_light_kwargs(init_path, bands)
    warmup_model = create_multiband_prob_model(
        bands,
        prob_model.lens_mass_params_list,
        prob_model.lens_mass_type_list,
        args,
        fixed_lens_mass=kwargs_lens,
        fixed_lens_light_by_band=kwargs_lens_light,
    )
    active_sites = set(get_active_sample_sites(warmup_model, rng_seed=args.random_seed))
    warmup_init = {key: value for key, value in init_params.items() if key in active_sites}
    warmup_args = SimpleNamespace(**vars(args))
    warmup_args.max_iterations_svi = max_iterations
    print(f'[multiband:svi-warmup] Optimizing source parameters for {max_iterations} iterations.')
    best_warmup, _ = run_svi(warmup_model, None, warmup_args, warmup_init)
    for key, value in best_warmup.items():
        if key in init_params:
            init_params[key] = value
    return init_params


def _initialize_pixelated_sources_from_previous_source(bands, args, init_params, init_path, seed):
    """Fit each pixelated Matérn source initialization to a prior analytic source."""
    from herculens_wrapper.models import PowerSpectrum, _project_analytic_kwargs_to_pixel_source

    init_root = os.path.abspath(init_path)
    max_iterations = int(getattr(args, 'num_iterations_warmup', 0))
    if max_iterations <= 0:
        print('[multiband:source-init] num_iterations_warmup <= 0; retaining prior pixelated draws.')
        return init_params

    for band in bands:
        if (
            band['type_list'].get('source_light_type_list') != ['PIXELATED']
            or getattr(band['prob_model'], 'prior_type', 'matern') != 'matern'
        ):
            continue
        result_path = os.path.join(init_root, band['name'], 'kwargs_result.json')
        with open(result_path) as handle:
            prior_kwargs = _materialize_pixel_arrays(
                json.load(handle), os.path.dirname(result_path),
            )
        analytic_source = prior_kwargs.get('kwargs_source', [])
        if not analytic_source or 'pixels' in analytic_source[0]:
            raise ValueError(
                f"pixelated_init_match='source' requires an analytic source in the prior "
                f"run for {band['name']!r}. Use pixelated_init_match='image' otherwise."
            )

        ny, nx = band['lens_image'].SourceModel.pixel_grid.num_pixel_axes
        k_values = PowerSpectrum.K_grid((ny, nx)).k
        pixelated_prior = band['param_list']['source_light_params_list'][0].get(
            'pixelated_prior', {}
        )
        source_image = _project_analytic_kwargs_to_pixel_source(
            band['lens_image'], analytic_source,
        )
        print(
            f"[multiband:source-init] Fitting {band['name']} Matérn source "
            f"to the prior analytic source ({max_iterations} iterations)."
        )
        fitted = PowerSpectrum.fit_power_spectrum_init(
            source_image, k_values, pixelated_prior,
            seed=seed + 7919, max_iterations=max_iterations,
        )
        prefix = f"{band['site_prefix']}/"
        for key, value in fitted.items():
            if prefix + key in init_params:
                init_params[prefix + key] = jnp.asarray(value)
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


def _zip_asymmetric_uncertainties(lower, upper):
    """Represent HMC 16th/84th-percentile offsets in the usual JSON format."""
    if isinstance(lower, dict) and isinstance(upper, dict):
        return {
            key: _zip_asymmetric_uncertainties(lower[key], upper[key])
            for key in lower
        }
    if isinstance(lower, list) and isinstance(upper, list):
        return [
            _zip_asymmetric_uncertainties(left, right)
            for left, right in zip(lower, upper)
        ]
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)
    if lower_array.ndim == 0:
        return [float(lower_array), float(upper_array)]
    return np.stack([lower_array, upper_array], axis=0)


def _rebase_pixel_array_references(payload, band_name):
    """Make pixel-array stubs in a joint result relative to its run directory."""
    if isinstance(payload, dict):
        result = {
            key: _rebase_pixel_array_references(value, band_name)
            for key, value in payload.items()
        }
        if result.get('_format') == 'pixelated_pixels_npy' and result.get('file'):
            result['file'] = os.path.join(band_name, result['file'])
        return result
    if isinstance(payload, list):
        return [_rebase_pixel_array_references(value, band_name) for value in payload]
    return copy.deepcopy(payload)


def _joint_log_likelihood(prob_model, params):
    """Sum likelihood terms from every band-scoped NumPyro observation site."""
    from numpyro.infer.util import log_likelihood

    terms_by_site = log_likelihood(prob_model.model, params, batch_ndims=0)
    return float(sum(np.sum(np.asarray(term)) for term in terms_by_site.values()))


def _save_multiband_hmc_batch_diagnostics(
    samples, batch_index, bands, args, run_path, save_hmc_diagnostics,
    prob_model, evaluate_mcmc_component_medians,
    evaluate_mcmc_source_pixels_summary, plot_multiband_composite,
    hmc_extra_fields=None,
):
    """Save the compact joint HMC diagnostic set for one checkpoint."""
    batch_root = os.path.join(run_path, 'diagnostics', f'batch_{batch_index}')
    os.makedirs(batch_root, exist_ok=True)
    combined_results = []
    for band in bands:
        band_samples = _band_hmc_samples(samples, band)
        median_params = {
            key: np.median(np.asarray(value), axis=0)
            for key, value in band_samples.items()
        }
        kwargs_best = band['prob_model'].params2kwargs(
            median_params,
            kwargs_lens_override=prob_model.mass_kwargs_from_params(median_params),
        )
        source_summary = evaluate_mcmc_source_pixels_summary(
            band['prob_model'], band_samples, batch_root, save_npy=False,
        )
        if source_summary is not None and kwargs_best.get('kwargs_source'):
            kwargs_best['kwargs_source'][0]['pixels'] = source_summary[0]

        component_medians = evaluate_mcmc_component_medians(
            band['prob_model'], band_samples,
            active_sites=band_samples.keys(),
            kwargs_lens_from_params=prob_model.mass_kwargs_from_params,
        )
        combined_results.append({
            'name': band['name'],
            'lens_image': band['lens_image'],
            'kwargs_result': kwargs_best,
            'image_data': band['image_data'],
            'noise_map': band['noise_map'],
            'pixel_scale': args.pixel_scale,
            'model_lens_light': component_medians['lens_light'],
            'model_lensed_source': component_medians['source'],
            'model_total': component_medians['total'],
        })

    plot_multiband_composite(
        combined_results, batch_root,
        residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
        output_filename=f'multiband_composite_batch_{batch_index}.png',
    )
    save_hmc_diagnostics(
        samples, int(args.num_chains_hmc_numpyro), batch_root,
        f'batch_{batch_index}', prob_model,
        hmc_extra_fields=hmc_extra_fields,
    )
    print(f'[hmc] Saved multi-band diagnostics for batch {batch_index + 1} to {batch_root}')


def build_and_run_multiband(config_path=None):
    if config_path is None:
        config_path = _resolve_single_config_spec(sys.argv[1] if len(sys.argv) > 1 else 'config.py')
    config_path = os.path.abspath(config_path)
    config_module, args = _load_config(config_path)
    _configure_cuda_from_args(args)

    if not bool(getattr(args, 'use_multiband', False)):
        raise ValueError('run_multiband.py requires use_multiband=True.')
    band_names = list(getattr(args, 'band_names', []))
    if not band_names:
        raise ValueError('band_names must contain at least one band.')
    save_path = resolve_project_path(args.save_path, config_dir=os.path.dirname(config_path))
    os.makedirs(save_path, exist_ok=True)
    configured_n_runs = int(getattr(args, 'n_runs', 1))
    composite_log_file = None
    composite_log_stdout = None
    composite_log_stderr = None
    root_hmc_logging = args.sampler == 'hmc'
    if (args.sampler == 'svi' and configured_n_runs > 1) or root_hmc_logging:
        root_log_path = os.path.join(save_path, 'log.txt')
        resume_root_log = root_hmc_logging and os.path.isfile(root_log_path)
        composite_log_file = open(root_log_path, 'a' if resume_root_log else 'w')
        composite_log_stdout = sys.stdout
        composite_log_stderr = sys.stderr
        root_log_marker = 'Resume' if resume_root_log else 'Start'
        composite_log_file.write(
            f"{root_log_marker} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        composite_log_file.flush()
        sys.stdout = Tee(sys.stdout, composite_log_file)
        sys.stderr = Tee(sys.stderr, composite_log_file)
    shutil.copy(config_path, os.path.join(save_path, os.path.basename(config_path)))
    with open(os.path.join(save_path, 'args.json'), 'w') as handle:
        json.dump(vars(args), handle, indent=4, default=json_serializer)

    print(f'Invoked: {shlex.join([sys.executable, *sys.argv])}')
    print(f'Starting multi-band run in: {save_path} (sampler={args.sampler!r})')
    log_jax_device_layout(args)

    from herculens_wrapper.models import create_lens_image, validate_param_list
    from herculens_wrapper.multiband import band_site_prefix, create_multiband_prob_model
    from herculens_wrapper.samplers import (
        evaluate_mcmc_component_medians,
        evaluate_mcmc_source_pixels_summary,
        _save_hmc_pixels_wn_summary,
        run_hmc,
        run_optax,
        run_svi,
        save_hmc_diagnostics,
    )
    from herculens_wrapper.visualizations import (
        generate_run_plots,
        plot_input_data,
        plot_corner_traced_params,
        plot_multiband_composite,
        plot_multiband_source_reconstructions,
        plot_loss_curve,
        plot_mass_and_convergence,
        save_lens_mass_ellipticity_summary,
    )

    lens_mass_config = getattr(config_module, 'lens_mass_config', empty_config)
    lens_light_config = getattr(config_module, 'lens_light_config', empty_config)
    source_light_config = getattr(config_module, 'source_light_config', empty_config)
    point_source_config = getattr(config_module, 'point_source_config', empty_config)
    kwargs_lens_equation_solver_model = {
        'nsolutions': getattr(args, 'ps_nsolutions', 5),
        'niter': getattr(args, 'ps_niter', 10),
        'scale_factor': getattr(args, 'ps_scale_factor', 2),
        'nsubdivisions': getattr(args, 'ps_nsubdivisions', 3),
    }

    bands = []
    shared_mass_params = None
    shared_mass_types = None
    for index, band_name in enumerate(band_names):
        paths = _band_paths(args, band_name)
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

        background_offset = 0.0
        background_size = int(getattr(args, 'background_subtract_corner', 0))
        background_corner = str(
            getattr(args, 'background_subtract_which_corner', 'bottom_left')
        ).lower().strip()
        if background_size > 0:
            if background_size > min(image_data.shape):
                raise ValueError(
                    f"background_subtract_corner={background_size} is larger than "
                    f"the {band_name!r} image dimensions {image_data.shape}"
                )
            corner_slices = {
                'bottom_left': (slice(0, background_size), slice(0, background_size)),
                'bottom_right': (slice(0, background_size), slice(-background_size, None)),
                'top_left': (slice(-background_size, None), slice(0, background_size)),
                'top_right': (slice(-background_size, None), slice(-background_size, None)),
            }
            if background_corner not in corner_slices:
                raise ValueError(
                    'background_subtract_which_corner must be one of: '
                    'bottom_left, bottom_right, top_left, top_right'
                )
            background_offset = float(np.nanmedian(image_data[corner_slices[background_corner]]))
            image_data = image_data - background_offset
            print(
                f'[bkg:{band_name}] Derived global background offset of {background_offset:.6f} '
                f'from {background_corner} corner ({background_size}x{background_size} pixels) '
                'and subtracted it.'
            )

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
        validate_param_list(type_list, param_list)
        print(f'[{band_name}] Data shape: {image_data.shape}; active source-mask pixels: {int(source_arc_mask.sum())}')
        print(f'[{band_name}] Lens mass type list: {mass_types}')
        print(f'[{band_name}] Lens light type list: {lens_light_types}')
        print(f'[{band_name}] Source light type list: {source_types}')
        print(f'[{band_name}] Point source type list: {point_types}')
        lens_image = create_lens_image(
            param_list, type_list, image_data, noise_map, psf_data, args.pixel_scale,
            kwargs_numerics={'supersampling_factor': args.supersampling_factor},
            kwargs_lens_equation_solver=kwargs_lens_equation_solver_model,
            source_arc_mask=source_arc_mask,
            source_grid_scale=float(getattr(args, 'source_grid_scale', 1.0)),
            conjugate_points=getattr(args, 'conjugate_points', None),
        )
        bands.append({
            'name': band_name,
            'site_prefix': band_site_prefix(index, band_name),
            'lens_image': lens_image,
            'image_data': image_data,
            'noise_map': noise_map,
            'psf_data': psf_data,
            'type_list': type_list,
            'param_list': param_list,
            'background_offset': background_offset,
            'save_path': None,
        })

    prob_model = create_multiband_prob_model(bands, shared_mass_params, shared_mass_types, args)
    for band, band_model in zip(bands, prob_model.band_models):
        band['prob_model'] = band_model
    sampler = args.sampler
    base_save_path = save_path
    n_runs = int(getattr(args, 'n_runs', 1))
    if sampler == 'hmc':
        n_runs = 1
    base_random_seed = int(args.random_seed)
    init_root = getattr(args, 'init_params_path', None)
    if init_root:
        init_root = resolve_init_run_dir(init_root, config_dir=os.path.dirname(config_path))
        if not os.path.isdir(init_root):
            raise FileNotFoundError(f'Multiband initialization path does not exist: {init_root!r}')
        print(f'[multiband] Selected initialization run: {init_root}')
    if sampler == 'hmc':
        if not init_root:
            raise ValueError('Joint multiband HMC requires init_params_path from a pixelated SVI run.')
    comparison = {}
    if sampler == 'svi' and n_runs > 1:
        print(f'Starting joint SVI multi-run in: {base_save_path} (n_runs={n_runs})')
    elif sampler == 'hmc':
        print(
            f'Starting joint HMC in: {base_save_path} '
            f'(num_chains={int(args.num_chains_hmc_numpyro)})'
        )
    for run_index in range(n_runs):
        run_path = base_save_path if n_runs == 1 else os.path.join(base_save_path, f'run_{run_index}')
        run_seed = base_random_seed + run_index
        os.makedirs(run_path, exist_ok=True)
        run_log_path = os.path.join(run_path, 'log.txt')
        if sampler == 'hmc' and _hmc_run_finished(run_log_path):
            completed_draws = _hmc_checkpoint_samples_per_chain(run_path, args)
            if completed_draws is None:
                raise FileNotFoundError(
                    f'HMC run at {run_path!r} is marked complete but has no checkpoint. '
                    'Cannot safely extend it without restarting the chain.'
                )
            requested_draws = int(args.num_samples_hmc_numpyro)
            if completed_draws >= requested_draws:
                print(
                    f'[hmc] Existing run is complete with {completed_draws} draws per chain '
                    f'(requested {requested_draws}); skipping it.'
                )
                if composite_log_file is not None:
                    composite_log_file.write(
                        f"End at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    composite_log_file.close()
                    sys.stdout = composite_log_stdout
                    sys.stderr = composite_log_stderr
                return base_save_path
            print(
                f'[hmc] Extending completed run from {completed_draws} to '
                f'{requested_draws} draws per chain without repeating warm-up.'
            )
        if root_hmc_logging:
            # HMC has one run at ``save_path``. Its root log is already
            # active, so do not reopen the same file or nest a second Tee.
            run_log_file = composite_log_file
            original_stdout = None
            original_stderr = None
        else:
            resume_run = sampler == 'hmc' and os.path.isfile(run_log_path)
            run_log_file = open(run_log_path, 'a' if resume_run else 'w')
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run_log_file.write(f"{'Resume' if resume_run else 'Start'} at {timestamp}\n")
            run_log_file.flush()
            sys.stdout = Tee(sys.stdout, run_log_file)
            sys.stderr = Tee(sys.stderr, run_log_file)

        print(f'\n========================================')
        print(f'Starting multi-band run {run_index} (seed={run_seed}, sampler={sampler!r})')
        print(f'========================================')
        args.save_path = run_path
        args.random_seed = run_seed
        shutil.copy(config_path, os.path.join(run_path, os.path.basename(config_path)))
        with open(os.path.join(run_path, 'args.json'), 'w') as handle:
            json.dump(vars(args), handle, indent=4, default=json_serializer)
        run_init_path = init_root
        for band in bands:
            band['save_path'] = os.path.join(run_path, band['name'])
            os.makedirs(band['save_path'], exist_ok=True)
            plot_input_data(
                band['image_data'], band['noise_map'], band['psf_data'], args.pixel_scale,
                band['save_path'], band['type_list']['point_source_type_list'],
                band['param_list']['point_source_params_list'],
                getattr(band['lens_image'], 'source_arc_mask', None),
                background_offset=band['background_offset'],
            )

        init_params = (
            _load_joint_initialization(
                prob_model, bands, run_init_path, run_seed,
                require_pixelated_svi=(sampler == 'hmc'),
            )
            if run_init_path else prob_model.get_sample(jax.random.PRNGKey(run_seed))
        )
        num_params = prob_model.count_sampled_parameters()
        with open(os.path.join(run_path, 'config.json'), 'w') as handle:
            json.dump({
                'bands': {
                    band['name']: {
                        'type_list': band['type_list'],
                        'param_list': band['param_list'],
                        'background_offset': band['background_offset'],
                    }
                    for band in bands
                },
                'shared_lens_mass_type_list': shared_mass_types,
                'shared_lens_mass_params_list': shared_mass_params,
                'num_params': num_params,
                'sampler': sampler,
                'init_params_path': run_init_path,
                'kwargs_numerics_fit': {
                    'supersampling_factor': args.supersampling_factor,
                },
                'kwargs_lens_equation_solver_model': kwargs_lens_equation_solver_model,
            }, handle, indent=4, default=json_serializer)
        mcmc_samples = None
        if sampler == 'svi':
            pixelated_match = str(getattr(args, 'pixelated_init_match', 'image')).lower()
            if pixelated_match not in ('image', 'source'):
                raise ValueError("pixelated_init_match must be 'image' or 'source'.")
            if pixelated_match == 'image':
                init_params = _run_pixelated_svi_warmup(
                    prob_model, bands, args, init_params, run_init_path,
                )
            elif (
                run_init_path
                and any(
                    band['type_list'].get('source_light_type_list') == ['PIXELATED']
                    for band in bands
                )
            ):
                init_params = _initialize_pixelated_sources_from_previous_source(
                    bands, args, init_params, run_init_path, run_seed,
                )

        init_log_prob = float(np.sum(prob_model.log_prob(init_params, constrained=True)))
        init_log_likelihood = _joint_log_likelihood(prob_model, init_params)
        print(f'Number of sampled parameters: {num_params}')
        print(
            f'Initial joint log-prob: {init_log_prob:.2f} '
            f'(log-likelihood: {init_log_likelihood:.2f})'
        )
        try:
            initial_kwargs_by_band = prob_model.params2kwargs_by_band(init_params)
            if sampler == 'hmc':
                _report_hmc_warm_start_reproduction(
                    run_init_path, bands, initial_kwargs_by_band,
                )
            initial_kwargs_json_by_band = {}
            initial_band_results = []
            for band in bands:
                initial_kwargs_json = kwargs_best_to_json_pixelated_npy(
                    initial_kwargs_by_band[band['name']],
                    band['save_path'],
                    band['type_list'],
                    pixels_filename='kwargs_source_pixels_init.npy',
                    pixels_wn_filename='kwargs_source_pixels_wn_init.npy',
                )
                initial_kwargs_json_by_band[band['name']] = _rebase_pixel_array_references(
                    initial_kwargs_json, band['name'],
                )
                initial_model = band['lens_image'].model(**initial_kwargs_by_band[band['name']])
                initial_chi2 = float(np.sum(
                    ((initial_model - band['image_data']) / band['noise_map']) ** 2
                ))
                print(f"Initial {band['name']} chi^2: {initial_chi2:.2f}")
                initial_band_results.append({
                    'name': band['name'],
                    'lens_image': band['lens_image'],
                    'kwargs_result': initial_kwargs_by_band[band['name']],
                    'image_data': band['image_data'],
                    'noise_map': band['noise_map'],
                    'pixel_scale': args.pixel_scale,
                })
            with open(os.path.join(run_path, 'kwargs_init.json'), 'w') as handle:
                json.dump({
                    'kwargs_lens': initial_kwargs_by_band[band_names[0]]['kwargs_lens'],
                    'kwargs_by_band': initial_kwargs_json_by_band,
                }, handle, indent=4, default=json_serializer)
            plot_multiband_composite(
                initial_band_results,
                run_path,
                residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
                output_filename='initial_guess_model.png',
            )
            plot_multiband_source_reconstructions(
                initial_band_results,
                run_path,
                output_filename='initial_source_model.png',
            )
        except Exception as error:
            print(f'[init] Initial multi-band diagnostics skipped: {error}')
        if sampler == 'svi':
            # Match the single-band behavior: without a warm start the guide
            # chooses its own init_to_median location from prior samples.
            svi_init_params = init_params if run_init_path else None
            best_params, extra = run_svi(prob_model, None, args, svi_init_params)
            if 'loss_history' in extra:
                with open(os.path.join(run_path, 'svi_loss_history.json'), 'w') as handle:
                    json.dump(
                        {'loss_history': np.asarray(extra['loss_history']).tolist()}, handle, indent=4,
                    )
                try:
                    plot_loss_curve(np.asarray(extra['loss_history']), run_path)
                except Exception as error:
                    print(f'[plots] loss_curve.png skipped: {error}')
            if 'result' in extra:
                with open(os.path.join(run_path, 'svi_guide_params.pkl'), 'wb') as handle:
                    pickle.dump(extra['result'].params, handle)
        elif sampler == 'optax':
            best_params, extra = run_optax(prob_model, args, init_params)
        elif sampler == 'hmc':
            batch_callback = lambda samples, batch_index, hmc_extra_fields: _save_multiband_hmc_batch_diagnostics(
                samples, batch_index, bands, args, run_path, save_hmc_diagnostics,
                prob_model, evaluate_mcmc_component_medians,
                evaluate_mcmc_source_pixels_summary, plot_multiband_composite,
                hmc_extra_fields,
            )
            mcmc_samples, best_params, extra = run_hmc(
                prob_model, args, init_params, run_init_path,
                batch_diagnostics_callback=batch_callback,
            )
            np.savez_compressed(os.path.join(run_path, 'hmc_samples.npz'), **{
                key: np.asarray(value) for key, value in mcmc_samples.items()
            })
        else:
            raise ValueError(f'Unsupported multiband sampler {sampler!r}.')

        posterior_samples = mcmc_samples
        if sampler == 'svi' and 'guide' in extra and 'result' in extra:
            try:
                cpu_device = jax.devices('cpu')[0]
                guide_params = jax.tree_util.tree_map(
                    lambda value: jax.device_put(value, cpu_device), extra['result'].params,
                )
                with jax.default_device(cpu_device):
                    posterior_samples = extra['guide'].sample_posterior(
                        jax.random.PRNGKey(run_seed + 12345), guide_params, sample_shape=(2000,),
                    )
            except Exception as error:
                print(f'[svi] Failed to draw guide samples for kwargs_sigma.json: {error}')

        kwargs_by_band = prob_model.params2kwargs_by_band(best_params)
        kwargs_sigma_by_band = None
        if posterior_samples is not None:
            try:
                if sampler == 'svi':
                    joint_sigma_params = {
                        key: np.std(np.asarray(value), axis=0)
                        for key, value in posterior_samples.items()
                    }
                    kwargs_sigma_by_band = prob_model.params2kwargs_by_band(joint_sigma_params)
                else:
                    joint_p16 = {
                        key: np.percentile(np.asarray(value), 16, axis=0)
                        for key, value in posterior_samples.items()
                    }
                    joint_p50 = {
                        key: np.percentile(np.asarray(value), 50, axis=0)
                        for key, value in posterior_samples.items()
                    }
                    joint_p84 = {
                        key: np.percentile(np.asarray(value), 84, axis=0)
                        for key, value in posterior_samples.items()
                    }
                    lower_by_band = prob_model.params2kwargs_by_band({
                        key: joint_p50[key] - joint_p16[key] for key in joint_p50
                    })
                    upper_by_band = prob_model.params2kwargs_by_band({
                        key: joint_p84[key] - joint_p50[key] for key in joint_p50
                    })
                    kwargs_sigma_by_band = {
                        band_name: _zip_asymmetric_uncertainties(
                            lower_by_band[band_name], upper_by_band[band_name],
                        )
                        for band_name in band_names
                    }
            except Exception as error:
                print(f'[{sampler}] Failed to derive per-band kwargs_sigma.json: {error}')
        if posterior_samples is not None:
            try:
                plot_corner_traced_params(
                    posterior_samples,
                    run_path,
                    filename='corner_multiband.png',
                    site_order=prob_model.posterior_site_order(),
                )
            except Exception as error:
                print(f'[plots] corner_multiband.png skipped: {error}')
        shared_lens = kwargs_by_band[band_names[0]]['kwargs_lens']
        with open(os.path.join(run_path, 'kwargs_lens_shared.json'), 'w') as handle:
            json.dump({'kwargs_lens': shared_lens}, handle, indent=4, default=json_serializer)
        comparison[f'run_{run_index}'] = {'seed': run_seed, 'bands': {}}
        combined_band_results = []
        combined_kwargs_by_band = {}
        result_arrays = {}
        total_chi2 = 0.0
        total_data_pixels = 0
        for band in bands:
            kwargs_best = kwargs_by_band[band['name']]
            band_samples = _band_hmc_samples(mcmc_samples, band) if mcmc_samples is not None else None
            component_medians = None
            if band_samples is not None:
                _save_hmc_pixels_wn_summary(band_samples, band['save_path'])
                source_summary = evaluate_mcmc_source_pixels_summary(
                    band['prob_model'], band_samples, band['save_path'], save_npy=True,
                )
                if source_summary is not None and kwargs_best.get('kwargs_source'):
                    kwargs_best['kwargs_source'][0]['pixels'] = source_summary[0]
                try:
                    component_medians = evaluate_mcmc_component_medians(
                        band['prob_model'], band_samples,
                        active_sites=band_samples.keys(),
                        kwargs_lens_from_params=prob_model.mass_kwargs_from_params,
                    )
                except Exception as error:
                    print(f"[plots] HMC component medians for {band['name']} skipped: {error}")
            kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_best, band['save_path'], band['type_list'])
            with open(os.path.join(band['save_path'], 'kwargs_result.json'), 'w') as handle:
                json.dump(kwargs_json, handle, indent=4, default=json_serializer)
            combined_kwargs_by_band[band['name']] = _rebase_pixel_array_references(
                kwargs_json, band['name'],
            )
            with open(os.path.join(band['save_path'], 'kwargs_lens_shared.json'), 'w') as handle:
                json.dump({'kwargs_lens': shared_lens}, handle, indent=4, default=json_serializer)
            if kwargs_sigma_by_band is not None:
                try:
                    kwargs_sigma_json = kwargs_best_to_json_pixelated_npy(
                        kwargs_sigma_by_band[band['name']], band['save_path'], band['type_list'],
                        save_pixel_arrays=False,
                    )
                    with open(os.path.join(band['save_path'], 'kwargs_sigma.json'), 'w') as handle:
                        json.dump(kwargs_sigma_json, handle, indent=4, default=json_serializer)
                    print(f"[{sampler}] Saved {band['name']} kwargs_sigma.json")
                except Exception as error:
                    print(f"[{sampler}] Failed to save {band['name']} kwargs_sigma.json: {error}")
            if band_samples is not None:
                save_hmc_diagnostics(
                    band_samples, int(args.num_chains_hmc_numpyro), band['save_path'],
                    'final', band['prob_model'],
                    hmc_extra_fields=extra.get('hmc_sampler_health'),
                )
            best_fit_model = band['lens_image'].model(**kwargs_best)
            metrics_model = component_medians.get('total') if component_medians else best_fit_model
            chi2 = float(np.sum(((metrics_model - band['image_data']) / band['noise_map']) ** 2))
            total_chi2 += chi2
            total_data_pixels += int(band['image_data'].size)
            comparison[f'run_{run_index}']['bands'][band['name']] = {
                'chi2': chi2,
            }
            generate_run_plots(
                lens_image=band['lens_image'], kwargs_best=kwargs_best,
                image_data=band['image_data'], noise_map=band['noise_map'], psf_data=band['psf_data'],
                pixel_scale=args.pixel_scale, save_path=band['save_path'], sampler=sampler,
                best_fit_model=best_fit_model, chi2=chi2, reduced_chi2=None,
                extra=None,
                mcmc_samples=band_samples, flat_samples=None, prob_model=band['prob_model'],
                init_params=None, point_source_type_list=band['type_list']['point_source_type_list'],
                point_source_params_list=band['param_list']['point_source_params_list'],
                regul_model=None, param_list=band['param_list'],
                residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
                mcmc_component_medians=component_medians,
            )
            if sampler == 'svi' and posterior_samples is not None:
                try:
                    plot_corner_traced_params(
                        _band_hmc_samples(posterior_samples, band),
                        band['save_path'],
                        filename='corner_svi.png',
                        param_list=band['param_list'],
                    )
                except Exception as error:
                    print(f"[plots] {band['name']} corner_svi.png skipped: {error}")
            combined_band_results.append({
                'name': band['name'],
                'lens_image': band['lens_image'],
                'kwargs_result': kwargs_best,
                'image_data': band['image_data'],
                'noise_map': band['noise_map'],
                'pixel_scale': args.pixel_scale,
                'model_lens_light': (
                    component_medians.get('lens_light') if component_medians else None
                ),
                'model_lensed_source': (
                    component_medians.get('source') if component_medians else None
                ),
                'model_total': component_medians.get('total') if component_medians else None,
            })
            result_arrays[f'{band["name"]}_best_fit_model'] = np.asarray(metrics_model)
            result_arrays[f'{band["name"]}_image_data'] = np.asarray(band['image_data'])
            result_arrays[f'{band["name"]}_noise_map'] = np.asarray(band['noise_map'])
            result_arrays[f'{band["name"]}_source_arc_mask'] = np.asarray(
                getattr(band['lens_image'], 'source_arc_mask', None)
            )

        total_log_likelihood = _joint_log_likelihood(prob_model, best_params)
        with open(os.path.join(run_path, 'kwargs_result.json'), 'w') as handle:
            json.dump({
                'kwargs_lens': shared_lens,
                'kwargs_by_band': combined_kwargs_by_band,
            }, handle, indent=4, default=json_serializer)
        num_params_free = num_params
        for band in bands:
            if band['type_list'].get('source_light_type_list') != ['PIXELATED']:
                continue
            ny, nx = band['lens_image'].SourceModel.pixel_grid.num_pixel_axes
            if getattr(band['prob_model'], 'prior_type', 'matern') == 'wavelet_sparsity':
                num_params_free -= int(getattr(band['prob_model'], 'nscales', 1)) * ny * nx
            else:
                num_params_free -= ny * nx
        num_params_free = max(int(num_params_free), 0)
        dof = max(total_data_pixels - num_params_free, 1)
        metrics = {
            'BIC': float(num_params_free * np.log(total_data_pixels) - 2 * total_log_likelihood),
            'CHI2': float(total_chi2),
            'CHI2_NPIX2': float(total_chi2 / total_data_pixels),
            'REDUCED_CHI2': float(total_chi2 / dof),
            'CHI2_DOF': int(dof),
            'N_DATA_PIXELS': int(total_data_pixels),
            'N_PARAMS_FITTED': int(num_params),
            'N_PARAMS_FREE': num_params_free,
            'LOG_LIKELIHOOD': total_log_likelihood,
            'POSTERIOR_MEDIAN_PARAMETER_LOG_LIKELIHOOD': total_log_likelihood,
            'POSTERIOR_MEDIAN_MODEL_CHI2': float(total_chi2),
        }
        with open(os.path.join(run_path, 'metrics.json'), 'w') as handle:
            json.dump(metrics, handle, indent=4, default=json_serializer)
        comparison[f'run_{run_index}']['metrics'] = metrics
        np.savez_compressed(
            os.path.join(run_path, 'modeling_result.npz'),
            **result_arrays,
        )
        print(
            f"Joint reduced chi^2: {metrics['REDUCED_CHI2']:.4f} "
            f"(chi^2={metrics['CHI2']:.2f}, dof={metrics['CHI2_DOF']}); "
            f"BIC: {metrics['BIC']:.2f}, "
            f"parameter-median log-likelihood: {total_log_likelihood:.2f}"
        )
        try:
            plot_multiband_composite(
                combined_band_results,
                run_path,
                residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
            )
        except Exception as error:
            print(f'[plots] multiband_composite.png skipped: {error}')
        if sampler == 'hmc':
            # Lens mass is shared, so save this final mass-only diagnostic once
            # at the joint result root rather than duplicating it per band.
            shared_mass_result = {'kwargs_lens': shared_lens}
            try:
                mass_summary = save_lens_mass_ellipticity_summary(
                    bands[0]['lens_image'], shared_mass_result, run_path,
                )
                plot_mass_and_convergence(
                    bands[0]['lens_image'], shared_mass_result,
                    args.pixel_scale, run_path, mass_summary,
                )
                print('[plots] mass_profile_convergence.png')
            except Exception as error:
                print(f'[plots] joint mass-profile diagnostics skipped: {error}')
        print(f'[multiband] Run {run_index} complete. Outputs in {run_path}')
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not root_hmc_logging:
            run_log_file.write(f'End at {end_timestamp}\n')
            run_log_file.flush()
            run_log_file.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        print(f'[multiband] Run {run_index} logged complete at {end_timestamp}')

    with open(os.path.join(base_save_path, 'comparison.json'), 'w') as handle:
        json.dump(comparison, handle, indent=4, default=json_serializer)
    if comparison:
        comparison_path = os.path.join(base_save_path, 'comparison.json')
        print('\n' + '=' * 40)
        print('All runs completed.')
        print(f'Comparison summary saved to {comparison_path}')
        print('=' * 40)
        for run_name, run_info in comparison.items():
            metrics = run_info.get('metrics', {})
            if not metrics:
                continue
            print(
                f"{run_name} (seed={run_info.get('seed')}): "
                f"log-likelihood={metrics.get('LOG_LIKELIHOOD', float('nan')):.2f}, "
                f"chi2={metrics.get('CHI2', float('nan')):.2f}, "
                f"chi2/N_pix^2={metrics.get('CHI2_NPIX2', float('nan')):.4f}, "
                f"reduced_chi2={metrics.get('REDUCED_CHI2', float('nan')):.4f}, "
                f"BIC={metrics.get('BIC', float('nan')):.2f}"
            )
        print('=' * 40)
    if composite_log_file is not None:
        composite_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        composite_log_file.write(f'End at {composite_end}\n')
        composite_log_file.close()
        sys.stdout = composite_log_stdout
        sys.stderr = composite_log_stderr
    return base_save_path


if __name__ == '__main__':
    build_and_run_multiband()
