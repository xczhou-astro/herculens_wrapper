"""
Main entry point for Herculens wrapper runs.

Point optimization (optax, jaxopt) and posterior sampling (hmc_numpyro,
hmc_blackjax, emcee) are implemented in samplers.py.

Pipeline: run optax/jaxopt first, then set init_params_path to that run
directory and switch sampler to emcee or hmc_* for MCMC warm-started at the MAP.
"""

import datetime
import importlib.util
import json
import os
import shlex
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from herculens_wrapper.utils import (
    MCMC_SAMPLERS,
    OPTIMIZATION_SAMPLERS,
    Tee,
    _configure_cuda_from_args,
    log_jax_device_layout,
    _resolve_single_config_spec,
    center_crop,
    configure_import_paths,
    empty_config,
    fit_dof_and_reduced_chi2,
    get_fits_data,
    json_serializer,
    kwargs_best_to_json_pixelated_npy,
    normalize_run_args_paths,
    resolve_init_run_dir,
    run_arguments_namespace,
)

configure_import_paths()

import numpy as np
from astropy.io import fits


def write_parameter_comparison(save_path, init_params_path, current_kwargs, type_list=None):
    if type_list is None:
        type_list = {}

    inherited_kwargs = None
    if init_params_path:
        try:
            init_run = resolve_init_run_dir(init_params_path)
            init_json_path = os.path.join(init_run, 'kwargs_result.json')
            if os.path.exists(init_json_path):
                with open(init_json_path, 'r') as f:
                    inherited_kwargs = json.load(f)
        except Exception as e:
            print(f"[comparison] Warning: failed to load inherited kwargs from {init_params_path}: {e}")

    category_mapping = {
        'kwargs_lens': ('lens_mass', 'lens_mass_type_list'),
        'kwargs_lens_light': ('lens_light', 'lens_light_type_list'),
        'kwargs_source': ('source_light', 'source_light_type_list'),
        'kwargs_ps': ('point_source', 'point_source_type_list'),
    }

    output_lines = []

    for kw_key, (category_name, type_key) in category_mapping.items():
        curr_list = current_kwargs.get(kw_key, [])
        if not curr_list:
            continue

        model_types = type_list.get(type_key, [])
        category_header_written = False

        for idx, comp in enumerate(curr_list):
            if not isinstance(comp, dict):
                continue

            model_type = model_types[idx] if idx < len(model_types) else "UNKNOWN"
            type_count = model_types.count(model_type)
            if type_count > 1:
                model_name = f"{model_type}_{idx}"
            else:
                model_name = model_type

            model_header_written = False

            for k, val in comp.items():
                if k == 'pixels' or isinstance(val, dict) or isinstance(val, np.ndarray):
                    continue
                if isinstance(val, list):
                    continue
                if not isinstance(val, (int, float, np.number)):
                    continue

                inherited_val = None
                if inherited_kwargs is not None:
                    try:
                        inh_list = inherited_kwargs.get(kw_key, [])
                        if idx < len(inh_list):
                            inherited_val = inh_list[idx].get(k, None)
                    except Exception:
                        pass

                if len(curr_list) > 1:
                    key_str = f"{k}_{idx}"
                else:
                    key_str = k

                val_curr_str = f"{float(val):.3f}"
                val_inh_str = f"{float(inherited_val):.3f}" if inherited_val is not None else "null"

                if not category_header_written:
                    output_lines.append(f"{category_name}:")
                    category_header_written = True

                if not model_header_written:
                    output_lines.append(f"     {model_name}:")
                    model_header_written = True

                output_lines.append(f"            {key_str}: {val_inh_str} -> {val_curr_str}")

    file_path = os.path.join(save_path, 'parameter_changes.txt')
    try:
        with open(file_path, 'w') as f:
            f.write('\n'.join(output_lines) + '\n')
        print(f"[plots] Saved parameter comparison to {file_path}")
    except Exception as e:
        print(f"Failed to write parameter comparison to {file_path}: {e}")


def build_and_run(config_path=None):
    if config_path is None:
        if len(sys.argv) > 1 and not str(sys.argv[1]).startswith('-'):
            config_path = _resolve_single_config_spec(sys.argv[1])
        else:
            config_path = _resolve_single_config_spec('config.py')

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = config_module
    spec.loader.exec_module(config_module)

    args = run_arguments_namespace(config_module, config_path)
    args = normalize_run_args_paths(args)
    _configure_cuda_from_args(args)

    if args.save_path is None:
        args.save_path = os.path.join(
            _PROJECT_ROOT,
            f'workspace{datetime.datetime.now().strftime("%Y%m%d%H%M")}',
        )

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    print(f'Starting run in: {save_path} (sampler={args.sampler!r})')

    import shutil
    shutil.copy(config_path, os.path.join(save_path, os.path.basename(config_path)))

    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, default=json_serializer)

    log_file = open(os.path.join(save_path, 'log.txt'), 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f'Invoked: {shlex.join([sys.executable, *sys.argv])}')

    import jax
    jax.config.update('jax_enable_x64', True)
    log_jax_device_layout(args)

    from herculens_wrapper.models import (
        create_lens_image,
        create_prob_model,
        get_init_params,
        resolve_fixed_kwargs,
        validate_param_list,
    )
    from herculens_wrapper.samplers import (
        run_svi,
        run_optax,
        run_hmc,
        save_metrics,
    )
    from herculens_wrapper.visualizations import (
        display_init,
        generate_run_plots,
        plot_input_data,
    )

    from herculens.RegulModel.regul_model import RegularizationModel

    lens_mass_config = getattr(config_module, 'lens_mass_config', empty_config)
    lens_light_config = getattr(config_module, 'lens_light_config', empty_config)
    source_light_config = getattr(config_module, 'source_light_config', empty_config)
    point_source_config = getattr(config_module, 'point_source_config', empty_config)

    image_data = get_fits_data(args.data_path)
    noise_map = get_fits_data(args.noise_path)
    psf_data = get_fits_data(args.psf_path)
    image_size = image_data.shape[0]

    if args.crop_size is not None:
        image_data = center_crop(image_data, args.crop_size)
        noise_map = center_crop(noise_map, args.crop_size)

    background_subtract_corner = int(getattr(args, 'background_subtract_corner', 0))
    background_subtract_which_corner = str(getattr(args, 'background_subtract_which_corner', 'bottom_left')).lower().strip()
    background_offset = 0.0

    if background_subtract_corner > 0:
        c = background_subtract_corner
        if c > image_data.shape[0] or c > image_data.shape[1]:
            raise ValueError(
                f"background_subtract_corner={c} is larger than image dimensions {image_data.shape}"
            )
        
        if background_subtract_which_corner == 'bottom_left':
            corner_region = image_data[:c, :c]
        elif background_subtract_which_corner == 'bottom_right':
            corner_region = image_data[:c, -c:]
        elif background_subtract_which_corner == 'top_left':
            corner_region = image_data[-c:, :c]
        elif background_subtract_which_corner == 'top_right':
            corner_region = image_data[-c:, -c:]
        else:
            raise ValueError(
                f"Unknown background_subtract_which_corner: {background_subtract_which_corner}. "
                "Must be one of: 'bottom_left', 'bottom_right', 'top_left', 'top_right'"
            )
        
        background_offset = float(np.nanmedian(corner_region))
        image_data = image_data - background_offset
        print(
            f"[bkg] Derived global background offset of {background_offset:.6f} "
            f"from {background_subtract_which_corner} corner ({c}x{c} pixels) and subtracted it."
        )

    args.background_offset = background_offset
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, default=json_serializer)

    source_arc_mask = None
    source_arc_mask_path = getattr(args, 'source_arc_mask_path', None)
    if source_arc_mask_path is not None:
        source_arc_mask = get_fits_data(source_arc_mask_path).astype(bool)
        if args.crop_size is not None:
            source_arc_mask = center_crop(source_arc_mask, args.crop_size)
    source_grid_scale = float(getattr(args, 'source_grid_scale', 1.0))
    conjugate_points = getattr(args, 'conjugate_points', None)
    if conjugate_points is not None:
        conjugate_points = np.asarray(conjugate_points, dtype=np.float64)

    mask_bool = None
    if args.ps_mask_path is not None:
        mask_file = fits.open(args.ps_mask_path)
        all_mask = mask_file[0].data
        if args.relieve_mask_indices is not None:
            for i in np.array(args.relieve_mask_indices, dtype=int):
                mask_comp = mask_file[i].data
                mask_comp = np.where(mask_comp > 0.5, 0.0, 1.0)
                all_mask = all_mask + mask_comp
        mask_bool = all_mask > 0.5
        image_data = image_data * mask_bool
        noise_map = np.where(mask_bool, noise_map, 1e10)

    lens_mass_type_list, lens_mass_params_list = lens_mass_config(
        image_size=image_size, pixel_scale=args.pixel_scale, args=args,
    )
    lens_light_type_list, lens_light_params_list = lens_light_config(
        image_size=image_size, pixel_scale=args.pixel_scale, args=args,
    )
    source_light_type_list, source_light_params_list = source_light_config(
        image_size=image_size, pixel_scale=args.pixel_scale, args=args,
    )

    if not args.exclude_ps:
        point_source_type_list, point_source_params_list = point_source_config(
            image_size=image_size, pixel_scale=args.pixel_scale, args=args,
        )
    else:
        point_source_type_list, point_source_params_list = [], []

    try:
        plot_input_data(
            image_data=image_data,
            noise_map=noise_map,
            psf_data=psf_data,
            pixel_scale=args.pixel_scale,
            save_path=save_path,
            point_source_type_list=point_source_type_list,
            point_source_params_list=point_source_params_list,
            source_arc_mask=source_arc_mask,
            background_subtract_corner=background_subtract_corner,
            background_subtract_which_corner=background_subtract_which_corner,
            background_offset=background_offset,
        )
    except Exception as e:
        print(f'[plots] input_data.png skipped: {e}')

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
    validate_param_list(type_list, param_list)

    print(f'Lens mass type list: {lens_mass_type_list}')
    print(f'Lens light type list: {lens_light_type_list}')
    print(f'Source light type list: {source_light_type_list}')
    print(f'Point source type list: {point_source_type_list}')

    kwargs_numerics_fit = {'supersampling_factor': args.supersampling_factor}
    kwargs_lens_equation_solver_model = {
        'nsolutions': args.ps_nsolutions,
        'niter': args.ps_niter,
        'scale_factor': args.ps_scale_factor,
        'nsubdivisions': args.ps_nsubdivisions,
    }

    lens_image = create_lens_image(
        param_list=param_list,
        type_list=type_list,
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=args.pixel_scale,
        kwargs_numerics=kwargs_numerics_fit,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver_model,
        source_arc_mask=source_arc_mask,
        source_grid_scale=source_grid_scale,
        conjugate_points=conjugate_points,
    )

    fix_components = getattr(args, 'fix_component', [])
    if isinstance(fix_components, str):
        fix_components = [fix_components]
    elif fix_components is None:
        fix_components = []

    fix_lens_light_legacy = bool(getattr(args, 'fix_lens_light', False))
    if fix_lens_light_legacy and 'lens_light' not in fix_components:
        fix_components.append('lens_light')

    if any(c in fix_components for c in ('lens_mass', 'lens_light', 'source_light')):
        if not args.init_params_path:
            raise ValueError(f"Fixing components {fix_components} requires init_params_path.")

    fix_lens_mass = 'lens_mass' in fix_components
    fix_lens_light = 'lens_light' in fix_components
    fix_source_light = 'source_light' in fix_components

    kwargs_lens_fixed = None
    kwargs_lens_light_fixed = None
    kwargs_source_light_fixed = None

    if fix_lens_mass:
        kwargs_lens_fixed = resolve_fixed_kwargs(args.init_params_path, 'lens_mass')
    if fix_lens_light:
        kwargs_lens_light_fixed = resolve_fixed_kwargs(args.init_params_path, 'lens_light')
    if fix_source_light:
        kwargs_source_light_fixed = resolve_fixed_kwargs(args.init_params_path, 'source_light')

    sample_wavelets = bool(getattr(args, 'sample_wavelets', False))
    regul_model = None
    n_runs = int(getattr(args, 'n_runs', 1))
    sampler = args.sampler

    if sampler != 'svi':
        n_runs = 1

    def run_one_iteration(n, run_save_path, run_seed):
        from types import SimpleNamespace
        import shutil

        os.makedirs(run_save_path, exist_ok=True)
        run_args = SimpleNamespace(**vars(args))
        run_args.save_path = run_save_path
        run_args.random_seed = run_seed



        print(f"\n========================================")
        print(f"Starting Run {n} (seed={run_seed})")
        print(f"========================================")

        try:
            run_prob_model = create_prob_model(
                param_list, type_list, lens_image, image_data, noise_map,
                regul_model=None,
                fix_lens_light=fix_lens_light,
                kwargs_lens_light_fixed=kwargs_lens_light_fixed,
                fix_lens_mass=fix_lens_mass,
                kwargs_lens_fixed=kwargs_lens_fixed,
                fix_source_light=fix_source_light,
                kwargs_source_light_fixed=kwargs_source_light_fixed,
                init_params_path=run_args.init_params_path,
                args=run_args,
            )

            shutil.copy(config_path, os.path.join(run_save_path, os.path.basename(config_path)))
            with open(os.path.join(run_save_path, 'args.json'), 'w') as f:
                json.dump(vars(run_args), f, indent=4, default=json_serializer)

            num_params = run_prob_model.count_sampled_parameters()

            with open(os.path.join(run_save_path, 'config.json'), 'w') as f:
                json.dump({
                    'type_list': type_list,
                    'param_list': param_list,
                    'num_params': num_params,
                    'sampler': run_args.sampler,
                    'init_params_path': run_args.init_params_path,
                    'kwargs_numerics_fit': kwargs_numerics_fit,
                    'kwargs_lens_equation_solver_model': kwargs_lens_equation_solver_model,
                }, f, indent=4, default=json_serializer)

            if run_args.init_params_path:
                init_run = resolve_init_run_dir(run_args.init_params_path)
                print(
                    f'[init] Warm-starting from prior run: {init_run} '
                    f'(sampler={run_args.sampler!r})'
                )

            init_params = get_init_params(
                run_prob_model, param_list, type_list,
                init_params_path=run_args.init_params_path,
                random_seed=run_seed,
                fix_lens_light=fix_lens_light,
                fix_lens_mass=fix_lens_mass,
                fix_source_light=fix_source_light,
                lens_image=lens_image,
                pixel_init_jitter=getattr(run_args, 'pixel_init_jitter', 0.0),
                sample_wavelets=sample_wavelets,
                regul_model=None,
            )
            print(f'Number of sampled parameters: {num_params}')
            init_log_prob = float(run_prob_model.log_prob(init_params, constrained=True))
            init_log_likelihood = float(run_prob_model.log_likelihood(init_params))
            print(
                f'Initial log-prob: {init_log_prob:.2f} '
                f'(log-likelihood: {init_log_likelihood:.2f})'
            )

            try:
                display_init(
                    prob_model=run_prob_model,
                    init_params=init_params,
                    lens_image=lens_image,
                    image_data=image_data,
                    noise_map=noise_map,
                    pixel_scale=run_args.pixel_scale,
                    save_path=run_save_path,
                    num_params=num_params,
                    type_list=type_list,
                )
            except Exception as e:
                print(f'[plots] initial_guess_model.png skipped: {e}')

            mcmc_samples = None
            flat_samples = None

            if run_args.sampler == 'svi':
                best_params, extra = run_svi(run_prob_model, image_data, run_args, init_params)
                if 'loss_history' in extra:
                    history = {'loss_history': np.asarray(extra['loss_history']).tolist()}
                    with open(os.path.join(run_save_path, 'svi_loss_history.json'), 'w') as f:
                        json.dump(history, f, indent=4)
                if 'guide' in extra and 'result' in extra:
                    try:
                        import jax
                        rng_key = jax.random.PRNGKey(run_args.random_seed + 12345)
                        cpu_device = jax.devices('cpu')[0]
                        params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), extra['result'].params)
                        with jax.default_device(cpu_device):
                            guide_samples = extra['guide'].sample_posterior(
                                rng_key, params_cpu, sample_shape=(2000,)
                            )
                        sigma_params = {k: np.std(np.asarray(v), axis=0) for k, v in guide_samples.items()}
                        kwargs_sigma = run_prob_model.params2kwargs(sigma_params)
                        kwargs_sigma_json = kwargs_best_to_json_pixelated_npy(kwargs_sigma, run_save_path, type_list)
                        with open(os.path.join(run_save_path, 'kwargs_sigma.json'), 'w') as f:
                            json.dump(kwargs_sigma_json, f, indent=4, default=json_serializer)
                        print(f"[svi] Saved parameter uncertainties to kwargs_sigma.json")
                    except Exception as e:
                        print(f"[svi] Failed to compute/save kwargs_sigma.json: {e}")
            elif run_args.sampler == 'optax':
                best_params, extra = run_optax(run_prob_model, run_args, init_params)
                if 'loss_history' in extra:
                    history = {'loss_history': np.asarray(extra['loss_history']).tolist()}
                    with open(os.path.join(run_save_path, 'svi_loss_history.json'), 'w') as f:
                        json.dump(history, f, indent=4)
            elif run_args.sampler in MCMC_SAMPLERS:
                mcmc_samples, best_params, extra = run_hmc(run_prob_model, run_args, init_params)
                flat_samples = extra.get('flat_samples', None)

                # Save MCMC samples
                samples_npz_path = os.path.join(run_save_path, f'{run_args.sampler}_samples.npz')
                npz_dict = {k: np.asarray(v) for k, v in mcmc_samples.items()}
                np.savez_compressed(samples_npz_path, **npz_dict)
                print(f"Saved MCMC samples to {samples_npz_path}")

                # Save extra info (excluding flat_samples)
                extra_json_path = os.path.join(run_save_path, f'{run_args.sampler}_extra.json')
                extra_dict = {k: v for k, v in extra.items() if k != 'flat_samples'}
                with open(extra_json_path, 'w') as f:
                    json.dump(extra_dict, f, indent=4, default=json_serializer)
            else:
                raise ValueError(f"Unknown sampler: {run_args.sampler}")

            kwargs_best = run_prob_model.params2kwargs(best_params)
            kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_best, run_save_path, type_list)
            with open(os.path.join(run_save_path, 'kwargs_result.json'), 'w') as f:
                json.dump(kwargs_json, f, indent=4, default=json_serializer)
            write_parameter_comparison(run_save_path, run_args.init_params_path, kwargs_json, type_list)

            num_params_free = num_params
            if type_list.get('source_light_type_list') == ['PIXELATED']:
                prior_type = getattr(run_prob_model, 'prior_type', 'matern')
                ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
                if prior_type == 'wavelet_sparsity':
                    nscales = getattr(run_prob_model, 'nscales', 1)
                    num_params_free = num_params - (nscales * ny * nx)
                else:
                    num_params_free = num_params - (ny * nx)

            best_fit_model = lens_image.model(**kwargs_best)
            chi2 = float(np.sum(((best_fit_model - image_data) / noise_map) ** 2))
            log_likelihood = float(run_prob_model.log_likelihood(best_params))
            metrics = save_metrics(
                run_save_path, chi2, image_data, num_params, log_likelihood, fit_dof_and_reduced_chi2,
                num_params_free=num_params_free,
                mask_bool=mask_bool,
            )
            reduced_chi2 = metrics['REDUCED_CHI2']

            generate_run_plots(
                lens_image=lens_image,
                kwargs_best=kwargs_best,
                image_data=image_data,
                noise_map=noise_map,
                psf_data=psf_data,
                pixel_scale=run_args.pixel_scale,
                save_path=run_save_path,
                sampler=run_args.sampler,
                best_fit_model=best_fit_model,
                chi2=chi2,
                reduced_chi2=reduced_chi2,
                extra=extra,
                mcmc_samples=mcmc_samples,
                flat_samples=flat_samples,
                prob_model=run_prob_model,
                init_params=init_params,
                point_source_type_list=point_source_type_list,
                point_source_params_list=point_source_params_list,
                regul_model=getattr(run_prob_model, 'regul_model', None),
                param_list=param_list,
                residual_vis_max=getattr(run_args, 'residual_vis_max', 0.0),
            )

            np.savez_compressed(
                os.path.join(run_save_path, 'modeling_result.npz'),
                best_fit_model=np.asarray(best_fit_model),
                image_data=np.asarray(image_data),
                noise_map=np.asarray(noise_map),
                source_arc_mask=np.asarray(source_arc_mask) if source_arc_mask is not None else None,
            )
            print(f'Run {n} complete. Outputs in {run_save_path}')
            return metrics

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Run {n} failed: {e}")
            return None


    if n_runs > 1:


        print(f'Invoked: {shlex.join([sys.executable, *sys.argv])}')
        print(f'Starting SVI multi-run in: {save_path} (n_runs={n_runs})')

        comparison_results = {}
        for n in range(n_runs):
            run_save_path = os.path.join(save_path, f'run_{n}')
            run_seed = args.random_seed + n
            metrics_data = run_one_iteration(n, run_save_path, run_seed)
            if metrics_data is not None:
                comparison_results[f"run_{n}"] = {
                    "seed": run_seed,
                    "metrics": metrics_data
                }

        comparison_file_path = os.path.join(save_path, 'comparison.json')
        with open(comparison_file_path, 'w') as f:
            json.dump(comparison_results, f, indent=4)

        print("\n========================================")
        print("All runs completed.")
        print(f"Comparison summary saved to {comparison_file_path}")
        print("========================================")
        for run_name, run_info in comparison_results.items():
            m = run_info["metrics"]
            print(f"{run_name} (seed={run_info['seed']}): log-likelihood={m['LOG_LIKELIHOOD']:.2f}, chi2={m['CHI2']:.2f}, chi2/N_pix^2={m['CHI2_NPIX2']:.4f}, reduced_chi2={m['REDUCED_CHI2']:.4f}, BIC={m['BIC']:.2f}")
        print("========================================\n")


        return save_path

    else:
        # Original backward-compatible single run logic
        prob_model = create_prob_model(
            param_list, type_list, lens_image, image_data, noise_map,
            regul_model=None,
            fix_lens_light=fix_lens_light,
            kwargs_lens_light_fixed=kwargs_lens_light_fixed,
            fix_lens_mass=fix_lens_mass,
            kwargs_lens_fixed=kwargs_lens_fixed,
            fix_source_light=fix_source_light,
            kwargs_source_light_fixed=kwargs_source_light_fixed,
            init_params_path=args.init_params_path,
            args=args,
        )

        num_params = prob_model.count_sampled_parameters()

        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump({
                'type_list': type_list,
                'param_list': param_list,
                'num_params': num_params,
                'sampler': args.sampler,
                'init_params_path': args.init_params_path,
                'kwargs_numerics_fit': kwargs_numerics_fit,
                'kwargs_lens_equation_solver_model': kwargs_lens_equation_solver_model,
            }, f, indent=4, default=json_serializer)

        if args.init_params_path:
            init_run = resolve_init_run_dir(args.init_params_path)
            print(
                f'[init] Warm-starting from prior run: {init_run} '
                f'(sampler={args.sampler!r})'
            )

        init_params = get_init_params(
            prob_model, param_list, type_list,
            init_params_path=args.init_params_path,
            random_seed=args.random_seed,
            fix_lens_light=fix_lens_light,
            fix_lens_mass=fix_lens_mass,
            fix_source_light=fix_source_light,
            lens_image=lens_image,
            pixel_init_jitter=getattr(args, 'pixel_init_jitter', 0.0),
            sample_wavelets=sample_wavelets,
            regul_model=regul_model,
        )
        print(f'Number of sampled parameters: {num_params}')
        init_log_prob = float(prob_model.log_prob(init_params, constrained=True))
        init_log_likelihood = float(prob_model.log_likelihood(init_params))
        print(
            f'Initial log-prob: {init_log_prob:.2f} '
            f'(log-likelihood: {init_log_likelihood:.2f})'
        )

        try:
            display_init(
                prob_model=prob_model,
                init_params=init_params,
                lens_image=lens_image,
                image_data=image_data,
                noise_map=noise_map,
                pixel_scale=args.pixel_scale,
                save_path=save_path,
                num_params=num_params,
                type_list=type_list,
            )
        except Exception as e:
            print(f'[plots] initial_guess_model.png skipped: {e}')

        extra = None
        mcmc_samples = None
        flat_samples = None
        reduced_chi2 = None

        if sampler in OPTIMIZATION_SAMPLERS:
            if sampler == 'svi':
                best_params, extra = run_svi(prob_model, image_data, args, init_params)
                if 'guide' in extra and 'result' in extra:
                    try:
                        import jax
                        rng_key = jax.random.PRNGKey(args.random_seed + 12345)
                        cpu_device = jax.devices('cpu')[0]
                        params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), extra['result'].params)
                        with jax.default_device(cpu_device):
                            guide_samples = extra['guide'].sample_posterior(
                                rng_key, params_cpu, sample_shape=(2000,)
                            )
                        sigma_params = {k: np.std(np.asarray(v), axis=0) for k, v in guide_samples.items()}
                        kwargs_sigma = prob_model.params2kwargs(sigma_params)
                        kwargs_sigma_json = kwargs_best_to_json_pixelated_npy(kwargs_sigma, save_path, type_list)
                        with open(os.path.join(save_path, 'kwargs_sigma.json'), 'w') as f:
                            json.dump(kwargs_sigma_json, f, indent=4, default=json_serializer)
                        print(f"[svi] Saved parameter uncertainties to kwargs_sigma.json")
                    except Exception as e:
                        print(f"[svi] Failed to compute/save kwargs_sigma.json: {e}")
            elif sampler == 'optax':
                best_params, extra = run_optax(prob_model, args, init_params)

            kwargs_best = prob_model.params2kwargs(best_params)
            kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_best, save_path, type_list)
            with open(os.path.join(save_path, 'kwargs_result.json'), 'w') as f:
                json.dump(kwargs_json, f, indent=4, default=json_serializer)
            write_parameter_comparison(save_path, args.init_params_path, kwargs_json, type_list)

            if 'loss_history' in extra:
                history = {'loss_history': np.asarray(extra['loss_history']).tolist()}
                with open(os.path.join(save_path, 'svi_loss_history.json'), 'w') as f:
                    json.dump(history, f, indent=4)

            # Calculate effective/free parameters for regularized pixelated models
            num_params_free = num_params
            if type_list.get('source_light_type_list') == ['PIXELATED']:
                prior_type = getattr(prob_model, 'prior_type', 'matern')
                ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
                if prior_type == 'wavelet_sparsity':
                    nscales = getattr(prob_model, 'nscales', 1)
                    num_params_free = num_params - (nscales * ny * nx)
                else:
                    num_params_free = num_params - (ny * nx)

            best_fit_model = lens_image.model(**kwargs_best)
            chi2 = float(np.sum(((best_fit_model - image_data) / noise_map) ** 2))
            log_likelihood = float(prob_model.log_likelihood(best_params))
            metrics = save_metrics(
                save_path, chi2, image_data, num_params, log_likelihood, fit_dof_and_reduced_chi2,
                num_params_free=num_params_free,
                mask_bool=mask_bool,
            )
            reduced_chi2 = metrics['REDUCED_CHI2']

        elif sampler in MCMC_SAMPLERS:
            mcmc_samples, best_params, extra = run_hmc(prob_model, args, init_params)
            flat_samples = extra.get('flat_samples', None)

            kwargs_best = prob_model.params2kwargs(best_params)
            kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_best, save_path, type_list)
            with open(os.path.join(save_path, 'kwargs_result.json'), 'w') as f:
                json.dump(kwargs_json, f, indent=4, default=json_serializer)
            write_parameter_comparison(save_path, args.init_params_path, kwargs_json, type_list)

            # Save MCMC samples
            samples_npz_path = os.path.join(save_path, f'{sampler}_samples.npz')
            npz_dict = {k: np.asarray(v) for k, v in mcmc_samples.items()}
            np.savez_compressed(samples_npz_path, **npz_dict)
            print(f"Saved MCMC samples to {samples_npz_path}")

            # Save extra info (excluding flat_samples)
            extra_json_path = os.path.join(save_path, f'{sampler}_extra.json')
            extra_dict = {k: v for k, v in extra.items() if k != 'flat_samples'}
            with open(extra_json_path, 'w') as f:
                json.dump(extra_dict, f, indent=4, default=json_serializer)

            best_fit_model = lens_image.model(**kwargs_best)
            chi2 = float(np.sum(((best_fit_model - image_data) / noise_map) ** 2))
            log_likelihood = float(prob_model.log_likelihood(best_params))
            metrics = save_metrics(
                save_path, chi2, image_data, num_params, log_likelihood, fit_dof_and_reduced_chi2,
                num_params_free=num_params,
                mask_bool=mask_bool,
            )
            reduced_chi2 = metrics['REDUCED_CHI2']

        else:
            raise ValueError(f'Unsupported sampler {sampler!r}')

        generate_run_plots(
            lens_image=lens_image,
            kwargs_best=kwargs_best,
            image_data=image_data,
            noise_map=noise_map,
            psf_data=psf_data,
            pixel_scale=args.pixel_scale,
            save_path=save_path,
            sampler=sampler,
            best_fit_model=best_fit_model,
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            extra=extra,
            mcmc_samples=mcmc_samples,
            flat_samples=flat_samples,
            prob_model=prob_model,
            init_params=init_params,
            point_source_type_list=point_source_type_list,
            point_source_params_list=point_source_params_list,
            regul_model=regul_model,
            param_list=param_list,
            residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
        )

        np.savez_compressed(
            os.path.join(save_path, 'modeling_result.npz'),
            best_fit_model=np.asarray(best_fit_model),
            image_data=np.asarray(image_data),
            noise_map=np.asarray(noise_map),
            source_arc_mask=np.asarray(source_arc_mask) if source_arc_mask is not None else None,
        )
        print(f'Run complete. Outputs in {save_path}')
        return save_path


if __name__ == '__main__':
    start_time = time.time()
    build_and_run()
    end_time = time.time()
    print(f'Time taken: {end_time - start_time} seconds')