"""Herculens inference backends: point optimization and posterior sampling."""

from numpyro.distributions import biject_to
import json
import os
import pickle

import numpy as np
import optax
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.infer as infer
import numpyro.infer.autoguide as autoguide
from functools import partial


def to_unconstrained(prob_model, params):
    """
    Constrained -> unconstrained NumPyro site dict.

    Use for optax/jaxopt/HMC/emcee inputs. ``get_init_params()`` and
    ``params2kwargs()`` work in constrained (physical) space.
    """
    return prob_model.unconstrain(params)


def to_constrained(prob_model, params):
    """
    Unconstrained -> constrained NumPyro site dict.

    Use on optimizer / MCMC outputs before ``params2kwargs()`` or
    ``log_likelihood(..., constrained=True)``.
    """
    return prob_model.constrain(params)


def init_params_unconstrained(prob_model, init_params):
    """Alias for :func:`to_unconstrained` (backward compatibility)."""
    return to_unconstrained(prob_model, init_params)


def tree_median(tree):
    import jax
    return jax.tree_util.tree_map(lambda x: np.median(x, axis=0), tree)


def get_active_sample_sites(prob_model, rng_seed=0):
    """Return latent sample-site names in the NumPyro model."""
    from numpyro.handlers import trace, seed

    with seed(rng_seed=rng_seed):
        model_trace = trace(prob_model.model).get_trace()
    return [
        name for name, site in model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]


def evaluate_model_deterministics(prob_model, params, rng_seed=0, active_sites=None):
    """Evaluate NumPyro deterministic sites after conditioning on constrained params."""
    from numpyro.handlers import substitute, trace, seed

    if active_sites is None:
        active_sites = get_active_sample_sites(prob_model, rng_seed=rng_seed)
    active_sites = set(active_sites)
    conditioned_params = {k: v for k, v in params.items() if k in active_sites}
    missing = sorted(k for k in active_sites if k not in conditioned_params)
    if missing:
        raise KeyError(
            "Cannot evaluate deterministic model outputs; missing conditioned "
            f"sample sites: {missing}"
        )

    with seed(rng_seed=rng_seed):
        model_trace = trace(
            substitute(prob_model.model, data=params)
        ).get_trace()
    return {
        name: site["value"]
        for name, site in model_trace.items()
        if site["type"] == "deterministic"
    }


def median_deterministics_from_samples(samples, active_sites=None):
    """Median deterministic arrays stored in HMC samples."""
    active_sites = set(active_sites or [])
    deterministics = {}
    for key, value in samples.items():
        if key in active_sites:
            continue
        deterministics[key] = np.median(np.asarray(value), axis=0)
    return deterministics


def kwargs_with_deterministics(prob_model, params, deterministics=None, rng_seed=0, active_sites=None):
    """
    Convert constrained parameters to kwargs, replacing model-derived outputs
    with NumPyro deterministic values where available.
    """
    kwargs = prob_model.params2kwargs(params)
    if deterministics is None:
        deterministics = {}

    kwargs_source = kwargs.get('kwargs_source', None)
    needs_pixels = (
        kwargs_source is not None
        and len(kwargs_source) > 0
        and isinstance(kwargs_source[0], dict)
        and 'pixels' in kwargs_source[0]
        and 'pixels_source_grid' not in deterministics
    )
    if 'model_image' not in deterministics or needs_pixels:
        computed_deterministics = evaluate_model_deterministics(
            prob_model,
            params,
            rng_seed=rng_seed,
            active_sites=active_sites,
        )
        computed_deterministics.update(deterministics)
        deterministics = computed_deterministics

    if (
        kwargs_source is not None
        and len(kwargs_source) > 0
        and isinstance(kwargs_source[0], dict)
        and 'pixels_source_grid' in deterministics
    ):
        kwargs_source[0]['pixels'] = deterministics['pixels_source_grid']

    return kwargs, deterministics


def model_image_from_deterministics(prob_model, kwargs, deterministics=None):
    """Return deterministic model_image, falling back to lens_image.model()."""
    if deterministics is not None and 'model_image' in deterministics:
        return np.asarray(deterministics['model_image'])
    lens_image = getattr(prob_model, 'lens_image', None)
    if lens_image is None:
        raise ValueError("prob_model does not expose lens_image for model image fallback.")
    return lens_image.model(**kwargs)


def save_hmc_diagnostics(samples, num_chains, target_dir, suffix, prob_model=None):
    try:
        import arviz as az
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        # Focus on lens mass and power spectrum related parameters
        target_keys = [
            k for k in samples.keys() 
            if (('lens_' in k and 'lens_light_' not in k) or k in ('n_source_grid', 'rho_source_grid', 'sigma_source_grid'))
        ]

        if not target_keys:
            return

        # Sort the target keys following the order defined in the mass configuration if available
        ordered_keys = []
        if prob_model is not None and hasattr(prob_model, 'param_list'):
            lens_mass_params_list = prob_model.param_list.get('lens_mass_params_list', [])
            for i, mass_profile in enumerate(lens_mass_params_list):
                for param_name in mass_profile.keys():
                    expected_key = f"lens_{param_name}_{i}"
                    if expected_key in target_keys and expected_key not in ordered_keys:
                        ordered_keys.append(expected_key)

        # Append any remaining target_keys (like power spectrum related parameters)
        for k in target_keys:
            if k not in ordered_keys:
                ordered_keys.append(k)

        # Format the data for arviz: dict of shape (num_chains, samples_per_chain)
        arviz_data = {}
        for k in ordered_keys:
            val = np.asarray(samples[k])
            total_samples = val.shape[0]
            samples_per_chain = total_samples // num_chains
            if samples_per_chain > 0:
                arviz_data[k] = val.reshape((num_chains, samples_per_chain) + val.shape[1:])

        if not arviz_data:
            return

        # Convert dictionary to InferenceData first to support new ArviZ 1.2+ API
        idata = az.from_dict({'posterior': arviz_data})

        # 1. Generate convergence summary using arviz
        try:
            summary_df = az.summary(idata)
            summary_path = os.path.join(target_dir, f"mcmc_summary_{suffix}.txt")
            with open(summary_path, 'w') as f:
                f.write(summary_df.to_string())
            print(f"[hmc] Saved arviz summary to {summary_path}")
        except Exception as es:
            print(f"[warning] Failed to compute arviz summary: {es}")

        # 2. Generate trace and density plots using arviz
        try:
            axes = az.plot_trace_dist(idata, var_names=ordered_keys)
            fig = plt.gcf()
            fig.tight_layout()
            plot_path = os.path.join(target_dir, f"mcmc_diagnostics_{suffix}.png")
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close('all')
            print(f"[hmc] Saved arviz diagnostics plots to {plot_path}")
        except Exception as ep:
            print(f"[warning] Failed to plot arviz trace: {ep}")

    except Exception as e:
        print(f"[warning] Failed to generate arviz diagnostics: {e}")


def _sample_at_index(samples, idx, include_keys=None, exclude=('model_image',)):
    if include_keys is not None:
        include_keys = set(include_keys)
    return {
        k: np.asarray(v)[idx]
        for k, v in samples.items()
        if include_keys is None or k in include_keys
        if k not in exclude
    }


def _save_hmc_max_loglike_outputs(
    samples,
    prob_model,
    save_path,
    args,
    active_sites=None,
    kwargs_filename='kwargs_loglike.json',
    pixels_filename='kwargs_loglike_source_pixels.npy',
    pixels_wn_filename='kwargs_loglike_source_pixels_wn.npy',
    save_pixel_arrays=True,
    linear_plot_filename='best_fit_model_loglike_linear.png',
    log_plot_filename='best_fit_model_loglike_log.png',
    log_likelihoods_filename='hmc_log_likelihoods.npy',
):
    try:
        from herculens_wrapper.utils import kwargs_best_to_json_pixelated_npy, json_serializer
        from herculens_wrapper.visualizations import display

        n_total_samples = len(samples[list(samples.keys())[0]])
        log_likelihoods = []
        for idx in range(n_total_samples):
            sample_params = _sample_at_index(samples, idx, include_keys=active_sites)
            log_likelihoods.append(float(prob_model.log_likelihood(sample_params)))

        log_likelihoods = np.asarray(log_likelihoods)
        if not np.any(np.isfinite(log_likelihoods)):
            print("[warning] Could not identify max-log-likelihood HMC sample: all values are non-finite.")
            return {}
        if log_likelihoods_filename is not None:
            np.save(os.path.join(save_path, log_likelihoods_filename), log_likelihoods)

        best_idx = int(np.nanargmax(log_likelihoods))
        best_loglike = float(log_likelihoods[best_idx])
        best_params_loglike = _sample_at_index(samples, best_idx, include_keys=active_sites)
        sample_deterministics = {
            k: np.asarray(v)[best_idx]
            for k, v in samples.items()
            if active_sites is None or k not in set(active_sites)
        }
        kwargs_loglike, sample_deterministics = kwargs_with_deterministics(
            prob_model,
            best_params_loglike,
            deterministics=sample_deterministics,
            rng_seed=getattr(args, 'random_seed', 0),
            active_sites=active_sites,
        )
        type_list = getattr(prob_model, 'type_list', {})
        kwargs_loglike_json = kwargs_best_to_json_pixelated_npy(
            kwargs_loglike,
            save_path,
            type_list,
            pixels_filename=pixels_filename,
            pixels_wn_filename=pixels_wn_filename,
            save_pixel_arrays=save_pixel_arrays,
        )
        kwargs_loglike_json['_hmc_log_likelihood'] = best_loglike
        kwargs_loglike_json['_hmc_sample_index'] = best_idx
        with open(os.path.join(save_path, kwargs_filename), 'w') as f:
            json.dump(kwargs_loglike_json, f, indent=4, default=json_serializer)
        print(f"[hmc] Saved max-log-likelihood kwargs to {os.path.join(save_path, kwargs_filename)}")

        img_data = getattr(prob_model, 'image_data', None)
        ns_map = getattr(prob_model, 'noise_map', None)
        l_image = getattr(prob_model, 'lens_image', None)
        if img_data is not None and ns_map is not None and l_image is not None:
            best_fit_model = model_image_from_deterministics(
                prob_model,
                kwargs_loglike,
                sample_deterministics,
            )
            p_scale = getattr(prob_model, 'pixel_scale', 0.08)
            chi2 = float(np.sum(((best_fit_model - img_data) / ns_map) ** 2))
            mask = getattr(l_image, 'source_arc_mask', None)
            if mask is not None:
                mask = np.asarray(mask)
            residual_vis_max = getattr(args, 'residual_vis_max', 0.0)

            display(
                [best_fit_model, img_data, (best_fit_model - img_data) / ns_map],
                titles=[
                    'Max loglike model',
                    'Image data',
                    f'Residuals (chi^2 = {chi2:.2f})',
                ],
                pixel_scale=p_scale,
                savefilename=os.path.join(save_path, linear_plot_filename),
                plot_scale='linear',
                contour_mask=mask,
                residual_vis_max=residual_vis_max,
            )
            display(
                [best_fit_model, img_data, (best_fit_model - img_data) / ns_map],
                titles=[
                    'Max loglike model',
                    'Image data',
                    f'Residuals (chi^2 = {chi2:.2f})',
                ],
                pixel_scale=p_scale,
                savefilename=os.path.join(save_path, log_plot_filename),
                plot_scale='log',
                contour_mask=mask,
                residual_vis_max=residual_vis_max,
            )
            print("[hmc] Saved max-log-likelihood model plots.")

        return {
            'log_likelihoods': log_likelihoods,
            'max_log_likelihood': best_loglike,
            'max_log_likelihood_index': best_idx,
        }
    except Exception as e:
        print(f"[warning] Failed to save max-log-likelihood HMC outputs: {e}")
        return {}


def _save_hmc_pixels_wn_summary(
    samples,
    save_path,
    plot_filename='source_pixels_wn_median_uncertainties.png',
    median_filename='source_pixels_wn_median.npy',
    lower_filename='source_pixels_wn_sigma_lower.npy',
    upper_filename='source_pixels_wn_sigma_upper.npy',
):
    key = 'pixels_wn_source_grid'
    if key not in samples:
        return
    try:
        import matplotlib.pyplot as plt

        arr = np.asarray(samples[key])
        if arr.ndim < 2:
            return
        median = np.median(arr, axis=0)
        p16 = np.percentile(arr, 16, axis=0)
        p84 = np.percentile(arr, 84, axis=0)
        lower = median - p16
        upper = p84 - median

        if median_filename is not None:
            np.save(os.path.join(save_path, median_filename), median)
        if lower_filename is not None:
            np.save(os.path.join(save_path, lower_filename), lower)
        if upper_filename is not None:
            np.save(os.path.join(save_path, upper_filename), upper)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        panels = [
            (median, 'Median pixels_wn'),
            (lower, 'Lower 1 sigma'),
            (upper, 'Upper 1 sigma'),
        ]
        for ax, (panel, title) in zip(axes, panels):
            im = ax.imshow(panel, origin='lower', cmap='twilight')
            ax.set_title(title)
            ax.set_xlabel('Fourier x')
            ax.set_ylabel('Fourier y')
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        out_path = os.path.join(save_path, plot_filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[hmc] Saved median pixels_wn uncertainty plot to {out_path}")
    except Exception as e:
        print(f"[warning] Failed to save median pixels_wn uncertainty plot: {e}")


def _select_hmc_jitter_keys(init_params_unconst, args, vars_mass):
    mode = str(getattr(args, 'hmc_init_jitter_sites', 'lens_mass')).strip().lower()
    keys = list(init_params_unconst.keys())
    if mode in ('none', 'false', 'off'):
        return []
    if mode == 'lens_mass':
        return [k for k in vars_mass if k in init_params_unconst]
    if mode == 'all_non_pixel':
        return [k for k in keys if 'pixels_wn_' not in k]
    if mode == 'all':
        return keys
    requested = [k.strip() for k in mode.split(',') if k.strip()]
    return [k for k in requested if k in init_params_unconst]


def _build_hmc_chain_init_params(prob_model, init_params_unconst, args, num_chains, vars_mass):
    if num_chains <= 1 or init_params_unconst is None:
        return init_params_unconst

    jitter_scale = float(getattr(args, 'hmc_init_jitter_scale', 0.0))
    jitter_keys = _select_hmc_jitter_keys(init_params_unconst, args, vars_mass)
    if jitter_scale <= 0.0 or not jitter_keys:
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + jnp.shape(x)),
            init_params_unconst,
        )

    max_tries = int(getattr(args, 'hmc_init_jitter_max_tries', 200))
    rng_key = jax.random.PRNGKey(int(getattr(args, 'random_seed', 0)) + 7919)
    chain_params = []
    print(
        f"[hmc] Jittering initial parameters for {num_chains} chains "
        f"(scale={jitter_scale:g}, sites={jitter_keys})"
    )

    for chain_idx in range(num_chains):
        accepted = None
        accepted_log_prob = None
        for attempt in range(max_tries):
            proposal = dict(init_params_unconst)
            for key in jitter_keys:
                rng_key, noise_key = jax.random.split(rng_key)
                base = jnp.asarray(init_params_unconst[key])
                proposal[key] = base + jitter_scale * jax.random.normal(
                    noise_key,
                    shape=base.shape,
                    dtype=base.dtype,
                )
            try:
                constrained = to_constrained(prob_model, proposal)
                log_prob = float(prob_model.log_prob(constrained, constrained=True))
                if np.isfinite(log_prob):
                    accepted = proposal
                    accepted_log_prob = log_prob
                    break
            except Exception:
                pass
        if accepted is None:
            raise ValueError(
                f"Failed to generate a finite-log-prob HMC initial jitter "
                f"for chain {chain_idx} after {max_tries} attempts."
            )
        print(
            f"[hmc] Accepted jittered init for chain {chain_idx} "
            f"(log_prob={accepted_log_prob:.2f})"
        )
        chain_params.append(accepted)

    return {
        key: jnp.stack([jnp.asarray(chain[key]) for chain in chain_params], axis=0)
        for key in init_params_unconst.keys()
    }


def save_metrics(save_path, chi2, image_data, num_params, log_likelihood, fit_dof_and_reduced_chi2, num_params_free=None, mask_bool=None, source_pixel_scale=None):
    if num_params_free is None:
        num_params_free = num_params
    reduced_chi2, n_pix, n_fit_free, dof = fit_dof_and_reduced_chi2(chi2, image_data, num_params_free, mask_bool=mask_bool)
    bic = num_params_free * np.log(n_pix) - 2 * log_likelihood
    metrics = {
        'BIC': float(bic),
        'CHI2': float(chi2),
        'CHI2_NPIX2': float(chi2 / n_pix),
        'REDUCED_CHI2': float(reduced_chi2),
        'CHI2_DOF': int(dof),
        'N_DATA_PIXELS': int(n_pix),
        'N_PARAMS_FITTED': int(num_params),
        'N_PARAMS_FREE': int(num_params_free),
        'LOG_LIKELIHOOD': float(log_likelihood),
    }
    if source_pixel_scale is not None:
        metrics['SOURCE_PIXEL_SCALE'] = float(source_pixel_scale)
        
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(
        f'Reduced chi^2: {reduced_chi2:.4f} (chi^2={chi2:.2f}, dof={dof}, p={num_params_free}), chi^2/N_pix^2={chi2 / n_pix:.4f}'
    )
    print(f'BIC: {bic:.2f}, log-likelihood: {log_likelihood:.2f}')
    return metrics


def run_svi(
    prob_model,
    image_data,
    args,
    init_params,
    init_values=None,
    max_iterations=None,
    learning_rate=None,
    init_scale=None,
    loss_kind=None,
):
    if max_iterations is None:
        max_iterations = int(getattr(args, 'max_iterations_svi', 10000))
    if learning_rate is None:
        learning_rate = float(getattr(args, 'init_learning_rate_svi', 0.01))
    if init_scale is None:
        init_scale = float(getattr(args, 'init_scale_svi', 0.1))
    if loss_kind is None:
        loss_kind = getattr(args, 'loss_kind_svi', 'trace_elbo')

    def init_to_value_or_defer(site, values=None, defer=infer.init_to_median(num_samples=25)):
        if values is None:
            values = {}
        if site["type"] == "sample" and not site["is_observed"]:
            if site["name"] in values:
                return values[site["name"]]
            return defer(site)

    merged_init = {}
    if init_params:
        merged_init.update(init_params)
    if init_values:
        merged_init.update(init_values)

    init_fun = partial(init_to_value_or_defer, values=merged_init) if merged_init else infer.init_to_median(num_samples=25)

    guide = autoguide.AutoLowRankMultivariateNormal(
        prob_model.model,
        init_loc_fn=init_fun,
        init_scale=init_scale
    )

    boundary = int(max_iterations * 0.5)
    scheduler1 = optax.exponential_decay(
        init_value=learning_rate,
        decay_rate=0.99,
        transition_steps=200,
    )
    scheduler2 = optax.exponential_decay(
        init_value=scheduler1(boundary),
        decay_rate=0.99,
        transition_steps=10,
    )
    scheduler = optax.join_schedules([scheduler1, scheduler2], boundaries=[boundary])
    optim = optax.adabelief(learning_rate=scheduler)

    if loss_kind == 'trace_meanfield_elbo':
        loss = infer.TraceMeanField_ELBO()
    elif loss_kind == 'trace_elbo':
        loss = infer.Trace_ELBO(num_particles=10)
    else:
        raise ValueError(f"Unknown SVI loss_kind: {loss_kind}")

    svi = infer.SVI(prob_model.model, guide, optim, loss)
    
    print(f"[svi] Running NumPyro SVI (max_iterations={max_iterations}, loss={loss_kind})...")
    result = svi.run(
        jax.random.PRNGKey(args.random_seed),
        max_iterations,
        progress_bar=True,
        stable_update=True,
        init_params=init_params,
    )
    median = guide.median(result.params)
    for k, v in result.params.items():
        if k not in median:
            median[k] = v
    
    return median, {
        'loss_history': np.asarray(result.losses).tolist(),
        'result': result,
        'guide': guide
    }


def run_optax(prob_model, args, init_params):
    from herculens.Inference.loss import Loss
    from herculens.Inference.Optimization.optax import OptaxOptimizer

    init_params_unconst = to_unconstrained(prob_model, init_params)
    loss = Loss(prob_model, constrained_space=False)
    optimizer = OptaxOptimizer(loss)

    algorithm = getattr(args, 'algorithm_optax', 'adabelief')
    max_iterations = int(getattr(args, 'max_iterations_optax', 2000))
    init_learning_rate = float(getattr(args, 'init_learning_rate_optax', 1e-2))
    schedule_learning_rate = bool(getattr(args, 'schedule_learning_rate_optax', True))
    stop_at_loss_increase = bool(getattr(args, 'stop_at_loss_increase_optax', False))
    progress_bar = bool(getattr(args, 'progress_bar_optax', True))

    print(f"[optax] Running Herculens OptaxOptimizer (max_iterations={max_iterations}, algorithm={algorithm})...")
    best_fit_unconst, logL, extra_fields, runtime = optimizer.run(
        init_params_unconst,
        algorithm=algorithm,
        max_iterations=max_iterations,
        init_learning_rate=init_learning_rate,
        schedule_learning_rate=schedule_learning_rate,
        stop_at_loss_increase=stop_at_loss_increase,
        progress_bar=progress_bar,
    )

    best_fit = to_constrained(prob_model, best_fit_unconst)

    return best_fit, {
        'loss_history': np.asarray(extra_fields['loss_history']).tolist(),
        'logL': float(logL),
        'runtime': runtime
    }


def pixelated_stage_init_from_parametric(params):
    allowed_prefixes = ('lens_', 'lens_light_', 'ps_', 'RMS')
    return {k: v for k, v in params.items() if k.startswith(allowed_prefixes)}


def run_hmc(prob_model, args, init_params, init_params_path=None):
    if init_params_path is None:
        raise ValueError("HMC sampler requires a prior SVI run path (init_params_path) for warm-start.")
        
    def _concatenate_batches(all_samples, num_chains):
        samples = {}
        for k in all_samples[0].keys():
            reshaped_batches = []
            for b in all_samples:
                val = b[k]
                batch_samples_per_chain = val.shape[0] // num_chains
                reshaped_val = val.reshape((num_chains, batch_samples_per_chain) + val.shape[1:])
                reshaped_batches.append(reshaped_val)
            concat_val = np.concatenate(reshaped_batches, axis=1)
            samples[k] = concat_val.reshape((-1,) + concat_val.shape[2:])
        return samples

    from herculens_wrapper.custom_gibbs import MultiHMCGibbs
    from numpyro.handlers import trace, seed
    
    # Trace the model to find active latent sample sites
    with seed(rng_seed=args.random_seed):
        model_trace = trace(prob_model.model).get_trace()
        
    active_sites = [
        name for name, site in model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]
    
    # Filter the input physical init_params to only keep active sites
    init_params = {k: v for k, v in init_params.items() if k in active_sites}
    init_params = {k: v for k, v in init_params.items() if k != 'pixels_source_grid'}

    # print(init_params)

    # debug
    # from herculens_wrapper.models import PowerSpectrum
    # ny, nx = 50, 50
    # k_grid = PowerSpectrum.K_grid((ny, nx))
    # k_values = k_grid.k

    # pixels_wn = init_params['pixels_wn_source_grid']
    # n = init_params['n_source_grid']
    # rho = init_params['rho_source_grid']
    # sigma = init_params['sigma_source_grid']

    # params = {
    #     'pixels_wn_source_grid': jnp.asarray(pixels_wn),
    #     'n_source_grid': jnp.asarray(n),
    #     'rho_source_grid': jnp.asarray(rho),
    #     'sigma_source_grid': jnp.asarray(sigma),
    # }
    # # 4. Generate the final physical pixels_source_grid
    # pixels_source_grid = PowerSpectrum.pixels_from_params(
    #     params,
    #     param_name='source_grid',
    #     k_values=k_values,
    #     positive=True,   # Softplus positivity constraint (standard in SVI/HMC)
    #     k_zero=0.0       # k_zero value used in prior (normally 0.0)
    # )
    # import matplotlib.pyplot as plt
    # plt.imshow(pixels_source_grid, cmap='twilight')
    # plt.savefig('debug.png')


    # Classify parameter names dynamically
    vars_pixel = [k for k in init_params.keys() if 'pixels_wn_' in k]
    vars_power = [k for k in init_params.keys() if k in ('n_source_grid', 'rho_source_grid', 'sigma_source_grid')]
    vars_lens_light_hmc = [k for k in init_params.keys() if k.startswith('lens_light_')]
    vars_mass = [k for k in init_params.keys() if k.startswith('lens_') and not k.startswith('lens_light_')]
    vars_other = [k for k in init_params.keys() if k not in vars_pixel + vars_power + vars_lens_light_hmc + vars_mass]
    vars_other = [k for k in vars_other if k != 'pixels_source_grid']
    
    print(f"[hmc] Grouped parameters for Gibbs-within-HMC sampling:")
    print(f"  Pixelated source: {vars_pixel}")
    print(f"  Matérn power spectrum: {vars_power}")
    print(f"  Lens light: {vars_lens_light_hmc}")
    print(f"  Lens mass: {vars_mass}")
    print(f"  Other parameters: {vars_other}")
    
    # Map physical parameters to unconstrained space
    init_params_unconst = to_unconstrained(prob_model, init_params)
    init_params_unconst = {k: v.astype(jnp.float64) for k, v in init_params_unconst.items()}
    
    def init_to_value_or_defer(site, values=None, defer=infer.init_to_median(num_samples=25)):
        if values is None:
            values = {}
        if site["type"] == "sample" and not site["is_observed"]:
            if site["name"] in values:
                return values[site["name"]]
            return defer(site)
            
    init_fun = partial(init_to_value_or_defer, values=init_params)
    
    disable_gibbs = bool(getattr(args, 'disable_gibbs', False))
    
    if disable_gibbs:
        print("[hmc] Gibbs sampling is disabled. Running joint NUTS sampler...")
        dense_mass_blocks = []
        if vars_power:
            dense_mass_blocks.append(tuple(vars_power))
            
        from collections import defaultdict
        
        # Group lens light parameters by component index
        lens_light_by_idx = defaultdict(list)
        for k in vars_lens_light_hmc:
            try:
                idx = int(k.split('_')[-1])
                lens_light_by_idx[idx].append(k)
            except ValueError:
                pass
        for idx, params_group in sorted(lens_light_by_idx.items()):
            dense_mass_blocks.append(tuple(params_group))
            
        # Group lens mass parameters by component index
        lens_mass_by_idx = defaultdict(list)
        for k in vars_mass:
            try:
                idx = int(k.split('_')[-1])
                lens_mass_by_idx[idx].append(k)
            except ValueError:
                pass
        for idx, params_group in sorted(lens_mass_by_idx.items()):
            dense_mass_blocks.append(tuple(params_group))
            
        outer_kernel = infer.NUTS(
            prob_model.model,
            init_strategy=init_fun,
            target_accept_prob=0.9,
            max_tree_depth=10,
            dense_mass=dense_mass_blocks if dense_mass_blocks else False,
        )
    else:
        # Set up inner kernels
        # Kernel 1: NUTS for source pixels, Matérn, lens light, and other variables
        dense_mass_blocks_1 = []
        if vars_power:
            dense_mass_blocks_1.append(tuple(vars_power))
            
        # Group lens light parameters by component index
        from collections import defaultdict
        lens_light_by_idx = defaultdict(list)
        for k in vars_lens_light_hmc:
            try:
                idx = int(k.split('_')[-1])
                lens_light_by_idx[idx].append(k)
            except ValueError:
                pass
        for idx, params_group in sorted(lens_light_by_idx.items()):
            dense_mass_blocks_1.append(tuple(params_group))
            
        kernel_1 = infer.NUTS(
            prob_model.model,
            init_strategy=init_fun,
            target_accept_prob=0.95,
            max_tree_depth=10,
            dense_mass=dense_mass_blocks_1 if dense_mass_blocks_1 else False,
        )
        
        # Kernel 2: NUTS for lens mass
        dense_mass_blocks_2 = []
        lens_mass_by_idx = defaultdict(list)
        for k in vars_mass:
            try:
                idx = int(k.split('_')[-1])
                lens_mass_by_idx[idx].append(k)
            except ValueError:
                pass
        for idx, params_group in sorted(lens_mass_by_idx.items()):
            dense_mass_blocks_2.append(tuple(params_group))
            
        kernel_2 = infer.NUTS(
            prob_model.model,
            init_strategy=init_fun,
            target_accept_prob=0.9,
            max_tree_depth=10,
            dense_mass=dense_mass_blocks_2 if dense_mass_blocks_2 else False,
        )
        
        inner_kernels = [kernel_1, kernel_2]
        
        # Outer Gibbs kernel
        outer_kernel = MultiHMCGibbs(
            inner_kernels,
            gibbs_sites_list=[
                vars_pixel + vars_power + vars_lens_light_hmc + vars_other,
                vars_mass
            ],
        )
    
    num_warmup = int(getattr(args, 'num_warmup_hmc_numpyro', 1000))
    num_samples_total = int(getattr(args, 'num_samples_hmc_numpyro', 1000))
    checkpoint_interval = int(getattr(args, 'checkpoint_interval_hmc_numpyro', 250))
    num_chains = int(getattr(args, 'num_chains_hmc_numpyro', 1))
    from herculens_wrapper.utils import resolve_chain_method_hmc_numpyro
    chain_method = resolve_chain_method_hmc_numpyro(args)
    progress_bar = bool(getattr(args, 'progress_bar_hmc_numpyro', True))
    
    if checkpoint_interval <= 0 or checkpoint_interval > num_samples_total:
        checkpoint_interval = num_samples_total
        
    batch_sizes = []
    current_samples = 0
    while current_samples < num_samples_total:
        size = min(checkpoint_interval, num_samples_total - current_samples)
        batch_sizes.append(size)
        current_samples += size
        
    rng_key = jax.random.PRNGKey(args.random_seed)
    rng_key, rng_key_ = jax.random.split(rng_key)
    
    all_samples = []
    save_path = getattr(args, 'save_path', '.')
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_path = os.path.join(save_path, "hmc_checkpoint.pkl")
    start_batch_idx = 0
    last_state = None
    
    if os.path.exists(checkpoint_path):
        print(f"[hmc] Found existing checkpoint at {checkpoint_path}. Attempting to resume...")
        try:
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
            all_samples = ckpt['all_samples']
            # Convert loaded samples to CPU NumPy arrays to prevent GPU OOM if checkpoint is old
            all_samples = [
                {k: np.asarray(v) for k, v in batch.items()}
                for batch in all_samples
            ]
            last_state = ckpt['last_state']
            start_batch_idx = ckpt['completed_batches']
            print(f"[hmc] Resuming from batch {start_batch_idx+1} (completed {start_batch_idx} batches).")
        except Exception as e:
            print(f"[hmc] Failed to load checkpoint: {e}. Starting from scratch.")
            all_samples = []
            last_state = None
            start_batch_idx = 0
            
    for i, size in enumerate(batch_sizes):
        if i < start_batch_idx:
            print(f"[hmc] Batch {i+1} already completed. Skipping.")
            continue
            
        if disable_gibbs:
            print(f"[hmc] Running joint NUTS batch {i+1}/{len(batch_sizes)} (drawing {size} samples, total {num_samples_total})...")
        else:
            print(f"[hmc] Running Gibbs-within-HMC batch {i+1}/{len(batch_sizes)} (drawing {size} samples, total {num_samples_total})...")
        if i == 0:
            mcmc = infer.MCMC(
                outer_kernel,
                num_warmup=num_warmup,
                num_samples=size,
                num_chains=num_chains,
                progress_bar=progress_bar,
                chain_method=chain_method,
            )
            init_params_unconst_chain = _build_hmc_chain_init_params(
                prob_model,
                init_params_unconst,
                args,
                num_chains,
                vars_mass,
            )
            mcmc.run(
                rng_key_,
                init_params=init_params_unconst_chain,
            )
        else:
            # Re-instantiate MCMC for subsequent batches bypassing warmup
            mcmc = infer.MCMC(
                outer_kernel,
                num_warmup=0,
                num_samples=size,
                num_chains=num_chains,
                progress_bar=progress_bar,
                chain_method=chain_method,
            )
            mcmc.post_warmup_state = last_state
            if disable_gibbs:
                rng_key_to_pass = last_state.rng_key
            else:
                rng_key_to_pass = last_state.rng_key[..., 0, :]
            mcmc.run(
                rng_key_to_pass,
            )
            
        last_state = mcmc.last_state
        
        # Get samples from this batch
        batch_samples = mcmc.get_samples(group_by_chain=False)
        # Convert to CPU NumPy arrays to prevent GPU OOM
        batch_samples = {k: np.asarray(v) for k, v in batch_samples.items()}
        all_samples.append(batch_samples)
        
        batch_path = os.path.join(save_path, f"hmc_samples_batch_{i}.npz")
        npz_dict = {k: np.asarray(v) for k, v in batch_samples.items()}
        np.savez_compressed(batch_path, **npz_dict)
        print(f"[hmc] Saved MCMC batch {i+1} to: {batch_path}")
        
        # Save checkpoint pkl for resumption
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'last_state': last_state,
                    'all_samples': all_samples,
                    'completed_batches': i + 1,
                }, f)
            print(f"[hmc] Saved checkpoint to: {checkpoint_path}")
        except Exception as e:
            print(f"[warning] Failed to save checkpoint pkl: {e}")
            
        # Generate intermediate diagnostics after each batch (image plane and source plane plots)
        try:
            # 1. Concatenate all samples collected so far
            temp_samples = _concatenate_batches(all_samples, num_chains)
            
            # 2. Compute current medians
            temp_medians = {
                k: np.median(np.asarray(v), axis=0)
                for k, v in temp_samples.items()
                if k in active_sites
            }
            temp_deterministics = median_deterministics_from_samples(temp_samples, active_sites=active_sites)
            temp_kwargs, temp_deterministics = kwargs_with_deterministics(
                prob_model,
                temp_medians,
                deterministics=temp_deterministics,
                rng_seed=getattr(args, 'random_seed', 0),
                active_sites=active_sites,
            )
            
            # 3. Create diagnostics subfolder
            diag_dir = os.path.join(save_path, 'diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            
            # Write intermediate kwargs_result JSON to diagnostics
            try:
                from herculens_wrapper.utils import kwargs_best_to_json_pixelated_npy, json_serializer
                
                type_list = getattr(prob_model, 'type_list', {})
                temp_kwargs_json = kwargs_best_to_json_pixelated_npy(
                    temp_kwargs, 
                    diag_dir, 
                    type_list, 
                    save_pixel_arrays=False,
                    # Uncomment to save diagnostic source-pixel arrays.
                    # pixels_filename=f'kwargs_source_pixels_batch_{i}.npy',
                    # pixels_wn_filename=f'kwargs_source_pixels_wn_batch_{i}.npy',
                )
                
                kwargs_json_path = os.path.join(diag_dir, f'kwargs_result_batch_{i}.json')
                with open(kwargs_json_path, 'w') as f:
                    json.dump(temp_kwargs_json, f, indent=4, default=json_serializer)
                print(f"[hmc] Saved intermediate kwargs_result JSON to {kwargs_json_path}")

                _save_hmc_max_loglike_outputs(
                    temp_samples,
                    prob_model,
                    diag_dir,
                    args,
                    active_sites=active_sites,
                    kwargs_filename=f'kwargs_loglike_batch_{i}.json',
                    save_pixel_arrays=False,
                    # Uncomment to save diagnostic max-loglike source-pixel arrays.
                    # pixels_filename=f'kwargs_loglike_source_pixels_batch_{i}.npy',
                    # pixels_wn_filename=f'kwargs_loglike_source_pixels_wn_batch_{i}.npy',
                    linear_plot_filename=f'best_fit_model_loglike_linear_batch_{i}.png',
                    log_plot_filename=f'best_fit_model_loglike_log_batch_{i}.png',
                    # Uncomment to save diagnostic per-sample log-likelihood arrays.
                    # log_likelihoods_filename=f'hmc_log_likelihoods_batch_{i}.npy',
                    log_likelihoods_filename=None,
                )
                _save_hmc_pixels_wn_summary(
                    temp_samples,
                    diag_dir,
                    plot_filename=f'source_pixels_wn_median_uncertainties_batch_{i}.png',
                    # Uncomment to save diagnostic pixels_wn summary arrays.
                    # median_filename=f'source_pixels_wn_median_batch_{i}.npy',
                    # lower_filename=f'source_pixels_wn_sigma_lower_batch_{i}.npy',
                    # upper_filename=f'source_pixels_wn_sigma_upper_batch_{i}.npy',
                    median_filename=None,
                    lower_filename=None,
                    upper_filename=None,
                )
            except Exception as ex_json:
                print(f"[warning] Failed to save intermediate kwargs_result JSON: {ex_json}")
            
            # 4. Generate plots using current medians
            from herculens_wrapper.visualizations import (
                plot_image_plane,
                plot_ring_model_comparison,
                plot_source_plane,
                display,
            )
            
            img_data = getattr(prob_model, 'image_data', None)
            ns_map = getattr(prob_model, 'noise_map', None)
            l_image = getattr(prob_model, 'lens_image', None)
            p_scale = getattr(prob_model, 'pixel_scale', 0.08)
            
            if img_data is not None and l_image is not None:
                # Image plane plot
                try:
                    plot_image_plane(
                        l_image,
                        temp_kwargs,
                        p_scale,
                        img_data,
                        ns_map,
                        diag_dir,
                        output_filename=f"image_plane_batch_{i}.png",
                    )
                    print(f"[hmc] Saved intermediate image plane visualization to {diag_dir}/image_plane_batch_{i}.png")
                except Exception as ex:
                    print(f"[warning] Failed to plot intermediate image plane: {ex}")
                
                # Best fit model plots (linear and log)
                try:
                    best_fit_model = model_image_from_deterministics(
                        prob_model,
                        temp_kwargs,
                        temp_deterministics,
                    )
                    chi2 = float(np.sum(((best_fit_model - img_data) / ns_map) ** 2))
                    mask = getattr(l_image, 'source_arc_mask', None)
                    if mask is not None:
                        mask = np.asarray(mask)
                    residual_vis_max = getattr(args, 'residual_vis_max', 0.0)
                    
                    display(
                        [best_fit_model, img_data, (best_fit_model - img_data) / ns_map],
                        titles=[
                            'Best fit model',
                            'Image data',
                            f'Residuals (chi^2 = {chi2:.2f})',
                        ],
                        pixel_scale=p_scale,
                        savefilename=os.path.join(diag_dir, f"best_fit_model_linear_batch_{i}.png"),
                        plot_scale='linear',
                        contour_mask=mask,
                        residual_vis_max=residual_vis_max,
                    )
                    print(f"[hmc] Saved intermediate best fit model (linear) visualization to {diag_dir}/best_fit_model_linear_batch_{i}.png")
                    
                    display(
                        [best_fit_model, img_data, (best_fit_model - img_data) / ns_map],
                        titles=[
                            'Best fit model',
                            'Image data',
                            f'Residuals (chi^2 = {chi2:.2f})',
                        ],
                        pixel_scale=p_scale,
                        savefilename=os.path.join(diag_dir, f"best_fit_model_log_batch_{i}.png"),
                        plot_scale='log',
                        contour_mask=mask,
                        residual_vis_max=residual_vis_max,
                    )
                    print(f"[hmc] Saved intermediate best fit model (log) visualization to {diag_dir}/best_fit_model_log_batch_{i}.png")
                except Exception as ex:
                    print(f"[warning] Failed to plot intermediate best fit model: {ex}")

                # Ring-focused model/data comparison after lens-light subtraction
                try:
                    plot_ring_model_comparison(
                        l_image,
                        temp_kwargs,
                        p_scale,
                        img_data,
                        ns_map,
                        diag_dir,
                        plot_scale='linear',
                        residual_vis_max=residual_vis_max,
                        output_filename=f"ring_model_comparison_linear_batch_{i}.png",
                    )
                    print(f"[hmc] Saved intermediate ring comparison (linear) visualization to {diag_dir}/ring_model_comparison_linear_batch_{i}.png")

                    plot_ring_model_comparison(
                        l_image,
                        temp_kwargs,
                        p_scale,
                        img_data,
                        ns_map,
                        diag_dir,
                        plot_scale='log',
                        residual_vis_max=residual_vis_max,
                        output_filename=f"ring_model_comparison_log_batch_{i}.png",
                    )
                    print(f"[hmc] Saved intermediate ring comparison (log) visualization to {diag_dir}/ring_model_comparison_log_batch_{i}.png")
                except Exception as ex:
                    print(f"[warning] Failed to plot intermediate ring comparison: {ex}")
                
                # Source plane plot (linear)
                try:
                    plot_source_plane(
                        l_image,
                        temp_kwargs,
                        diag_dir,
                        plot_scale='linear',
                        output_filename=f"source_plane_linear_batch_{i}.png",
                    )
                    print(f"[hmc] Saved intermediate source plane (linear) visualization to {diag_dir}/source_plane_linear_batch_{i}.png")
                except Exception as ex:
                    print(f"[warning] Failed to plot intermediate source plane linear: {ex}")
                    
                # Source plane plot (log)
                try:
                    plot_source_plane(
                        l_image,
                        temp_kwargs,
                        diag_dir,
                        plot_scale='log',
                        output_filename=f"source_plane_log_batch_{i}.png",
                    )
                    print(f"[hmc] Saved intermediate source plane (log) visualization to {diag_dir}/source_plane_log_batch_{i}.png")
                except Exception as ex:
                    print(f"[warning] Failed to plot intermediate source plane log: {ex}")
                
                # Save ArviZ diagnostics for this batch
                save_hmc_diagnostics(temp_samples, num_chains, diag_dir, f"batch_{i}", prob_model=prob_model)
        except Exception as e:
            print(f"[warning] Failed to generate intermediate diagnostics: {e}")
            
        # Free MCMC object and trigger garbage collection to release GPU memory
        if 'mcmc' in locals():
            del mcmc
        import gc
        gc.collect()
            
    # Concatenate all batches along the sample axis
    samples = _concatenate_batches(all_samples, num_chains)
        
    param_samples = {k: v for k, v in samples.items() if k in active_sites}
    map_params = tree_median(param_samples)

    loglike_extra = _save_hmc_max_loglike_outputs(samples, prob_model, save_path, args, active_sites=active_sites)
    _save_hmc_pixels_wn_summary(samples, save_path)
    
    # Flatten unconstrained samples for trace analysis
    try:
        flat_samples_list = []
        n_total_samples = len(samples[list(samples.keys())[0]])
        for i in range(n_total_samples):
            sample_c = {k: v[i] for k, v in samples.items() if k in active_sites}
            sample_u = to_unconstrained(prob_model, sample_c)
            from jax.flatten_util import ravel_pytree
            flat_val, _ = ravel_pytree(sample_u)
            flat_samples_list.append(np.asarray(flat_val))
        flat_samples = np.array(flat_samples_list)
    except Exception as e:
        print(f"[warning] Failed to flatten samples: {e}")
        flat_samples = None
        
    # Save final ArviZ diagnostics
    save_hmc_diagnostics(samples, num_chains, save_path, "final", prob_model=prob_model)
        
    extra_fields = {
        'flat_samples': flat_samples,
    }
    if loglike_extra:
        extra_fields.update({
            k: v
            for k, v in loglike_extra.items()
            if k != 'log_likelihoods'
        })
    
    return samples, map_params, extra_fields
