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


def save_metrics(save_path, chi2, image_data, num_params, log_likelihood, fit_dof_and_reduced_chi2, num_params_free=None, mask_bool=None):
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
            init_params_unconst_chain = init_params_unconst
            if num_chains > 1 and init_params_unconst is not None:
                init_params_unconst_chain = jax.tree_util.tree_map(
                    lambda x: jnp.broadcast_to(x, (num_chains,) + jnp.shape(x)),
                    init_params_unconst
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
            mcmc.run(
                last_state.rng_key,
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
            temp_medians = {k: np.median(np.asarray(v), axis=0) for k, v in temp_samples.items()}
            temp_kwargs = prob_model.params2kwargs(temp_medians)
            
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
                    pixels_filename=f'kwargs_source_pixels_batch_{i}.npy'
                )
                
                kwargs_json_path = os.path.join(diag_dir, f'kwargs_result_batch_{i}.json')
                with open(kwargs_json_path, 'w') as f:
                    json.dump(temp_kwargs_json, f, indent=4, default=json_serializer)
                print(f"[hmc] Saved intermediate kwargs_result JSON to {kwargs_json_path}")
            except Exception as ex_json:
                print(f"[warning] Failed to save intermediate kwargs_result JSON: {ex_json}")
            
            # 4. Generate plots using current medians
            from herculens_wrapper.visualizations import plot_image_plane, plot_source_plane, display
            
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
                    if 'model_image' in temp_samples:
                        best_fit_model = np.median(temp_samples['model_image'], axis=0)
                    else:
                        best_fit_model = l_image.model(**temp_kwargs)
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
        
    map_params = tree_median(samples)
    
    # Flatten unconstrained samples for trace analysis
    try:
        flat_samples_list = []
        n_total_samples = len(samples[list(samples.keys())[0]])
        for i in range(n_total_samples):
            sample_c = {k: v[i] for k, v in samples.items()}
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
    
    return samples, map_params, extra_fields
