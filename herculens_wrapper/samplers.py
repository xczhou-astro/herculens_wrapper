"""Herculens inference backends: point optimization and posterior sampling."""

import json
import os

import numpy as np
import optax
import jax
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
    import jax.numpy as jnp
    return jax.tree_util.tree_map(lambda x: jnp.median(x, axis=0), tree)


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
        
    import pickle
    import numpyro.infer.autoguide as autoguide
    from herculens_wrapper.custom_gibbs import MultiHMCGibbs
    from herculens_wrapper.utils import resolve_project_path
    
    init_dir = resolve_project_path(init_params_path)
    params_pkl_path = os.path.join(init_dir, 'svi_guide_params.pkl')
    if not os.path.exists(params_pkl_path):
        raise FileNotFoundError(f"svi_guide_params.pkl not found in prior run: {init_dir}")
        
    print(f"[hmc] Loading SVI guide parameters from {params_pkl_path}...")
    with open(params_pkl_path, 'rb') as f:
        guide_params = pickle.load(f)
        
    # Recreate and prime the SVI guide
    guide = autoguide.AutoLowRankMultivariateNormal(prob_model.model)
    with numpyro.handlers.seed(rng_seed=args.random_seed):
        guide()
        
    # Extract physical parameter medians
    init_params = guide.median(guide_params)
    
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
    
    num_warmup = int(getattr(args, 'num_warmup_hmc_numpyro', 500))
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
            mcmc.run(
                rng_key_,
                init_params=init_params_unconst,
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
            
    # Concatenate all batches along the sample axis (axis 0)
    samples = {}
    for k in all_samples[0].keys():
        samples[k] = jnp.concatenate([b[k] for b in all_samples], axis=0)
        
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
        
    extra_fields = {
        'flat_samples': flat_samples,
    }
    
    return samples, map_params, extra_fields
