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


def run_hmc(prob_model, args, init_params):
    sampler_type = getattr(args, 'sampler_type_hmc_numpyro', 'nuts').lower()
    num_warmup = int(getattr(args, 'num_warmup_hmc_numpyro', 500))
    num_samples = int(getattr(args, 'num_samples_hmc_numpyro', 1000))
    num_chains = int(getattr(args, 'num_chains_hmc_numpyro', 1))
    from herculens_wrapper.utils import resolve_chain_method_hmc_numpyro
    chain_method = resolve_chain_method_hmc_numpyro(args)
    progress_bar = bool(getattr(args, 'progress_bar_hmc_numpyro', True))

    def init_to_value_or_defer(site, values=None, defer=infer.init_to_median(num_samples=25)):
        if values is None:
            values = {}
        if site["type"] == "sample" and not site["is_observed"]:
            if site["name"] in values:
                return values[site["name"]]
            return defer(site)

    init_fun = partial(init_to_value_or_defer, values=init_params) if init_params else infer.init_to_median(num_samples=25)

    if sampler_type == 'nuts':
        kernel = infer.NUTS(prob_model.model, init_strategy=init_fun)
    elif sampler_type == 'hmc':
        kernel = infer.HMC(prob_model.model, init_strategy=init_fun)
    else:
        raise ValueError(f"Unknown sampler_type_hmc_numpyro: {sampler_type}")

    print(f"[{sampler_type}] Running NumPyro MCMC (warmup={num_warmup}, samples={num_samples}, chains={num_chains}, chain_method={chain_method!r})...")
    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )

    rng_key = jax.random.PRNGKey(args.random_seed)
    mcmc.run(rng_key)

    samples = mcmc.get_samples(group_by_chain=False)
    map_params = tree_median(samples)

    # Flatten unconstrained samples for plotting / trace analysis
    try:
        unconstrained_samples = mcmc.get_samples(group_by_chain=False, raw_samples=True)
        from jax.flatten_util import ravel_pytree
        first_key = list(unconstrained_samples.keys())[0]
        n_samples = len(unconstrained_samples[first_key])
        flat_samples_list = []
        for i in range(n_samples):
            sample_u = {k: v[i] for k, v in unconstrained_samples.items()}
            flat_val, _ = ravel_pytree(sample_u)
            flat_samples_list.append(np.asarray(flat_val))
        flat_samples = np.array(flat_samples_list)
    except Exception as e:
        print(f"[warning] Failed to get raw unconstrained samples directly: {e}. Falling back to converting constrained samples.")
        from jax.flatten_util import ravel_pytree
        first_key = list(samples.keys())[0]
        n_samples = len(samples[first_key])
        flat_samples_list = []
        for i in range(n_samples):
            sample_c = {k: v[i] for k, v in samples.items()}
            sample_u = to_unconstrained(prob_model, sample_c)
            flat_val, _ = ravel_pytree(sample_u)
            flat_samples_list.append(np.asarray(flat_val))
        flat_samples = np.array(flat_samples_list)

    extra_fields = {
        'flat_samples': flat_samples,
    }

    return samples, map_params, extra_fields
