"""Herculens inference backends: point optimization and posterior sampling."""

from numpyro.distributions import biject_to
import json
import os
import pickle
from glob import glob
from urllib.parse import quote, unquote

import h5py
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


def evaluate_parameter_components(prob_model, params, *, rng_seed=0):
    """Evaluate and cache image products for one constrained parameter set."""
    kwargs, deterministics = kwargs_with_deterministics(
        prob_model, params, rng_seed=rng_seed,
    )
    lens_image = getattr(prob_model, "lens_image", None)
    if lens_image is None:
        raise ValueError("prob_model does not expose lens_image for component evaluation.")
    type_list = getattr(prob_model, "type_list", {})
    total = np.asarray(model_image_from_deterministics(prob_model, kwargs, deterministics))
    source = np.asarray(lens_image.model(
        **kwargs, source_add=True, lens_light_add=False, point_source_add=False,
    ))
    if type_list.get("lens_light_type_list"):
        lens_light = np.asarray(lens_image.model(
            **kwargs, source_add=False, lens_light_add=True, point_source_add=False,
        ))
    else:
        lens_light = np.zeros_like(total)
    if type_list.get("point_source_type_list"):
        point_source = np.asarray(lens_image.model(
            **kwargs, source_add=False, lens_light_add=False, point_source_add=True,
        ))
        no_lens_light = np.asarray(lens_image.model(
            **kwargs, source_add=True, lens_light_add=False, point_source_add=True,
        ))
    else:
        point_source = np.zeros_like(total)
        no_lens_light = source
    components = {
        "total": total,
        "source": source,
        "lens_light": lens_light,
        "point_source": point_source,
        "no_lens_light": no_lens_light,
    }
    derived = {
        "kwargs": kwargs,
        "deterministics": deterministics,
        "components": components,
        "model": total,
        "lensed_source": source,
        "lens_light": lens_light,
        "point_source": point_source,
    }
    image_data = getattr(prob_model, "image_data", None)
    if image_data is not None:
        derived["data_minus_lens_light"] = np.asarray(image_data) - lens_light
    source_kwargs = kwargs.get("kwargs_source", [])
    if source_kwargs and isinstance(source_kwargs[0], dict) and "pixels" in source_kwargs[0]:
        derived["source_plane"] = np.asarray(source_kwargs[0]["pixels"])
    return derived


def evaluate_mcmc_component_medians(
    prob_model,
    samples,
    batch_size=500,
    active_sites=None,
    kwargs_lens_from_params=None,
    lens_image_override=None,
):
    """
    Evaluates total, source-only, lens-light-only, and no-lens-light model images 
    for all MCMC samples using fast vectorized JAX vmap and computes their pixel-by-pixel medians.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    lens_image = lens_image_override or getattr(prob_model, 'lens_image', None)
    if lens_image is None:
        raise ValueError("prob_model does not expose lens_image for MCMC median evaluation.")

    active_sites = set(
        get_active_sample_sites(prob_model) if active_sites is None else active_sites
    )
    sample_keys = [k for k in samples.keys() if k in active_sites]
    if not sample_keys:
        sample_keys = list(samples.keys())

    n_total_samples = len(samples[sample_keys[0]])
    type_list = getattr(prob_model, 'type_list', {})
    has_lens_light = bool(type_list.get('lens_light_type_list'))
    has_point_source = bool(type_list.get('point_source_type_list'))

    def eval_single(sample_dict):
        kwargs_lens = (
            kwargs_lens_from_params(sample_dict)
            if kwargs_lens_from_params is not None else None
        )
        kw = prob_model.params2kwargs(
            sample_dict, kwargs_lens_override=kwargs_lens,
        )
        img_total = jnp.squeeze(lens_image.model(**kw))
        img_source = jnp.squeeze(lens_image.model(
            **kw, source_add=True, lens_light_add=False, point_source_add=False
        ))
        if has_lens_light:
            img_lens_light = jnp.squeeze(lens_image.model(
                **kw, source_add=False, lens_light_add=True, point_source_add=False
            ))
        else:
            img_lens_light = jnp.zeros_like(img_total)

        if has_point_source:
            img_no_lens_light = jnp.squeeze(lens_image.model(
                **kw, source_add=True, lens_light_add=False, point_source_add=True
            ))
            img_point_source = jnp.squeeze(lens_image.model(
                **kw, source_add=False, lens_light_add=False, point_source_add=True
            ))
        else:
            img_no_lens_light = img_source
            img_point_source = jnp.zeros_like(img_total)

        return img_total, img_source, img_lens_light, img_no_lens_light, img_point_source

    vmap_eval = jax.jit(jax.vmap(eval_single))

    totals_list, sources_list, lens_lights_list, no_lens_lights_list, point_sources_list = [], [], [], [], []
    image_data = getattr(prob_model, "image_data", None)
    noise_map = getattr(prob_model, "noise_map", None)
    likelihood_mask = getattr(prob_model, "likelihood_mask", None)
    valid = None
    log_normalization = 0.0
    if image_data is not None and noise_map is not None:
        image_data, noise_map = np.asarray(image_data), np.asarray(noise_map)
        valid = np.isfinite(image_data) & np.isfinite(noise_map) & (noise_map > 0)
        if likelihood_mask is not None:
            valid &= np.asarray(likelihood_mask, dtype=bool)
        log_normalization = float(np.sum(np.log(2.0 * np.pi * noise_map[valid] ** 2)))
    likelihood_scale = float(getattr(prob_model, "likelihood_scale", 1.0))
    max_log_likelihood, chi2_at_max_loglike, max_sample_index = -np.inf, None, None
    for b_start in range(0, n_total_samples, batch_size):
        b_end = min(b_start + batch_size, n_total_samples)
        b_samples = {
            k: jnp.asarray(samples[k][b_start:b_end])
            for k in sample_keys
        }
        b_total, b_source, b_lens_light, b_no_lens_light, b_point_source = vmap_eval(b_samples)
        total_cpu = np.asarray(b_total)
        totals_list.append(total_cpu)
        sources_list.append(np.asarray(b_source))
        lens_lights_list.append(np.asarray(b_lens_light))
        no_lens_lights_list.append(np.asarray(b_no_lens_light))
        point_sources_list.append(np.asarray(b_point_source))
        if valid is not None:
            residual = (total_cpu - image_data[None, ...]) / noise_map[None, ...]
            chi2_batch = np.sum(np.square(residual[..., valid]), axis=1)
            loglike_batch = -0.5 * likelihood_scale * (chi2_batch + log_normalization)
            local_index = int(np.argmax(loglike_batch))
            local_loglike = float(loglike_batch[local_index])
            if local_loglike > max_log_likelihood:
                max_log_likelihood = local_loglike
                chi2_at_max_loglike = float(chi2_batch[local_index])
                max_sample_index = b_start + local_index

    result = {
        'total': np.median(np.concatenate(totals_list, axis=0), axis=0),
        'source': np.median(np.concatenate(sources_list, axis=0), axis=0),
        'lens_light': np.median(np.concatenate(lens_lights_list, axis=0), axis=0),
        'no_lens_light': np.median(np.concatenate(no_lens_lights_list, axis=0), axis=0),
        'point_source': np.median(np.concatenate(point_sources_list, axis=0), axis=0),
    }
    if valid is not None:
        result['_sample_likelihood_summary'] = {
            'max_log_likelihood': float(max_log_likelihood),
            'chi2_max_loglike': chi2_at_max_loglike,
            'max_loglike_sample_index': max_sample_index,
        }
    return result


def evaluate_mcmc_median_model_image(prob_model, samples, batch_size=500):
    """
    Evaluates model images for all MCMC samples using fast vectorized JAX vmap
    and computes the pixel-by-pixel median across all samples.
    """
    res = evaluate_mcmc_component_medians(prob_model, samples, batch_size=batch_size)
    return res['total']




def _hmc_health_report(extra_fields, num_chains, max_tree_depth=10):
    """Summarize NUTS health fields collected by NumPyro MCMC."""
    if not extra_fields:
        return (
            'HMC sampler health\n'
            'No sampler-health fields were collected for these draws.\n'
        )

    fields = {
        key: np.asarray(value)
        for key, value in extra_fields.items()
        if key in ('diverging', 'accept_prob', 'num_steps', 'energy')
    }
    if not fields:
        return (
            'HMC sampler health\n'
            'No divergence, acceptance, tree-depth, or energy fields are available.\n'
        )

    first = next(iter(fields.values()))
    if first.shape[0] % num_chains:
        return 'HMC sampler health\nInvalid chain layout; health metrics were skipped.\n'
    draws_per_chain = first.shape[0] // num_chains
    grouped = {
        key: value.reshape((num_chains, draws_per_chain) + value.shape[1:])
        for key, value in fields.items()
    }
    trailing_shape = first.shape[1:]
    n_blocks = int(np.prod(trailing_shape)) if trailing_shape else 1
    max_steps = 2 ** int(max_tree_depth) - 1
    lines = [
        'HMC sampler health',
        f'draws_per_chain: {draws_per_chain}',
        'Interpretation:',
        '  divergences: require 0; any nonzero value needs investigation.',
        '  acceptance probability: values near the configured target are normal; persistently <0.6 is concerning.',
        f'  tree-depth saturation: require 0 at the configured max_tree_depth={max_tree_depth}.',
        '  BFMI: prefer >0.3 per chain; lower values indicate poor energy exploration.',
    ]

    for block in range(n_blocks):
        label = 'joint_nuts' if n_blocks == 1 else f'gibbs_block_{block}'
        lines.append(f'[{label}]')
        index = np.unravel_index(block, trailing_shape) if trailing_shape else ()

        def select(name):
            if name not in grouped:
                return None
            values = grouped[name]
            return values[(slice(None), slice(None)) + index]

        diverging = select('diverging')
        if diverging is not None:
            per_chain = np.sum(diverging.astype(bool), axis=1)
            lines.append(
                f'  divergences: total={int(np.sum(per_chain))}, '
                f'per_chain={per_chain.astype(int).tolist()}'
            )

        accept_prob = select('accept_prob')
        if accept_prob is not None:
            per_chain = np.mean(accept_prob, axis=1)
            lines.append(
                f'  acceptance_probability: mean={float(np.mean(per_chain)):.3f}, '
                f'per_chain={[round(float(value), 3) for value in per_chain]}'
            )

        num_steps = select('num_steps')
        if num_steps is not None:
            saturated = num_steps >= max_steps
            per_chain = np.sum(saturated, axis=1)
            max_depth_seen = int(np.ceil(np.log2(max(float(np.max(num_steps)), 1.0) + 1.0)))
            lines.append(
                f'  tree_depth: max_seen={max_depth_seen}, saturation_total={int(np.sum(per_chain))}, '
                f'per_chain={per_chain.astype(int).tolist()}'
            )

        energy = select('energy')
        if energy is not None:
            bfmi = []
            for values in energy:
                variance = float(np.var(values))
                numerator = float(np.mean(np.diff(values) ** 2)) if values.size > 1 else np.nan
                bfmi.append(numerator / variance if variance > 0 else np.nan)
            lines.append(
                f'  BFMI: mean={float(np.nanmean(bfmi)):.3f}, '
                f'per_chain={[round(float(value), 3) for value in bfmi]}'
            )
    return '\n'.join(lines) + '\n'


def save_hmc_diagnostics(
    samples,
    num_chains,
    target_dir,
    suffix,
    prob_model=None,
    hmc_extra_fields=None,
    max_tree_depth=10,
):
    try:
        import arviz as az
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        # Prefer lens-mass parameters, but lens mass may legitimately be
        # fixed when fitting conditional lens light/source models.
        def local_site_name(key):
            return key.rsplit('/', 1)[-1]

        target_keys = [
            key for key in samples.keys()
            if (
                'lens_' in local_site_name(key)
                and 'lens_light_' not in local_site_name(key)
            )
        ]

        if not target_keys:
            # Fall back to all low-dimensional latent sites.  Pixel Fourier
            # coefficients are deliberately excluded: their high dimension
            # makes ArviZ trace/density panels unusable.
            for key, values in samples.items():
                local = local_site_name(key)
                values = np.asarray(values)
                if local.startswith('pixels_wn_') or local in {
                    'source_pixels', 'source_scales', 'source_coarse',
                }:
                    continue
                if values.ndim == 1 or (values.ndim == 2 and values.shape[1] <= 32):
                    target_keys.append(key)

        if not target_keys:
            # A pixels-only model has no sensible scalar trace plot.  Still
            # write sampler health so a successful HMC output remains
            # inspectable rather than silently omitting diagnostics.
            health_report = _hmc_health_report(
                hmc_extra_fields, num_chains, max_tree_depth=max_tree_depth,
            )
            summary_path = os.path.join(target_dir, f"mcmc_summary_{suffix}.txt")
            with open(summary_path, 'w') as stream:
                stream.write('No scalar free parameters available for ArviZ diagnostics.\n\n')
                stream.write(health_report)
            print(f"[hmc] Saved sampler-health summary to {summary_path}")
            return

        # Joint multi-band models provide their complete configured order.
        # Single-band models retain the existing mass-config ordering.
        ordered_keys = []
        if prob_model is not None and hasattr(prob_model, 'posterior_site_order'):
            ordered_keys.extend(
                key for key in prob_model.posterior_site_order()
                if key in target_keys
            )
        elif prob_model is not None and hasattr(prob_model, 'param_list'):
            lens_mass_params_list = prob_model.param_list.get('lens_mass_params_list', [])
            for i, mass_profile in enumerate(lens_mass_params_list):
                for param_name in mass_profile.keys():
                    expected_key = f"lens_{param_name}_{i}"
                    if expected_key in target_keys and expected_key not in ordered_keys:
                        ordered_keys.append(expected_key)

        # Append any remaining lens-mass keys not represented in the config order.
        for k in target_keys:
            if k not in ordered_keys:
                ordered_keys.append(k)

        # Format the data for arviz: dict of shape (num_chains, samples_per_chain)
        arviz_data = {}
        arviz_names = {}
        for k in ordered_keys:
            # ArviZ/DataTree treats '/' as a path separator.  NumPyro scopes
            # multi-band sites with '/', so use a reversible display-safe name
            # only in the temporary InferenceData object.
            arviz_key = k.replace('/', '__')
            if arviz_key in arviz_data:
                raise ValueError(
                    f"ArviZ diagnostic name collision after sanitizing {k!r}."
                )
            val = np.asarray(samples[k])
            total_samples = val.shape[0]
            samples_per_chain = total_samples // num_chains
            if samples_per_chain > 0:
                arviz_data[arviz_key] = val.reshape(
                    (num_chains, samples_per_chain) + val.shape[1:]
                )
                arviz_names[k] = arviz_key

        if not arviz_data:
            return

        # Convert dictionary to InferenceData first to support new ArviZ 1.2+ API
        idata = az.from_dict({'posterior': arviz_data})

        health_report = _hmc_health_report(
            hmc_extra_fields, num_chains, max_tree_depth=max_tree_depth,
        )
        print(health_report.rstrip())

        # 1. Generate convergence summary and sampler-health report.
        try:
            summary_df = az.summary(idata)
            summary_path = os.path.join(target_dir, f"mcmc_summary_{suffix}.txt")
            with open(summary_path, 'w') as f:
                f.write(summary_df.to_string())
                f.write('\n\n')
                f.write(health_report)
            print(f"[hmc] Saved arviz summary to {summary_path}")
        except Exception as es:
            print(f"[warning] Failed to compute arviz summary: {es}")
            summary_path = os.path.join(target_dir, f"mcmc_summary_{suffix}.txt")
            with open(summary_path, 'w') as f:
                f.write(health_report)

        # 2. Generate trace and density plots using arviz
        try:
            axes = az.plot_trace_dist(
                idata,
                var_names=[arviz_names[k] for k in ordered_keys if k in arviz_names],
                aes={'color': ['chain']},
                visuals={'trace': {'linestyle': '-'}, 'dist': {'linestyle': '-'}},
            )
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





def _save_hmc_pixels_wn_summary(
    samples,
    save_path,
    plot_filename='source_pixels_wn_median_uncertainties.png',
    median_filename=None,
    lower_filename=None,
    upper_filename=None,
):
    key = 'pixels_wn_source_grid'
    if key not in samples:
        return
    try:
        import matplotlib.pyplot as plt
        from herculens_wrapper.utils import save_array_fits

        arr = np.asarray(samples[key])
        if arr.ndim < 2:
            return
        median = np.median(arr, axis=0)
        p16 = np.percentile(arr, 16, axis=0)
        p84 = np.percentile(arr, 84, axis=0)
        lower = median - p16
        upper = p84 - median

        if median_filename is not None:
            save_array_fits(os.path.join(save_path, median_filename), median)
        if lower_filename is not None:
            save_array_fits(os.path.join(save_path, lower_filename), lower)
        if upper_filename is not None:
            save_array_fits(os.path.join(save_path, upper_filename), upper)

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


def evaluate_mcmc_source_pixels_summary(prob_model, samples, save_path, save_npy=True):
    """
    Evaluate physical 2D source pixel-flux images across ALL MCMC samples using
    JAX vmap and compute the sample-wise median, 16th, and 84th percentiles.

    Herculens stores pixelated-source values as flux per image-data pixel, so
    these arrays can be compared directly with image-plane pixel fluxes without
    an additional pixel-area conversion.
    """
    key = 'pixels_wn_source_grid'
    if key not in samples:
        return None
    try:
        import jax
        import jax.numpy as jnp
        from herculens_wrapper.models import PowerSpectrum

        p_wn_arr = jnp.asarray(samples['pixels_wn_source_grid'], dtype=jnp.float64)
        n_arr = jnp.asarray(np.ravel(samples['n_source_grid']), dtype=jnp.float64)
        sigma_arr = jnp.asarray(np.ravel(samples['sigma_source_grid']), dtype=jnp.float64)
        rho_arr = jnp.asarray(np.ravel(samples['rho_source_grid']), dtype=jnp.float64)

        ny, nx = p_wn_arr.shape[1], p_wn_arr.shape[2]
        k_grid = PowerSpectrum.K_grid((ny, nx))
        k_values = jnp.asarray(k_grid.k)

        is_positive = True
        if hasattr(prob_model, 'pixelated_prior') and isinstance(prob_model.pixelated_prior, dict):
            is_positive = bool(prob_model.pixelated_prior.get('positive', True))

        def single_source_pixels(n, sigma, rho, p_wn):
            scale = jnp.sqrt(PowerSpectrum.P_Matern(k_values, n, sigma, rho, k_zero=0.0))
            pixels = jnp.fft.irfft2(PowerSpectrum.pack_fft_values(p_wn * scale), s=scale.shape, norm='ortho')
            if is_positive:
                return jax.nn.softplus(100.0 * pixels) / 100.0
            return pixels

        vmap_fn = jax.jit(jax.vmap(single_source_pixels))

        n_samples_total = len(p_wn_arr)
        batch_size = 200
        all_rec_sources = []
        for b in range(0, n_samples_total, batch_size):
            b_end = min(b + batch_size, n_samples_total)
            batch_srcs = vmap_fn(n_arr[b:b_end], sigma_arr[b:b_end], rho_arr[b:b_end], p_wn_arr[b:b_end])
            all_rec_sources.append(np.asarray(batch_srcs))

        rec_sources = np.concatenate(all_rec_sources, axis=0)

        median_src = np.median(rec_sources, axis=0)
        p16_src = np.percentile(rec_sources, 16, axis=0)
        p84_src = np.percentile(rec_sources, 84, axis=0)

        lower_src = median_src - p16_src
        upper_src = p84_src - median_src

        if save_npy and save_path is not None:
            from herculens_wrapper.utils import append_array_fits, save_array_fits
            source_path = os.path.join(save_path, 'kwargs_source_pixels.fits')
            # Keep all posterior source summaries together.  LOWER and UPPER
            # are the asymmetric 16th/84th-percentile offsets from PRIMARY,
            # rather than a lossy symmetrised ``SIGMA`` map.
            save_array_fits(source_path, median_src)
            append_array_fits(source_path, lower_src, extension_name='LOWER')
            append_array_fits(source_path, upper_src, extension_name='UPPER')
            for legacy_name in (
                'kwargs_source_pixels_lower.fits',
                'kwargs_source_pixels_upper.fits',
            ):
                legacy_path = os.path.join(save_path, legacy_name)
                if os.path.isfile(legacy_path):
                    os.remove(legacy_path)
            print(f"[hmc] Saved source-pixel median/lower/upper FITS extensions to {source_path}")

        return median_src, lower_src, upper_src
    except Exception as e:
        print(f"[warning] Failed to evaluate MCMC physical source pixels summary: {e}")
        return None


def _build_hmc_chain_init_params(
    prob_model,
    init_params,
    args,
    num_chains,
    init_params_path,
):
    """Initialize chains from valid draws within the SVI guide's 1-sigma region."""
    init_params_unconst = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in to_unconstrained(prob_model, init_params).items()
    }
    from herculens_wrapper.utils import resolve_init_run_dir

    try:
        init_run = resolve_init_run_dir(init_params_path)
        guide_path = os.path.join(init_run, 'svi_guide_params.pkl')
        if not os.path.isfile(guide_path):
            raise FileNotFoundError(guide_path)
        with open(guide_path, 'rb') as handle:
            guide_params = pickle.load(handle)
        guide_params = jax.tree_util.tree_map(jnp.asarray, guide_params)

        guide = autoguide.AutoLowRankMultivariateNormal(prob_model.model)
        with numpyro.handlers.seed(
            rng_seed=int(getattr(args, 'random_seed', 0)) + 104729,
        ):
            guide._setup_prototype()

        prefix = guide.prefix
        loc = jnp.asarray(guide_params[f'{prefix}_loc'])
        scale = jnp.asarray(guide_params[f'{prefix}_scale'])
        cov_factor = jnp.asarray(guide_params[f'{prefix}_cov_factor'])
        if loc.shape != scale.shape or cov_factor.shape[0] != loc.shape[0]:
            raise ValueError('saved AutoLowRank guide parameter shapes are inconsistent')
        if int(loc.size) != int(guide.latent_dim):
            raise ValueError(
                f'saved guide latent dimension {loc.size} does not match '
                f'the HMC model dimension {guide.latent_dim}'
            )

        guide_median = guide.median(guide_params)
        for key, expected in init_params.items():
            if key not in guide_median:
                raise ValueError(f'saved SVI guide is missing active site {key!r}')
            if np.shape(guide_median[key]) != np.shape(expected):
                raise ValueError(
                    f'saved SVI guide shape for {key!r} is {np.shape(guide_median[key])}, '
                    f'but HMC expects {np.shape(expected)}'
                )

        effective_factor = cov_factor * scale[..., None]
        marginal_std = scale * jnp.sqrt(
            1.0 + jnp.sum(jnp.square(cov_factor), axis=-1)
        )
        posterior = numpyro.distributions.LowRankMultivariateNormal(
            loc,
            effective_factor,
            jnp.square(scale),
        )
    except Exception as error:
        print(
            '[hmc:init] Could not load the joint SVI posterior '
            f'({error}); using the saved SVI median initialization.'
        )
        if num_chains == 1:
            return init_params_unconst
        return jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(
                value, (num_chains,) + jnp.shape(value),
            ),
            init_params_unconst,
        )

    max_retries = max(int(getattr(args, 'hmc_init_max_retries', 100)), 0)
    rng_key = jax.random.PRNGKey(int(getattr(args, 'random_seed', 0)) + 130363)
    chain_params = []
    print(
        f'[hmc:init] Drawing {num_chains} chain initializations from the joint '
        f'SVI posterior within 1 sigma (max_retries={max_retries} per chain).'
    )

    for chain_index in range(num_chains):
        accepted = None
        for attempt in range(max_retries):
            rng_key, sample_key = jax.random.split(rng_key)
            latent = posterior.sample(sample_key)

            # Direct rejection is impractical with thousands of latent
            # coordinates. Keep the joint draw direction and contract it into
            # the guide's marginal 1-sigma box in unconstrained space.
            delta = latent - loc
            standardized_max = jnp.max(
                jnp.abs(delta) / jnp.maximum(marginal_std, 1e-12)
            )
            contraction = jnp.minimum(
                1.0,
                0.999 / jnp.maximum(standardized_max, 1e-12),
            )
            latent_1sigma = loc + contraction * delta
            candidate_all = guide._unpack_and_constrain(
                latent_1sigma, guide_params,
            )

            try:
                candidate = {}
                for key, expected in init_params.items():
                    if key not in candidate_all:
                        raise KeyError(f'candidate is missing active site {key!r}')
                    value = jnp.asarray(candidate_all[key], dtype=jnp.float64)
                    if value.shape != jnp.shape(expected):
                        raise ValueError(
                            f'candidate shape mismatch for {key!r}: '
                            f'{value.shape} != {jnp.shape(expected)}'
                        )
                    if not bool(jnp.all(jnp.isfinite(value))):
                        raise ValueError(
                            f'candidate contains non-finite values at {key!r}'
                        )
                    candidate[key] = value

                candidate_unconst = to_unconstrained(prob_model, candidate)
                if not all(
                    bool(jnp.all(jnp.isfinite(jnp.asarray(value))))
                    for value in candidate_unconst.values()
                ):
                    raise ValueError('candidate is non-finite in unconstrained space')
                log_prob = float(np.sum(
                    prob_model.log_prob(candidate, constrained=True)
                ))
                if not np.isfinite(log_prob):
                    raise ValueError('candidate has non-finite joint log-probability')

                accepted = {
                    key: jnp.asarray(value, dtype=jnp.float64)
                    for key, value in candidate_unconst.items()
                }
                print(
                    f'[hmc:init] Chain {chain_index}: accepted SVI draw on attempt '
                    f'{attempt + 1} (log_prob={log_prob:.2f}, '
                    f'contraction={float(contraction):.3f}).'
                )
                break
            except Exception as error:
                if attempt + 1 == max_retries:
                    print(
                        f'[hmc:init] Chain {chain_index}: no valid SVI draw after '
                        f'{max_retries} attempts ({error}); using the SVI median.'
                    )

        if accepted is None:
            if max_retries == 0:
                print(
                    f'[hmc:init] Chain {chain_index}: posterior initialization '
                    'retries are disabled; using the SVI median.'
                )
            accepted = init_params_unconst
        chain_params.append(accepted)

    if num_chains == 1:
        return chain_params[0]
    return {
        key: jnp.stack(
            [jnp.asarray(chain[key]) for chain in chain_params], axis=0,
        )
        for key in init_params_unconst
    }


def save_metrics(
    save_path, chi2, image_data, num_params, log_likelihood,
    fit_dof_and_reduced_chi2, num_params_free=None, num_params_physical=None,
    mask_bool=None, source_pixel_scale=None, metric_summary=None,
):
    """Save legacy metrics, or an explicit API median/max-likelihood summary."""
    if metric_summary is not None:
        metrics = {
            'BIC_PHYSICAL_MEDIAN': metric_summary['bic_physical_median'],
            'BIC_PHYSICAL_MAX_LOGLIKE': metric_summary['bic_physical_max_loglike'],
            'CHI2_MEDIAN': metric_summary['chi2_median'],
            'CHI2_MAX_LOGLIKE': metric_summary['chi2_max_loglike'],
            'CHI2_PER_DATA_PIXEL_MEDIAN': metric_summary['chi2_median'] / metric_summary['n_data_pixels'],
            'CHI2_PER_DATA_PIXEL_MAX_LOGLIKE': (
                None if metric_summary['chi2_max_loglike'] is None
                else metric_summary['chi2_max_loglike'] / metric_summary['n_data_pixels']
            ),
            'REDUCED_CHI2_MEDIAN': metric_summary['reduced_chi2_median'],
            'REDUCED_CHI2_MAX_LOGLIKE': metric_summary['reduced_chi2_max_loglike'],
            'N_DATA_PIXELS': metric_summary['n_data_pixels'],
            'N_PARAMS_FITTED': int(num_params),
            'N_PARAMS_FREE': metric_summary['n_free_parameters'],
            'N_PARAMS_PHYSICAL': metric_summary['n_physical_parameters'],
            'CHI2_DOF': metric_summary['degrees_of_freedom'],
            'LOG_LIKELIHOOD_MEDIAN': metric_summary['log_likelihood_median'],
            'MAX_LOG_LIKELIHOOD': metric_summary['max_log_likelihood'],
            'MAX_LOGLIKE_SAMPLE_INDEX': metric_summary['max_loglike_sample_index'],
        }
        if source_pixel_scale is not None:
            metrics['SOURCE_PIXEL_SCALE'] = float(source_pixel_scale)
        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics
    if num_params_free is None:
        num_params_free = num_params
    if num_params_physical is None:
        num_params_physical = num_params_free
    reduced_chi2, n_pix, n_fit_free, dof = fit_dof_and_reduced_chi2(chi2, image_data, num_params_free, mask_bool=mask_bool)
    bic_physical = num_params_physical * np.log(n_pix) - 2 * log_likelihood
    metrics = {
        'BIC_PHYSICAL_MEDIAN': float(bic_physical),
        'BIC_PHYSICAL_MAX_LOGLIKE': None,
        'CHI2_MEDIAN': float(chi2),
        'CHI2_MAX_LOGLIKE': None,
        'CHI2_PER_DATA_PIXEL_MEDIAN': float(chi2 / n_pix),
        'CHI2_PER_DATA_PIXEL_MAX_LOGLIKE': None,
        'REDUCED_CHI2_MEDIAN': float(reduced_chi2),
        'REDUCED_CHI2_MAX_LOGLIKE': None,
        'CHI2_DOF': int(dof),
        'N_DATA_PIXELS': int(n_pix),
        'N_PARAMS_FITTED': int(num_params),
        'N_PARAMS_FREE': int(num_params_free),
        'N_PARAMS_PHYSICAL': int(num_params_physical),
        'LOG_LIKELIHOOD_MEDIAN': float(log_likelihood),
        'MAX_LOG_LIKELIHOOD': None,
        'MAX_LOGLIKE_SAMPLE_INDEX': None,
    }
    if source_pixel_scale is not None:
        metrics['SOURCE_PIXEL_SCALE'] = float(source_pixel_scale)
        
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(
        f'Reduced chi^2 (median): {reduced_chi2:.4f} '
        f'(chi^2={chi2:.2f}, dof={dof}, p={num_params_free}), '
        f'chi^2/N_pix={chi2 / n_pix:.4f}'
    )
    print(f'BIC_physical (median): {bic_physical:.2f}, log-likelihood (median): {log_likelihood:.2f}')
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
    num_particles=None,
):
    if max_iterations is None:
        max_iterations = int(getattr(args, 'max_iterations_svi', 10000))
    if learning_rate is None:
        learning_rate = float(getattr(args, 'init_learning_rate_svi', 0.01))
    if init_scale is None:
        init_scale = float(getattr(args, 'init_scale_svi', 0.1))
    if loss_kind is None:
        loss_kind = getattr(args, 'loss_kind_svi', 'trace_elbo')
    if num_particles is None:
        num_particles = int(getattr(args, 'num_particles_svi', 10))
    if num_particles < 1:
        raise ValueError("num_particles_svi must be at least 1.")

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

    warmup_steps = min(500, int(max_iterations * 0.05))
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max_iterations,
        end_value=1e-5,
    )
    optim = optax.adabelief(learning_rate=scheduler)

    if loss_kind == 'trace_meanfield_elbo':
        loss = infer.TraceMeanField_ELBO(num_particles=num_particles)
    elif loss_kind == 'trace_elbo':
        loss = infer.Trace_ELBO(num_particles=num_particles)
    else:
        raise ValueError(f"Unknown SVI loss_kind: {loss_kind}")

    svi = infer.SVI(prob_model.model, guide, optim, loss)
    
    print(f"[svi] Running NumPyro SVI (max_iterations={max_iterations}, loss={loss_kind}, num_particles={num_particles})...")
    result = svi.run(
        jax.random.PRNGKey(args.random_seed),
        max_iterations,
        progress_bar=True,
        stable_update=True,
        init_params=init_params,
    )
    median = guide.median(result.params)
    # ``result.params`` also contains AutoGuide internals (``auto_loc``,
    # covariance factors, ...).  They are variational parameters, not model
    # parameters.  ``guide.median`` additionally contains model
    # deterministics.  Keep exactly the constrained sample sites supplied by
    # the API initialization, so neither category leaks into FitResult, BIC,
    # or saved kwargs.
    if init_params:
        median = {name: value for name, value in median.items() if name in init_params}

    return median, {
        'loss_history': np.asarray(result.losses).tolist(),
        'result': result,
        'guide': guide
    }


def run_optax(prob_model, args, init_params):
    from herculens.Inference.loss import Loss
    from herculens.Inference.Optimization.optax import OptaxOptimizer

    init_params_unconst = to_unconstrained(prob_model, init_params)
    clean_unconst = {}
    for k, v in init_params_unconst.items():
        arr = jnp.asarray(v)
        if jnp.issubdtype(arr.dtype, jnp.integer):
            arr = arr.astype(jnp.float64)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            clean_unconst[k] = arr
    init_params_unconst = clean_unconst

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


HMC_SAMPLES_HDF5_FILENAME = 'hmc_samples.h5'


def _hmc_hdf5_path(save_path):
    return os.path.join(save_path, HMC_SAMPLES_HDF5_FILENAME)


def _hdf5_site_name(name):
    """Encode NumPyro site names, which may contain '/' in joint models."""
    return quote(name, safe='')


def _is_pixel_wn_site(name):
    """Whether a site is the high-dimensional Fourier white-noise latent."""
    return name.rsplit('/', 1)[-1].startswith('pixels_wn_')


def _hdf5_group_arrays(group):
    return {
        unquote(name): np.asarray(dataset)
        for name, dataset in group.items()
        if isinstance(dataset, h5py.Dataset)
    }


def _convert_hmc_pixel_latents_hdf5_to_float32(path):
    """Shrink existing float64 pixel-latent datasets without touching sampler state."""
    with h5py.File(path, 'r') as source:
        samples_group = source.get('samples')
        requires_conversion = samples_group is not None and any(
            _is_pixel_wn_site(unquote(name))
            and np.issubdtype(dataset.dtype, np.floating)
            and dataset.dtype.itemsize > np.dtype(np.float32).itemsize
            for name, dataset in samples_group.items()
        )
    if not requires_conversion:
        return False

    temporary_path = f'{path}.float32.tmp'
    try:
        with h5py.File(path, 'r') as source, h5py.File(temporary_path, 'w') as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            for group_name, source_group in source.items():
                target_group = target.create_group(group_name)
                for key, value in source_group.attrs.items():
                    target_group.attrs[key] = value
                for name, dataset in source_group.items():
                    is_pixel_latent = group_name == 'samples' and _is_pixel_wn_site(unquote(name))
                    dtype = np.float32 if is_pixel_latent and np.issubdtype(dataset.dtype, np.floating) else dataset.dtype
                    chunk_rows = max(1, min(int(dataset.shape[0]), 128))
                    target_dataset = target_group.create_dataset(
                        name,
                        shape=dataset.shape,
                        dtype=dtype,
                        maxshape=dataset.maxshape,
                        chunks=(chunk_rows,) + dataset.shape[1:],
                        compression='gzip',
                        compression_opts=4,
                        shuffle=True,
                    )
                    for key, value in dataset.attrs.items():
                        target_dataset.attrs[key] = value
                    for start in range(0, dataset.shape[0], chunk_rows):
                        end = min(start + chunk_rows, dataset.shape[0])
                        target_dataset[start:end] = dataset[start:end]
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return True


def _append_hmc_samples_hdf5(path, samples, extra_fields, num_chains):
    """Append one flattened MCMC batch to the single on-disk posterior store."""
    if not samples:
        return
    batch_size = int(np.asarray(next(iter(samples.values()))).shape[0])
    if batch_size % num_chains:
        raise ValueError('HMC batch sample count is incompatible with num_chains.')

    with h5py.File(path, 'a') as handle:
        saved_chains = handle.attrs.get('num_chains')
        if saved_chains is not None and int(saved_chains) != int(num_chains):
            raise ValueError(
                f'HDF5 samples use num_chains={saved_chains}, but the current '
                f'configuration requests {num_chains}.'
            )
        handle.attrs['num_chains'] = int(num_chains)
        handle.attrs['format_version'] = 1

        for group_name, values in (('samples', samples), ('sampler_health', extra_fields)):
            if not values:
                continue
            group = handle.require_group(group_name)
            for name, value in values.items():
                array = np.asarray(value)
                if group_name == 'samples' and _is_pixel_wn_site(name):
                    # Archive-only downcast: HMC itself and its checkpoint remain float64.
                    array = array.astype(np.float32, copy=False)
                if array.shape[0] != batch_size:
                    raise ValueError(
                        f'HMC {group_name} field {name!r} has an incompatible batch length.'
                    )
                dataset_name = _hdf5_site_name(name)
                if dataset_name not in group:
                    chunk_rows = max(1, min(batch_size, 128))
                    group.create_dataset(
                        dataset_name,
                        data=array,
                        maxshape=(None,) + array.shape[1:],
                        chunks=(chunk_rows,) + array.shape[1:],
                        compression='gzip',
                        compression_opts=4,
                        shuffle=True,
                    )
                else:
                    dataset = group[dataset_name]
                    if dataset.shape[1:] != array.shape[1:]:
                        raise ValueError(
                            f'HMC {group_name} field {name!r} changed shape from '
                            f'{dataset.shape[1:]} to {array.shape[1:]}.'
                        )
                    old_rows = dataset.shape[0]
                    dataset.resize(old_rows + batch_size, axis=0)
                    dataset[old_rows:] = array
        handle.flush()


def _load_hmc_samples_hdf5(path):
    """Load flattened posterior and sampler-health arrays from HDF5."""
    with h5py.File(path, 'r') as handle:
        if 'samples' not in handle:
            raise ValueError(f'HMC samples file {path!r} does not contain a samples group.')
        samples = _hdf5_group_arrays(handle['samples'])
        extra_fields = (
            _hdf5_group_arrays(handle['sampler_health'])
            if 'sampler_health' in handle else {}
        )
    return samples, extra_fields


def _hmc_hdf5_sample_rows(path):
    with h5py.File(path, 'r') as handle:
        if 'samples' not in handle or not handle['samples']:
            return 0
        lengths = {dataset.shape[0] for dataset in handle['samples'].values()}
        if len(lengths) != 1:
            raise ValueError(f'HMC samples file {path!r} has inconsistent dataset lengths.')
        return lengths.pop()


def _truncate_hmc_hdf5(path, rows):
    """Discard an uncheckpointed HDF5 tail left by an interrupted batch write."""
    with h5py.File(path, 'r+') as handle:
        for group_name in ('samples', 'sampler_health'):
            if group_name not in handle:
                continue
            for dataset in handle[group_name].values():
                if dataset.shape[0] > rows:
                    dataset.resize(rows, axis=0)
        handle.flush()


def _write_hmc_checkpoint(path, last_state, completed_samples, completed_batches,
                          num_chains, checkpoint_interval):
    """Persist only the state and metadata required to resume NumPyro MCMC."""
    payload = {
        'format_version': 2,
        'last_state': last_state,
        'completed_samples_per_chain': int(completed_samples),
        'completed_batches': int(completed_batches),
        'num_chains': int(num_chains),
        'checkpoint_interval': int(checkpoint_interval),
    }
    temporary_path = f'{path}.tmp'
    with open(temporary_path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def run_hmc(prob_model, args, init_params, init_params_path=None, batch_diagnostics_callback=None):
    resume_checkpoint_path = os.path.join(getattr(args, 'save_path', '.'), 'hmc_checkpoint.pkl')
    if init_params_path is None and not os.path.isfile(resume_checkpoint_path):
        raise ValueError(
            "HMC sampler requires a prior SVI run path (init_params_path) when starting a new chain."
        )
        
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

    def _concatenate_hmc_extra_fields(all_fields, num_chains):
        if not all_fields or not all_fields[0]:
            return {}
        return _concatenate_batches(all_fields, num_chains)

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


    # Band-scoped sites are named ``F150W/site_name``.  Classify them
    # from their local name while retaining the full name for NumPyro.
    def local_site_name(name):
        return name.rsplit('/', 1)[-1]

    vars_pixel = [k for k in init_params if local_site_name(k).startswith('pixels_wn_')]
    vars_power = [
        k for k in init_params
        if local_site_name(k) in ('n_source_grid', 'rho_source_grid', 'sigma_source_grid')
    ]
    vars_lens_light_hmc = [
        k for k in init_params if local_site_name(k).startswith('lens_light_')
    ]
    # Shared mass sites are unscoped. Multi-band astrometric centres are
    # band-scoped but still belong in the mass Gibbs block.
    vars_mass = [
        k for k in init_params
        if (
            ('/' not in k and k.startswith('lens_') and not k.startswith('lens_light_'))
            or local_site_name(k).startswith(('lens_center_x_', 'lens_center_y_'))
        )
    ]
    vars_other = [k for k in init_params.keys() if k not in vars_pixel + vars_power + vars_lens_light_hmc + vars_mass]
    vars_other = [k for k in vars_other if k != 'pixels_source_grid']

    def component_group(name):
        scope, _, local_name = name.rpartition('/')
        return scope, int(local_name.rsplit('_', 1)[-1])
    
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
    
    if not init_params:
        raise ValueError(
            "HMC requires at least one free sampling parameter; all declared "
            "parameters are fixed or linked."
        )

    # Gibbs needs two non-empty conditional blocks.  With a fixed lens mass
    # there is only the light/source block, for which ordinary joint NUTS is
    # the correct and simpler kernel.
    use_joint_nuts = (
        bool(getattr(args, 'disable_gibbs', False))
        or not vars_mass
        or not (vars_pixel + vars_power + vars_lens_light_hmc + vars_other)
    )

    if use_joint_nuts:
        if not vars_mass:
            print("[hmc] Lens mass is fixed; running joint NUTS for remaining free parameters.")
        elif not (vars_pixel + vars_power + vars_lens_light_hmc + vars_other):
            print("[hmc] Only lens-mass parameters are free; running joint NUTS.")
        else:
            print("[hmc] Gibbs sampling is disabled. Running joint NUTS sampler...")
        dense_mass_blocks = []
        if vars_power:
            dense_mass_blocks.append(tuple(vars_power))
            
        from collections import defaultdict
        
        # Group lens light parameters by component index
        lens_light_by_idx = defaultdict(list)
        for k in vars_lens_light_hmc:
            try:
                lens_light_by_idx[component_group(k)].append(k)
            except ValueError:
                pass
        for idx, params_group in sorted(lens_light_by_idx.items()):
            dense_mass_blocks.append(tuple(params_group))
            
        # Group lens mass parameters by component index
        lens_mass_by_idx = defaultdict(list)
        for k in vars_mass:
            try:
                lens_mass_by_idx[component_group(k)].append(k)
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
                lens_light_by_idx[component_group(k)].append(k)
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
                lens_mass_by_idx[component_group(k)].append(k)
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
        
    rng_key = jax.random.PRNGKey(args.random_seed)
    rng_key, rng_key_ = jax.random.split(rng_key)
    
    all_samples = []
    all_hmc_extra_fields = []
    save_path = getattr(args, 'save_path', '.')
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_path = os.path.join(save_path, "hmc_checkpoint.pkl")
    samples_hdf5_path = _hmc_hdf5_path(save_path)
    start_batch_idx = 0
    last_state = None
    completed_samples = 0
    record_hmc_health = True
    
    if os.path.exists(checkpoint_path):
        print(f"[hmc] Found existing checkpoint at {checkpoint_path}. Attempting to resume...")
        try:
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
            saved_num_chains = ckpt.get('num_chains')
            saved_interval = ckpt.get('checkpoint_interval')
            if saved_num_chains is not None and int(saved_num_chains) != num_chains:
                raise ValueError(
                    f'Checkpoint uses num_chains={saved_num_chains}, but the current '
                    f'configuration requests num_chains_hmc_numpyro={num_chains}.'
                )
            if saved_interval is not None and int(saved_interval) != checkpoint_interval:
                raise ValueError(
                    f'Checkpoint uses checkpoint_interval_hmc_numpyro={saved_interval}, but '
                    f'the current configuration requests {checkpoint_interval}. Keep the '
                    'checkpoint interval unchanged when extending HMC sampling.'
                )

            # Checkpoints created before HDF5 embedded all draws. Migrate them once,
            # then rewrite the checkpoint in the compact v2 format below.
            if 'all_samples' in ckpt:
                legacy_samples = [
                    {k: np.asarray(v) for k, v in batch.items()}
                    for batch in ckpt.get('all_samples', [])
                    if batch
                ]
                legacy_health = [
                    {k: np.asarray(v) for k, v in batch.items()}
                    for batch in ckpt.get('all_hmc_extra_fields', [])
                ]
                health_is_complete = (
                    len(legacy_health) == len(legacy_samples)
                    and all(bool(fields) for fields in legacy_health)
                )
                if not health_is_complete:
                    legacy_health = [{} for _ in legacy_samples]
                    record_hmc_health = False
                    print('[hmc] Legacy checkpoint has no complete sampler-health history; '
                          'health diagnostics will cover only a fresh HMC run.')

                with h5py.File(samples_hdf5_path, 'w'):
                    pass
                for batch_index, batch in enumerate(legacy_samples):
                    _append_hmc_samples_hdf5(
                        samples_hdf5_path,
                        batch,
                        legacy_health[batch_index] if record_hmc_health else {},
                        num_chains,
                    )
                all_samples = legacy_samples
                all_hmc_extra_fields = legacy_health
                for batch in legacy_samples:
                    first_value = next(iter(batch.values()))
                    batch_total = int(np.asarray(first_value).shape[0])
                    if batch_total % num_chains:
                        raise ValueError(
                            'Checkpoint sample count is incompatible with the configured number of chains.'
                        )
                    completed_samples += batch_total // num_chains
                print(f'[hmc] Migrated existing samples to {samples_hdf5_path}.')
                for legacy_batch_path in glob(os.path.join(save_path, 'hmc_samples_batch_*.npz')):
                    os.remove(legacy_batch_path)
                if legacy_samples:
                    print('[hmc] Removed legacy per-batch NPZ archives after HDF5 migration.')
            else:
                completed_samples = int(ckpt.get('completed_samples_per_chain', 0))
                if completed_samples and not os.path.isfile(samples_hdf5_path):
                    raise ValueError(
                        'Checkpoint records completed HMC draws but hmc_samples.h5 is missing. '
                        'Cannot resume without the saved posterior samples.'
                    )
                expected_rows = completed_samples * num_chains
                if expected_rows:
                    actual_rows = _hmc_hdf5_sample_rows(samples_hdf5_path)
                    if actual_rows < expected_rows:
                        raise ValueError(
                            f'HDF5 samples contain {actual_rows} rows, but the checkpoint '
                            f'requires {expected_rows} rows.'
                        )
                    if actual_rows > expected_rows:
                        _truncate_hmc_hdf5(samples_hdf5_path, expected_rows)
                        print('[hmc] Discarded an uncheckpointed HDF5 sample tail from an interrupted write.')
                    if _convert_hmc_pixel_latents_hdf5_to_float32(samples_hdf5_path):
                        print('[hmc] Converted archived pixels_wn samples to float32 in HDF5.')
                    saved_samples, saved_health = _load_hmc_samples_hdf5(samples_hdf5_path)
                    all_samples = [saved_samples]
                    all_hmc_extra_fields = [saved_health] if saved_health else [{}]
                    if not saved_health:
                        record_hmc_health = False
                elif os.path.isfile(samples_hdf5_path):
                    # A checkpoint with zero draws is a fresh start; stale samples must not leak in.
                    with h5py.File(samples_hdf5_path, 'w'):
                        pass
            last_state = ckpt['last_state']
            start_batch_idx = ckpt['completed_batches']
            if 'all_samples' in ckpt:
                _write_hmc_checkpoint(
                    checkpoint_path, last_state, completed_samples, start_batch_idx,
                    num_chains, checkpoint_interval,
                )
                print('[hmc] Rewrote checkpoint in compact format (posterior draws are in HDF5).')
            print(
                f"[hmc] Resuming after {completed_samples} draws per chain "
                f"({start_batch_idx} completed batches)."
            )
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Failed to resume the HMC checkpoint at {checkpoint_path}. "
                "The existing chain was left unchanged."
            ) from error

    elif os.path.isfile(samples_hdf5_path):
        # No checkpoint means this is intentionally a new chain, not a continuation.
        with h5py.File(samples_hdf5_path, 'w'):
            pass
        print(f'[hmc] Replaced stale HDF5 samples at {samples_hdf5_path} for a new run.')

    if completed_samples > num_samples_total:
        raise ValueError(
            f'Checkpoint already contains {completed_samples} draws per chain, exceeding '
            f'num_samples_hmc_numpyro={num_samples_total}. Use a larger target or start a new run.'
        )
    batch_sizes = []
    remaining_samples = num_samples_total - completed_samples
    while remaining_samples > 0:
        size = min(checkpoint_interval, remaining_samples)
        batch_sizes.append(size)
        remaining_samples -= size
    total_batch_count = start_batch_idx + len(batch_sizes)
            
    for batch_offset, size in enumerate(batch_sizes):
        i = start_batch_idx + batch_offset
            
        if use_joint_nuts:
            print(f"[hmc] Running joint NUTS batch {i+1}/{total_batch_count} (drawing {size} samples, total {num_samples_total})...")
        else:
            print(f"[hmc] Running Gibbs-within-HMC batch {i+1}/{total_batch_count} (drawing {size} samples, total {num_samples_total})...")
        if last_state is None:
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
                init_params,
                args,
                num_chains,
                init_params_path,
            )
            mcmc.run(
                rng_key_,
                init_params=init_params_unconst_chain,
                extra_fields=('diverging', 'accept_prob', 'num_steps', 'energy'),
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
            if use_joint_nuts:
                rng_key_to_pass = last_state.rng_key
            else:
                rng_key_to_pass = last_state.rng_key[..., 0, :]
            mcmc.run(
                rng_key_to_pass,
                extra_fields=('diverging', 'accept_prob', 'num_steps', 'energy'),
            )
            
        last_state = mcmc.last_state
        
        # Get samples from this batch
        batch_samples = mcmc.get_samples(group_by_chain=False)
        # Convert to CPU NumPy arrays to prevent GPU OOM
        batch_samples = {k: np.asarray(v) for k, v in batch_samples.items()}
        all_samples.append(batch_samples)
        batch_hmc_extra_fields = {
            key: np.asarray(value)
            for key, value in mcmc.get_extra_fields(group_by_chain=False).items()
            if key in ('diverging', 'accept_prob', 'num_steps', 'energy')
        }
        all_hmc_extra_fields.append(batch_hmc_extra_fields)

        _append_hmc_samples_hdf5(
            samples_hdf5_path,
            batch_samples,
            batch_hmc_extra_fields if record_hmc_health else {},
            num_chains,
        )
        print(f"[hmc] Updated HDF5 posterior after batch {i + 1}: {samples_hdf5_path}")
        
        # Save only the sampler state and progress; posterior draws live in HDF5.
        try:
            _write_hmc_checkpoint(
                checkpoint_path,
                last_state,
                completed_samples + sum(batch_sizes[:batch_offset + 1]),
                i + 1,
                num_chains,
                checkpoint_interval,
            )
            print(f"[hmc] Saved checkpoint to: {checkpoint_path}")
        except Exception as e:
            print(f"[warning] Failed to save checkpoint pkl: {e}")
            
        if batch_diagnostics_callback is not None:
            try:
                batch_diagnostics_callback(
                    _concatenate_batches(all_samples, num_chains),
                    i,
                    _concatenate_hmc_extra_fields(all_hmc_extra_fields, num_chains),
                )
            except Exception as e:
                print(f"[warning] Failed to generate multi-band batch diagnostics: {e}")
            del mcmc
            import gc
            gc.collect()
            continue

        # Generate the compact single-band diagnostic set for this batch.
        try:
            temp_samples = _concatenate_batches(all_samples, num_chains)
            temp_hmc_extra_fields = _concatenate_hmc_extra_fields(
                all_hmc_extra_fields, num_chains,
            )
            temp_medians = {
                k: np.median(np.asarray(v), axis=0)
                for k, v in temp_samples.items()
                if k in active_sites
            }
            temp_kwargs = prob_model.params2kwargs(temp_medians)
            diag_dir = os.path.join(save_path, 'diagnostics')
            os.makedirs(diag_dir, exist_ok=True)

            source_summary = evaluate_mcmc_source_pixels_summary(
                prob_model, temp_samples, diag_dir, save_npy=False,
            )
            if source_summary is not None and temp_kwargs.get('kwargs_source'):
                temp_kwargs['kwargs_source'][0]['pixels'] = source_summary[0]

            from herculens_wrapper.visualizations import (
                plot_composite_2x3_panel,
                plot_hmc_chain_comparison,
            )
            img_data = getattr(prob_model, 'image_data', None)
            ns_map = getattr(prob_model, 'noise_map', None)
            l_image = getattr(prob_model, 'lens_image', None)
            p_scale = getattr(prob_model, 'pixel_scale', 0.08)
            if img_data is not None and l_image is not None:
                temp_comp_medians = evaluate_mcmc_component_medians(
                    prob_model, temp_samples,
                )
                plot_composite_2x3_panel(
                    l_image, temp_kwargs, p_scale, img_data, ns_map, diag_dir,
                    residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
                    output_filename=f"composite_batch_{i}.png",
                    model_extended_override=temp_comp_medians['source'],
                    model_lens_light_override=temp_comp_medians['lens_light'],
                    model_composite_override=temp_comp_medians['total'],
                )
                print(f"[hmc] Saved compact composite diagnostic for batch {i + 1}.")
                plot_hmc_chain_comparison(
                    prob_model,
                    temp_samples,
                    num_chains,
                    p_scale,
                    img_data,
                    ns_map,
                    diag_dir,
                    residual_vis_max=getattr(args, 'residual_vis_max', 0.0),
                    output_filename=f'hmc_chain_comparison_batch_{i}.png',
                )
                print(f"[hmc] Saved chain comparison diagnostic for batch {i + 1}.")
            save_hmc_diagnostics(
                temp_samples, num_chains, diag_dir, f"batch_{i}", prob_model=prob_model,
                hmc_extra_fields=temp_hmc_extra_fields,
            )
        except Exception as e:
            print(f"[warning] Failed to generate intermediate diagnostics: {e}")
            
        # Free MCMC object and trigger garbage collection to release GPU memory
        if 'mcmc' in locals():
            del mcmc
        import gc
        gc.collect()
            
    # Concatenate all batches along the sample axis
    samples = _concatenate_batches(all_samples, num_chains)
    hmc_extra_fields = _concatenate_hmc_extra_fields(all_hmc_extra_fields, num_chains)
        
    param_samples = {k: v for k, v in samples.items() if k in active_sites}
    map_params = tree_median(param_samples)

    loglike_extra = {}
    derived = {}
    _save_hmc_pixels_wn_summary(samples, save_path)
    source_summary = evaluate_mcmc_source_pixels_summary(prob_model, samples, save_path)
    if source_summary is not None:
        source_plane, source_plane_lower, source_plane_upper = source_summary
        derived.update({
            'source_plane': source_plane,
            'source_plane_lower': source_plane_lower,
            'source_plane_upper': source_plane_upper,
        })
    try:
        if hasattr(prob_model, 'bands'):
            # Joint multiband models have one LensImage per band and cache
            # their batch/result diagnostics through the API callback.
            raise RuntimeError('multiband component caching is handled per band by MultiBandModel')
        component_medians = evaluate_mcmc_component_medians(prob_model, samples)
        sample_likelihood = component_medians.pop('_sample_likelihood_summary', None)
        derived['component_medians'] = component_medians
        derived['components'] = component_medians
        derived['model'] = component_medians['total']
        derived['lensed_source'] = component_medians['source']
        derived['lens_light'] = component_medians['lens_light']
        derived['point_source'] = component_medians['point_source']
        if sample_likelihood is not None:
            derived['sample_likelihood_summary'] = sample_likelihood
        image_data = getattr(prob_model, 'image_data', None)
        if image_data is not None:
            derived['data_minus_lens_light'] = (
                np.asarray(image_data) - component_medians['lens_light']
            )
        print('[hmc] Cached posterior-median component images for result output.')
    except RuntimeError as error:
        if str(error) != 'multiband component caching is handled per band by MultiBandModel':
            print(f'[warning] Failed to cache HMC posterior component images: {error}')
    except Exception as error:
        print(f'[warning] Failed to cache HMC posterior component images: {error}')

    
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
    save_hmc_diagnostics(
        samples, num_chains, save_path, "final", prob_model=prob_model,
        hmc_extra_fields=hmc_extra_fields,
    )
        
    extra_fields = {
        'flat_samples': flat_samples,
        'hmc_sampler_health': hmc_extra_fields,
        'derived': derived,
    }
    if loglike_extra:
        extra_fields.update({
            k: v
            for k, v in loglike_extra.items()
            if k != 'log_likelihoods'
        })
    
    return samples, map_params, extra_fields
