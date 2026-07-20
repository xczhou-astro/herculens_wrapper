import json
import os
from functools import partial

def safe_float(val, default):
    return float(val) if val is not None else default


import jax
jax.config.update('jax_enable_x64', True)
import optax
import numpy as np
import scipy.ndimage
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import lazy_property
import jax.numpy as jnp

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Inference.ProbModel.numpyro import NumpyroModel
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.LensImage.lens_image import LensImage
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source_model import PointSourceModel

# Monkey patch LightModel.surface_brightness to clean extra keys from kwargs_list (kwargs_source)
_original_surface_brightness = LightModel.surface_brightness

def _patched_surface_brightness(self, x, y, kwargs_list, k=None, **kwargs):
    if kwargs_list is not None:
        clean_kwargs_list = []
        for kw in kwargs_list:
            if isinstance(kw, dict):
                clean_kw = {key: val for key, val in kw.items() 
                            if key not in ['pixels_wn', 'n_source_grid', 'rho_source_grid', 'sigma_source_grid', 'rho_soure_grid']}
                clean_kwargs_list.append(clean_kw)
            else:
                clean_kwargs_list.append(kw)
        kwargs_list = clean_kwargs_list
    return _original_surface_brightness(self, x, y, kwargs_list, k=k, **kwargs)

LightModel.surface_brightness = _patched_surface_brightness


def _split_scheduler(
    max_iterations,
    init_value=0.01,
    decay_rates=(0.99, 0.99),
    transition_steps=(200, 10),
    boundary=0.5,
):
    boundary = int(max_iterations * boundary)
    scheduler1 = optax.exponential_decay(
        init_value=init_value,
        decay_rate=decay_rates[0],
        transition_steps=transition_steps[0],
    )
    scheduler2 = optax.exponential_decay(
        init_value=scheduler1(boundary),
        decay_rate=decay_rates[1],
        transition_steps=transition_steps[1],
    )
    return optax.join_schedules([scheduler1, scheduler2], boundaries=[boundary])

class TruncatedWedge(dist.Distribution):
    def __init__(self, a, low, b):
        self.a = a
        self.b = b
        self.low = low
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(a),
            jnp.shape(low),
            jnp.shape(b),
        )
        self._support = dist.constraints.interval(low, b)
        self.norm = (self.b - self.a) ** 2 - (self.low - self.a) ** 2
        super().__init__(batch_shape=batch_shape, event_shape=())

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def log_prob(self, value):
        return jnp.log(2) + jnp.log(value - self.a) - jnp.log(self.norm)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        u = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
        return self.a + jnp.sqrt(self.norm * u + (self.low - self.a) ** 2)

class PowerSpectrum:
    class K_grid:
        def __init__(self, shape, scale=1):
            self.Ny, self.Nx = shape
            self.scale = scale

        @lazy_property
        def rk(self):
            kx = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.scale)
            ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.scale)
            return np.sqrt(ky.reshape(-1, 1) ** 2 + kx ** 2)

        @lazy_property
        def k(self):
            kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.scale)
            ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.scale)
            return np.sqrt(ky.reshape(-1, 1) ** 2 + kx ** 2)

    @staticmethod
    @partial(jax.jit, static_argnums=(5,))
    def P_Matern(k, n, sigma, rho, c=1e-20, k_zero=None):
        r = 2 * n / rho ** 2
        norm = sigma ** 2 * 4 * jnp.pi * n * jnp.power(r, n)
        power = norm * jnp.power(r + k ** 2, -(n + 1))
        if k_zero is not None:
            power = jnp.where(k == 0, k_zero, power)
        return power

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _odd_pack(values, n_pix):
        n1 = n_pix // 2 + 1
        thin_real = jax.lax.dynamic_slice(values, (0, 1), (n_pix, n1 - 1))
        thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (n_pix, n1 - 1)), axis=1)

        first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 1, 1))
        first_real = jnp.vstack([
            2 * values[0, 0].reshape(1, 1),
            first_real_slice,
            jnp.flip(first_real_slice, axis=0),
        ])

        first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 1, 1))
        first_imag = jnp.vstack([
            jnp.zeros((1, 1)),
            -jnp.flip(first_imag_slice, axis=0),
            first_imag_slice,
        ])

        fft_real = jnp.hstack([first_real[:thin_real.shape[0]], thin_real])
        fft_imag = jnp.hstack([first_imag[:thin_imag.shape[0]], thin_imag])
        return fft_real + 1j * fft_imag

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _even_pack(values, n_pix):
        n1 = n_pix // 2 + 1
        thin_real = jax.lax.dynamic_slice(values, (0, 1), (n_pix, n1 - 2))
        thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (n_pix, n1 - 2)), axis=1)

        first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 2, 1))
        first_real = jnp.vstack([
            2 * jax.lax.dynamic_slice(values, (0, 0), (1, 1)),
            first_real_slice,
            2 * jax.lax.dynamic_slice(values, (n1 - 1, 0), (1, 1)),
            jnp.flip(first_real_slice, axis=0),
        ])

        last_real_slice = jax.lax.dynamic_slice(values, (1, n1 - 1), (n1 - 2, 1))
        last_real = jnp.vstack([
            2 * jax.lax.dynamic_slice(values, (0, n1 - 1), (1, 1)),
            last_real_slice,
            2 * jax.lax.dynamic_slice(values, (n1 - 1, n1 - 1), (1, 1)),
            jnp.flip(last_real_slice, axis=0),
        ])

        first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 2, 1))
        first_imag = jnp.vstack([
            jnp.zeros((1, 1)),
            -jnp.flip(first_imag_slice, axis=0),
            jnp.zeros((1, 1)),
            first_imag_slice,
        ])

        last_imag_slice = jax.lax.dynamic_slice(values, (n1, n1 - 1), (n1 - 2, 1))
        last_imag = jnp.vstack([
            jnp.zeros((1, 1)),
            -jnp.flip(last_imag_slice, axis=0),
            jnp.zeros((1, 1)),
            last_imag_slice,
        ])

        delta = thin_real.shape[0] - first_real.shape[0]
        first_real = jnp.pad(first_real, ((0, delta), (0, 0)))
        last_real = jnp.pad(last_real, ((0, delta), (0, 0)))
        first_imag = jnp.pad(first_imag, ((0, delta), (0, 0)))
        last_imag = jnp.pad(last_imag, ((0, delta), (0, 0)))

        fft_real = jnp.hstack([first_real, thin_real, last_real])
        fft_imag = jnp.hstack([first_imag, thin_imag, last_imag])
        return fft_real + 1j * fft_imag

    @staticmethod
    @jax.jit
    def pack_fft_values(values):
        ny, nx = values.shape
        assert ny == nx, 'Input array must be square'
        return jax.lax.cond(
            nx % 2 == 0,
            partial(PowerSpectrum._even_pack, n_pix=nx),
            partial(PowerSpectrum._odd_pack, n_pix=nx),
            jnp.sqrt(0.5) * values,
        )

    @staticmethod
    def matern_power_spectrum(
        plate_name,
        param_name,
        k,
        k_zero=None,
        n_high=100,
        n_value=None,
        sigma_low=1e-5,
        sigma_high=10,
        positive=True,
        n_low=0.0001,
    ):
        with numpyro.plate(f'{plate_name} power spectrum params - [1]', 1):
            if n_value is None:
                n = numpyro.sample(
                    f'n_{param_name}',
                    TruncatedWedge(-1, n_low, n_high),
                )
            else:
                n = numpyro.deterministic(f'n_{param_name}', jnp.atleast_1d(n_value))
            sigma = numpyro.sample(f'sigma_{param_name}', dist.LogUniform(sigma_low, sigma_high))
            rho = numpyro.sample(f'rho_{param_name}', dist.LogNormal(2.1, 1.1))

        P = PowerSpectrum.P_Matern(k, n[0], sigma[0], rho[0], k_zero=k_zero)
        scale = jnp.sqrt(P)

        ny, nx = scale.shape
        with numpyro.plate(f'{plate_name} fft y - [{ny}]', ny):
            with numpyro.plate(f'{plate_name} fft x - [{nx}]', nx):
                pixels_wn = numpyro.sample(
                    f'pixels_wn_{param_name}',
                    dist.Normal(0, 1),
                )

        gp = jnp.fft.irfft2(
            PowerSpectrum.pack_fft_values(pixels_wn * scale),
            s=scale.shape,
            norm='ortho',
        )
        if positive:
            gp = jax.nn.softplus(100 * gp) / 100.0
        pixels = numpyro.deterministic(f'pixels_{param_name}', gp)
        return {'pixels': pixels}

    @staticmethod
    def pixels_from_params(params, param_name, k_values, *, positive=True, n_value=None, k_zero=None):
        pixel_key = f'pixels_{param_name}'
        if pixel_key in params:
            return jnp.asarray(params[pixel_key], dtype=jnp.float64)

        if f'n_{param_name}' in params:
            n = jnp.ravel(jnp.asarray(params[f'n_{param_name}']))[0]
        elif n_value is not None:
            n = jnp.asarray(n_value, dtype=jnp.float64)
        else:
            raise KeyError(f'Missing n_{param_name}; provide n_value when n is fixed in the model.')

        sigma = jnp.ravel(jnp.asarray(params[f'sigma_{param_name}']))[0]
        rho = jnp.ravel(jnp.asarray(params[f'rho_{param_name}']))[0]
        pixels_wn = jnp.asarray(params[f'pixels_wn_{param_name}'], dtype=jnp.float64)
        scale = jnp.sqrt(PowerSpectrum.P_Matern(k_values, n, sigma, rho, k_zero=k_zero))
        pixels = jnp.fft.irfft2(PowerSpectrum.pack_fft_values(pixels_wn * scale), s=scale.shape, norm='ortho')
        if positive:
            pixels = jax.nn.softplus(100 * pixels) / 100.0
        return pixels

    @staticmethod
    def params2kwargs_power_spectrum(params, param_name, k_values, *, positive=True, n_value=None, k_zero=None):
        return {
            'pixels': PowerSpectrum.pixels_from_params(
                params,
                param_name,
                k_values,
                positive=positive,
                n_value=n_value,
                k_zero=k_zero,
            )
        }

    @staticmethod
    def fit_power_spectrum_init(
        image,
        k_values,
        pixelated_prior,
        seed,
        max_iterations=30000,
        learning_rate=0.01,
        noise_factor=0.001,
        progress_bar=True,
        param_name='source_grid',
    ):
        image = jnp.asarray(np.asarray(image, dtype=np.float64), dtype=jnp.float64)
        noise_level = max(noise_factor * float(np.nanmax(np.asarray(image))), 1e-6)

        def power_init_model(image_obs):
            source = PowerSpectrum.matern_power_spectrum(
                'Source grid',
                param_name,
                k_values,
                k_zero=pixelated_prior.get('power_init_k_zero', None),
                n_value=pixelated_prior.get('n_value'),
                n_low=safe_float(pixelated_prior.get('n_value_low'), 0.0001),
                n_high=safe_float(pixelated_prior.get('n_value_high'), 100.0),
                sigma_low=safe_float(pixelated_prior.get('sigma_low'), 1e-5),
                sigma_high=safe_float(pixelated_prior.get('sigma_high'), 10.0),
                positive=bool(pixelated_prior.get('positive', True)),
            )
            numpyro.sample('obs', dist.Normal(source['pixels'], noise_level).to_event(2), obs=image_obs)

        import numpyro.infer as infer
        import optax
        import numpyro.infer.autoguide as autoguide
        guide = autoguide.AutoDiagonalNormal(power_init_model, init_loc_fn=infer.init_to_median(num_samples=25))
        scheduler = _split_scheduler(
            max_iterations,
            init_value=learning_rate,
            decay_rates=(0.99, 0.995),
            transition_steps=(100, 25),
        )
        svi = infer.SVI(power_init_model, guide, optax.adabelief(learning_rate=scheduler), infer.TraceMeanField_ELBO())
        result = svi.run(
            jax.random.PRNGKey(seed),
            max_iterations,
            image,
            progress_bar=progress_bar,
            stable_update=True,
        )
        return guide.median(result.params)

def _is_correlated_param(param):
    """
    Flexible correlation syntax:
      ['correlated', component, index, param_name]
    """
    return isinstance(param, (tuple, list)) and len(param) == 4 and param[0] == 'correlated'

def _normalize_link_spec(param):
    """
    Convert a correlation spec into canonical (component, index, key).
    Component names are normalized to lenstronomy constraint groups.
    Returns None if param is not a correlation spec.
    """
    if _is_correlated_param(param):
        _, comp, idx, key = param
        comp_map = {
            'lens': 'lens',
            'lens_mass': 'lens',
            'mass': 'lens',
            'lens_light': 'lens_light',
            'source': 'source',
            'source_light': 'source',
            'point_source': 'point_source',
            'ps': 'point_source',
        }
        if comp not in comp_map:
            raise ValueError(f"Unknown correlated component '{comp}'. Expected one of {sorted(comp_map.keys())}.")
        return (comp_map[comp], int(idx), str(key))

    return None

def _resolve_link(bank, spec, *, context=""):
    """Resolve (component, index, key) against already-built component dicts."""
    comp, idx, key = spec
    if comp not in bank:
        raise ValueError(f"Cannot resolve linked param {spec} ({context}): component '{comp}' not available yet.")
    arr = bank[comp]
    if idx < 0 or idx >= len(arr):
        raise IndexError(
            f"Cannot resolve linked param {spec} ({context}): index {idx} out of range for component '{comp}' "
            f"(len={len(arr)})."
        )
    if key not in arr[idx]:
        raise KeyError(
            f"Cannot resolve linked param {spec} ({context}): key '{key}' not found in {comp}[{idx}]. "
            f"Available keys: {sorted(arr[idx].keys())}"
        )
    return arr[idx][key]


def _kwargs_list_to_jax(kw_list):
    return [{k: jnp.asarray(v) for k, v in comp.items()} for comp in kw_list]


def param_list_to_init_kwargs(param_list, type_list, lens_image):
    kwargs = {}
    
    # 1. Lens mass
    kwargs['kwargs_lens'] = []
    for model in param_list.get('lens_mass_params_list', []):
        kwargs_model = {}
        for k, v in model.items():
            if isinstance(v, list):
                kwargs_model[k] = v[0]
            else:
                kwargs_model[k] = v
        kwargs['kwargs_lens'].append(kwargs_model)
        
    # 2. Lens light
    kwargs['kwargs_lens_light'] = []
    for model in param_list.get('lens_light_params_list', []):
        kwargs_model = {}
        for k, v in model.items():
            if isinstance(v, list):
                kwargs_model[k] = v[0]
            else:
                kwargs_model[k] = v
        kwargs['kwargs_lens_light'].append(kwargs_model)
        
    # 3. Source light
    kwargs['kwargs_source'] = []
    for i, model in enumerate(param_list.get('source_light_params_list', [])):
        src_type = type_list.get('source_light_type_list', [])[i]
        if src_type == 'PIXELATED':
            ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
            kwargs_model = {'pixels': jnp.zeros((ny, nx))}
        else:
            kwargs_model = {}
            for k, v in model.items():
                if isinstance(v, list):
                    kwargs_model[k] = v[0]
                else:
                    kwargs_model[k] = v
        kwargs['kwargs_source'].append(kwargs_model)

    # 4. Point source
    kwargs_point_source = []
    for model in param_list.get('point_source_params_list', []):
        kwargs_model = {}
        for k, v in model.items():
            if isinstance(v, list):
                kwargs_model[k] = v[0]
            else:
                kwargs_model[k] = v
        kwargs_point_source.append(kwargs_model)
    if len(kwargs_point_source) > 0:
        kwargs['kwargs_point_source'] = kwargs_point_source
        
    return kwargs


def _source_support_mask_from_lens(lens_image, kwargs_lens, args=None):
    if kwargs_lens is None or getattr(lens_image, 'source_arc_mask', None) is None:
        return None
    if not np.any(np.asarray(lens_image.source_arc_mask)):
        return None

    percentile = 0.5
    padding = 0
    if args is not None:
        percentile = float(getattr(args, 'source_support_mask_percentile', percentile))
        padding = int(getattr(args, 'source_support_mask_padding', padding))
    percentile = min(max(percentile, 0.0), 49.0)

    npix_src, npix_src_y = lens_image.SourceModel.pixel_grid.num_pixel_axes
    if npix_src_y != npix_src:
        raise ValueError('Source support mask currently requires a square source grid.')

    x_src_axis, y_src_axis, _ = lens_image.get_source_coordinates(
        kwargs_lens,
        npix_src=npix_src,
        source_grid_scale=getattr(lens_image, '_source_grid_scale', 1.0),
    )
    x_src_axis = np.asarray(x_src_axis)
    y_src_axis = np.asarray(y_src_axis)
    if x_src_axis.ndim == 1 and y_src_axis.ndim == 1:
        xx_src, yy_src = np.meshgrid(x_src_axis, y_src_axis)
    else:
        xx_src, yy_src = x_src_axis, y_src_axis

    x_img, y_img = lens_image.ImageNumerics.coordinates_evaluate
    x_ray, y_ray = lens_image.MassModel.ray_shooting(
        x_img,
        y_img,
        kwargs_lens,
    )
    mask_flat = np.asarray(lens_image._source_arc_mask_flat).astype(bool)
    x_masked = np.asarray(x_ray)[mask_flat]
    y_masked = np.asarray(y_ray)[mask_flat]
    finite = np.isfinite(x_masked) & np.isfinite(y_masked)
    x_masked = x_masked[finite]
    y_masked = y_masked[finite]
    if x_masked.size == 0 or y_masked.size == 0:
        return None

    xmin = float(np.nanpercentile(x_masked, percentile))
    xmax = float(np.nanpercentile(x_masked, 100.0 - percentile))
    ymin = float(np.nanpercentile(y_masked, percentile))
    ymax = float(np.nanpercentile(y_masked, 100.0 - percentile))
    support_mask = (
        (xx_src >= xmin) & (xx_src <= xmax)
        & (yy_src >= ymin) & (yy_src <= ymax)
    )
    if padding > 0:
        support_mask = scipy.ndimage.binary_dilation(support_mask, iterations=padding)
    return support_mask.astype(bool)


def create_prob_model(
    param_list,
    type_list,
    lens_image,
    image_data,
    noise_map,
    regul_model=None,
    fix_lens_light=False,
    kwargs_lens_light_fixed=None,
    fix_lens_mass=False,
    kwargs_lens_fixed=None,
    fix_source_light=False,
    kwargs_source_light_fixed=None,
    sample_wavelets=False,
    init_params_path=None,
    args=None,
):
    refine_prior_range = None
    refine_prior_min_frac = None
    if args is not None:
        refine_prior_range = getattr(args, 'refine_prior_range', None)
        refine_prior_min_frac = getattr(args, 'refine_prior_min_frac', None)
    
    if refine_prior_range is not None and init_params_path is None:
        print("[create_prob_model] Warning: refine_prior_range is set, but init_params_path is None. Prior range refinement is skipped.")
    
    if init_params_path is not None and refine_prior_range is not None:
        import os
        import json
        import copy
        
        refine_prior_range = float(refine_prior_range)
        if refine_prior_min_frac is not None:
            refine_prior_min_frac = float(refine_prior_min_frac)
            
        res_path = os.path.join(init_params_path, 'kwargs_result.json')
        sig_path = os.path.join(init_params_path, 'kwargs_sigma.json')
        
        if os.path.exists(res_path) and os.path.exists(sig_path):
            try:
                with open(res_path, 'r') as f:
                    kwargs_result = json.load(f)
                with open(sig_path, 'r') as f:
                    kwargs_sigma = json.load(f)
                
                msg = f"[create_prob_model] Refining parameter prior ranges using {refine_prior_range}-sigma limits"
                if refine_prior_min_frac is not None:
                    msg += f" (with min threshold fraction {refine_prior_min_frac})"
                msg += f" from {init_params_path}"
                print(msg)
                
                param_list = copy.deepcopy(param_list)
                
                mapping = [
                    ('lens_mass_params_list', 'kwargs_lens'),
                    ('lens_light_params_list', 'kwargs_lens_light'),
                    ('source_light_params_list', 'kwargs_source'),
                    ('point_source_params_list', 'kwargs_point_source'),
                ]
                
                for list_key, kw_key in mapping:
                    if list_key in param_list and kw_key in kwargs_result and kw_key in kwargs_sigma:
                        res_list = kwargs_result[kw_key]
                        sig_list = kwargs_sigma[kw_key]
                        for i, comp_model in enumerate(param_list[list_key]):
                            if i < len(res_list) and i < len(sig_list):
                                res_comp = res_list[i]
                                sig_comp = sig_list[i]
                                for p_key, p_val in comp_model.items():
                                    if isinstance(p_val, list) and len(p_val) == 4:
                                        if p_key in res_comp and p_key in sig_comp:
                                            median_val = res_comp[p_key]
                                            sigma_val = sig_comp[p_key]
                                            if isinstance(median_val, (int, float)) and isinstance(sigma_val, (int, float)):
                                                half_width = refine_prior_range * sigma_val
                                                if refine_prior_min_frac is not None:
                                                    min_half_width = refine_prior_min_frac * abs(median_val)
                                                    if half_width < min_half_width:
                                                        half_width = min_half_width
                                                
                                                half_width = max(half_width, 1e-6)
                                                effective_sigma = half_width / refine_prior_range
                                                
                                                low_lim = median_val - half_width
                                                high_lim = median_val + half_width
                                                original_low = p_val[2]
                                                original_high = p_val[3]
                                                
                                                low_lim = max(original_low, low_lim)
                                                high_lim = min(original_high, high_lim)
                                                
                                                if low_lim >= high_lim:
                                                    low_lim = max(original_low, median_val - 0.1 * effective_sigma)
                                                    high_lim = min(original_high, median_val + 0.1 * effective_sigma)
                                                
                                                p_val[2] = float(low_lim)
                                                p_val[3] = float(high_lim)
                                                p_val[0] = float(median_val)
                                                p_val[1] = float(effective_sigma)
                                                
                                                print(f"  {kw_key}[{i}].{p_key}: refined prior to [{p_val[0]:.4f}, {p_val[1]:.4f}, {p_val[2]:.4f}, {p_val[3]:.4f}]")
            except Exception as e:
                print(f"[create_prob_model] Error applying refined prior ranges: {e}")
        else:
            print(f"[create_prob_model] Prior refinement skipped. Result or sigma json file missing in {init_params_path}.")

    noise = Noise(nx=image_data.shape[0], ny=image_data.shape[0], noise_map=noise_map)

    # For wavelet_sparsity prior, we need to initialize the RegularizationModel
    pixelated_prior = {}
    if type_list.get('source_light_type_list') == ['PIXELATED']:
        pixelated_prior = param_list['source_light_params_list'][0].get('pixelated_prior', {})
    prior_type = pixelated_prior.get('prior_type', 'matern')

    sampler_name = getattr(args, 'sampler', None) if args is not None else None
    use_source_support_mask = bool(getattr(args, 'use_source_support_mask', False)) if args is not None else False
    use_source_support_mask_hmc = bool(getattr(args, 'use_source_support_mask_hmc', True)) if args is not None else True
    source_support_mask = None
    if (
        type_list.get('source_light_type_list') == ['PIXELATED']
        and (use_source_support_mask or (sampler_name == 'hmc' and use_source_support_mask_hmc))
    ):
        try:
            support_kwargs = None
            if init_params_path is not None:
                support_kwargs = load_kwargs_init_json(init_params_path)
            else:
                support_kwargs = param_list_to_init_kwargs(param_list, type_list, lens_image)
            source_support_mask = _source_support_mask_from_lens(
                lens_image,
                support_kwargs.get('kwargs_lens', None),
                args=args,
            )
            if source_support_mask is not None:
                active = int(np.sum(source_support_mask))
                total = int(source_support_mask.size)
                print(f"[source_support_mask] Active source pixels: {active}/{total} ({active / total:.1%})")
                save_path = getattr(args, 'save_path', None) if args is not None else None
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, 'source_support_mask.npy'), source_support_mask)
        except Exception as e:
            print(f"[source_support_mask] Warning: failed to build source support mask: {e}")
            source_support_mask = None
    source_support_mask_jax = (
        jnp.asarray(source_support_mask, dtype=jnp.float64)
        if source_support_mask is not None
        else None
    )
    
    regul_weights = None
    starlet = None
    nscales = None
    
    if type_list.get('source_light_type_list') == ['PIXELATED'] and prior_type in ('wavelet_sparsity', 'wavelet_penalty'):
        from herculens.RegulModel.regul_model import RegularizationModel
        
        # 1. Load or estimate kwargs_best
        if init_params_path is not None:
            init_info = load_kwargs_init_json(init_params_path)
            try:
                ks = init_info.get('kwargs_source', [])
                if (
                    isinstance(ks, list) and len(ks) > 0
                    and isinstance(ks[0], dict)
                    and isinstance(ks[0].get('pixels'), dict)
                    and ks[0]['pixels'].get('_format') == 'pixelated_pixels_npy'
                ):
                    init_dir = init_params_path if os.path.isdir(str(init_params_path)) else os.path.dirname(
                        os.path.abspath(str(init_params_path))
                    )
                    npy_name = ks[0]['pixels'].get('file')
                    npy_path = os.path.join(init_dir, npy_name)
                    ks0 = dict(ks[0])
                    ks0['pixels'] = np.load(npy_path)
                    if 'pixels_wn' in ks0 and isinstance(ks0['pixels_wn'], dict) and ks0['pixels_wn'].get('_format') == 'pixelated_pixels_npy':
                        npy_wn_name = ks0['pixels_wn'].get('file')
                        npy_wn_path = os.path.join(init_dir, npy_wn_name)
                        ks0['pixels_wn'] = np.load(npy_wn_path)
                    ks = list(ks)
                    ks[0] = ks0
                    init_info['kwargs_source'] = ks
            except Exception as e:
                print(f"[Wavelet Init] Could not resolve pixelated source stub: {e}")
            kwargs_best = init_info
        else:
            kwargs_best = param_list_to_init_kwargs(param_list, type_list, lens_image)
            
        # 2. Setup noise variance
        try:
            model_image = lens_image.model(**kwargs_best)
            noise_var = lens_image.Noise.C_D_model(model_image)
        except Exception as e:
            print(f"[Wavelet Init] Model-based noise variance failed ({e}); falling back to noise map.")
            noise_var = noise_map ** 2
            
        # 3. Create and initialize RegularizationModel
        if prior_type == 'wavelet_sparsity':
            w_regul_model = RegularizationModel([
                ('source', 0, 'SPARSITY_STARLET_2'),
                ('source', 0, 'POSITIVITY'),
            ])
        else:
            w_regul_model = RegularizationModel([
                ('source', 0, 'SPARSITY_STARLET'),
                ('source', 0, 'SPARSITY_BLWAVELET'),
                ('source', 0, 'POSITIVITY'),
            ])
        
        print(f"[Wavelet Init] Propagating noise to compute wavelet weights ({prior_type})...")
        num_samples = 2000
        if args is not None:
            num_samples = int(getattr(args, 'regul_num_samples', 2000))
        w_regul_model.initialize(
            lens_image, 
            kwargs_best, 
            noise_var=noise_var, 
            num_samples=num_samples
        )
        
        starlet = w_regul_model.method_list[0].transform
        weights_list = w_regul_model.get_weights()
        regul_weights = jnp.asarray(weights_list[0])
        nscales = regul_weights.shape[0] - 1
        
        if prior_type == 'wavelet_penalty':
            regul_model = w_regul_model

    class ProbModel(NumpyroModel):

        def model(self):

            prior_lens_mass = []
            if fix_lens_mass and kwargs_lens_fixed is not None:
                prior_lens_mass = _kwargs_list_to_jax(kwargs_lens_fixed)
            
            bank = {'lens': prior_lens_mass}
            
            if not (fix_lens_mass and kwargs_lens_fixed is not None):
                for i, lens_mass_model in enumerate(param_list['lens_mass_params_list']):
                    model = {}
                    for key, param in lens_mass_model.items():
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            model[key] = _resolve_link(bank, link_spec, context=f"lens_mass[{i}].{key}")
                        elif isinstance(param, list):
                            if key == 'amp':
                                model[key] = numpyro.sample(
                                    f'lens_{key}_{i}',
                                    dist.LogNormal(param[0], param[1]),
                                )
                            else:
                                model[key] = numpyro.sample(
                                    f'lens_{key}_{i}',
                                    dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3]),
                                )
                        else:
                            model[key] = param

                    prior_lens_mass.append(model)

            prior_lens_light = []
            if fix_lens_light and kwargs_lens_light_fixed is not None:
                prior_lens_light = _kwargs_list_to_jax(kwargs_lens_light_fixed)
            
            if 'lens_light_params_list' in param_list:
                bank['lens_light'] = prior_lens_light
                if not (fix_lens_light and kwargs_lens_light_fixed is not None):
                    for i, lens_light_model in enumerate(param_list['lens_light_params_list']):
                        model = {}
                        for key, param in lens_light_model.items():
                            link_spec = _normalize_link_spec(param)
                            if link_spec is not None:
                                model[key] = _resolve_link(bank, link_spec, context=f"lens_light[{i}].{key}")
                            elif isinstance(param, list):
                                if key == 'amp':
                                    model[key] = numpyro.sample(
                                        f'lens_light_{key}_{i}',
                                        dist.LogNormal(param[0], param[1]),
                                    )
                                else:
                                    model[key] = numpyro.sample(
                                        f'lens_light_{key}_{i}',
                                        dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3]),
                                    )
                            else:
                                model[key] = param

                        prior_lens_light.append(model)

            if type_list['source_light_type_list'] == ['PIXELATED']:
                if fix_source_light and kwargs_source_light_fixed is not None:
                    prior_source_light = _kwargs_list_to_jax(kwargs_source_light_fixed)
                else:
                    if prior_type == 'wavelet_sparsity':
                        # Detail scales: dist.Laplace(0, b_scales)
                        lambda0, lambda1 = pixelated_prior.get('regul_strengths', (5.0, 5.0))
                        lambdas = jnp.array([lambda0] + [lambda1]*(nscales-1))[:, None, None]
                        
                        ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
                        mu_scales = jnp.zeros((nscales, ny, nx))
                        
                        # Laplace scale parameter: b = 1 / (lambdas * regul_weights[:-1])
                        b_scales = 1. / (lambdas * regul_weights[:-1] + 1e-12)
                        
                        dist_scales = dist.Laplace(mu_scales, b_scales)
                        source_scales = numpyro.sample('source_scales', dist.Independent(dist_scales, 3))
                        
                        # Coarse scale parameter
                        source_coarse = numpyro.param(
                            'source_coarse', 
                            init_value=1e-2*jnp.ones((ny, nx)), 
                            constraint=constraints.greater_than(0.), 
                            event_dim=2
                        )
                        
                        # Reconstruct source pixels
                        all_coeffs = jnp.concatenate([source_scales, source_coarse[jnp.newaxis, :, :]], axis=0)
                        source_pixels = starlet.reconstruct(all_coeffs)
                        
                        # Enforce positivity if configured
                        if bool(pixelated_prior.get('positive', True)):
                            source_pixels = jax.nn.softplus(100 * source_pixels) / 100.0
                        if source_support_mask_jax is not None:
                            source_pixels = source_pixels * source_support_mask_jax
                            
                        prior_source_light = [{'pixels': source_pixels}]
                    else:
                        ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
                        if prior_type == 'wavelet_penalty':
                            source_pixels = numpyro.param(
                                'source_pixels',
                                init_value=1e-2 * jnp.ones((ny, nx)),
                                constraint=constraints.greater_than(0.),
                                event_dim=2
                            )
                            if source_support_mask_jax is not None:
                                source_pixels = source_pixels * source_support_mask_jax
                            prior_source_light = [{'pixels': source_pixels}]
                        else:
                            k_grid = PowerSpectrum.K_grid((ny, nx))
                            k_values = k_grid.k
                            res = PowerSpectrum.matern_power_spectrum(
                                'Source grid',
                                'source_grid',
                                k_values,
                                k_zero=pixelated_prior.get('k_zero', None),
                                n_value=pixelated_prior.get('n_value', None),
                                n_low=safe_float(pixelated_prior.get('n_value_low'), 0.0001),
                                n_high=safe_float(pixelated_prior.get('n_value_high'), 100.0),
                                sigma_low=safe_float(pixelated_prior.get('sigma_low'), 1e-5),
                                sigma_high=safe_float(pixelated_prior.get('sigma_high'), 10.0),
                                positive=bool(pixelated_prior.get('positive', True)),
                            )
                            if source_support_mask_jax is not None:
                                res = dict(res)
                                res['pixels'] = res['pixels'] * source_support_mask_jax
                            prior_source_light = [{'pixels': res['pixels']}]
            else:
                prior_source_light = []
                if fix_source_light and kwargs_source_light_fixed is not None:
                    prior_source_light = _kwargs_list_to_jax(kwargs_source_light_fixed)
                
                bank['source'] = prior_source_light
                if not (fix_source_light and kwargs_source_light_fixed is not None):
                    for i, source_light_model in enumerate(param_list['source_light_params_list']):
                        model = {}
                        for key, param in source_light_model.items():
                            link_spec = _normalize_link_spec(param)
                            if link_spec is not None:
                                model[key] = _resolve_link(bank, link_spec, context=f"source_light[{i}].{key}")
                            elif isinstance(param, list):
                                if key == 'amp':
                                    model[key] = numpyro.sample(
                                        f'source_{key}_{i}',
                                        dist.LogNormal(param[0], param[1]),
                                    )
                                else:
                                    model[key] = numpyro.sample(
                                        f'source_{key}_{i}',
                                        dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3]),
                                    )
                            else:
                                model[key] = param

                        prior_source_light.append(model)

            prior_point_source = []
            if 'point_source_params_list' in param_list:
                bank['point_source'] = prior_point_source
                for i, point_source_model in enumerate(param_list['point_source_params_list']):
                    ps_type = type_list.get(
                        'point_source_type_list',
                        [None] * len(param_list['point_source_params_list']),
                    )[i]
                    n_img = int(point_source_model.get('n_images', 4)) if isinstance(point_source_model, dict) else 4
                    sigma_image = (
                        float(point_source_model.get('sigma_image', 3e-3))
                        if isinstance(point_source_model, dict)
                        else 3e-3
                    )
                    model = {}
                    for key, param in point_source_model.items():
                        if key in ('n_images', 'sigma_image', 'sigma_source'):
                            continue
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            model[key] = _resolve_link(bank, link_spec, context=f"point_source[{i}].{key}")
                        elif ps_type == 'IMAGE_POSITIONS' and key in ('ra', 'dec'):
                            if isinstance(param, (list, tuple, np.ndarray)) and len(param) == n_img and all(
                                isinstance(v, (int, float, np.floating)) for v in param
                            ):
                                loc = jnp.asarray(param)
                                pos_bound = (
                                    float(point_source_model.get('pos_bound', 0.1))
                                    if isinstance(point_source_model, dict)
                                    else 0.1
                                )
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.TruncatedNormal(
                                        loc=loc,
                                        scale=sigma_image,
                                        low=loc - pos_bound,
                                        high=loc + pos_bound,
                                    ).to_event(1),
                                )
                            else:
                                raise ValueError(
                                    f"For IMAGE_POSITIONS, point_source[{i}].{key} must be a length-{n_img} "
                                    f"list/array of observed image positions."
                                )
                        elif ps_type == 'IMAGE_POSITIONS' and key == 'amp':
                            if isinstance(param, (list, tuple, np.ndarray)) and len(param) == n_img and all(
                                isinstance(v, (int, float, np.floating)) for v in param
                            ):
                                model[key] = jnp.asarray(param)
                            elif isinstance(param, list) and len(param) == 2:
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.LogNormal(param[0], param[1]).expand((n_img,)).to_event(1),
                                )
                            elif isinstance(param, (int, float, np.floating)):
                                model[key] = jnp.ones((n_img,)) * float(param)
                            else:
                                raise ValueError(
                                    f"For IMAGE_POSITIONS, point_source[{i}].amp must be a length-{n_img} "
                                    f"list/array, a scalar, or a LogNormal prior [mu, sigma]."
                                )
                        elif isinstance(param, list):
                            if key == 'amp':
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.LogNormal(param[0], param[1]),
                                )
                            else:
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3]),
                                )
                        else:
                            model[key] = param
                    prior_point_source.append(model)

            model_params = dict(
                kwargs_lens=prior_lens_mass,
                kwargs_source=prior_source_light,
            )

            if len(prior_lens_light) > 0:
                model_params['kwargs_lens_light'] = prior_lens_light

            if len(prior_point_source) > 0:
                model_params['kwargs_point_source'] = prior_point_source

            model_image = lens_image.model(**model_params)
            numpyro.deterministic('model_image', model_image)
            model_var = noise.C_D_model(model_image)
            model_std = jnp.sqrt(model_var)
            obs = jnp.asarray(image_data)
            numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=obs)

            hyperparams = []
            if type_list['source_light_type_list'] == ['PIXELATED']:
                if prior_type == 'wavelet_penalty':
                    lambda_0, lambda_1 = pixelated_prior.get('regul_strengths', (3.0, 3.0))
                    hyperparams = [
                        {'lambda_0': lambda_0, 'lambda_1': lambda_1},
                        {'lambda_0': lambda_0},
                        {'strength': lambda_0},
                    ]
                elif 'factors' in param_list['source_light_params_list'][0]:
                    hyperparams = param_list['source_light_params_list'][0]['factors']

            if regul_model is not None and not sample_wavelets:
                regul_ready = True
                for method in regul_model.method_list:
                    if hasattr(method, 'transform') and method.transform is None:
                        regul_ready = False
                        break
                if regul_ready:
                    numpyro.factor(
                        'source_regul',
                        regul_model.log_prob(model_params, hyperparams),
                    )

            if 'point_source_type_list' in type_list and 'IMAGE_POSITIONS' in type_list['point_source_type_list']:
                sigma_source = 1e-3
                try:
                    for ps in param_list.get('point_source_params_list', []):
                        if isinstance(ps, dict) and 'sigma_source' in ps:
                            sigma_source = float(ps['sigma_source'])
                            break
                except Exception:
                    pass
                numpyro.factor(
                    'ps_source_plane_penalty',
                    lens_image.PointSourceModel.log_prob_source_plane(
                        model_params,
                        sigma_source=sigma_source,
                    ),
                )

        def params2kwargs(self, params):

            kwargs_lens = []
            if fix_lens_mass and kwargs_lens_fixed is not None:
                kwargs_lens = _kwargs_list_to_jax(kwargs_lens_fixed)
            
            bank = {'lens': kwargs_lens}
            
            if not (fix_lens_mass and kwargs_lens_fixed is not None):
                for i, lens_mass_model in enumerate(param_list['lens_mass_params_list']):
                    kw = {}
                    for key, param in lens_mass_model.items():
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs lens_mass[{i}].{key}")
                        elif isinstance(param, list):
                            kw[key] = params[f'lens_{key}_{i}']
                        else:
                            kw[key] = param
                    kwargs_lens.append(kw)

            kwargs_lens_light = []
            if fix_lens_light and kwargs_lens_light_fixed is not None:
                kwargs_lens_light = _kwargs_list_to_jax(kwargs_lens_light_fixed)
            
            if 'lens_light_params_list' in param_list:
                bank['lens_light'] = kwargs_lens_light
                if not (fix_lens_light and kwargs_lens_light_fixed is not None):
                    for i, lens_light_model in enumerate(param_list['lens_light_params_list']):
                        kw = {}
                        for key, param in lens_light_model.items():
                            link_spec = _normalize_link_spec(param)
                            if link_spec is not None:
                                kw[key] = _resolve_link(
                                    bank, link_spec, context=f"params2kwargs lens_light[{i}].{key}"
                                )
                            elif isinstance(param, list):
                                kw[key] = params[f'lens_light_{key}_{i}']
                            else:
                                kw[key] = param
                        kwargs_lens_light.append(kw)

            kwargs_source = []
            if type_list['source_light_type_list'] == ['PIXELATED']:
                if fix_source_light and kwargs_source_light_fixed is not None:
                    kwargs_source = _kwargs_list_to_jax(kwargs_source_light_fixed)
                else:
                    pixelated_prior = param_list['source_light_params_list'][0].get('pixelated_prior', {})
                    if prior_type == 'wavelet_sparsity':
                        source_scales = params['source_scales']
                        source_coarse = params['source_coarse']
                        all_coeffs = jnp.concatenate([source_scales, source_coarse[jnp.newaxis, :, :]], axis=0)
                        source_pixels = starlet.reconstruct(all_coeffs)
                        if bool(pixelated_prior.get('positive', True)):
                            source_pixels = jax.nn.softplus(100 * source_pixels) / 100.0
                        if source_support_mask_jax is not None:
                            source_pixels = source_pixels * source_support_mask_jax
                        kwargs_source = [{'pixels': source_pixels}]
                    else:
                        if prior_type == 'wavelet_penalty':
                            source_pixels = params['source_pixels']
                            if source_support_mask_jax is not None:
                                source_pixels = source_pixels * source_support_mask_jax
                            kwargs_source = [{'pixels': source_pixels}]
                        else:
                            ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
                            k_grid = PowerSpectrum.K_grid((ny, nx))
                            k_values = k_grid.k
                            source_pixels = PowerSpectrum.pixels_from_params(
                                params,
                                'source_grid',
                                k_values,
                                k_zero=pixelated_prior.get('k_zero', None),
                                n_value=pixelated_prior.get('n_value', None),
                                positive=bool(pixelated_prior.get('positive', True)),
                            )
                            if source_support_mask_jax is not None:
                                source_pixels = source_pixels * source_support_mask_jax
                            n_val = params.get('n_source_grid', pixelated_prior.get('n_value', None))
                            rho_val = params.get('rho_source_grid', None)
                            sigma_val = params.get('sigma_source_grid', None)
                            
                            if n_val is not None:
                                n_val = jnp.ravel(jnp.asarray(n_val))[0]
                            if rho_val is not None:
                                rho_val = jnp.ravel(jnp.asarray(rho_val))[0]
                            if sigma_val is not None:
                                sigma_val = jnp.ravel(jnp.asarray(sigma_val))[0]
                                
                            kwargs_source = [{
                                'pixels': source_pixels,
                                'pixels_wn': params.get('pixels_wn_source_grid'),
                                'n_source_grid': n_val,
                                'rho_source_grid': rho_val,
                                'sigma_source_grid': sigma_val,
                            }]
            else:
                kwargs_source = []
                if fix_source_light and kwargs_source_light_fixed is not None:
                    kwargs_source = _kwargs_list_to_jax(kwargs_source_light_fixed)
                
                bank['source'] = kwargs_source
                if not (fix_source_light and kwargs_source_light_fixed is not None):
                    for i, source_light_model in enumerate(param_list['source_light_params_list']):
                        kw = {}
                        for key, param in source_light_model.items():
                            link_spec = _normalize_link_spec(param)
                            if link_spec is not None:
                                kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs source_light[{i}].{key}")
                            elif isinstance(param, list):
                                kw[key] = params[f'source_{key}_{i}']
                            else:
                                kw[key] = param
                        kwargs_source.append(kw)

            kwargs_point_source = []
            if 'point_source_params_list' in param_list:
                bank['point_source'] = kwargs_point_source
                for i, point_source_model in enumerate(param_list['point_source_params_list']):
                    ps_type = type_list.get(
                        'point_source_type_list',
                        [None] * len(param_list['point_source_params_list']),
                    )[i]
                    kw = {}
                    for key, param in point_source_model.items():
                        if key in ('n_images', 'sigma_image', 'sigma_source'):
                            continue
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs point_source[{i}].{key}")
                        elif ps_type == 'IMAGE_POSITIONS' and key in ('ra', 'dec', 'amp'):
                            kw[key] = params[f'ps_{key}_{i}']
                        elif isinstance(param, list):
                            kw[key] = params[f'ps_{key}_{i}']
                        else:
                            kw[key] = param
                    kwargs_point_source.append(kw)

            kw_model = {
                'kwargs_lens': kwargs_lens,
                'kwargs_source': kwargs_source,
            }

            if len(kwargs_lens_light) > 0:
                kw_model['kwargs_lens_light'] = kwargs_lens_light

            if len(kwargs_point_source) > 0:
                kw_model['kwargs_point_source'] = kwargs_point_source

            return kw_model

    model_instance = ProbModel()
    model_instance.prior_type = prior_type
    model_instance.starlet = starlet
    model_instance.regul_weights = regul_weights
    model_instance.nscales = nscales
    model_instance.lens_image = lens_image
    model_instance.image_data = image_data
    model_instance.noise_map = noise_map
    model_instance.param_list = param_list
    model_instance.type_list = type_list
    model_instance.source_support_mask = source_support_mask
    model_instance.source_support_mask_jax = source_support_mask_jax
    p_scale = 0.08
    try:
        p_scale = float(lens_image.Grid.pixel_width)
    except Exception:
        pass
    if args is not None:
        model_instance.pixel_scale = getattr(args, 'pixel_scale', p_scale)
    else:
        model_instance.pixel_scale = p_scale
    return model_instance


def load_kwargs_init_json(init_params_path):
    from herculens_wrapper.utils import resolve_project_path

    if init_params_path is None:
        raise ValueError("init_params_path is required.")
    path = resolve_project_path(init_params_path)
    if os.path.isdir(path):
        path = os.path.join(path, "kwargs_result.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"kwargs init file not found: {path!r}")
    with open(path) as f:
        return json.load(f)



def _infer_analytic_source_light_types(kwargs_source):
    """Infer LightModel profile names from parameterized kwargs_source entries."""
    types = []
    for i, kw in enumerate(kwargs_source):
        if not isinstance(kw, dict):
            raise TypeError(f'kwargs_source[{i}] must be a dict, got {type(kw).__name__}.')
        if 'pixels' in kw:
            raise ValueError(
                "kwargs_source contains 'pixels'; load the pixel .npy stub directly."
            )
        if any(k in kw for k in ('R_sersic', 'n_sersic')):
            types.append('SERSIC_ELLIPSE')
        elif 'sigma' in kw:
            if 'e1' in kw or 'e2' in kw:
                types.append('GAUSSIAN_ELLIPSE')
            else:
                types.append('GAUSSIAN')
        else:
            raise ValueError(
                f'Cannot infer analytic source type for kwargs_source[{i}] '
                f'(keys: {sorted(kw.keys())}). Add "source_light_type_list" to the init JSON.'
            )
    return types


def _project_analytic_kwargs_to_pixel_source(
    lens_image,
    kwargs_source_analytic,
    *,
    source_light_type_list=None,
):
    """
    Initialize pixel values from analytic kwargs_source on the PIXELATED grid.

    Surface brightness is evaluated on the source pixel grid and scaled by the
    image-grid pixel area (Herculens convention for pixelated sources).
    """
    if source_light_type_list is None:
        source_light_type_list = _infer_analytic_source_light_types(kwargs_source_analytic)
    if len(source_light_type_list) != len(kwargs_source_analytic):
        raise ValueError(
            f'source_light_type_list length {len(source_light_type_list)} does not match '
            f'len(kwargs_source)={len(kwargs_source_analytic)}.'
        )
    lm = LightModel(source_light_type_list)
    x_src, y_src = lens_image.SourceModel.pixel_grid.pixel_coordinates
    sb = lm.surface_brightness(x_src, y_src, kwargs_source_analytic)
    pa = jnp.asarray(lens_image.Grid.pixel_area, dtype=jnp.float64)
    return jnp.asarray(sb, dtype=jnp.float64) * pa


def kwargs2params(
    param_list,
    kwargs,
    type_list=None,
    fix_lens_light=False,
    fix_lens_mass=False,
    fix_source_light=False,
    sample_wavelets=False,
    starlet_method=None,
):
    params = {}
    if not fix_lens_mass:
        for i, lens_mass_model in enumerate(param_list['lens_mass_params_list']):
            for key, param in lens_mass_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    if 'kwargs_lens' not in kwargs or i >= len(kwargs['kwargs_lens']):
                        continue
                    if key not in kwargs['kwargs_lens'][i]:
                        continue
                    params[f'lens_{key}_{i}'] = jnp.asarray(kwargs['kwargs_lens'][i][key])
    if (
        not fix_lens_light
        and 'lens_light_params_list' in param_list
        and 'kwargs_lens_light' in kwargs
    ):
        for i, lens_light_model in enumerate(param_list['lens_light_params_list']):
            for key, param in lens_light_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    if i >= len(kwargs['kwargs_lens_light']):
                        continue
                    if key not in kwargs['kwargs_lens_light'][i]:
                        continue
                    params[f'lens_light_{key}_{i}'] = jnp.asarray(kwargs['kwargs_lens_light'][i][key])
    
    if type_list is not None and type_list.get('source_light_type_list') == ['PIXELATED']:
        pass
    else:
        if not fix_source_light and 'source_light_params_list' in param_list:
            for i, source_light_model in enumerate(param_list['source_light_params_list']):
                for key, param in source_light_model.items():
                    if _normalize_link_spec(param) is None and isinstance(param, list):
                        if i >= len(kwargs.get('kwargs_source', [])):
                            continue
                        if key not in kwargs['kwargs_source'][i]:
                            continue
                        params[f'source_{key}_{i}'] = jnp.asarray(kwargs['kwargs_source'][i][key])
    if 'point_source_params_list' in param_list and 'kwargs_point_source' in kwargs:
        ps_type_list = [None] * len(param_list['point_source_params_list'])
        for i, point_source_model in enumerate(param_list['point_source_params_list']):
            ps_type = ps_type_list[i]
            if isinstance(type_list, dict):
                ps_type = type_list.get('point_source_type_list', ps_type_list)[i]
            for key, param in point_source_model.items():
                if key in ('n_images', 'sigma_image', 'sigma_source'):
                    continue
                if _normalize_link_spec(param) is not None:
                    continue
                if isinstance(param, list):
                    if i >= len(kwargs['kwargs_point_source']) or key not in kwargs['kwargs_point_source'][i]:
                        continue
                    params[f'ps_{key}_{i}'] = jnp.asarray(kwargs['kwargs_point_source'][i][key])
    return params


def get_init_params(
    prob_model,
    param_list,
    type_list,
    init_params_path=None,
    random_seed=42,
    fix_lens_light=False,
    fix_lens_mass=False,
    fix_source_light=False,
    lens_image=None,
    pixel_init_jitter=0.0,
    sample_wavelets=False,
    regul_model=None,
):
    """
    Return constrained NumPyro site parameters (physical values).

    Pass through ``to_unconstrained()`` before optax/jaxopt/HMC/emcee.
    For PIXELATED sources, pass ``lens_image`` so analytic kwargs_result.json
    files can be projected onto the source pixel grid.
    ``pixel_init_jitter`` adds relative Gaussian noise to ``source_pixels`` after
    loading/projection (helps NUTS escape a overly sharp local mode).
    """
    key_init = jax.random.PRNGKey(random_seed)
    init_params = prob_model.get_sample(key_init)

    if init_params_path is not None:
        init_dir = init_params_path if os.path.isdir(str(init_params_path)) else os.path.dirname(
            os.path.abspath(str(init_params_path))
        )
        init_info = load_kwargs_init_json(init_params_path)
        print(f"[Init] Loading kwargs from prior run: {init_dir}")
        try:
            ks = init_info.get('kwargs_source', [])
            if (
                isinstance(ks, list) and len(ks) > 0
                and isinstance(ks[0], dict)
                and isinstance(ks[0].get('pixels'), dict)
                and ks[0]['pixels'].get('_format') == 'pixelated_pixels_npy'
            ):
                npy_name = ks[0]['pixels'].get('file')
                npy_path = os.path.join(init_dir, npy_name)
                ks0 = dict(ks[0])
                ks0['pixels'] = np.load(npy_path)
                if 'pixels_wn' in ks0 and isinstance(ks0['pixels_wn'], dict) and ks0['pixels_wn'].get('_format') == 'pixelated_pixels_npy':
                    npy_wn_name = ks0['pixels_wn'].get('file')
                    npy_wn_path = os.path.join(init_dir, npy_wn_name)
                    ks0['pixels_wn'] = np.load(npy_wn_path)
                ks = list(ks)
                ks[0] = ks0
                init_info['kwargs_source'] = ks
        except Exception as e:
            print(f"[Init] Could not resolve pixelated source stub: {e}")

        if isinstance(init_info, dict) and 'kwargs_lens' in init_info:
            loaded_params = kwargs2params(
                param_list, init_info, type_list=type_list, fix_lens_light=fix_lens_light,
                fix_lens_mass=fix_lens_mass, fix_source_light=fix_source_light,
                sample_wavelets=sample_wavelets, starlet_method=None
            )
            src_types = type_list.get('source_light_type_list', [])
            if src_types == ['PIXELATED'] and not fix_source_light:
                ks = init_info.get('kwargs_source', [])
                if ks and isinstance(ks[0], dict) and 'pixels' in ks[0]:
                    has_pixel_array = not isinstance(ks[0]['pixels'], dict)
                    if has_pixel_array:
                        pixels_proj = jnp.asarray(ks[0]['pixels'], dtype=jnp.float64)
                    else:
                        pixels_proj = np.load(os.path.join(init_dir, ks[0]['pixels'].get('file')))
                    
                    prior_type = getattr(prob_model, 'prior_type', 'matern')
                    if prior_type == 'wavelet_sparsity':
                        starlet = prob_model.starlet
                        print("[Init] Decomposing target source image into Starlet space...")
                        coeffs = starlet.decompose(pixels_proj)
                        loaded_params['source_scales'] = coeffs[:-1]
                        loaded_params['source_coarse'] = coeffs[-1]
                    elif prior_type == 'wavelet_penalty':
                        print("[Init] Loading target source image directly for wavelet_penalty...")
                        loaded_params['source_pixels'] = pixels_proj
                    else:
                        ks0 = ks[0]
                        if 'pixels_wn' in ks0 and ks0['pixels_wn'] is not None:
                            print("[Init] Loading Matérn power spectrum parameters and pixels_wn directly from prior run...")
                            loaded_params['pixels_wn_source_grid'] = jnp.asarray(ks0['pixels_wn'], dtype=jnp.float64)
                            loaded_params['n_source_grid'] = jnp.asarray(ks0['n_source_grid'], dtype=jnp.float64)
                            loaded_params['rho_source_grid'] = jnp.asarray(ks0['rho_source_grid'], dtype=jnp.float64)
                            loaded_params['sigma_source_grid'] = jnp.asarray(ks0['sigma_source_grid'], dtype=jnp.float64)
                else:
                    print("[Init] Prior run was parametric. Source light parameters (pixels_wn, n, rho, sigma) will be randomly sampled from their prior distributions.")

            n_matched = 0
            n_skipped = 0
            for k, v in loaded_params.items():
                if k not in init_params:
                    n_skipped += 1
                    continue
                v_arr = jnp.asarray(v)
                ref_arr = jnp.asarray(init_params[k])
                if v_arr.shape != ref_arr.shape:
                    if v_arr.size == ref_arr.size:
                        v_arr = jnp.reshape(v_arr, ref_arr.shape)
                    else:
                        n_skipped += 1
                        continue
                init_params[k] = v_arr
                n_matched += 1
            print(f"[Init] Inherited kwargs from prior run: matched={n_matched}, skipped={n_skipped}")
        else:
            init_params = {k: jnp.asarray(v) for k, v in init_info.items()}

    for k, v in list(init_params.items()):
        if '_amp_' not in k:
            continue
        arr = jnp.asarray(v)
        init_params[k] = jnp.where(arr == 0.0, 1e-8, arr)

    jitter = float(pixel_init_jitter)
    if jitter > 0.0:
        if 'pixels_wn_source_grid' in init_params:
            key_pix = jax.random.PRNGKey(int(random_seed) + 99)
            wn = init_params['pixels_wn_source_grid']
            noise = jitter * jax.random.normal(key_pix, wn.shape)
            init_params['pixels_wn_source_grid'] = wn + noise
            print(f'[Init] Applied pixel_init_jitter={jitter} to pixels_wn_source_grid')

    if fix_lens_light:
        init_params = {
            k: v for k, v in init_params.items() if not k.startswith('lens_light_')
        }
    if fix_lens_mass:
        init_params = {
            k: v for k, v in init_params.items() if not (k.startswith('lens_') and not k.startswith('lens_light_'))
        }
    if fix_source_light:
        init_params = {
            k: v for k, v in init_params.items() if not k.startswith('source_')
        }

    init_params = {
        k: v for k, v in init_params.items() if k != 'pixels_source_grid'
    }

    return init_params


def resolve_fixed_kwargs(init_params_path, component):
    info = load_kwargs_init_json(init_params_path)
    if component == 'lens_mass':
        key = 'kwargs_lens'
    elif component == 'lens_light':
        key = 'kwargs_lens_light'
    elif component == 'source_light':
        key = 'kwargs_source'
    else:
        raise ValueError(f"Unknown component to fix: {component}")
        
    fixed = info.get(key)
    if not fixed:
        raise ValueError(
            f"Fixing {component} requires {key!r} in the init kwargs file at {init_params_path!r}."
        )
        
    if component == 'source_light' and isinstance(fixed, list) and len(fixed) > 0:
        kw = fixed[0]
        if isinstance(kw, dict):
            import numpy as np
            init_dir = init_params_path if os.path.isdir(str(init_params_path)) else os.path.dirname(
                os.path.abspath(str(init_params_path))
            )
            fixed_copy = list(fixed)
            fixed_copy[0] = dict(kw)
            
            if isinstance(kw.get('pixels'), dict) and kw['pixels'].get('_format') == 'pixelated_pixels_npy':
                npy_name = kw['pixels'].get('file')
                npy_path = os.path.join(init_dir, npy_name)
                fixed_copy[0]['pixels'] = np.load(npy_path)
                
            if isinstance(kw.get('pixels_wn'), dict) and kw['pixels_wn'].get('_format') == 'pixelated_pixels_npy':
                npy_wn_name = kw['pixels_wn'].get('file')
                npy_wn_path = os.path.join(init_dir, npy_wn_name)
                fixed_copy[0]['pixels_wn'] = np.load(npy_wn_path)
                
            fixed = fixed_copy
            
    return fixed


def create_pixel_grids(npix, pix_scl):
    half_size = npix * pix_scl / 2
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2
    transform_pix2angle = pix_scl * np.eye(2)
    pixel_grid = PixelGrid(
        nx=npix, ny=npix,
        ra_at_xy_0=ra_at_xy_0, dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=transform_pix2angle,
    )
    ps_grid_npix = 2 * npix + 1
    ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix
    ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.
    ps_grid = PixelGrid(
        nx=ps_grid_npix, ny=ps_grid_npix,
        ra_at_xy_0=-ps_grid_half_size + ps_grid_pix_scl / 2.,
        dec_at_xy_0=-ps_grid_half_size + ps_grid_pix_scl / 2.,
        transform_pix2angle=ps_grid_pix_scl * np.eye(2),
    )
    return pixel_grid, ps_grid

class LensImageExtension(LensImage):
    def __init__(
        self,
        grid_class,
        psf_class,
        noise_class=None,
        lens_mass_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_model_class=None,
        source_arc_mask=None,
        source_grid_scale=1.0,
        conjugate_points=None,
        kwargs_numerics=None,
        kwargs_lens_equation_solver=None,
    ):
        super().__init__(
            grid_class,
            psf_class,
            noise_class=noise_class,
            lens_mass_model_class=lens_mass_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            point_source_model_class=point_source_model_class,
            source_arc_mask=source_arc_mask,
            kwargs_numerics=kwargs_numerics,
            kwargs_lens_equation_solver=kwargs_lens_equation_solver,
        )
        self._source_grid_scale = source_grid_scale
        self.conjugate_points = conjugate_points

        ssf = self.ImageNumerics.grid_supersampling_factor
        s_ones = np.ones([ssf, ssf])
        if source_arc_mask is None:
            nx, ny = grid_class.num_pixel_axes
            self.source_arc_mask = np.ones([nx, ny], dtype=bool)
        else:
            self.source_arc_mask = source_arc_mask
        self.source_arc_mask_ss = np.kron(self.source_arc_mask, s_ones)
        self._source_arc_mask_flat = self.source_arc_mask_ss.flatten()
        self._source_arc_mask_outline_flat = (
            self.source_arc_mask_ss - scipy.ndimage.binary_erosion(self.source_arc_mask_ss)
        ).flatten().astype(bool)

    def source_surface_brightness(
        self,
        kwargs_source,
        kwargs_lens=None,
        de_lensed=False,
        k=None,
        k_lens=None,
    ):
        if len(self.SourceModel.profile_type_list) == 0:
            return jnp.zeros(self.Grid.num_pixel_axes)

        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        if (self._src_adaptive_grid) or (not de_lensed):
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                x_grid_img,
                y_grid_img,
                kwargs_lens,
                k=k_lens,
            )
            pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
                x_grid_src,
                y_grid_src,
            )
        else:
            pixels_x_coord, pixels_y_coord = None, None
        if de_lensed:
            source_light = self.SourceModel.surface_brightness(
                x_grid_img,
                y_grid_img,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord,
            )
        else:
            source_light = self.SourceModel.surface_brightness(
                x_grid_src,
                y_grid_src,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord,
            )
        return source_light

    def lens_surface_brightness(self, kwargs_lens_light, k=None):
        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        return self.LensLightModel.surface_brightness(
            x_grid_img,
            y_grid_img,
            kwargs_lens_light,
            k=k,
        )

    @partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    def model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_point_source=None,
        unconvolved=False,
        supersampled=False,
        source_add=True,
        lens_light_add=True,
        point_source_add=True,
        k_lens=None,
        k_source=None,
        k_lens_light=None,
        k_point_source=None,
        kwargs_psf=None,
    ):
        model = jnp.zeros((self.ImageNumerics.grid_class.num_grid_points,)).flatten()
        if source_add is True:
            source_model = self.source_surface_brightness(
                kwargs_source,
                kwargs_lens,
                k=k_source,
                k_lens=k_lens,
            )
            if self._source_arc_mask_flat is not None:
                source_model *= self._source_arc_mask_flat
            model += source_model
        if lens_light_add is True:
            model += self.lens_surface_brightness(
                kwargs_lens_light,
                k=k_lens_light,
            )
        if not supersampled:
            model = self.ImageNumerics.re_size_convolve(
                model,
                unconvolved=unconvolved,
                kwargs_psf=kwargs_psf,
            )
        if point_source_add and getattr(self, 'PointSourceModel', None) is not None:
            ps_image = self.point_source_image(
                kwargs_point_source,
                kwargs_lens,
                kwargs_solver=self.kwargs_lens_equation_solver,
                k=k_point_source,
                kwargs_psf=kwargs_psf,
            )
            if model.ndim == 1:
                ps_image = ps_image.flatten()
            model += ps_image
        return model

    def trace_conjugate_points(self, kwargs_lens, k_lens=None):
        if self.conjugate_points is not None:
            x, y = self.conjugate_points.T
            conj_x, conj_y = self.MassModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
            return jnp.vstack([conj_x, conj_y]).T
        return None

    def mask_extent(self, x_grid_src, y_grid_src, npix_src, grid_scale=1):
        x_left, x_right = x_grid_src.min(), x_grid_src.max()
        y_bottom, y_top = y_grid_src.min(), y_grid_src.max()
        cx = 0.5 * (x_left + x_right)
        cy = 0.5 * (y_bottom + y_top)
        width = jnp.abs(x_left - x_right)
        height = jnp.abs(y_bottom - y_top)
        half_size = 0.5 * grid_scale * jnp.maximum(height, width)
        x_left = cx - half_size
        x_right = cx + half_size
        y_bottom = cy - half_size
        y_top = cy + half_size
        x_adapt = jnp.linspace(x_left, x_right, npix_src)
        y_adapt = jnp.linspace(y_bottom, y_top, npix_src)
        extent_adapt = [x_adapt[0], x_adapt[-1], y_adapt[0], y_adapt[-1]]
        return x_adapt, y_adapt, extent_adapt

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def adapt_source_coordinates(
        self,
        x_grid_src,
        y_grid_src,
        force=False,
        npix_src=100,
        source_grid_scale=1,
    ):
        if self._src_adaptive_grid or force:
            if not force:
                npix_src, npix_src_y = self.SourceModel.pixel_grid.num_pixel_axes
                if npix_src_y != npix_src:
                    raise ValueError('Adaptive source plane grid only works with square grids')
                grid_scale = self._source_grid_scale
            else:
                grid_scale = source_grid_scale
            if self.Grid.x_is_inverted or self.Grid.y_is_inverted:
                raise NotImplementedError('invert x and y not yet supported for adaptive source grid')
            return self.mask_extent(
                x_grid_src[self._source_arc_mask_outline_flat],
                y_grid_src[self._source_arc_mask_outline_flat],
                npix_src,
                grid_scale,
            )
        return None, None, None

    def get_source_coordinates(
        self,
        kwargs_lens,
        force=False,
        npix_src=100,
        source_grid_scale=1.0,
        k_lens=None,
    ):
        if (not self._src_adaptive_grid) and (self.SourceModel.pixel_grid is not None):
            x_grid, y_grid = self.SourceModel.pixel_grid.pixel_coordinates
            extent = self.SourceModel.pixel_grid.extent
        else:
            x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                x_grid_img,
                y_grid_img,
                kwargs_lens,
                k=k_lens,
            )
            x_grid, y_grid, extent = self.adapt_source_coordinates(
                x_grid_src,
                y_grid_src,
                force=force,
                npix_src=npix_src,
                source_grid_scale=source_grid_scale,
            )
        return x_grid, y_grid, extent



def get_best_pixel_size(lens_image, herc_dict, source_grid_scale, return_full=False):
    from sklearn.neighbors import NearestNeighbors

    x_ss_grid, y_ss_grid = lens_image.ImageNumerics.coordinates_evaluate
    mask = lens_image._source_arc_mask_flat.astype(bool)
    x_ss_trace, y_ss_trace = lens_image.MassModel.ray_shooting(
        x_ss_grid[mask].flatten(),
        y_ss_grid[mask].flatten(),
        herc_dict['kwargs_lens'],
    )

    _, _, extent = lens_image.mask_extent(
        x_ss_trace,
        y_ss_trace,
        100,
        grid_scale=source_grid_scale,
    )

    full_size = jax.device_get(extent[1] - extent[0])
    tdx = (
        (x_ss_trace >= extent[0])
        & (x_ss_trace <= extent[1])
        & (y_ss_trace >= extent[2])
        & (y_ss_trace <= extent[3])
    )

    jax.block_until_ready(x_ss_trace)
    jax.block_until_ready(y_ss_trace)
    x_trim = jax.device_get(x_ss_trace[tdx])
    y_trim = jax.device_get(y_ss_trace[tdx])
    samples = np.vstack([x_trim, y_trim]).T
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(samples)
    distances, _ = nbrs.kneighbors(samples)

    mean_distance = np.mean(distances[:, 1])
    pixel_grid_shape = int(full_size / (5 * mean_distance) + 1)
    if return_full:
        return pixel_grid_shape, x_trim, y_trim, mean_distance, full_size
    return pixel_grid_shape


def create_lens_image(
    param_list,
    type_list,
    image_data,
    noise_map,
    psf_data,
    pixel_scale,
    kwargs_numerics=None,
    kwargs_lens_equation_solver=None,
    source_arc_mask=None,
    source_grid_scale=1.0,
    conjugate_points=None,
):
    num_pixels = image_data.shape[0]
    psf = PSF(psf_type='PIXEL', kernel_point_source=psf_data, pixel_size=pixel_scale)
    noise = Noise(nx=num_pixels, ny=num_pixels, noise_map=noise_map)
    pixel_grid, ps_grid = create_pixel_grids(num_pixels, pixel_scale)

    lens_mass_model = MassModel(type_list['lens_mass_type_list'])
    lens_light_model = LightModel(type_list['lens_light_type_list'])
    src_types = type_list['source_light_type_list']
    if src_types == ['PIXELATED']:
        kwargs_pixelated = param_list['source_light_params_list'][0]
        source_kwargs_pixelated = kwargs_pixelated.get('pixel_grid', kwargs_pixelated)
        pixel_adaptive_grid = source_kwargs_pixelated.get('pixel_adaptive_grid', False)
        if pixel_adaptive_grid:
            pixel_grid_shape = int(source_kwargs_pixelated.get('pixel_grid_shape', 100))
            source_light_model = LightModel(
                src_types,
                pixel_adaptive_grid=True,
                pixel_interpol=source_kwargs_pixelated.get('pixel_interpol', 'fast_bilinear'),
                kwargs_pixelated={'num_pixels': pixel_grid_shape}
            )
        else:
            source_light_model = LightModel(src_types, kwargs_pixelated=source_kwargs_pixelated)
    else:
        source_light_model = LightModel(src_types)

    point_source_model = None
    if type_list.get('point_source_type_list'):
        point_source_model = PointSourceModel(
            type_list['point_source_type_list'], lens_mass_model, ps_grid
        )

    if kwargs_numerics is None:
        kwargs_numerics = {'supersampling_factor': 1}

    return LensImageExtension(
        grid_class=pixel_grid,
        psf_class=psf,
        noise_class=noise,
        lens_mass_model_class=lens_mass_model,
        lens_light_model_class=lens_light_model,
        source_model_class=source_light_model,
        point_source_model_class=point_source_model,
        source_arc_mask=source_arc_mask,
        source_grid_scale=source_grid_scale,
        conjugate_points=conjugate_points,
        kwargs_numerics=kwargs_numerics,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver,
    )


def validate_param_list(type_list, param_list):
    if not isinstance(type_list, dict) or not isinstance(param_list, dict):
        raise TypeError("type_list and param_list must be dicts.")
    pairs = (
        ("lens_mass_type_list", "lens_mass_params_list"),
        ("lens_light_type_list", "lens_light_params_list"),
        ("source_light_type_list", "source_light_params_list"),
        ("point_source_type_list", "point_source_params_list"),
    )
    for type_key, param_key in pairs:
        has_t = type_key in type_list
        has_p = param_key in param_list
        if has_t != has_p:
            raise ValueError(
                f"type_list and param_list must both contain '{type_key}' and '{param_key}'."
            )
        if has_t and len(type_list[type_key]) != len(param_list[param_key]):
            raise ValueError(f"Length mismatch for {type_key} / {param_key}.")
