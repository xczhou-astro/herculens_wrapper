import filecmp
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from herculens_wrapper import HERCULENS_PKG, PROJECT_ROOT, WRAPPER_DIR


SAMPLER_CHOICES = frozenset({
    'svi',
    'optax',
    'hmc',
})


OPTIMIZATION_SAMPLERS = frozenset({'svi', 'optax'})
MCMC_SAMPLERS = frozenset({'hmc'})
HMC_NUMPYRO_CHAIN_METHODS = frozenset({'auto', 'parallel', 'vectorized', 'sequential'})


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        """Provide the file-like close API without closing shared stdio streams."""
        self.flush()



def json_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import jax
        if isinstance(obj, jax.Array):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def save_config(args, save_path, *, filename='config.json'):
    """Save argparse arguments (or a mapping) as a JSON configuration file.

    ``save_path`` may be a directory, in which case ``filename`` is used, or
    an explicit ``.json`` path.  ``Path``, NumPy, and JAX values are converted
    through :func:`json_serializer`.
    """
    if isinstance(args, Mapping):
        payload = dict(args)
    elif hasattr(args, '__dict__'):
        payload = vars(args).copy()
    else:
        raise TypeError("args must be an argparse.Namespace, SimpleNamespace, or mapping.")
    output = Path(save_path).expanduser()
    if output.suffix.lower() != '.json':
        output = output / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=4, default=json_serializer)
    return output


def center_crop(image, crop_size):
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size
    h, w = image.shape[:2]
    start_y = max((h - crop_h) // 2, 0)
    start_x = max((w - crop_w) // 2, 0)
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]


def get_fits_data(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data.astype(np.float64)


def sanitize_noise_map(noise_map, fill_value=None):
    """
    Ensures noise_map is finite and strictly positive (> 0), replacing invalid values
    (NaN, Inf, <= 0) with a positive noise value (median of valid positive values, or fill_value).
    """
    noise_map = np.asarray(noise_map, dtype=np.float64)
    invalid_mask = ~np.isfinite(noise_map) | (noise_map <= 0.0)
    if np.any(invalid_mask):
        valid_vals = noise_map[np.isfinite(noise_map) & (noise_map > 0.0)]
        if fill_value is not None:
            fallback = float(fill_value)
        elif len(valid_vals) > 0:
            fallback = float(np.median(valid_vals))
        else:
            fallback = 1.0
        n_invalid = int(np.sum(invalid_mask))
        print(f"[noise] Warning: Found {n_invalid} invalid/non-positive pixel(s) in noise map. "
              f"Replacing them with positive noise value ({fallback:.6e}).")
        noise_map = np.where(invalid_mask, fallback, noise_map)
    return noise_map


def sanitize_image_data(image_data):
    """
    Ensures image_data has no NaN or Inf values by replacing them with 0.0.
    Negative values are left untouched as they are valid noisy sky pixels.
    """
    image_data = np.asarray(image_data, dtype=np.float64)
    invalid_mask = ~np.isfinite(image_data)
    if np.any(invalid_mask):
        n_invalid = int(np.sum(invalid_mask))
        print(f"[data] Warning: Found {n_invalid} NaN/Inf pixel(s) in image data. Replacing them with 0.0.")
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    return image_data


def exclude_bad_pixels(image_data, noise_map, excluded_noise=1e10):
    """Exclude invalid image/noise locations from the likelihood.

    A bad location has a non-finite image value or a non-finite/non-positive
    noise value. Its data value is set to zero and its noise is made large,
    giving it negligible likelihood weight without introducing NaNs.
    """
    image_data = np.asarray(image_data, dtype=np.float64)
    noise_map = np.asarray(noise_map, dtype=np.float64)
    if image_data.shape != noise_map.shape:
        raise ValueError(
            f"image_data and noise_map must have the same shape; got "
            f"{image_data.shape} and {noise_map.shape}."
        )

    bad_pixel_mask = (
        ~np.isfinite(image_data)
        | ~np.isfinite(noise_map)
        | (noise_map <= 0.0)
    )
    if np.any(bad_pixel_mask):
        count = int(np.sum(bad_pixel_mask))
        print(
            f"[bad_pixels] Excluding {count} invalid image/noise pixel(s) "
            f"from the likelihood (noise={float(excluded_noise):.1e})."
        )
        image_data = np.where(bad_pixel_mask, 0.0, image_data)
        noise_map = np.where(bad_pixel_mask, float(excluded_noise), noise_map)
    return image_data, noise_map, bad_pixel_mask




def fit_dof_and_reduced_chi2(chi2, image_data, num_params, mask_bool=None):
    if mask_bool is not None:
        n_data = int(np.sum(mask_bool))
    else:
        n_data = int(np.asarray(image_data).size)
    n_fit = int(num_params)
    dof = max(n_data - n_fit, 1)
    return float(chi2 / dof), n_data, n_fit, dof


def resolve_project_path(path, config_dir=None):
    """Resolve a config path relative to config_dir, CWD, or project root."""
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
    if config_dir is not None:
        cand = os.path.abspath(os.path.join(config_dir, path))
        if os.path.exists(cand):
            return cand
    cand_cwd = os.path.abspath(path)
    if os.path.exists(cand_cwd):
        return cand_cwd
    base_dir = config_dir if config_dir is not None else PROJECT_ROOT
    return os.path.abspath(os.path.join(base_dir, path))


def _resolve_single_config_spec(spec):
    token = str(spec).strip()
    if not token:
        raise ValueError("Empty config spec is not allowed.")
    candidate = token if token.endswith('.py') else f'{token}.py'
    search_paths = [
        os.path.abspath(candidate),
        os.path.join(PROJECT_ROOT, candidate),
        os.path.join(WRAPPER_DIR, candidate),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Could not resolve config spec '{spec}'. Tried: {search_paths}")


def configure_import_paths():
    """Ensure project root and the Herculens package are importable."""
    import sys

    for path in (PROJECT_ROOT, HERCULENS_PKG):
        if path not in sys.path:
            sys.path.insert(0, path)


def run_arguments_namespace(config_module, config_path):
    if not hasattr(config_module, 'arguments'):
        raise AttributeError("config module must define arguments().")
    cfg = config_module.arguments()
    if not isinstance(cfg, dict):
        raise TypeError("config.arguments() must return a dict.")
    # Retired controls are accepted in old config files but intentionally ignored.
    for retired_key in (
        'refine_prior_range',
        'refine_prior_min_frac',
        'pixel_init_jitter',
    ):
        cfg.pop(retired_key, None)
    sampler_val = cfg.get('sampler')
    if isinstance(sampler_val, list):
        for s in sampler_val:
            if s not in SAMPLER_CHOICES:
                raise ValueError(
                    f"Unknown sampler {s!r} in list. "
                    f"Choose one of: {sorted(SAMPLER_CHOICES)}"
                )
    elif sampler_val not in SAMPLER_CHOICES:
        raise ValueError(
            f"Unknown sampler {sampler_val!r}. "
            f"Choose one of: {sorted(SAMPLER_CHOICES)}"
        )
    ns = SimpleNamespace(**cfg)
    ns.config_file = os.path.abspath(config_path)
    return ns


def _configure_cuda_from_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def count_configured_gpus(gpus):
    """Count GPU ids listed in config ``gpus`` (e.g. ``'0,1,2'`` -> 3)."""
    spec = str(gpus).strip()
    if not spec:
        return 0
    return len([part for part in spec.split(',') if part.strip() != ''])


def resolve_chain_method_hmc_numpyro(args):
    """
    Choose NumPyro MCMC chain execution strategy.

    ``auto`` (default):
      - multiple JAX devices and num_chains > 1 -> ``parallel`` (pmap, 1 chain/GPU)
      - single device and num_chains > 1 -> ``vectorized`` (vmap on one GPU)
      - otherwise -> ``parallel``
    """
    method = str(getattr(args, 'chain_method_hmc_numpyro', 'auto')).strip().lower()
    if method not in HMC_NUMPYRO_CHAIN_METHODS:
        raise ValueError(
            f"Unknown chain_method_hmc_numpyro {method!r}. "
            f"Choose one of: {sorted(HMC_NUMPYRO_CHAIN_METHODS)}"
        )

    import jax

    n_devices = jax.local_device_count()
    n_chains = int(getattr(args, 'num_chains_hmc_numpyro', 1))
    if method != 'auto':
        return method
    if n_chains > 1 and n_devices > 1:
        return 'parallel'
    if n_chains > 1:
        return 'vectorized'
    return 'parallel'


def log_jax_device_layout(args):
    """Print JAX devices and MCMC chain/GPU layout hints."""
    import jax

    n_devices = jax.local_device_count()
    n_config_gpus = count_configured_gpus(args.gpus)

    devices = jax.devices()
    for i, device in enumerate(devices):
        stats = device.memory_stats()
        if stats is not None:
            bytes_limit = stats['bytes_limit'] / 1024**2
            bytes_in_use = stats['bytes_in_use'] / 1024**2
            # bytes_reserved = stats['bytes_reserved'] / 1024 ** 2
            bytes_available = bytes_limit - bytes_in_use
            print(f'Device {i}: {bytes_in_use:.2f} MB in use, {bytes_available:.2f} MB available')
        else:
            print(f'Device {i}: (No memory stats available)')

    n_chains = int(getattr(args, 'num_chains_hmc_numpyro', 1))
    print(
        f'JAX devices: {jax.devices()} '
        f'(local_count={n_devices}, config_gpus={n_config_gpus}, '
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    )
    if getattr(args, 'sampler', None) == 'hmc' and n_chains > 1:
        chain_method = resolve_chain_method_hmc_numpyro(args)
        print(
            f'[hmc] chain layout: num_chains={n_chains}, '
            f'chain_method={chain_method!r}'
        )
        if chain_method == 'parallel':
            if n_devices >= n_chains:
                print(f'  -> {n_chains} chains in parallel (1 chain per device)')
            else:
                print(
                    f'  -> {n_chains} chains on {n_devices} device(s): '
                    f'first {n_devices} in parallel, remainder sequential'
                )
        elif chain_method == 'vectorized':
            print(f'  -> {n_chains} chains vectorized on {n_devices} device(s)')
        elif chain_method == 'sequential':
            print(f'  -> {n_chains} chains run sequentially')


def empty_config(*args, **kwargs):
    return [], []


_RESOLVED_INIT_PATHS_LOGGED = set()


def resolve_init_run_dir(init_params_path, verbose=True, config_dir=None):
    """Return an existing run directory or parent dir of a kwargs/init JSON file.

    If init_params_path is a directory containing comparison.json (or contains run_* subfolders),
    it automatically selects the best run (highest log-likelihood).

    If init_params_path points to a specific run folder (or file within a run folder) whose parent
    contains comparison.json, it checks if that run is the best run and issues a warning if it is not.
    """
    if not init_params_path:
        return init_params_path

    path = resolve_project_path(init_params_path, config_dir=config_dir)

    if os.path.isfile(path):
        target_dir = os.path.dirname(path)
    elif os.path.isdir(path):
        target_dir = path
    else:
        return path

    is_specific_run_target = (
        os.path.isfile(os.path.join(target_dir, 'kwargs_result.json')) or
        os.path.isfile(os.path.join(target_dir, 'kwargs_init.json'))
    )

    comp_file_in_target = os.path.join(target_dir, 'comparison.json')
    parent_dir = os.path.dirname(os.path.abspath(target_dir))
    comp_file_in_parent = os.path.join(parent_dir, 'comparison.json')

    def _find_best_run_from_comp(comp_json_path):
        if not os.path.isfile(comp_json_path):
            return None, None, None
        try:
            with open(comp_json_path, 'r') as f:
                comp_data = json.load(f)
            best_key = None
            best_ll = -float('inf')
            runs_info = {}
            for run_name, run_info in comp_data.items():
                if not isinstance(run_info, dict):
                    continue
                metrics = run_info.get('metrics', run_info)
                ll = None
                for k in ('LOG_LIKELIHOOD', 'log_likelihood', 'log_like', 'LOGLIKE', 'LOG_LIKELIHOOD_MEDIAN', 'log_likelihood_median'):
                    if k in metrics:
                        ll = float(metrics[k])
                        break
                if ll is not None:
                    runs_info[run_name] = ll
                    if ll > best_ll:
                        best_ll = ll
                        best_key = run_name
            return best_key, best_ll, runs_info
        except Exception:
            return None, None, None

    # Case 1: Directory specified contains comparison.json (e.g. '../modeling_F277W/parametric_restricted_gamma/')
    if os.path.isfile(comp_file_in_target) and not is_specific_run_target:
        best_key, best_ll, _ = _find_best_run_from_comp(comp_file_in_target)
        if best_key:
            best_run_dir = os.path.join(target_dir, best_key)
            if os.path.isdir(best_run_dir):
                cache_key = (os.path.abspath(target_dir), 'auto_select', best_key)
                if verbose and cache_key not in _RESOLVED_INIT_PATHS_LOGGED:
                    _RESOLVED_INIT_PATHS_LOGGED.add(cache_key)
                    print(f"[init_params] Automatically selected best run '{best_key}' (highest log-likelihood {best_ll:.2f}) from '{init_params_path}'")
                return best_run_dir

    # Case 2: Target directory has run_* subfolders but no comparison.json
    if os.path.isdir(target_dir) and not is_specific_run_target:
        subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and d.startswith('run_')]
        if subdirs:
            best_key = None
            best_ll = -float('inf')
            for sd in subdirs:
                m_file = os.path.join(target_dir, sd, 'metrics.json')
                if os.path.isfile(m_file):
                    try:
                        with open(m_file) as f:
                            m_data = json.load(f)
                        ll = float(m_data.get('LOG_LIKELIHOOD', m_data.get('LOG_LIKELIHOOD_MEDIAN', -float('inf'))))
                        if ll > best_ll:
                            best_ll = ll
                            best_key = sd
                    except Exception:
                        pass
            if best_key:
                best_run_dir = os.path.join(target_dir, best_key)
                cache_key = (os.path.abspath(target_dir), 'auto_select_scan', best_key)
                if verbose and cache_key not in _RESOLVED_INIT_PATHS_LOGGED:
                    _RESOLVED_INIT_PATHS_LOGGED.add(cache_key)
                    print(f"[init_params] Automatically selected best run '{best_key}' (highest log-likelihood {best_ll:.2f}) from '{init_params_path}'")
                return best_run_dir

    # Case 3: Target is a specific run directory (e.g. '../modeling_F277W/parametric_restricted_gamma/run_0')
    if os.path.isfile(comp_file_in_parent):
        best_key, best_ll, runs_info = _find_best_run_from_comp(comp_file_in_parent)
        current_run_key = os.path.basename(os.path.normpath(target_dir))
        if best_key and current_run_key in runs_info:
            current_ll = runs_info[current_run_key]
            if current_run_key != best_key:
                cache_key = (os.path.abspath(target_dir), 'warning_not_best', current_run_key)
                if verbose and cache_key not in _RESOLVED_INIT_PATHS_LOGGED:
                    _RESOLVED_INIT_PATHS_LOGGED.add(cache_key)
                    print(
                        f"[init_params] WARNING: User specified run '{current_run_key}' (log-likelihood {current_ll:.2f}), "
                        f"which is NOT the best run in '{parent_dir}'. Best run is '{best_key}' with log-likelihood {best_ll:.2f}."
                    )

    return target_dir


def normalize_run_args_paths(args, config_dir=None):
    """Resolve relative filesystem paths in the run namespace against config_dir or PROJECT_ROOT."""
    path_keys = (
        'data_path',
        'noise_path',
        'psf_path',
        'save_path',
        'init_params_path',
        'ps_mask_path',
        'image_positions_catalog',
        'source_arc_mask_path',
        'contaminate_mask_path',
    )
    for key in path_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is None:
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            normalized = []
            for item in values:
                if not isinstance(item, str):
                    normalized.append(item)
                elif key == 'init_params_path':
                    normalized.append(resolve_init_run_dir(item))
                else:
                    normalized.append(resolve_project_path(item, config_dir=config_dir))
            setattr(args, key, type(value)(normalized) if isinstance(value, tuple) else (
                normalized if isinstance(value, list) else normalized[0]
            ))
    return args


def load_binary_exclusion_mask(mask_path, image_shape, crop_size=None, role='contaminate mask'):
    """Load a binary FITS mask whose non-zero pixels are excluded from fitting."""
    if mask_path is None:
        return None

    mask = np.asarray(get_fits_data(mask_path), dtype=np.float64)
    if crop_size is not None:
        mask = center_crop(mask, crop_size)
    if mask.shape != tuple(image_shape):
        raise ValueError(
            f'{role} shape {mask.shape} does not match image shape {tuple(image_shape)}.'
        )
    if not np.all(np.isfinite(mask)):
        raise ValueError(f'{role} must contain only finite binary values (0 or 1).')
    binary_values = np.isclose(mask, 0.0) | np.isclose(mask, 1.0)
    if not np.all(binary_values):
        invalid = np.unique(mask[~binary_values])
        raise ValueError(
            f'{role} must contain only 0 (fit) and 1 (exclude); '
            f'found values such as {invalid[:5].tolist()}.'
        )
    return mask > 0.5


def archive_input_files(input_paths, destination_dir):
    """Copy resolved run inputs without replacing an existing snapshot."""
    os.makedirs(destination_dir, exist_ok=True)
    archived = {}
    destinations = {}
    for role, source_path in input_paths.items():
        if source_path is None:
            continue
        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f'Cannot archive missing {role} input: {source_path}')

        destination_path = os.path.join(destination_dir, os.path.basename(source_path))
        previous_source = destinations.get(destination_path)
        if previous_source is not None and previous_source != source_path:
            raise ValueError(
                f'Cannot archive {role}: {source_path!r} and {previous_source!r} '
                f'have the same filename {os.path.basename(source_path)!r}.'
            )
        destinations[destination_path] = source_path

        if os.path.abspath(destination_path) == source_path:
            archived[role] = destination_path
            continue
        if os.path.exists(destination_path):
            if not filecmp.cmp(source_path, destination_path, shallow=False):
                raise FileExistsError(
                    f'Refusing to overwrite archived {role} input: {destination_path}. '
                    'Use a new save_path or remove the existing snapshot explicitly.'
                )
            print(f'[data] Preserved existing {role}: {destination_path}')
        else:
            shutil.copy2(source_path, destination_path)
            print(f'[data] Archived {role}: {destination_path}')
        archived[role] = destination_path
    return archived


def create_source_arc_mask_from_radius(image_shape, pixel_scale, radius_config):
    """
    Create a 2D boolean ring (annulus) mask on the image plane from radius limits (in arcseconds).

    Parameters
    ----------
    image_shape : tuple of (int, int)
        Shape (ny, nx) of the image grid.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    radius_config : dict or tuple or list
        Specifies inner and outer radius in arcseconds.
        Example: {'inner': 0.2, 'outer': 0.4} or (0.2, 0.4).

    Returns
    -------
    np.ndarray (bool)
        2D boolean array of shape image_shape.
    """
    if radius_config is None:
        return None

    r_inner = 0.0
    r_outer = np.inf

    if isinstance(radius_config, dict):
        r_inner = float(radius_config.get('inner', radius_config.get('r_in', radius_config.get('min', 0.0))))
        r_outer = float(radius_config.get('outer', radius_config.get('r_out', radius_config.get('max', np.inf))))
    elif isinstance(radius_config, (list, tuple)) and len(radius_config) >= 2:
        r_inner = float(radius_config[0])
        r_outer = float(radius_config[1])
    else:
        raise ValueError(
            f"Invalid source_arc_mask_radius format: {radius_config}. "
            "Must be a dict like {{'inner': 0.2, 'outer': 0.4}} or tuple/list (0.2, 0.4)."
        )

    ny, nx = image_shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * float(pixel_scale)
    y = (np.arange(ny) - (ny - 1) / 2.0) * float(pixel_scale)
    xx, yy = np.meshgrid(x, y)
    r = np.hypot(xx, yy)

    mask = (r >= r_inner) & (r <= r_outer)
    return mask



def pytree_flat_param_labels(params_pytree):
    """
    Build flat parameter labels matching jax.flatten_util.ravel_pytree() order.
    """
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    flat_ref, unflatten_fn = ravel_pytree(params_pytree)
    n = int(np.asarray(flat_ref).size)
    zero_tree = unflatten_fn(jnp.zeros_like(flat_ref))
    labels = []
    for i in range(n):
        probe = jnp.zeros_like(flat_ref).at[i].set(1.0)
        tree = unflatten_fn(probe)
        if isinstance(tree, dict):
            for name, leaf in tree.items():
                delta = np.asarray(leaf, dtype=np.float64) - np.asarray(zero_tree[name], dtype=np.float64)
                if np.max(np.abs(delta)) < 0.5:
                    continue
                arr = np.asarray(leaf)
                if arr.ndim == 0:
                    labels.append(str(name))
                else:
                    for idx in np.ndindex(arr.shape):
                        if abs(float(delta[idx])) > 0.5:
                            idx_txt = ','.join(str(j) for j in idx)
                            labels.append(f'{name}[{idx_txt}]')
        else:
            labels.append(f'param_{i}')
    if len(labels) != n:
        return [f'param_{i}' for i in range(n)]
    return labels


def save_array_fits(path, values, *, extname=None):
    """Write one numerical array as a FITS primary image."""
    from astropy.io import fits
    array = np.asarray(values)
    header = fits.Header()
    if array.dtype == bool:
        header['ORIGTYPE'] = ('bool', 'Original NumPy dtype')
        array = array.astype(np.uint8)
    hdu = fits.PrimaryHDU(data=array, header=header)
    if extname is not None:
        hdu.header['EXTNAME'] = str(extname)
    hdu.writeto(path, overwrite=True)


def load_array_file(path):
    """Load a new FITS array or a legacy ``.npy`` array."""
    path = str(path)
    if path.lower().endswith(('.fits', '.fit', '.fts')):
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data)
            if hdul[0].header.get('ORIGTYPE') == 'bool':
                data = data.astype(bool)
            return data
    return np.load(path)


def save_named_arrays_fits(path, arrays):
    """Write named numerical arrays as FITS image extensions."""
    from astropy.io import fits
    hdus = [fits.PrimaryHDU()]
    for name, values in arrays.items():
        if values is None:
            continue
        array = np.asarray(values)
        header = fits.Header()
        if array.dtype == bool:
            header['ORIGTYPE'] = ('bool', 'Original NumPy dtype')
            array = array.astype(np.uint8)
        hdus.append(fits.ImageHDU(data=array, header=header, name=str(name).upper()[:68]))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def save_rtu_source_fits(path, pixels, x_corners, y_corners, *, polynomial_order=None):
    """Write a self-contained RTU source reconstruction FITS file.

    The primary HDU is the regular ``(ny, nx)`` RTU brightness array.  The
    ``X_CORNERS`` and ``Y_CORNERS`` image extensions hold matching physical
    source-plane cell corners in arcsec, each of shape ``(ny + 1, nx + 1)``.
    """
    source = np.asarray(pixels)
    x_corners, y_corners = np.asarray(x_corners), np.asarray(y_corners)
    expected = (source.shape[0] + 1, source.shape[1] + 1)
    if source.ndim != 2 or x_corners.shape != expected or y_corners.shape != expected:
        raise ValueError(
            "RTU FITS requires 2-D pixels and matching (ny + 1, nx + 1) physical corner arrays."
        )
    header = fits.Header()
    header['GRIDKIND'] = ('ray_transformed_uniform', 'Source-grid coordinate system')
    header['BUNIT'] = ('pixel_flux', 'Source brightness unit')
    header['XEXT'] = ('X_CORNERS', 'Physical source x-cell corners [arcsec]')
    header['YEXT'] = ('Y_CORNERS', 'Physical source y-cell corners [arcsec]')
    if polynomial_order is not None:
        header['RTUORDER'] = (int(polynomial_order), 'RTU inverse-CDF polynomial order')
    fits.HDUList([
        fits.PrimaryHDU(data=source, header=header),
        fits.ImageHDU(data=x_corners, name='X_CORNERS'),
        fits.ImageHDU(data=y_corners, name='Y_CORNERS'),
    ]).writeto(path, overwrite=True)


def load_named_arrays_fits(path):
    """Return a mapping of FITS extension name to its numerical array."""
    from astropy.io import fits
    arrays = {}
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul[1:]:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data)
            if hdu.header.get('ORIGTYPE') == 'bool':
                data = data.astype(bool)
            arrays[hdu.name.lower()] = data
    return arrays


def kwargs_best_to_json_pixelated_npy(
    kwargs_best, save_path, type_list, 
    pixels_filename='kwargs_source_pixels.fits',
    pixels_wn_filename='kwargs_source_pixels_wn.fits',
    lens_light_pixels_prefix='kwargs_lens_light_pixels',
    save_pixel_arrays=True,
):
    import copy
    out = copy.deepcopy(kwargs_best)
    if type_list.get('source_light_type_list') == ['PIXELATED']:
        ks = out.get('kwargs_source', [])
        if ks and isinstance(ks[0], dict):
            ks0 = dict(ks[0])
            if 'pixels' in ks0 and ks0['pixels'] is not None:
                if save_pixel_arrays:
                    pixels = np.asarray(ks0['pixels'])
                    save_array_fits(os.path.join(save_path, pixels_filename), pixels)
                    ks0['pixels'] = {
                        '_format': 'pixelated_pixels_fits',
                        'file': pixels_filename,
                        '_unit': 'pixel_flux',
                        '_pixel_area_reference': 'image_data_pixel',
                    }
                else:
                    ks0['pixels'] = {
                        '_format': 'pixelated_pixels_fits',
                        'file': pixels_filename,
                        '_unit': 'pixel_flux',
                        '_pixel_area_reference': 'image_data_pixel',
                        '_save_disabled': True,
                    }
            if 'pixels_wn' in ks0 and ks0['pixels_wn'] is not None:
                if save_pixel_arrays:
                    pixels_wn = np.asarray(ks0['pixels_wn'])
                    save_array_fits(os.path.join(save_path, pixels_wn_filename), pixels_wn)
                    ks0['pixels_wn'] = {'_format': 'pixelated_pixels_fits', 'file': pixels_wn_filename}
                else:
                    ks0['pixels_wn'] = {'_format': 'pixelated_pixels_fits', 'file': pixels_wn_filename, '_save_disabled': True}
            ks = list(ks)
            ks[0] = ks0
            out['kwargs_source'] = ks
    lens_types = type_list.get('lens_light_type_list', [])
    lens_kwargs = out.get('kwargs_lens_light', [])
    if lens_kwargs and any(profile_type == 'PIXELATED' for profile_type in lens_types):
        updated = list(lens_kwargs)
        for index, profile_type in enumerate(lens_types):
            if profile_type != 'PIXELATED' or index >= len(updated):
                continue
            values = updated[index]
            if not isinstance(values, dict):
                continue
            values = dict(values)
            for key, suffix in (("pixels", ""), ("pixels_wn", "_wn")):
                if values.get(key) is None:
                    continue
                filename = f"{lens_light_pixels_prefix}_{index}{suffix}.fits"
                if save_pixel_arrays:
                    save_array_fits(os.path.join(save_path, filename), np.asarray(values[key]))
                values[key] = {
                    '_format': 'pixelated_pixels_fits',
                    'file': filename,
                    **({'_unit': 'pixel_flux', '_pixel_area_reference': 'image_data_pixel'} if key == 'pixels' else {}),
                    **({} if save_pixel_arrays else {'_save_disabled': True}),
                }
            updated[index] = values
        out['kwargs_lens_light'] = updated
    return out
