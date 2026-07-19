import json
import os
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


def fit_dof_and_reduced_chi2(chi2, image_data, num_params, mask_bool=None):
    if mask_bool is not None:
        n_data = int(np.sum(mask_bool))
    else:
        n_data = int(np.asarray(image_data).size)
    n_fit = int(num_params)
    dof = max(n_data - n_fit, 1)
    return float(chi2 / dof), n_data, n_fit, dof


def resolve_project_path(path, config_dir=None):
    """Resolve a config path relative to the config_dir (if provided) or project root."""
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
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


def resolve_init_run_dir(init_params_path):
    """Return an existing run directory or parent dir of a kwargs/init JSON file."""
    path = resolve_project_path(init_params_path)
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


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
    )
    for key in path_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None and isinstance(value, str):
                setattr(args, key, resolve_project_path(value, config_dir=config_dir))
    return args


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


def kwargs_best_to_json_pixelated_npy(
    kwargs_best, save_path, type_list, 
    pixels_filename='kwargs_source_pixels.npy',
    pixels_wn_filename='kwargs_source_pixels_wn.npy'
):
    import copy
    out = copy.deepcopy(kwargs_best)
    if type_list.get('source_light_type_list') == ['PIXELATED']:
        ks = out.get('kwargs_source', [])
        if ks and isinstance(ks[0], dict):
            ks0 = dict(ks[0])
            if 'pixels' in ks0 and ks0['pixels'] is not None:
                pixels = np.asarray(ks0['pixels'])
                np.save(os.path.join(save_path, pixels_filename), pixels)
                ks0['pixels'] = {'_format': 'pixelated_pixels_npy', 'file': pixels_filename}
            if 'pixels_wn' in ks0 and ks0['pixels_wn'] is not None:
                pixels_wn = np.asarray(ks0['pixels_wn'])
                np.save(os.path.join(save_path, pixels_wn_filename), pixels_wn)
                ks0['pixels_wn'] = {'_format': 'pixelated_pixels_npy', 'file': pixels_wn_filename}
            ks = list(ks)
            ks[0] = ks0
            out['kwargs_source'] = ks
    return out
