"""Export explicitly declared API models as legacy-wrapper config modules."""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pformat
import subprocess
from typing import Any, Mapping

import numpy as np

from .samplers import SamplerConfig


def _python_value(value: Any) -> Any:
    """Convert NumPy values to literals which are valid in a Python config."""
    if isinstance(value, np.ndarray):
        return _python_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _python_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_python_value(item) for item in value)
    if isinstance(value, list):
        return [_python_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"Cannot export {type(value).__name__} to a wrapper config.py literal. "
        "Use only scalar, list, tuple, dictionary, or NumPy-array profile settings."
    )


def _literal(value: Any) -> str:
    return pformat(_python_value(value), width=100, sort_dicts=False)


def detect_gpus() -> str:
    """Return GPU ids visible to a wrapper subprocess, or ``''`` for CPU.

    A user-selected ``CUDA_VISIBLE_DEVICES`` takes precedence.  Otherwise
    ``nvidia-smi`` is queried without importing JAX, so exporting a config
    never initializes an accelerator backend.
    """
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return "" if configured.strip() == "-1" else configured.strip()
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return ",".join(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )


def export_wrapper_config(
    model: Any,
    path: str | Path,
    sampler: SamplerConfig,
    *,
    save_path: str | Path,
    n_runs: int = 1,
    gpus: str | int | None = None,
    init_params_path: str | Path | None = None,
    residual_vis_max: float = 0.0,
    wrapper_options: Mapping[str, Any] | None = None,
) -> Path:
    """Write a runnable legacy-wrapper ``config.py`` from an API model.

    The model must use :meth:`SingleBandData.from_fits`, so the wrapper can
    re-read the original image, noise, and PSF.  The generated config retains
    source/contamination masks, crop/background choices, profile priors and
    parameter links, numerical settings, and the selected sampler settings.
    ``wrapper_options`` is for legacy-only controls such as ``ps_mask_path``.
    """
    if not isinstance(sampler, SamplerConfig):
        raise TypeError("sampler must be a SamplerConfig instance.")
    if not isinstance(n_runs, int) or isinstance(n_runs, bool) or n_runs < 1:
        raise ValueError("n_runs must be a positive integer.")

    data_paths = getattr(model.data, "_input_paths", {})
    missing = [name for name in ("image", "noise", "psf") if name not in data_paths]
    if missing:
        raise ValueError(
            "Wrapper config export requires SingleBandData.from_fits(); missing "
            f"input FITS path(s): {', '.join(missing)}."
        )
    types, parameters = model.definition.as_dicts()
    component_functions = []
    mapping = (
        ("lens_mass", "lens_mass_config"),
        ("lens_light", "lens_light_config"),
        ("source_light", "source_light_config"),
        ("point_source", "point_source_config"),
    )
    for component, function in mapping:
        type_key = f"{component}_type_list"
        parameter_key = f"{component}_params_list"
        component_functions.append(
            f"def {function}(image_size=None, pixel_scale=None, args=None, init_params=None):\n"
            f"    return {_literal(types[type_key])}, {_literal(parameters[parameter_key])}\n"
        )

    corner = str(model.data.background_subtract["corner"])
    wrapper_corner = {
        "upper left": "top_left", "upper right": "top_right",
        "lower left": "bottom_left", "lower right": "bottom_right",
    }[corner]
    arguments: dict[str, Any] = {
        "data_path": str(data_paths["image"].resolve()),
        "noise_path": str(data_paths["noise"].resolve()),
        "psf_path": str(data_paths["psf"].resolve()),
        "source_arc_mask_path": model.data.source_arc_mask_path,
        "source_arc_mask_radius": model.data.source_arc_mask_radius,
        "contaminate_mask_path": model.data.contaminate_mask_path,
        "save_path": str(Path(save_path).expanduser()),
        "random_seed": sampler.random_seed,
        "pixel_scale": model.data.pixel_scale,
        "psf_supersampling_factor": model.data.psf_supersampling_factor,
        "crop_size": model.data.crop_size,
        "background_subtract_corner": model.data.background_subtract["num_pixels"],
        "background_subtract_which_corner": wrapper_corner,
        "residual_vis_max": residual_vis_max,
        "supersampling_factor": model.numerics.get("supersampling_factor", 1),
        "supersampling_convolution": model.numerics.get("supersampling_convolution", False),
        "sampler": sampler.name,
        # initialize(..., init_params_path=...) already establishes this on
        # the model.  An explicit argument remains useful for export before
        # initialization or when intentionally choosing another warm start.
        "init_params_path": str(Path(init_params_path or model.initialization_path).expanduser())
        if (init_params_path or model.initialization_path) is not None else None,
        "source_grid_scale": model.source_grid_scale,
        "likelihood_scale": model.likelihood_scale,
        "gpus": detect_gpus() if gpus is None else str(gpus),
        "n_runs": n_runs,
        "fix_component": [],
        "regul_num_samples": 1000,
        "conjugate_points": None,
        "pixelated_init_match": "image",
        "num_iterations_warmup": 2_000,
        "max_iterations_svi": 10_000,
        "init_learning_rate_svi": 1e-2,
        "init_scale_svi": 0.1,
        "loss_kind_svi": "trace_elbo",
        "num_particles_svi": 10,
        "num_warmup_hmc_numpyro": 1_000,
        "num_samples_hmc_numpyro": 1_000,
        "num_chains_hmc_numpyro": 1,
        "checkpoint_interval_hmc_numpyro": 250,
        "chain_method_hmc_numpyro": "auto",
        "progress_bar_hmc_numpyro": True,
        "hmc_init_max_retries": 100,
        "ps_nsolutions": 5,
        "ps_niter": 10,
        "ps_scale_factor": 2,
        "ps_nsubdivisions": 3,
        "ps_mask_path": None,
        "image_positions_catalog": None,
        "num_point_sources": 1,
        "relieve_mask_indices": None,
        "exclude_ps": True,
    }
    arguments.update(sampler.options)
    if wrapper_options is not None:
        arguments.update(dict(wrapper_options))
    if arguments.get("num_point_sources", 1) < 0:
        raise ValueError("wrapper_options['num_point_sources'] must be non-negative.")

    index_lines = "\n".join(
        f"    args['images_indices_{index}'] = None"
        for index in range(int(arguments["num_point_sources"]))
    )
    source = (
        '"""Generated by herculens_wrapper.api; runnable with ``python run.py config.py``."""\n\n'
        + "\n".join(component_functions)
        + "\ndef arguments():\n"
        + f"    args = {_literal(arguments)}\n"
        + (index_lines + "\n" if index_lines else "")
        + "    return args\n"
    )
    output = Path(path).expanduser()
    if output.suffix != ".py":
        output = output / "config.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    return output
