"""Sersic pipeline: parametric/pixelated SVI and HMC.

Examples
--------
python herculens_modelling.py --data-dir /path/to/data --stage pipeline
python herculens_modelling.py --data-dir /path/to/data --stage pixelated_svi --pixelated-grid-kind ray_transformed_uniform
python herculens_modelling.py --data-dir /path/to/data --stage parametric_hmc
python herculens_modelling.py --data-dir /path/to/data --stage single --sampler svi --source-method parametric
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
LOCAL_WRAPPER = SCRIPT_DIR.parents[1] / "herculens_wrapper"
if LOCAL_WRAPPER.is_dir() and str(LOCAL_WRAPPER) not in sys.path:
    sys.path.insert(0, str(LOCAL_WRAPPER))

from herculens_wrapper.utils import save_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-Sersic single-band modelling.")
    parser.add_argument("--stage", choices=("pipeline", "parametric_svi", "parametric_hmc", "pixelated_svi", "pixelated_hmc", "single"), default="pipeline")
    parser.add_argument("--sampler", choices=("svi", "hmc"), default=None)
    parser.add_argument("--source-method", choices=("parametric", "pixelated"), default=None)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing sim_sl.fits, sim_sl_noise.fits, and sim_sl_psf.fits.",
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpus", default=None,
        help=(
            "One CUDA/MIG identifier or comma-separated MIG UUIDs. Multiple "
            "identifiers run independent SVI fits or parallel HMC chains."
        ),
    )
    parser.add_argument("--svi-runs", type=int, default=4)
    parser.add_argument("--svi-iterations", type=int, default=10_000)
    parser.add_argument("--pixelated-warmup", type=int, default=2_000)
    parser.add_argument("--hmc-warmup", type=int, default=1_000)
    parser.add_argument("--hmc-samples", type=int, default=2_000)
    parser.add_argument("--hmc-chains", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--init-params-path", type=Path, default=None, help="Warm start for a single selected stage.")
    parser.add_argument("--residual-vis-max", type=float, default=3.0)
    parser.add_argument("--crop-size", type=int, default=80)
    parser.add_argument("--pixel-scale", type=float, default=0.08)
    parser.add_argument(
        "--psf-supersampling-factor",
        type=int,
        default=1,
        help="Sampling factor of the supplied PSF kernel relative to image pixels.",
    )
    parser.add_argument("--supersampling-factor", type=int, default=2)
    parser.add_argument(
        "--supersampling-convolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convolve on the supersampled model grid before binning to the data grid.",
    )
    parser.add_argument("--source-grid-scale", type=float, default=0.8)
    parser.add_argument(
        "--pixelated-grid-kind",
        choices=("uniform", "ray_transformed_uniform"),
        default="uniform",
        help="Pixelated-source grid: legacy uniform grid or the RTU adaptive physical grid.",
    )
    parser.add_argument(
        "--rtu-polynomial-order", type=int, default=11,
        help="Odd inverse-CDF polynomial order used only by the RTU source grid.",
    )
    args = parser.parse_args()
    for name in ("svi_runs", "svi_iterations", "pixelated_warmup", "hmc_warmup", "hmc_samples", "hmc_chains", "checkpoint_interval", "psf_supersampling_factor"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.stage == "single" and (args.sampler is None or args.source_method is None):
        parser.error("--stage single requires both --sampler and --source-method.")
    if args.rtu_polynomial_order < 3 or args.rtu_polynomial_order % 2 == 0:
        parser.error("--rtu-polynomial-order must be an odd integer of at least 3.")
    return args


def make_data(args: argparse.Namespace):
    from herculens_wrapper.api import SingleBandData
    data_dir = args.data_dir.expanduser().resolve()
    return SingleBandData.from_fits(
        data_dir / "sim_sl.fits", data_dir / "sim_sl_noise.fits", data_dir / "sim_sl_psf.fits",
        pixel_scale=args.pixel_scale,
        psf_supersampling_factor=args.psf_supersampling_factor,
        crop_size=args.crop_size,
        source_arc_mask_radius={"inner": 0.8, "outer": 2.5},
    )


def make_profiles(
    source_method: str,
    pixel_scale: float,
    crop_size: int,
    *,
    pixelated_grid_kind: str = "uniform",
    rtu_polynomial_order: int = 11,
):
    from herculens_wrapper.api import LensProfileCollection, LightProfile, MassProfile, PixelatedSource
    lens_mass = MassProfile(["SIE", "SHEAR"], prior=[
        {"theta_E": [1.0, 2.0], "center_x": [0.0, 0.1, -0.3, 0.3],
         "center_y": [0.0, 0.1, -0.3, 0.3], "e1": [-0.5, 0.5], "e2": [-0.5, 0.5]},
        {"ra_0": 0.0, "dec_0": 0.0, "gamma1": [-0.3, 0.3], "gamma2": [-0.3, 0.3]},
    ])
    
    # lens_light = LightProfile("SERSIC_ELLIPSE", prior={
    #     "amp": [2.0, 0.1], "R_sersic": [0.01, 2.0], "n_sersic": [1.0, 5.0],
    #     "e1": [-0.5, 0.5], "e2": [-0.5, 0.5],
    #     "center_x": [0.0, 0.1, -0.3, 0.3], "center_y": [0.0, 0.1, -0.3, 0.3],
    # })

    if source_method == "parametric":
        source_light = LightProfile("SERSIC_ELLIPSE", prior={
            "amp": [1.3, 0.3], "R_sersic": [0.01, 2.0], "n_sersic": [0.5, 2.0],
            "e1": [-0.5, 0.5], "e2": [-0.5, 0.5],
            "center_x": [0.0, 0.1, -0.3, 0.3], "center_y": [0.0, 0.1, -0.3, 0.3],
        })
    elif source_method == "pixelated":
        source_light = PixelatedSource(
            pixel_grid={"grid_kind": pixelated_grid_kind,
                        "rtu_polynomial_order": rtu_polynomial_order,
                        "pixel_adaptive_grid": True, "pixel_grid_shape": 80,
                        "pixel_interpol": "fast_bilinear", "pixel_scale_factor": 0.5,
                        "grid_center": (0.0, 0.0), "grid_shape": (2.0, 2.0)},
            pixelated_prior={"prior_type": "matern", "regul_strengths": (3.0, 3.0),
                              "k_zero": 0.0, "n_value_low": 1e-4, "n_value_high": 100.0,
                              "sigma_low": 1e-5, "sigma_high": 10.0,
                              "rho_low": None, "rho_high": None, "positive": True},
        )
    else:
        raise ValueError(f"Unsupported source method {source_method!r}.")
    return LensProfileCollection(lens_mass=lens_mass, lens_light=None, source_light=source_light)


def make_model(args: argparse.Namespace, source_method: str):
    from herculens_wrapper.api import SingleBandModel
    return SingleBandModel(
        profiles=make_profiles(
            source_method, args.pixel_scale, args.crop_size,
            pixelated_grid_kind=args.pixelated_grid_kind,
            rtu_polynomial_order=args.rtu_polynomial_order,
        ),
        observation=make_data(args),
        numerics={
            "supersampling_factor": args.supersampling_factor,
            "supersampling_convolution": args.supersampling_convolution,
        },
        source_grid_scale=args.source_grid_scale,
    )


def prepare_stage(model, args: argparse.Namespace, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_config(args, directory)
    model.data.show(scale="linear", save_path=directory / "input_data_linear.png")
    model.data.show(scale="log", save_path=directory / "input_data_log.png")
    model.data.save(directory / "data")
    if SCRIPT_PATH != (directory / SCRIPT_PATH.name).resolve():
        shutil.copy2(SCRIPT_PATH, directory / SCRIPT_PATH.name)


def prepare_stage_data_only(args: argparse.Namespace, directory: Path) -> None:
    """Prepare shared SVI-stage inputs without importing/initialising JAX."""
    directory.mkdir(parents=True, exist_ok=True)
    save_config(args, directory)
    data = make_data(args)
    data.show(scale="linear", save_path=directory / "input_data_linear.png")
    data.show(scale="log", save_path=directory / "input_data_log.png")
    data.save(directory / "data")
    if SCRIPT_PATH != (directory / SCRIPT_PATH.name).resolve():
        shutil.copy2(SCRIPT_PATH, directory / SCRIPT_PATH.name)


def _split_visible_devices(gpus: str | None) -> list[str]:
    """Split comma-separated CUDA/MIG identifiers without resolving them."""
    return [] if gpus is None else [item.strip() for item in gpus.split(",") if item.strip()]


def run_svi(args: argparse.Namespace, source_method: str, directory: Path, init_path: Path | None) -> None:
    from herculens_wrapper.api import (
        SamplerConfig, SingleBandResultsCombination, is_completed_svi_run,
    )

    devices = _split_visible_devices(args.gpus)
    # The API owns multi-MIG scheduling.  The run script only selects the
    # requested devices and prepares shared stage products.
    if len(devices) > 1:
        directory.mkdir(parents=True, exist_ok=True)
        prepare_stage_data_only(args, directory)
        model = make_model(args, source_method)
        if init_path is not None:
            # Do not initialize in the parent: it could initialize JAX on all
            # visible devices before the spawned workers restrict themselves
            # to one MIG.  Each child loads this path and initializes with its
            # own seed instead.
            model.initialization_path = init_path.expanduser()
        sampler = SamplerConfig.svi(
            max_iterations=args.svi_iterations, learning_rate=1e-2, init_scale=0.01,
            loss_kind="trace_elbo", num_particles=10, random_seed=args.seed,
        )
        results = model.run(
            sampler, save_path=directory, n_runs=args.svi_runs,
            parallel=True, gpus=devices, pixelated_init_match="image",
            num_iterations_warmup=args.pixelated_warmup if source_method == "pixelated" else 0,
            residual_vis_max=args.residual_vis_max,
        )
        results.output(directory, residual_vis_max=args.residual_vis_max)
        return

    directory.mkdir(parents=True, exist_ok=True)
    model = make_model(args, source_method)
    prepare_stage(model, args, directory)
    results = []
    print(f"Starting {source_method} SVI with {args.svi_runs} runs: {directory}")
    for run_id in range(args.svi_runs):
        run_dir = directory / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if is_completed_svi_run(run_dir):
            print(f"[svi] Run {run_id} is complete; loading it and continuing.")
            model.load(run_dir, seed=args.seed + run_id)
            results.append(model.get_results(random_seed=args.seed + run_id))
            continue
        initial = model.initialize(
            seed=args.seed + run_id, run_id=run_id, init_params_path=init_path,
            pixelated_init_match="image",
            num_iterations_warmup=args.pixelated_warmup if source_method == "pixelated" else 0,
        )
        model.plot_initial_model(scale="linear", save_path=run_dir / "initial_guess_model.png", residual_vis_max=args.residual_vis_max)
        if source_method == "pixelated":
            model.plot_initial_source(scale="linear", save_path=run_dir / "initial_source_plane.png")
        sampler = SamplerConfig.svi(max_iterations=args.svi_iterations, learning_rate=1e-2, init_scale=0.01,
                                    loss_kind="trace_elbo", num_particles=10, random_seed=args.seed + run_id)
        result = model.run(sampler, init_params=initial, save_path=run_dir)
        result.output(residual_vis_max=args.residual_vis_max)
        results.append(result)
    SingleBandResultsCombination(results).output(
        directory, residual_vis_max=args.residual_vis_max,
    )


def run_hmc(
    args: argparse.Namespace,
    directory: Path,
    init_path: Path,
    source_method: str = "pixelated",
) -> None:
    from herculens_wrapper.api import SamplerConfig

    if not init_path.exists():
        raise FileNotFoundError(
            f"{source_method.capitalize()} HMC needs matching SVI output: {init_path}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    model = make_model(args, source_method)
    prepare_stage(model, args, directory)
    # Restore the matching SVI posterior. HMC then draws chain-specific
    # initial values from that guide; no image-match warmup is used here.
    initial = model.initialize(
        seed=args.seed, run_id=0, init_params_path=init_path,
        num_iterations_warmup=0,
    )
    model.plot_initial_model(scale="linear", save_path=directory / "initial_guess_model.png", residual_vis_max=args.residual_vis_max)
    if source_method == "pixelated":
        model.plot_initial_source(scale="linear", save_path=directory / "initial_source_plane.png")
    sampler = SamplerConfig.hmc(num_warmup=args.hmc_warmup, num_samples=args.hmc_samples,
                                num_chains=args.hmc_chains, checkpoint_interval=args.checkpoint_interval,
                                random_seed=args.seed)
    result = model.run(
        sampler, init_params=initial, save_path=directory,
        residual_vis_max=args.residual_vis_max,
    )
    result.output(residual_vis_max=args.residual_vis_max)


def requested_pair(args: argparse.Namespace) -> tuple[str, str]:
    canonical = {
        "parametric_svi": ("svi", "parametric"),
        "parametric_hmc": ("hmc", "parametric"),
        "pixelated_svi": ("svi", "pixelated"),
        "pixelated_hmc": ("hmc", "pixelated"),
    }
    if args.stage == "single":
        return args.sampler, args.source_method
    sampler, source = canonical[args.stage]
    if args.sampler not in (None, sampler) or args.source_method not in (None, source):
        raise ValueError(f"--stage {args.stage} requires --sampler {sampler} --source-method {source}.")
    return sampler, source


def main() -> None:
    args = parse_arguments()
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    root = args.output_root.expanduser().resolve()
    paths = {
        name: root / name
        for name in ("parametric_svi", "parametric_hmc", "pixelated_svi", "pixelated_hmc")
    }
    if args.stage == "pipeline":
        if args.sampler is not None or args.source_method is not None or args.init_params_path is not None:
            raise ValueError("--stage pipeline chooses its preceding-stage outputs automatically; do not set sampler, source-method, or init-params-path.")
        run_svi(args, "parametric", paths["parametric_svi"], None)
        run_svi(args, "pixelated", paths["pixelated_svi"], paths["parametric_svi"])
        run_hmc(args, paths["pixelated_hmc"], paths["pixelated_svi"], "pixelated")
        return
    sampler, source = requested_pair(args)
    directory = paths.get(f"{source}_{sampler}", root / f"{source}_{sampler}")
    if sampler == "svi":
        init_path = args.init_params_path
        if source == "pixelated":
            init_path = init_path or paths["parametric_svi"]
            if not init_path.exists():
                raise FileNotFoundError(f"Pixelated SVI needs parametric-SVI output: {init_path}")
        run_svi(args, source, directory, init_path)
    else:
        default_init = paths["parametric_svi"] if source == "parametric" else paths["pixelated_svi"]
        run_hmc(args, directory, args.init_params_path or default_init, source)


if __name__ == "__main__":
    main()
