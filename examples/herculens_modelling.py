"""Sersic pipeline: parametric/pixelated SVI and HMC.

Examples
--------
python sersic_modelling.py --stage pipeline
python sersic_modelling.py --stage pixelated_svi
python sersic_modelling.py --stage parametric_hmc
python sersic_modelling.py --stage single --sampler svi --source-method parametric
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DATA_DIR = SCRIPT_DIR.parent / "simulation_sersic_no_lens_light_finer"
LOCAL_WRAPPER = SCRIPT_DIR.parents[1] / "herculens_wrapper"
if LOCAL_WRAPPER.is_dir() and str(LOCAL_WRAPPER) not in sys.path:
    sys.path.insert(0, str(LOCAL_WRAPPER))

from herculens_wrapper.utils import Tee, save_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-Sersic single-band modelling.")
    parser.add_argument("--stage", choices=("pipeline", "parametric_svi", "parametric_hmc", "pixelated_svi", "pixelated_hmc", "single"), default="pipeline")
    parser.add_argument("--sampler", choices=("svi", "hmc"), default=None)
    parser.add_argument("--source-method", choices=("parametric", "pixelated"), default=None)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value; omitted leaves the environment unchanged.")
    parser.add_argument("--svi-runs", type=int, default=4)
    parser.add_argument("--svi-iterations", type=int, default=10_000)
    parser.add_argument("--pixelated-warmup", type=int, default=2_000)
    parser.add_argument("--hmc-warmup", type=int, default=1_000)
    parser.add_argument("--hmc-samples", type=int, default=2_000)
    parser.add_argument("--hmc-chains", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--init-params-path", type=Path, default=None, help="Warm start for a single selected stage.")
    parser.add_argument("--pixelated-init-path", type=Path, default=None, help="Parametric-SVI warm start in pipeline mode.")
    parser.add_argument("--hmc-init-path", type=Path, default=None, help="Pixelated-SVI warm start in pipeline mode.")
    parser.add_argument("--residual-vis-max", type=float, default=3.0)
    parser.add_argument("--crop-size", type=int, default=80)
    parser.add_argument("--pixel-scale", type=float, default=0.08)
    parser.add_argument("--supersampling-factor", type=int, default=2)
    parser.add_argument("--source-grid-scale", type=float, default=0.8)
    args = parser.parse_args()
    for name in ("svi_runs", "svi_iterations", "pixelated_warmup", "hmc_warmup", "hmc_samples", "hmc_chains", "checkpoint_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.stage == "single" and (args.sampler is None or args.source_method is None):
        parser.error("--stage single requires both --sampler and --source-method.")
    return args


def make_data(args: argparse.Namespace):
    from herculens_wrapper.api import SingleBandData
    return SingleBandData.from_fits(
        DATA_DIR / "sim_sl.fits", DATA_DIR / "sim_sl_noise.fits", DATA_DIR / "sim_sl_psf.fits",
        pixel_scale=args.pixel_scale, crop_size=args.crop_size,
        source_arc_mask_radius={"inner": 0.8, "outer": 2.5},
    )


def make_profiles(source_method: str, pixel_scale: float, crop_size: int):
    from herculens_wrapper.api import LensProfileCollection, LightProfile, MassProfile
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
        source_light = LightProfile("PIXELATED", prior={
            "pixel_grid": {"pixel_adaptive_grid": True, "pixel_grid_shape": 80,
                           "pixel_interpol": "fast_bilinear", "pixel_scale_factor": 0.5,
                           "grid_center": (0.0, 0.0), "grid_shape": (2.0, 2.0)},
            "pixelated_prior": {"prior_type": "matern", "regul_strengths": (3.0, 3.0),
                                 "k_zero": 0.0, "n_value_low": 1e-4, "n_value_high": 100.0,
                                 "sigma_low": 1e-5, "sigma_high": 10.0,
                                 "rho_low": None, "rho_high": None, "positive": True},
        })
    else:
        raise ValueError(f"Unsupported source method {source_method!r}.")
    return LensProfileCollection(lens_mass=lens_mass, lens_light=None, source_light=source_light)


def make_model(args: argparse.Namespace, source_method: str):
    from herculens_wrapper.api import SingleBandModel
    return SingleBandModel(
        profiles=make_profiles(source_method, args.pixel_scale, args.crop_size), observation=make_data(args),
        numerics={"supersampling_factor": args.supersampling_factor}, source_grid_scale=args.source_grid_scale,
    )


@contextmanager
def stage_logging(directory: Path):
    """Mirror stage stdout/stderr to ``directory/log.txt``."""
    log_path = directory / "log.txt"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            yield log_path
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def prepare_stage(model, args: argparse.Namespace, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_config(args, directory)
    model.data.show(scale="linear", save_path=directory / "input_data_linear.png")
    model.data.show(scale="log", save_path=directory / "input_data_log.png")
    model.data.save(directory / "data")
    if SCRIPT_PATH != (directory / SCRIPT_PATH.name).resolve():
        shutil.copy2(SCRIPT_PATH, directory / SCRIPT_PATH.name)


def run_svi(args: argparse.Namespace, source_method: str, directory: Path, init_path: Path | None) -> None:
    from herculens_wrapper.api import SamplerConfig, SingleBandResultsCombination

    directory.mkdir(parents=True, exist_ok=True)
    with stage_logging(directory):
        model = make_model(args, source_method)
        prepare_stage(model, args, directory)
        results = []
        print(f"Starting {source_method} SVI with {args.svi_runs} runs: {directory}")
        for run_id in range(args.svi_runs):
            run_dir = directory / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
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
            result = model.run(sampler, init_params=initial)
            result.output(save_path=run_dir, residual_vis_max=args.residual_vis_max)
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
    with stage_logging(directory):
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
        result = model.run(sampler, init_params=initial, save_path=directory)
        result.output(save_path=directory, residual_vis_max=args.residual_vis_max)


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
            raise ValueError("For pipeline warm starts use --pixelated-init-path and --hmc-init-path.")
        run_svi(args, "parametric", paths["parametric_svi"], None)
        run_svi(args, "pixelated", paths["pixelated_svi"], args.pixelated_init_path or paths["parametric_svi"])
        run_hmc(args, paths["pixelated_hmc"], args.hmc_init_path or paths["pixelated_svi"], "pixelated")
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
