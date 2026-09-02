"""Single-band model initialization, inference, and visualization."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import numpy as np
from .collections import LensProfileCollection
from .data import SingleBandData
from .models import ModelDefinition, count_physical_parameters
from .samplers import FitResult, SamplerConfig
from .visualization import PlotScale, _extent, _normalization

def _model_backend():
    from ..models import create_lens_image, create_prob_model, get_init_params, validate_param_list
    return create_lens_image, create_prob_model, get_init_params, validate_param_list

def _sampler_backend():
    from ..samplers import run_hmc, run_optax, run_svi
    return run_hmc, run_optax, run_svi


def _svi_many_worker(spec, run_id, device):
    """Spawn worker for one independent SVI run (sets CUDA before JAX use)."""
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    from .samplers import is_completed_svi_run
    model = SingleBandModel(**spec["model"])
    run_dir = Path(spec["directory"]) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_completed_svi_run(run_dir):
        return
    sampler = SamplerConfig("svi", random_seed=spec["seed"] + run_id, options=dict(spec["options"]))
    initial = model.initialize(
        seed=sampler.random_seed, run_id=run_id, init_params_path=spec["init_path"],
        pixelated_init_match=spec["pixelated_init_match"], num_iterations_warmup=spec["warmup"],
    )
    model.plot_initial_model(
        scale="linear", save_path=run_dir / "initial_guess_model.png",
        residual_vis_max=spec["residual_vis_max"],
    )
    source_types = model.definition.as_dicts()[0].get("source_light_type_list", [])
    if any("PIXELATED" in str(profile_type).upper() for profile_type in source_types):
        model.plot_initial_source(scale="linear", save_path=run_dir / "initial_source_plane.png")
    model.run(sampler, init_params=initial).output(run_dir, residual_vis_max=spec["residual_vis_max"])

class SingleBandModel:
    """Config-free, immediately built controller for a single band."""
    def __init__(self, *, profiles: LensProfileCollection, observation: SingleBandData,
                 numerics: Mapping[str, Any] | None = None,
                 source_grid_scale: float = 1.0,
                 likelihood_scale: float = 1.0):
        if not isinstance(profiles, LensProfileCollection):
            raise TypeError("profiles must be a LensProfileCollection.")
        if not isinstance(observation, SingleBandData):
            raise TypeError("observation must be a SingleBandData instance.")
        self.profiles, self.observation = profiles, observation
        self.data = observation
        self.definition: ModelDefinition = profiles.as_definition()
        self.numerics, self.lens_image, self.prob_model = dict(numerics or {"supersampling_factor": 1}), None, None
        if source_grid_scale <= 0:
            raise ValueError("source_grid_scale must be positive.")
        self.source_grid_scale = float(source_grid_scale)
        if not np.isfinite(likelihood_scale) or likelihood_scale <= 0:
            raise ValueError("likelihood_scale must be a finite positive number.")
        self.likelihood_scale = float(likelihood_scale)
        self.result: FitResult | None = None
        self.initial_parameters: Mapping[str, Any] | None = None
        self.initialization_path: Path | None = None
        # HMC is deliberately loaded separately from a point-estimate run:
        # the archive may be large, and posterior image products are only
        # evaluated when get_results() is requested.
        self._loaded_hmc_samples: Mapping[str, Any] | None = None
        self._loaded_hmc_details: dict[str, Any] | None = None
        self._build()

    def with_fixed(self, **selections: Any) -> "SingleBandModel":
        """Create an independently built next-stage model with fixed values.

        Selections follow :meth:`LensProfileCollection.with_fixed`; values are
        taken from the current initialized or fitted profile state.
        """
        return SingleBandModel(
            profiles=self.profiles.with_fixed(**selections),
            observation=self.observation,
            numerics=deepcopy(self.numerics),
            source_grid_scale=self.source_grid_scale,
            likelihood_scale=self.likelihood_scale,
        )

    def load(
        self,
        save_path: str | Path,
        *,
        seed: int = 42,
        pixelated_init_match: str = "image",
        num_iterations_warmup: int = 0,
    ) -> Mapping[str, Any]:
        """Load saved parameters into this explicitly declared model.

        A collection directory such as ``pixelated_svi`` automatically selects
        its highest-likelihood ``run_i``.  Supplying ``pixelated_svi/run_2``
        uses that specific run.  The saved ``kwargs_result.json`` and pixelated
        source arrays provide all result state; data and profile declarations
        deliberately remain explicit in the calling notebook/script.
        """
        parameters = self.initialize(
            seed=seed, init_params_path=save_path,
            pixelated_init_match=pixelated_init_match,
            num_iterations_warmup=num_iterations_warmup,
        )
        self.result = None
        return parameters

    def export_wrapper_config(
        self,
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
        """Export this API declaration as a runnable legacy-wrapper config.

        ``path`` may be ``config.py`` or a directory.  The model's data must
        originate from :meth:`SingleBandData.from_fits`, because the wrapper
        reads FITS paths rather than in-memory arrays.
        """
        from .config_export import export_wrapper_config

        return export_wrapper_config(
            self, path, sampler, save_path=save_path, n_runs=n_runs,
            gpus=gpus, init_params_path=init_params_path,
            residual_vis_max=residual_vis_max, wrapper_options=wrapper_options,
        )

    def get_results(
        self,
        parameters: Mapping[str, Any] | None = None,
        *,
        random_seed: int = 42,
    ) -> FitResult:
        """Evaluate and cache derived products for fitted or loaded parameters.

        This includes the model image and its source, lens-light, point-source,
        and component products.  It intentionally remains separate from
        :meth:`load`, so loading a previous run does not immediately perform
        an expensive forward evaluation.
        """
        if parameters is None and self._loaded_hmc_samples is not None:
            return self._get_loaded_hmc_results(random_seed=random_seed)
        if parameters is None:
            if self.initial_parameters is not None:
                parameters = self.initial_parameters
            elif self.result is not None:
                parameters = self.result.parameters
            else:
                raise RuntimeError("Call load(), initialize(), or run() before get_results().")
        from ..samplers import evaluate_parameter_components

        derived = evaluate_parameter_components(
            self.prob_model, parameters, rng_seed=random_seed,
        )
        self.result = FitResult(
            parameters, {"loaded_from": str(self.initialization_path)},
            random_seed=random_seed, derived=derived, _model=self,
        )
        return self.result

    def load_hmc(self, save_path: str | Path) -> None:
        """Load an on-disk HMC posterior into this explicitly declared model.

        ``save_path`` must be the directory containing ``hmc_samples.h5``.
        This method validates the model fingerprint when an HMC manifest is
        present, reads only the posterior samples and sampler-health fields,
        and does *not* evaluate model images.  Call :meth:`get_results` to
        form the posterior-median result and cache its derived products.

        The original ``SingleBandData`` and ``LensProfileCollection`` must
        still be declared by the caller, exactly as for :meth:`load`.
        """
        output = Path(save_path).expanduser()
        samples_path = output / "hmc_samples.h5"
        if not samples_path.is_file():
            raise FileNotFoundError(
                f"No HMC posterior found at {samples_path}. Supply the HMC run directory."
            )

        manifest_path = output / "hmc_manifest.json"
        if manifest_path.is_file():
            stored = json.loads(manifest_path.read_text())
            try:
                expected = self._hmc_manifest(SamplerConfig.hmc(
                    num_chains=int(stored["num_chains"]),
                    checkpoint_interval=int(stored["checkpoint_interval"]),
                ))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid HMC manifest: {manifest_path}") from error
            if stored.get("model_fingerprint") != expected["model_fingerprint"]:
                raise ValueError(
                    "The declared data, profiles, or numerics do not match this HMC posterior. "
                    "Recreate the original SingleBandModel before loading it."
                )

        from ..samplers import _load_hmc_samples_hdf5
        samples, sampler_health = _load_hmc_samples_hdf5(str(samples_path))
        if not samples:
            raise ValueError(f"The HMC posterior at {samples_path} contains no samples.")
        lengths = {np.asarray(values).shape[0] for values in samples.values()}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise ValueError("HMC posterior sample arrays have inconsistent or empty lengths.")
        try:
            import h5py
            with h5py.File(samples_path, "r") as handle:
                num_chains = int(handle.attrs.get("num_chains", 1))
        except Exception as error:
            raise RuntimeError(f"Could not read HMC archive metadata from {samples_path}.") from error
        if num_chains < 1 or next(iter(lengths)) % num_chains:
            raise ValueError("HMC archive has an invalid number of chains or sample layout.")

        self.initialization_path = output
        self._loaded_hmc_samples = samples
        self._loaded_hmc_details = {
            "loaded_from": str(output),
            "hmc_sampler_health": sampler_health,
            "num_chains_hmc_numpyro": num_chains,
            "num_samples_per_chain": next(iter(lengths)) // num_chains,
        }
        self.initial_parameters = None
        self.result = None

    def recompute_hmc_metrics(
        self,
        save_path: str | Path | None = None,
        *,
        random_seed: int = 42,
        write: bool = True,
    ) -> dict[str, float | int | None]:
        """Recompute median and maximum-likelihood metrics from an HMC archive.

        Pass an HMC run directory directly, or first call :meth:`load_hmc`
        and omit ``save_path``.  The posterior is not sampled again: the
        archived draws are evaluated in GPU-sized chunks.  When ``write`` is
        true, the run directory's ``metrics.json`` is replaced with the
        explicit median/max-log-likelihood metric schema.
        """
        if save_path is not None:
            self.load_hmc(save_path)
        if self._loaded_hmc_samples is None:
            raise RuntimeError(
                "Call load_hmc(save_path) first, or pass an HMC run directory."
            )

        # ``get_results`` performs the one necessary posterior forward pass
        # and caches both pixel-wise component medians and the best-likelihood
        # sample summary.  Reuse that cache on repeated calls.
        result = self.result
        if result is None or result.samples is None:
            result = self.get_results(random_seed=random_seed)
        metrics = result.metrics()
        if write:
            from ..samplers import save_metrics
            from ..utils import fit_dof_and_reduced_chi2

            assert self.initialization_path is not None
            save_metrics(
                str(self.initialization_path), metrics["chi2_median"],
                self.data.likelihood_image, self.num_sampling_parameters,
                metrics["log_likelihood_median"], fit_dof_and_reduced_chi2,
                num_params_free=metrics["n_free_parameters"],
                num_params_physical=metrics["n_physical_parameters"],
                mask_bool=self.data.likelihood_mask, metric_summary=metrics,
            )
        return metrics

    def _get_loaded_hmc_results(self, *, random_seed: int) -> FitResult:
        """Create posterior-median cached products from :meth:`load_hmc`."""
        from ..samplers import (
            evaluate_mcmc_component_medians,
            evaluate_mcmc_source_pixels_summary,
            get_active_sample_sites,
            kwargs_with_deterministics,
            tree_median,
        )

        samples = self._loaded_hmc_samples
        assert samples is not None  # narrowed by get_results()
        active_sites = get_active_sample_sites(self.prob_model, rng_seed=random_seed)
        missing = sorted(set(active_sites) - set(samples))
        if missing:
            raise ValueError(
                "The HMC archive is incompatible with this model; missing sample sites: "
                f"{missing}"
            )
        parameters = tree_median({name: samples[name] for name in active_sites})
        components = evaluate_mcmc_component_medians(
            self.prob_model, samples, active_sites=active_sites,
        )
        sample_likelihood = components.pop("_sample_likelihood_summary", None)
        source_summary = evaluate_mcmc_source_pixels_summary(
            self.prob_model, samples, save_path=None, save_npy=False,
        )
        kwargs, deterministics = kwargs_with_deterministics(
            self.prob_model, parameters, rng_seed=random_seed,
            active_sites=active_sites,
        )
        derived: dict[str, Any] = {
            "kwargs": kwargs,
            "deterministics": deterministics,
            "components": components,
            "component_medians": components,
            "model": components["total"],
            "lensed_source": components["source"],
            "lens_light": components["lens_light"],
            "point_source": components["point_source"],
        }
        if sample_likelihood is not None:
            derived["sample_likelihood_summary"] = sample_likelihood
        if self.data.likelihood_image is not None:
            derived["data_minus_lens_light"] = (
                np.asarray(self.data.likelihood_image) - components["lens_light"]
            )
        if source_summary is not None:
            median, lower, upper = source_summary
            derived.update({
                "source_plane": median,
                "source_plane_lower": lower,
                "source_plane_upper": upper,
            })
            # Source-plane visualizations should use the posterior median of
            # the reconstructed physical pixels, not pixels reconstructed
            # from the median Fourier coefficients.
            if kwargs.get("kwargs_source"):
                kwargs = deepcopy(kwargs)
                kwargs["kwargs_source"][0]["pixels"] = median
                derived["kwargs"] = kwargs

        details = dict(self._loaded_hmc_details or {})
        details["derived"] = derived
        self.definition.update_values(parameters)
        self.initial_parameters = parameters
        self.result = FitResult(
            parameters, details, samples, random_seed=random_seed,
            derived=derived, _model=self,
        )
        return self.result

    @property
    def num_sampling_parameters(self) -> int:
        """Number of scalar parameters declared for sampling.

        Scalar priors are fixed, while parameter links are derived from their
        target and therefore are not independent sampling parameters.
        """
        if self.initial_parameters is not None:
            return int(sum(np.asarray(value).size for value in self.initial_parameters.values()))
        _, parameter_lists = self.definition.as_dicts()
        return sum(
            1
            for component in parameter_lists.values()
            for profile in component
            for prior in profile.values()
            if isinstance(prior, (list, tuple))
            and not (len(prior) == 4 and prior[0] == "correlated")
        )
    def _build(self) -> None:
        """Create the backend model from the declared profiles and numerics."""
        create_lens_image, create_prob_model, _, validate_param_list = _model_backend()
        type_list, param_list = self.definition.as_dicts(); validate_param_list(type_list, param_list)
        self.lens_image = create_lens_image(
            param_list, type_list, self.data.likelihood_image,
            self.data.likelihood_noise, self.data.psf, self.data.pixel_scale,
            psf_supersampling_factor=self.data.psf_supersampling_factor,
            kwargs_numerics=self.numerics, source_arc_mask=self.data.source_arc_mask,
            source_grid_scale=self.source_grid_scale,
        )
        self.prob_model = create_prob_model(
            param_list, type_list, self.lens_image, self.data.likelihood_image,
            self.data.likelihood_noise, likelihood_mask=self.data.likelihood_mask,
            args=SimpleNamespace(likelihood_scale=self.likelihood_scale),
        )

    def _declared_pixelated_lens_light_path(self) -> Path | None:
        """Return the one explicit analytic/pixelated lens-light warm-start file."""
        collection = self.profiles.lens_light
        if collection is None:
            return None
        paths = {
            str(profile._initialization["path"])
            for profile in collection
            if profile.profile_type == "PIXELATED"
            and hasattr(profile, "pixelated_prior")
            and profile._initialization is not None
            and profile._initialization["component"] == "lens_light"
        }
        if len(paths) > 1:
            raise ValueError(
                "Pixelated lens-light profiles currently require one common "
                "initialize_from(.../kwargs_result.json) file."
            )
        return Path(next(iter(paths))).expanduser() if paths else None
    def initialize(
        self,
        *,
        seed: int = 42,
        run_id: int | str | None = None,
        init_params_path: str | Path | None = None,
        pixelated_init_match: str = "image",
        num_iterations_warmup: int = 0,
    ) -> Mapping[str, Any]:
        """Create the constrained SVI start point with ``init_to_median``.

        A fresh model uses NumPyro's ``init_to_median(num_samples=25)`` with
        ``seed``.  Passing the returned parameters to :meth:`run` therefore
        makes the displayed initial model the actual SVI initialization.  An
        ``init_params_path`` remains a deliberate warm start from a prior run.
        For a pixelated source, ``pixelated_init_match='image'`` runs a short
        SVI warmup with inherited lens mass/light held fixed; ``'source'``
        fits Matérn hyperparameters to the inherited analytic source.
        """
        if run_id is not None:
            print("\n========================================")
            print(f"Starting Run {run_id} (seed={seed})")
            print("========================================")
        if self.profiles.apply_initializations():
            # File-declared parameters change priors into fixed scalars.  The
            # immediately-built backend must therefore be regenerated before
            # NumPyro discovers its sample sites.
            self.definition = self.profiles.as_definition()
            self._build()
        declared_lens_light_path = self._declared_pixelated_lens_light_path()
        if init_params_path is None and declared_lens_light_path is not None:
            init_params_path = declared_lens_light_path
        _, _, get_init_params, _ = _model_backend()
        if init_params_path is None:
            self.initialization_path = None
        else:
            from ..utils import resolve_init_run_dir
            self.initialization_path = Path(resolve_init_run_dir(init_params_path)).expanduser()
        if init_params_path is None:
            import jax
            from numpyro import infer
            from numpyro.infer.util import initialize_model

            model_info = initialize_model(
                jax.random.PRNGKey(seed), self.prob_model.model,
                init_strategy=infer.init_to_median(num_samples=25),
                validate_grad=False,
            )
            initial = {
                name: site["value"]
                for name, site in model_info.model_trace.items()
                if site["type"] == "sample" and not site["is_observed"]
            }
        else:
            type_list, param_list = self.definition.as_dicts()
            initial = get_init_params(
                self.prob_model, param_list, type_list,
                init_params_path=self.initialization_path, random_seed=seed,
                lens_image=self.lens_image,
            )
            type_list, param_list = self.definition.as_dicts()
            is_pixelated = type_list.get("source_light_type_list") == ["PIXELATED"]
            if pixelated_init_match not in {"image", "source"}:
                raise ValueError("pixelated_init_match must be 'image' or 'source'.")
            if is_pixelated and pixelated_init_match == "source":
                from ..models import PowerSpectrum

                iterations = num_iterations_warmup or 2_000
                if not isinstance(iterations, int) or iterations <= 0:
                    raise ValueError("num_iterations_warmup must be a positive integer for source matching.")
                ny, nx = self.lens_image.SourceModel.pixel_grid.num_pixel_axes
                pixelated_prior = param_list["source_light_params_list"][0].get("pixelated_prior", {})
                print(
                    f"[pixelated-init: source] Fitting Matérn parameters "
                    f"({iterations} iterations) from the inherited analytic source..."
                )
                power_values = PowerSpectrum.fit_power_spectrum_init_from_parametric_source(
                    self.lens_image, str(init_params_path),
                    PowerSpectrum.K_grid((ny, nx)).k, pixelated_prior,
                    seed=seed + 7919, max_iterations=iterations,
                )
                for name, value in power_values.items():
                    if name in initial:
                        initial[name] = value
                print("[pixelated-init: source] Source-matched initialization complete.")
            if is_pixelated and pixelated_init_match == "image" and num_iterations_warmup > 0:
                if not isinstance(num_iterations_warmup, int):
                    raise TypeError("num_iterations_warmup must be an integer.")
                create_lens_image, create_prob_model, _, _ = _model_backend()
                _, _, run_svi = _sampler_backend()
                initial_kwargs = self.prob_model.params2kwargs(initial)
                warmup_model = create_prob_model(
                    param_list, type_list, self.lens_image,
                    self.data.likelihood_image, self.data.likelihood_noise,
                    fix_lens_mass=bool(initial_kwargs.get("kwargs_lens")),
                    kwargs_lens_fixed=initial_kwargs.get("kwargs_lens"),
                    fix_lens_light=bool(initial_kwargs.get("kwargs_lens_light")),
                    kwargs_lens_light_fixed=initial_kwargs.get("kwargs_lens_light"),
                    init_params_path=str(init_params_path),
                    likelihood_mask=self.data.likelihood_mask,
                    args=SimpleNamespace(likelihood_scale=self.likelihood_scale),
                )
                warmup = SamplerConfig.svi(
                    max_iterations=num_iterations_warmup, random_seed=seed,
                ).to_namespace()
                print(
                    f"[svi-warmup] Starting {num_iterations_warmup} iteration "
                    "pixelated-source image-match warmup..."
                )
                warmup_params, _ = run_svi(
                    warmup_model, self.data.likelihood_image, warmup, initial,
                    max_iterations=num_iterations_warmup,
                )
                for name, value in warmup_params.items():
                    if "source" in name or "pixels_wn" in name:
                        initial[name] = value
                print("[svi-warmup] Pixelated-source image-match warmup complete.")
            pixelated_lens_indices = [
                index for index, profile_type in enumerate(type_list.get("lens_light_type_list", []))
                if profile_type == "PIXELATED"
            ]
            if pixelated_lens_indices:
                from ..models import PowerSpectrum, load_kwargs_init_json

                saved = load_kwargs_init_json(self.initialization_path)
                saved_lens_light = saved.get("kwargs_lens_light", [])
                iterations = num_iterations_warmup or 2_000
                if not isinstance(iterations, int) or iterations <= 0:
                    raise ValueError(
                        "num_iterations_warmup must be positive for pixelated lens-light matching."
                    )
                for index in pixelated_lens_indices:
                    saved_values = (
                        saved_lens_light[index]
                        if index < len(saved_lens_light) and isinstance(saved_lens_light[index], Mapping)
                        else None
                    )
                    if saved_values is not None and saved_values.get("pixels_wn") is not None:
                        # get_init_params() already restored the saved latent.
                        continue
                    pixelated_prior = param_list["lens_light_params_list"][index].get("pixelated_prior", {})
                    ny, nx = self.lens_image.LensLightModel.pixel_grid.num_pixel_axes
                    print(
                        f"[pixelated-init: lens-light] Fitting Matérn parameters "
                        f"({iterations} iterations) from inherited analytic/MGE lens light..."
                    )
                    power_values = PowerSpectrum.fit_power_spectrum_init_from_parametric_lens_light(
                        self.lens_image, str(self.initialization_path),
                        PowerSpectrum.K_grid((ny, nx)).k, pixelated_prior,
                        seed=seed + 17863 + index, max_iterations=iterations,
                        lens_light_index=index,
                    )
                    for name, value in power_values.items():
                        if name in initial:
                            initial[name] = value
                print("[pixelated-init: lens-light] Lens-light-matched initialization complete.")
        self.definition.update_values(initial); self.initial_parameters = initial
        return initial

    def _hmc_manifest(self, sampler: SamplerConfig) -> dict[str, Any]:
        """Return the immutable identity of an HMC chain and its model."""
        types, parameters = self.definition.as_dicts()

        def _array_digest(values: Any) -> str:
            if values is None:
                return "none"
            array = np.ascontiguousarray(np.asarray(values))
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            return digest.hexdigest()

        def _json_value(value: Any):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"Cannot serialize {type(value).__name__} in the HMC manifest.")

        model_payload = {
            "type_list": types,
            "parameter_list": parameters,
            "numerics": self.numerics,
            "source_grid_scale": self.source_grid_scale,
            "likelihood_scale": self.likelihood_scale,
            "pixel_scale": self.data.pixel_scale,
            "data": {
                "image": _array_digest(self.data.likelihood_image),
                "noise": _array_digest(self.data.likelihood_noise),
                "psf": _array_digest(self.data.psf),
                "mask": _array_digest(self.data.likelihood_mask),
            },
        }
        encoded = json.dumps(model_payload, sort_keys=True, default=_json_value).encode()
        return {
            "format_version": 1,
            "model_fingerprint": hashlib.sha256(encoded).hexdigest(),
            "num_chains": int(sampler.options["num_chains_hmc_numpyro"]),
            "checkpoint_interval": int(sampler.options["checkpoint_interval_hmc_numpyro"]),
        }

    def _validate_hmc_run_directory(self, output: Path, sampler: SamplerConfig) -> bool:
        """Create or validate metadata required to safely continue HMC."""
        checkpoint = output / "hmc_checkpoint.pkl"
        manifest_path = output / "hmc_manifest.json"
        expected = self._hmc_manifest(sampler)
        if checkpoint.exists():
            if manifest_path.exists():
                stored = json.loads(manifest_path.read_text())
                for key in ("model_fingerprint", "num_chains", "checkpoint_interval"):
                    if stored.get(key) != expected[key]:
                        raise ValueError(
                            f"Cannot resume HMC: {key} differs from the existing chain. "
                            "Use the original model, num_chains, and checkpoint_interval."
                        )
            else:
                # Compatibility with chains produced before the API manifest
                # existed.  The backend still checks chain count and interval.
                manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
                print(f"[hmc] Existing checkpoint has no manifest; wrote {manifest_path} for future resumes.")
            return True
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
        return False

    def resume_hmc(self, sampler: SamplerConfig, *, save_path: str | Path) -> FitResult:
        """Continue an interrupted HMC chain stored in ``save_path``.

        ``sampler.num_samples`` is the desired *total* retained draws per
        chain, not the number of extra draws.  ``num_chains`` and
        ``checkpoint_interval`` must match the original run.  A prior SVI
        initialization is not needed here because the checkpoint contains the
        warmed-up NUTS state.
        """
        if sampler.name != "hmc":
            raise TypeError("resume_hmc() requires SamplerConfig.hmc(...).")
        output = Path(save_path).expanduser()
        if not (output / "hmc_checkpoint.pkl").is_file():
            raise FileNotFoundError(
                f"No HMC checkpoint found at {output / 'hmc_checkpoint.pkl'}. "
                "Use model.run(hmc, ...) to start a new chain."
            )
        from ..samplers import _load_hmc_samples_hdf5

        samples_path = output / "hmc_samples.h5"
        if not samples_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume HMC: {samples_path} is missing. The checkpoint alone "
                "does not contain the posterior state required by the API sampler."
            )
        archived_samples, _ = _load_hmc_samples_hdf5(str(samples_path))
        if not archived_samples:
            raise ValueError("Cannot resume HMC: the posterior archive contains no samples.")
        # ``run_hmc`` receives these only to reconstruct its NUTS/Gibbs kernel
        # structure; the saved warmup state, rather than this draw, continues
        # the Markov chains.
        initial = {name: np.asarray(values)[-1] for name, values in archived_samples.items()}
        return self.run(sampler, init_params=initial, save_path=output)
    def run(
        self,
        sampler: SamplerConfig,
        *,
        init_params: Mapping[str, Any] | None = None,
        save_path: str | Path | None = None,
        n_runs: int = 1,
        parallel: bool = False,
        gpus: str | Sequence[str] | None = None,
        pixelated_init_match: str = "image",
        num_iterations_warmup: int = 0,
        residual_vis_max: float = 0.0,
    ) -> FitResult | "SingleBandResultsCombination":
        """Run inference from supplied or automatically initialized parameters.

        ``n_runs > 1`` launches independent SVI restarts and returns a
        :class:`SingleBandResultsCombination`.  With ``parallel=True``, each
        selected entry in ``gpus`` is used by at most one spawned SVI process
        at a time; entries may be CUDA indices or MIG UUIDs.  This is distinct
        from HMC's within-run chain parallelism (``chain_method='parallel'``).
        """
        if not isinstance(n_runs, int) or n_runs < 1:
            raise ValueError("n_runs must be a positive integer.")
        if n_runs > 1:
            return self._run_svi_many(
                sampler,
                save_path=save_path,
                n_runs=n_runs,
                parallel=parallel,
                gpus=gpus,
                init_params=init_params,
                pixelated_init_match=pixelated_init_match,
                num_iterations_warmup=num_iterations_warmup,
                residual_vis_max=residual_vis_max,
            )
        if init_params is not None:
            initial = dict(init_params)
        elif self.initial_parameters is not None:
            initial = dict(self.initial_parameters)
        else:
            initial = self.initialize(seed=sampler.random_seed)
        self.definition.update_values(initial)
        run_hmc, run_optax, run_svi = _sampler_backend(); args = sampler.to_namespace()
        if sampler.name == "optax": params, details, samples = *run_optax(self.prob_model, args, initial), None
        elif sampler.name == "svi": params, details, samples = *run_svi(self.prob_model, self.data.likelihood_image, args, initial), None
        elif sampler.name == "hmc":
            if save_path is None:
                raise ValueError("HMC requires save_path for checkpoints and posterior samples.")
            output = Path(save_path).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            resuming = self._validate_hmc_run_directory(output, sampler)
            if self.initialization_path is None and not resuming:
                raise ValueError(
                    "HMC requires model.initialize(init_params_path=...) before model.run()."
                )
            args.save_path = str(output)
            samples, params, details = run_hmc(
                self.prob_model, args, initial,
                init_params_path=(str(self.initialization_path) if self.initialization_path is not None else None),
            )
            # ``run_hmc`` flattens chain and draw axes before returning the
            # posterior samples.  Retain the original chain count so result
            # visualizations can reconstruct one summary per chain.
            details["num_chains_hmc_numpyro"] = int(args.num_chains_hmc_numpyro)
        else: raise ValueError(f"Unknown sampler {sampler.name!r}.")
        from ..samplers import evaluate_parameter_components, kwargs_with_deterministics
        derived = dict(details.get("derived", {}))
        if sampler.name == "hmc":
            kwargs, deterministics = kwargs_with_deterministics(
                self.prob_model, params, rng_seed=sampler.random_seed,
            )
            derived.update({"kwargs": kwargs, "deterministics": deterministics})
        else:
            derived = evaluate_parameter_components(
                self.prob_model, params, rng_seed=sampler.random_seed,
            )
        details["derived"] = derived
        self.definition.update_values(params); self.result = FitResult(
            params, details, samples, random_seed=sampler.random_seed,
            derived=derived, _model=self,
        )
        return self.result

    def _run_svi_many(
        self,
        sampler: SamplerConfig,
        *,
        save_path: str | Path | None,
        n_runs: int,
        parallel: bool,
        gpus: str | Sequence[str] | None,
        init_params: Mapping[str, Any] | None,
        pixelated_init_match: str,
        num_iterations_warmup: int,
        residual_vis_max: float,
    ) -> "SingleBandResultsCombination":
        """Run independent SVI restarts, optionally one process per CUDA/MIG device."""
        if sampler.name != "svi":
            raise ValueError("n_runs > 1 is currently supported only for independent SVI runs.")
        if save_path is None:
            raise ValueError("Multi-run SVI requires save_path for run_i outputs and resuming.")
        if init_params is not None:
            raise ValueError(
                "Pass no init_params when n_runs > 1: each SVI restart must be initialized "
                "from its own seed.  Call model.initialize(init_params_path=...) first to "
                "set a previous-stage initialization path."
            )
        if num_iterations_warmup < 0:
            raise ValueError("num_iterations_warmup must be non-negative.")
        if residual_vis_max < 0:
            raise ValueError("residual_vis_max must be non-negative.")

        if gpus is None:
            devices: list[str] = []
        elif isinstance(gpus, str):
            devices = [item.strip() for item in gpus.split(",") if item.strip()]
        else:
            devices = [str(item).strip() for item in gpus if str(item).strip()]
        if parallel and not devices:
            raise ValueError("parallel=True requires one or more CUDA/MIG identifiers in gpus.")

        directory = Path(save_path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        # A self-contained worker specification deliberately copies the API
        # declarations, rather than the already-built JAX backend.  Spawned
        # children can therefore set CUDA_VISIBLE_DEVICES before JAX first
        # touches a GPU.
        spec = {
            "model": {
                "profiles": deepcopy(self.profiles),
                "observation": deepcopy(self.observation),
                "numerics": deepcopy(self.numerics),
                "source_grid_scale": self.source_grid_scale,
                "likelihood_scale": self.likelihood_scale,
            },
            "directory": str(directory),
            "seed": int(sampler.random_seed),
            "options": deepcopy(sampler.options),
            "init_path": None if self.initialization_path is None else str(self.initialization_path),
            "pixelated_init_match": pixelated_init_match,
            "warmup": int(num_iterations_warmup),
            "residual_vis_max": float(residual_vis_max),
        }
        from .samplers import SingleBandResultsCombination, is_completed_svi_run

        pending = [run_id for run_id in range(n_runs)
                   if not is_completed_svi_run(directory / f"run_{run_id}")]
        if pending:
            if parallel:
                print(
                    f"Starting {len(pending)} independent SVI run(s) across "
                    f"{len(devices)} CUDA/MIG device(s): {directory}"
                )
                context = mp.get_context("spawn")
                for first in range(0, len(pending), len(devices)):
                    batch_ids = pending[first:first + len(devices)]
                    workers = [
                        context.Process(
                            target=_svi_many_worker,
                            args=(spec, run_id, devices[index]),
                        )
                        for index, run_id in enumerate(batch_ids)
                    ]
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join()
                    failures = [worker.exitcode for worker in workers if worker.exitcode]
                    if failures:
                        raise RuntimeError(f"One or more parallel SVI runs failed (exit codes: {failures}).")
            else:
                for run_id in pending:
                    _svi_many_worker(spec, run_id, None)
        else:
            print(f"All {n_runs} SVI runs are already complete; reloading outputs from {directory}.")

        # Process-local result objects are intentionally reconstructed from
        # disk after workers exit.  This also gives serial and parallel paths
        # identical resumable behaviour.
        results: list[FitResult] = []
        for run_id in range(n_runs):
            restored = SingleBandModel(**deepcopy(spec["model"]))
            run_dir = directory / f"run_{run_id}"
            restored.load(run_dir, seed=int(sampler.random_seed) + run_id)
            results.append(restored.get_results(random_seed=int(sampler.random_seed) + run_id))
        return SingleBandResultsCombination(results)
    def model_image(self, parameters: Mapping[str, Any] | None = None) -> np.ndarray:
        if self.prob_model is None or self.lens_image is None: raise RuntimeError("The backend model is unavailable.")
        if parameters is None:
            if self.result is not None: parameters = self.result.parameters
            elif self.initial_parameters is not None: parameters = self.initial_parameters
            elif self.definition.has_free_parameters: raise RuntimeError("Supply parameters or call run() first.")
            else: parameters = {}
        return np.asarray(self.lens_image.model(**self.prob_model.params2kwargs(parameters)))

    def mass_component_convergence(
        self, parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return total, stellar, and gNFW convergence maps when present.

        The maps are evaluated directly from the same compiled mass profiles
        used during inference, so their sum is exactly the relevant part of
        the fitted lens mass model.
        """
        if self.prob_model is None or self.lens_image is None:
            raise RuntimeError("The backend model is unavailable.")
        if parameters is None:
            if self.result is None:
                raise RuntimeError("Supply parameters or call run() first.")
            parameters = self.result.parameters
        kwargs_lens = self.prob_model.params2kwargs(parameters)["kwargs_lens"]
        x_grid, y_grid = self.lens_image.Grid.pixel_coordinates
        types, _ = self.definition.as_dicts()
        result: dict[str, np.ndarray] = {}
        component_types = types["lens_mass_type_list"]
        labels = {"STELLAR_MGE": "stellar", "GNFW_MGE": "dark_matter"}
        for index, profile_type in enumerate(component_types):
            label = labels.get(profile_type)
            if label is None:
                continue
            image = np.asarray(self.lens_image.MassModel.kappa(x_grid, y_grid, kwargs_lens, k=index))
            result[label] = image if label not in result else result[label] + image
        result["total"] = np.asarray(self.lens_image.MassModel.kappa(x_grid, y_grid, kwargs_lens))
        return result
    def plot_fit(self, parameters: Mapping[str, Any] | None = None, *, scale: PlotScale = "linear",
                 residual_vis_max: float = 0.0, save_path: str | Path | None = None):
        """Plot data, model image, and normalized residual for a parameter set.

        ``scale`` applies to data and model; the normalized residual remains
        on a symmetric linear scale so its amplitude stays interpretable.
        """
        import matplotlib.pyplot as plt
        if scale not in ("linear", "log"):
            raise ValueError("scale must be either 'linear' or 'log'.")
        if residual_vis_max < 0:
            raise ValueError("residual_vis_max must be non-negative.")
        model = self.model_image(parameters); residual = (model - self.data.likelihood_image) / self.data.likelihood_noise
        valid = np.isfinite(residual)
        if self.data.likelihood_mask is not None:
            valid &= np.asarray(self.data.likelihood_mask, dtype=bool)
        chi2 = float(np.sum(np.square(residual[valid])))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        extent = _extent(self.data.image.shape, self.data.pixel_scale)
        for axis, image, title, cmap in zip(axes, (self.data.likelihood_image, model, residual), ("Data", "Model", "Normalized residual"), ("twilight", "twilight", "coolwarm")):
            if title == "Normalized residual":
                finite = np.abs(image[np.isfinite(image)])
                limit = float(residual_vis_max) if residual_vis_max > 0 else (float(np.max(finite)) if finite.size else 1.0)
                limit = limit if limit > 0 else 1.0
                rendered = axis.imshow(image, origin="lower", cmap=cmap, extent=extent, vmin=-limit, vmax=limit)
            else:
                rendered = axis.imshow(image, origin="lower", cmap=cmap, extent=extent, norm=_normalization(image, scale, signed=True))
            if title in ("Data", "Normalized residual"):
                if self.data.source_arc_mask is not None:
                    axis.contour(self.data.source_arc_mask, levels=[0.5], colors="lime", linewidths=1.0, extent=extent)
                if self.data.contaminate_mask is not None:
                    axis.contour(self.data.contaminate_mask, levels=[0.5], colors="orange", linewidths=1.2, linestyles="--", extent=extent)
            label = "Standardized residual" if title == "Normalized residual" else "Pixel flux"
            if scale == "log" and title != "Normalized residual": label += " (log scale)"
            plot_title = f"{title} ($\\chi^2$ = {chi2:.2f})" if title == "Normalized residual" else title
            axis.set(title=plot_title, xlabel="arcsec", ylabel="arcsec"); fig.colorbar(rendered, ax=axis, shrink=0.85, label=label)
        if save_path is not None:
            output = Path(save_path); output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=200, bbox_inches="tight")
        return fig, axes
    def plot_initial_model(self, *, scale: PlotScale = "linear", residual_vis_max: float = 0.0,
                     save_path: str | Path | None = None):
        """Visualize the random initial model, creating it first if needed."""
        if self.initial_parameters is None: self.initialize()
        return self.plot_fit(self.initial_parameters, scale=scale, residual_vis_max=residual_vis_max, save_path=save_path)

    def plot_initial_source(
        self,
        *,
        scale: PlotScale = "linear",
        residual_vis_max: float = 0.0,
        save_path: str | Path | None = None,
        source_pixel_scale: float = 0.01,
        num_pixel: int = 200,
    ) -> Path:
        """Plot the source plane corresponding to the current SVI start point.

        For pixelated models this displays the initialized source grid; for a
        parametric model it renders the initial analytic source.  The
        ``residual_vis_max`` argument is accepted for a uniform plotting API
        but does not apply to a source-plane image.
        """
        if scale not in ("linear", "log"):
            raise ValueError("scale must be either 'linear' or 'log'.")
        if residual_vis_max < 0:
            raise ValueError("residual_vis_max must be non-negative.")
        if self.initial_parameters is None:
            self.initialize()
        from ..visualizations import plot_source_plane

        if save_path is None:
            directory = Path(tempfile.mkdtemp(prefix="herculens_initial_source_"))
            filename = "initial_source_plane.png"
        else:
            target = Path(save_path).expanduser()
            directory, filename = (
                (target.parent, target.name) if target.suffix else (target, "initial_source_plane.png")
            )
            directory.mkdir(parents=True, exist_ok=True)
        plot_source_plane(
            self.lens_image, self.prob_model.params2kwargs(self.initial_parameters),
            str(directory), source_pixel_scale=source_pixel_scale, num_pixel=num_pixel,
            plot_scale=scale, output_filename=filename,
            source_arc_mask=self.data.source_arc_mask,
        )
        output = directory / filename
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import Image, display
                display(Image(filename=str(output)))
        except Exception:
            pass
        return output

    def _metrics(self, parameters: Mapping[str, Any]) -> dict[str, float | int | None]:
        """Evaluate metrics at the coordinate-wise posterior median parameters."""
        if self.prob_model is None:
            raise RuntimeError("Call build() before evaluating metrics.")
        model_image = self.model_image(parameters)
        valid = np.isfinite(self.data.likelihood_image) & np.isfinite(self.data.likelihood_noise) & (self.data.likelihood_noise > 0)
        if self.data.likelihood_mask is not None:
            valid &= np.asarray(self.data.likelihood_mask, dtype=bool)
        residual = (self.data.likelihood_image - model_image) / self.data.likelihood_noise
        chi2 = float(np.sum(np.square(residual[valid])))
        n_data = int(np.sum(valid))
        n_free = int(sum(np.asarray(value).size for value in parameters.values()))
        n_physical = count_physical_parameters(parameters)
        dof = max(n_data - n_free, 1)

        # NumPyro evaluates the exact model likelihood, priors, and any
        # registered factor penalties.  Observed sample sites are likelihoods.
        from numpyro.infer.util import log_density

        log_probability, trace = log_density(self.prob_model.model, (), {}, parameters)
        log_probability_value = float(np.asarray(log_probability))
        log_likelihood = 0.0
        for site in trace.values():
            if site.get("type") == "sample" and site.get("is_observed", False):
                scale = site.get("scale", 1.0)
                scale = 1.0 if scale is None else float(np.asarray(scale))
                log_likelihood += scale * float(np.asarray(site["fn"].log_prob(site["value"])).sum())
        log_prior_and_penalties = log_probability_value - log_likelihood
        bic_physical = n_physical * np.log(max(n_data, 1)) - 2.0 * log_likelihood
        return {
            "log_likelihood_median": float(log_likelihood),
            "log_probability_median": log_probability_value,
            "log_prior_and_penalties_median": float(log_prior_and_penalties),
            "chi2_median": chi2,
            "reduced_chi2_median": float(chi2 / dof),
            "bic_physical_median": float(bic_physical),
            "n_data_pixels": n_data,
            "n_free_parameters": n_free,
            "n_physical_parameters": n_physical,
            "degrees_of_freedom": int(dof),
        }
