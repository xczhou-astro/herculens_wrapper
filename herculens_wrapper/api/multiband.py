"""First-generation config-free API for joint multi-band modelling."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .collections import LensProfileCollection
from .data import SingleBandData
from .models import ModelDefinition, count_physical_parameters
from .samplers import SamplerConfig


_FILES = {
    "data": ("Data_cutout.fits", "data.fits", "image.fits"),
    "noise": ("noise.fits", "noise_map.fits"),
    "psf": ("psf.fits", "psf_modelled.fits"),
}


class MultiBandData:
    """Ordered band-name to :class:`SingleBandData` collection.

    A band may be a ``SingleBandData``, a directory path with standard FITS
    names, or a dictionary containing ``data``, ``noise`` and ``psf``.
    Directory inputs require the explicit ``pixel_scale={band: value}`` map.
    """
    def __init__(self, *, pixel_scale: Mapping[str, float] | None = None, **bands: Any):
        if not bands:
            raise ValueError("MultiBandData requires at least one named band.")
        if pixel_scale is not None and not isinstance(pixel_scale, Mapping):
            raise TypeError("pixel_scale must be a mapping such as {'F150W': 0.03}.")
        self._bands: dict[str, SingleBandData] = {}
        for name, value in bands.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Each band name must be a non-empty string.")
            scale = None if pixel_scale is None else pixel_scale.get(name)
            self._bands[name] = self._coerce(name, value, scale)

    @staticmethod
    def _find(directory: Path, role: str) -> Path:
        for filename in _FILES[role]:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"{directory}: no {role} file found among {_FILES[role]}.")

    def _coerce(self, name: str, value: Any, inherited_scale: float | None) -> SingleBandData:
        if isinstance(value, SingleBandData):
            return value
        if isinstance(value, (str, Path)):
            directory = Path(value).expanduser()
            if not directory.is_dir():
                raise NotADirectoryError(f"{name}: {directory} is not a band directory.")
            if inherited_scale is None:
                raise ValueError(f"{name}: provide pixel_scale={{'{name}': ...}} for directory input.")
            kwargs: dict[str, Any] = {}
            source = directory / "mask_1.fits"
            contaminate = directory / "mask_out.fits"
            if source.is_file(): kwargs["source_arc_mask_path"] = str(source)
            if contaminate.is_file(): kwargs["contaminate_mask_path"] = str(contaminate)
            return SingleBandData.from_fits(self._find(directory, "data"), self._find(directory, "noise"), self._find(directory, "psf"), pixel_scale=inherited_scale, **kwargs)
        if not isinstance(value, Mapping):
            raise TypeError(f"{name}: band input must be SingleBandData, a directory, or a dictionary.")
        spec = dict(value)
        scale = spec.pop("pixel_scale", inherited_scale)
        if scale is None:
            raise ValueError(f"{name}: pixel_scale is required.")
        data, noise, psf = (spec.pop(key, None) for key in ("data", "noise", "psf"))
        if any(item is None for item in (data, noise, psf)):
            raise ValueError(f"{name}: dictionary input requires data, noise, and psf.")
        if all(isinstance(item, (str, Path)) for item in (data, noise, psf)):
            return SingleBandData.from_fits(data, noise, psf, pixel_scale=scale, **spec)
        return SingleBandData(image=data, noise=noise, psf=psf, pixel_scale=scale, **spec)

    def __getitem__(self, name: str) -> SingleBandData: return self._bands[name]
    def __getattr__(self, name: str) -> SingleBandData:
        try: return self._bands[name]
        except KeyError as error: raise AttributeError(name) from error
    def items(self): return self._bands.items()
    @property
    def band_names(self) -> tuple[str, ...]: return tuple(self._bands)
    def save(self, path: str | Path) -> dict[str, dict[str, Path]]:
        root = Path(path); return {name: data.save(root / name) for name, data in self.items()}
    def show(
        self,
        *,
        scale: str = "linear",
        residual_vis_max: float = 0.0,
        save_path: str | Path | None = None,
    ):
        """Show all bands in a ``N_band × 4`` data diagnostic figure."""
        from .visualization import plot_multiband_data
        return plot_multiband_data(
            self._bands, scale=scale, residual_vis_max=residual_vis_max,
            save_path=save_path,
        )


class MultiBandProfileCollection:
    """Shared lens mass plus per-band light/source profile declarations."""
    def __init__(self, *, shared: LensProfileCollection, bands: Mapping[str, LensProfileCollection],
                 unshared: Mapping[str, Any] | None = None):
        if shared.lens_mass is None:
            raise ValueError("shared must define lens_mass.")
        if not bands: raise ValueError("bands must not be empty.")
        unknown = set(unshared or {}) - set(bands)
        if unknown: raise KeyError(f"unshared names unknown band(s): {sorted(unknown)}")
        self.shared, self.bands, self.unshared = shared, dict(bands), deepcopy(unshared or {})
        for name, collection in self.bands.items():
            if collection.lens_mass is not None:
                raise ValueError(f"{name}: lens_mass belongs in shared; use unshared for centre priors.")

    def band_definitions(self) -> dict[str, tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]]:
        shared_types, shared_params = self.shared.as_definition().as_dicts()
        result = {}
        for name, collection in self.bands.items():
            types, params = collection.as_definition().as_dicts()
            types["lens_mass_type_list"] = deepcopy(shared_types["lens_mass_type_list"])
            params["lens_mass_params_list"] = deepcopy(shared_params["lens_mass_params_list"])
            # ``unshared`` is retained temporarily for compatibility, while
            # MassProfile.set_independent() is the preferred public spelling.
            changes = deepcopy(self.unshared.get(name, {}).get("lens_mass", {}))
            for index, profile in enumerate(self.shared.lens_mass):
                declared = profile.independent_parameters.get(name, {})
                if declared:
                    changes.setdefault(index, {}).update(declared)
            for index, values in changes.items():
                if not isinstance(index, int) or not 0 <= index < len(params["lens_mass_params_list"]):
                    raise IndexError(f"{name}: invalid lens_mass profile index {index!r}.")
                for key, prior in values.items():
                    if key not in {"center_x", "center_y"}:
                        raise ValueError("First multiband API version only permits unshared lens_mass center_x/center_y.")
                    if key not in params["lens_mass_params_list"][index]:
                        raise KeyError(f"{name}: lens_mass[{index}] has no {key!r}.")
                    params["lens_mass_params_list"][index][key] = deepcopy(prior)
            result[name] = types, params
        return result

    def apply_initializations(self) -> bool:
        """Resolve any explicit ``Profile.initialize_from`` declarations."""
        changed = self.shared.apply_initializations()
        for collection in self.bands.values():
            changed = collection.apply_initializations() or changed
        return changed


@dataclass
class MultiBandFitResult:
    parameters: Mapping[str, Any]
    details: Mapping[str, Any]
    _model: Any = None
    random_seed: int | None = None
    initial_parameters: Mapping[str, Any] | None = None
    samples: Mapping[str, Any] | None = None
    def kwargs_by_band(self) -> dict[str, Any]:
        return self._model.prob_model.params2kwargs_by_band(self.parameters)

    def _shared_lens_kwargs(self, kwargs_by_band: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return the shared lens block, excluding explicitly independent keys."""
        reference_band = self._model.observations.band_names[0]
        shared = deepcopy(kwargs_by_band[reference_band]["kwargs_lens"])
        mass_profiles = self._model.profiles.shared.lens_mass
        if not isinstance(mass_profiles, (list, tuple)) and not hasattr(mass_profiles, "profiles"):
            mass_profiles = [mass_profiles]
        for index, profile in enumerate(mass_profiles):
            for keys in profile.independent_parameters.values():
                for key in keys:
                    shared[index].pop(key, None)
        return shared

    def _joint_corner_site_order(self) -> list[str]:
        """Return joint latent sites in physical declaration order."""
        order: list[str] = []
        for index, definition in enumerate(self._model.prob_model.lens_mass_params_list):
            for key in definition:
                if key not in {"center_x", "center_y"}:
                    order.append(f"lens_{key}_{index}")
        for band in self._model.bands:
            prefix = f"{band['site_prefix']}/"
            parameters = band["param_list"]
            for index, definition in enumerate(parameters.get("lens_mass_params_list", [])):
                for key in definition:
                    if key in {"center_x", "center_y"}:
                        order.append(f"{prefix}lens_{key}_{index}")
            for index, definition in enumerate(parameters.get("lens_light_params_list", [])):
                order.extend(f"{prefix}lens_light_{key}_{index}" for key in definition)
            for index, definition in enumerate(parameters.get("source_light_params_list", [])):
                order.extend(f"{prefix}source_{key}_{index}" for key in definition)
            for index, definition in enumerate(parameters.get("point_source_params_list", [])):
                order.extend(f"{prefix}ps_{key}_{index}" for key in definition)
        return order

    @property
    def loss_history(self) -> np.ndarray | None:
        history = self.details.get("loss_history")
        return None if history is None else np.asarray(history)

    def metrics(self) -> dict[str, Any]:
        """Return joint and per-band Gaussian-residual fit metrics."""
        per_band, total_chi2, total_pixels, log_likelihood = {}, 0.0, 0, 0.0
        for band, kwargs in zip(self._model.bands, self.kwargs_by_band().values()):
            residual = (np.asarray(band["lens_image"].model(**kwargs)) - band["image_data"]) / band["noise_map"]
            valid = np.isfinite(residual)
            if band["fit_mask_bool"] is not None: valid &= np.asarray(band["fit_mask_bool"], bool)
            chi2, n_pixels = float(np.sum(residual[valid] ** 2)), int(np.sum(valid))
            noise = np.asarray(band["noise_map"])[valid]
            per_band[band["name"]] = {"chi2": chi2, "n_data_pixels": n_pixels}
            total_chi2 += chi2; total_pixels += n_pixels
            log_likelihood += float(
                -0.5 * self._model.likelihood_scale
                * np.sum(residual[valid] ** 2 + np.log(2 * np.pi * noise ** 2))
            )
        n_parameters = int(sum(np.asarray(value).size for value in self.parameters.values()))
        n_physical = count_physical_parameters(self.parameters)
        dof = max(total_pixels - n_parameters, 1)
        bic_physical = n_physical * np.log(total_pixels) - 2 * log_likelihood
        metrics = {"chi2_median": total_chi2, "n_data_pixels": total_pixels, "n_free_parameters": n_parameters,
                "n_physical_parameters": n_physical,
                "degrees_of_freedom": dof, "reduced_chi2_median": total_chi2 / dof,
                "log_likelihood_median": log_likelihood,
                "bic_physical_median": bic_physical,
                "bands": per_band}
        summary = self.details.get("sample_likelihood_summary") if self.samples is not None else None
        if summary is None:
            metrics.update({"max_log_likelihood": None, "chi2_max_loglike": None,
                            "reduced_chi2_max_loglike": None,
                            "bic_physical_max_loglike": None,
                            "max_loglike_sample_index": None})
        else:
            max_loglike = float(summary["max_log_likelihood"])
            chi2 = float(summary["chi2_max_loglike"])
            metrics.update({
                "max_log_likelihood": max_loglike,
                "chi2_max_loglike": chi2,
                "reduced_chi2_max_loglike": chi2 / dof,
                "bic_physical_max_loglike": n_physical * np.log(total_pixels) - 2 * max_loglike,
                "max_loglike_sample_index": summary.get("max_loglike_sample_index"),
            })
        return metrics

    def output(self, save_path: str | Path, *, residual_vis_max: float = 0.0,
               include_corner: bool = True) -> dict[str, Any]:
        """Write wrapper-style per-band diagnostics and joint SVI products."""
        from ..utils import json_serializer, kwargs_best_to_json_pixelated_npy
        from ..visualizations import generate_run_plots, plot_corner_traced_params, plot_loss_curve, plot_multiband_composite
        root = Path(save_path).expanduser(); root.mkdir(parents=True, exist_ok=True)
        kwargs_by_band, metrics, files, skipped = self.kwargs_by_band(), self.metrics(), {}, {}
        shared_lens = self._shared_lens_kwargs(kwargs_by_band)
        hmc_components = self.details.get("component_medians_by_band", {})
        band_results, arrays, kwargs_json_by_band = [], {}, {}
        for band in self._model.bands:
            name, directory, kwargs = band["name"], root / band["name"], kwargs_by_band[band["name"]]
            directory.mkdir(parents=True, exist_ok=True)
            components = hmc_components.get(name)
            kwargs_for_plots = deepcopy(kwargs)
            if components is not None and kwargs_for_plots.get("kwargs_source"):
                kwargs_for_plots["kwargs_source"][0]["pixels"] = components["source_plane"]
            best = np.asarray(components["total"]) if components is not None else np.asarray(band["lens_image"].model(**kwargs))
            residual = (best - band["image_data"]) / band["noise_map"]
            valid = np.isfinite(residual)
            if band["fit_mask_bool"] is not None: valid &= np.asarray(band["fit_mask_bool"], bool)
            chi2 = float(np.sum(residual[valid] ** 2))
            kwargs_json = kwargs_best_to_json_pixelated_npy(
                kwargs_for_plots, str(directory), band["type_list"],
            )
            kwargs_json_by_band[name] = kwargs_json
            with (directory / "kwargs_result.json").open("w") as stream:
                json.dump(kwargs_json, stream, indent=2, default=json_serializer)
            with (directory / "kwargs_lens_shared.json").open("w") as stream:
                json.dump({"kwargs_lens": shared_lens}, stream, indent=2, default=json_serializer)
            try:
                generate_run_plots(
                    lens_image=band["lens_image"], kwargs_best=kwargs_for_plots, image_data=band["image_data"],
                    noise_map=band["noise_map"], psf_data=self._model.observations[name].psf,
                    pixel_scale=self._model.observations[name].pixel_scale, save_path=str(directory),
                    sampler="hmc" if self.samples is not None else "svi",
                    best_fit_model=best, chi2=chi2, reduced_chi2=None, extra=None,
                    param_list=band["param_list"], residual_vis_max=residual_vis_max,
                    mcmc_samples=self.samples,
                    mcmc_component_medians=components,
                    num_chains_hmc=int(self.details.get("num_chains_hmc_numpyro", 1)),
                )
            except Exception as error: skipped[f"{name}_plots"] = str(error)
            band_results.append({"name": name, "lens_image": band["lens_image"], "kwargs_result": kwargs_for_plots,
                                 "image_data": band["image_data"], "noise_map": band["noise_map"],
                                 "pixel_scale": self._model.observations[name].pixel_scale,
                                 "model_total": components["total"] if components is not None else None,
                                 "model_lensed_source": components["source"] if components is not None else None,
                                 "model_lens_light": components["lens_light"] if components is not None else None})
            arrays.update({f"{name}_best_fit_model": best, f"{name}_image_data": band["image_data"],
                           f"{name}_noise_map": band["noise_map"], f"{name}_fit_mask_bool": valid})
            files[f"{name}_kwargs"] = directory / "kwargs_result.json"
        with (root / "kwargs_result.json").open("w") as stream:
            json.dump(
                {"kwargs_by_band": kwargs_json_by_band, "kwargs_lens": shared_lens},
                stream, indent=2, default=json_serializer,
            )
        with (root / "kwargs_lens_shared.json").open("w") as stream:
            json.dump({"kwargs_lens": shared_lens}, stream, indent=2, default=json_serializer)
        with (root / "metrics.json").open("w") as stream: json.dump(metrics, stream, indent=2, default=json_serializer)
        from ..utils import save_named_arrays_fits
        save_named_arrays_fits(root / "modeling_result.fits", arrays)
        if self.loss_history is not None:
            with (root / "svi_loss_history.json").open("w") as stream: json.dump({"loss_history": self.loss_history.tolist()}, stream)
            try: plot_loss_curve(self.loss_history, str(root))
            except Exception as error: skipped["loss_curve"] = str(error)
        if self.details.get("result") is not None:
            with (root / "svi_guide_params.pkl").open("wb") as stream: pickle.dump(self.details["result"].params, stream)
        try: plot_multiband_composite(band_results, str(root), residual_vis_max=residual_vis_max)
        except Exception as error: skipped["multiband_composite"] = str(error)
        if self.initial_parameters is not None:
            try:
                initial_by_band = self._model.prob_model.params2kwargs_by_band(
                    self.initial_parameters,
                )
                initial_shared = self._shared_lens_kwargs(initial_by_band)
                initial_json_by_band = {}
                for band in self._model.bands:
                    name = band["name"]
                    initial_json_by_band[name] = kwargs_best_to_json_pixelated_npy(
                        initial_by_band[name], str(root / name), band["type_list"],
                        pixels_filename="kwargs_source_pixels_init.fits",
                        pixels_wn_filename="kwargs_source_pixels_wn_init.fits",
                        lens_light_pixels_prefix="kwargs_lens_light_pixels_init",
                    )
                with (root / "kwargs_init.json").open("w") as stream:
                    json.dump(
                        {"kwargs_lens": initial_shared, "kwargs_by_band": initial_json_by_band},
                        stream, indent=2, default=json_serializer,
                    )
                initial_rows = []
                for band in self._model.bands:
                    name = band["name"]
                    initial_rows.append({
                        "name": name,
                        "lens_image": band["lens_image"],
                        "kwargs_result": initial_by_band[name],
                        "image_data": band["image_data"],
                        "noise_map": band["noise_map"],
                        "pixel_scale": self._model.observations[name].pixel_scale,
                    })
                plot_multiband_composite(
                    initial_rows, str(root), residual_vis_max=residual_vis_max,
                    output_filename="initial_guess_model.png",
                )
            except Exception as error:
                skipped["initial_guess"] = str(error)
        if self.details.get("guide") is not None and self.details.get("result") is not None:
            try:
                import jax
                samples = self.details["guide"].sample_posterior(
                    jax.random.PRNGKey(42), self.details["result"].params,
                    sample_shape=(2_000,),
                )
                sigma_parameters = jax.tree_util.tree_map(
                    lambda value: np.asarray(value).std(axis=0), samples,
                )
                sigma_by_band = self._model.prob_model.params2kwargs_by_band(sigma_parameters)
                for band in self._model.bands:
                    name, directory = band["name"], root / band["name"]
                    sigma_json = kwargs_best_to_json_pixelated_npy(
                        sigma_by_band[name], str(directory), band["type_list"],
                        pixels_filename="kwargs_source_pixels_sigma.fits",
                        pixels_wn_filename="kwargs_source_pixels_wn_sigma.fits",
                        lens_light_pixels_prefix="kwargs_lens_light_pixels_sigma",
                    )
                    with (directory / "kwargs_sigma.json").open("w") as stream:
                        json.dump(sigma_json, stream, indent=2, default=json_serializer)
                    if include_corner:
                        prefix = f"{band['site_prefix']}/"
                        band_samples = {
                            key[len(prefix):] if key.startswith(prefix) else key: value
                            for key, value in samples.items()
                            if "/" not in key or key.startswith(prefix)
                        }
                        plot_corner_traced_params(
                            band_samples, str(directory), filename="corner_svi.png",
                            param_list=band["param_list"],
                        )
                if include_corner:
                    plot_corner_traced_params(
                        samples, str(root), filename="corner_multiband.png",
                        site_order=self._joint_corner_site_order(),
                    )
            except Exception as error: skipped["corner"] = str(error)
        elif self.samples is not None:
            try:
                from .samplers import hmc_one_sigma_kwargs

                for band in self._model.bands:
                    name, directory = band["name"], root / band["name"]
                    sigma_json = hmc_one_sigma_kwargs(
                        self._model.prob_model, self.samples, self.parameters,
                        band["type_list"], directory,
                        kwargs_from_params=lambda params, band_name=name: (
                            self._model.prob_model.params2kwargs_by_band(params)[band_name]
                        ),
                    )
                    with (directory / "kwargs_sigma.json").open("w") as stream:
                        json.dump(sigma_json, stream, indent=2, default=json_serializer)
                    files[f"{name}_kwargs_sigma"] = directory / "kwargs_sigma.json"
            except Exception as error:
                skipped["kwargs_sigma"] = str(error)
        if self.samples is not None and include_corner:
            try:
                samples = {key: np.asarray(value) for key, value in self.samples.items()}
                for band in self._model.bands:
                    prefix = f"{band['site_prefix']}/"
                    band_samples = {
                        key[len(prefix):] if key.startswith(prefix) else key: value
                        for key, value in samples.items()
                        if "/" not in key or key.startswith(prefix)
                    }
                    plot_corner_traced_params(
                        band_samples, str(root / band["name"]),
                        filename="corner_traced_params.png", param_list=band["param_list"],
                    )
                plot_corner_traced_params(
                    samples, str(root), filename="corner_multiband.png",
                    site_order=self._joint_corner_site_order(),
                )
            except Exception as error:
                skipped["hmc_corner"] = str(error)
        files.update({"metrics": root / "metrics.json", "kwargs_result": root / "kwargs_result.json", "modeling_result": root / "modeling_result.fits"})
        return {"directory": root, "files": files, "skipped": skipped, "metrics": metrics}


class MultiBandModel:
    """Joint multi-band model; first version supports parametric SVI."""
    def __init__(self, *, observations: MultiBandData, profiles: MultiBandProfileCollection,
                 numerics: Mapping[str, Any] | None = None, source_grid_scale: float = 1.0,
                 likelihood_scale: float = 1.0):
        if set(observations.band_names) != set(profiles.bands):
            raise ValueError("observations and profiles must have identical band names.")
        self.observations, self.profiles = observations, profiles
        self.numerics = dict(numerics or {"supersampling_factor": 1})
        self.source_grid_scale, self.likelihood_scale = float(source_grid_scale), float(likelihood_scale)
        self.prob_model = self.bands = None
        self.initial_parameters = None; self.result = None
        self._build()

    def _build(self) -> None:
        from ..models import create_lens_image, validate_param_list
        from ..multiband import band_site_prefix, create_multiband_prob_model
        definitions = self.profiles.band_definitions(); bands = []
        shared_types = shared_params = None
        for index, (name, data) in enumerate(self.observations.items()):
            types, params = definitions[name]; validate_param_list(types, params)
            if shared_types is None: shared_types, shared_params = types["lens_mass_type_list"], params["lens_mass_params_list"]
            lens_image = create_lens_image(params, types, data.likelihood_image, data.likelihood_noise, data.psf, data.pixel_scale,
                kwargs_numerics=self.numerics, source_arc_mask=data.source_arc_mask, source_grid_scale=self.source_grid_scale)
            bands.append({"name": name, "site_prefix": band_site_prefix(index, name), "lens_image": lens_image,
                "image_data": data.likelihood_image, "noise_map": data.likelihood_noise, "fit_mask_bool": data.likelihood_mask,
                "param_list": params, "type_list": types, "args": SimpleNamespace(likelihood_scale=self.likelihood_scale)})
        self.bands = bands
        self.prob_model = create_multiband_prob_model(bands, shared_params, shared_types, SimpleNamespace(likelihood_scale=self.likelihood_scale))

    def initialize(
        self,
        *,
        seed: int = 42,
        run_id: int | str | None = None,
        init_params_path: str | Path | None = None,
        pixelated_init_match: str = "image",
        num_iterations_warmup: int = 0,
    ) -> Mapping[str, Any]:
        """Create a start point, optionally inheriting a parametric multiband fit.

        A prior ``parametric_svi`` directory (or a specific ``run_i`` within
        it) supplies shared lens mass, band-specific centres, and lens light.
        For a pixelated source, ``pixelated_init_match='image'`` then runs a
        short source-only SVI warmup with those inherited components fixed.
        """
        if run_id is not None:
            print("\n========================================")
            print(f"Starting Run {run_id} (seed={seed})")
            print("========================================")
        if self.profiles.apply_initializations():
            self._build()
        import jax
        from numpyro import infer
        from numpyro.infer.util import initialize_model
        info = initialize_model(jax.random.PRNGKey(seed), self.prob_model.model, init_strategy=infer.init_to_median(num_samples=25), validate_grad=False)
        self.initial_parameters = {name: site["value"] for name, site in info.model_trace.items() if site["type"] == "sample" and not site["is_observed"]}
        self.initialization_path = None
        if init_params_path is not None:
            from ..utils import resolve_init_run_dir

            self.initialization_path = Path(resolve_init_run_dir(init_params_path)).expanduser()
            result_path = self.initialization_path / "kwargs_result.json"
            if not result_path.is_file():
                raise FileNotFoundError(
                    f"{self.initialization_path}: expected kwargs_result.json from a parametric multiband run."
                )
            with result_path.open() as stream:
                inherited = json.load(stream)
            kwargs_by_band = inherited.get("kwargs_by_band")
            if not isinstance(kwargs_by_band, Mapping):
                raise ValueError(
                    f"{result_path} does not contain kwargs_by_band; it is not a multiband API result."
                )
            missing = set(self.observations.band_names) - set(kwargs_by_band)
            if missing:
                raise ValueError(f"{result_path}: missing inherited band(s) {sorted(missing)}.")
            self._apply_inherited_parametric_kwargs(kwargs_by_band)
            source_kind = "pixelated SVI" if any(
                (source_values := values.get("kwargs_source", []))
                and isinstance(source_values[0], Mapping)
                and "pixels_wn" in source_values[0]
                for values in kwargs_by_band.values()
            ) else "parametric SVI"
            print(f"[Init] Loaded {source_kind} result from {self.initialization_path}")

            is_pixelated = all(
                band["type_list"].get("source_light_type_list") == ["PIXELATED"]
                for band in self.bands
            )
            if pixelated_init_match not in {"image"}:
                raise ValueError("Multiband pixelated initialization currently supports pixelated_init_match='image'.")
            if is_pixelated and num_iterations_warmup > 0:
                self._warmup_pixelated_source(
                    kwargs_by_band, seed=seed, num_iterations=num_iterations_warmup,
                )
            self._initialize_pixelated_lens_light_from_parametric(
                kwargs_by_band, seed=seed,
                num_iterations=num_iterations_warmup or 2_000,
            )
        return self.initial_parameters

    def _initialize_pixelated_lens_light_from_parametric(
        self,
        kwargs_by_band: Mapping[str, Any],
        *,
        seed: int,
        num_iterations: int,
    ) -> None:
        """Map each band analytic/MGE lens-light fit to its pixelated GP start."""
        from ..models import PowerSpectrum, _project_analytic_kwargs_to_pixel_lens_light
        import jax.numpy as jnp

        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations_warmup must be positive for pixelated lens-light matching.")
        initial = dict(self.initial_parameters)
        matched = False
        for band_index, band in enumerate(self.bands):
            pixelated_indices = [
                index for index, profile_type in enumerate(band["type_list"].get("lens_light_type_list", []))
                if profile_type == "PIXELATED"
            ]
            if not pixelated_indices:
                continue
            inherited = kwargs_by_band[band["name"]].get("kwargs_lens_light", [])
            if not inherited:
                raise ValueError(
                    f"{band['name']}: pixelated lens light needs inherited analytic/MGE "
                    "kwargs_lens_light for initialization."
                )
            for profile_index in pixelated_indices:
                saved = inherited[profile_index] if profile_index < len(inherited) else None
                prefix = f"{band['site_prefix']}/"
                pixel_site = f"{prefix}pixels_wn_lens_light_grid_{profile_index}"
                if isinstance(saved, Mapping) and isinstance(saved.get("pixels_wn"), Mapping):
                    file_path = self.initialization_path / band["name"] / saved["pixels_wn"]["file"]
                    if not file_path.is_file():
                        raise FileNotFoundError(f"Missing saved pixelated lens-light coefficients: {file_path}")
                    from ..utils import load_array_file
                    initial[pixel_site] = jnp.asarray(load_array_file(file_path))
                    for key in ("n", "rho", "sigma"):
                        value = saved.get(f"{key}_lens_light_grid")
                        site = f"{prefix}{key}_lens_light_grid_{profile_index}"
                        if value is not None and site in initial:
                            initial[site] = jnp.atleast_1d(value)
                    continue
                pixelated_prior = band["param_list"]["lens_light_params_list"][profile_index].get("pixelated_prior", {})
                ny, nx = band["lens_image"].LensLightModel.pixel_grid.num_pixel_axes
                print(
                    f"[pixelated-init: lens-light] {band['name']}: fitting Matérn parameters "
                    f"({num_iterations} iterations) from inherited analytic/MGE lens light..."
                )
                target = _project_analytic_kwargs_to_pixel_lens_light(band["lens_image"], inherited)
                fitted = PowerSpectrum.fit_power_spectrum_init(
                    target, PowerSpectrum.K_grid((ny, nx)).k, pixelated_prior,
                    seed=seed + 17863 + 101 * band_index + profile_index,
                    max_iterations=num_iterations,
                    param_name=f"lens_light_grid_{profile_index}",
                )
                for key, value in fitted.items():
                    site = f"{prefix}{key}"
                    if site in initial:
                        initial[site] = value
                matched = True
        if matched:
            print("[pixelated-init: lens-light] Multiband lens-light-matched initialization complete.")
        self.initial_parameters = initial

    def _apply_inherited_parametric_kwargs(self, kwargs_by_band: Mapping[str, Any]) -> None:
        """Map saved physical kwargs onto the joint NumPyro site names."""
        import jax.numpy as jnp

        initial = dict(self.initial_parameters)
        for band in self.bands:
            name, prefix, inherited = band["name"], f"{band['site_prefix']}/", kwargs_by_band[band["name"]]
            for index, definition in enumerate(band["param_list"]["lens_mass_params_list"]):
                values = inherited.get("kwargs_lens", [])
                if index >= len(values):
                    continue
                for key, specification in definition.items():
                    if not isinstance(specification, (list, tuple)) or key not in values[index]:
                        continue
                    site = f"{prefix}lens_{key}_{index}" if key in {"center_x", "center_y"} else f"lens_{key}_{index}"
                    if site in initial:
                        initial[site] = jnp.asarray(values[index][key])
            for index, definition in enumerate(band["param_list"].get("lens_light_params_list", [])):
                values = inherited.get("kwargs_lens_light", [])
                if index >= len(values):
                    continue
                for key, specification in definition.items():
                    site = f"{prefix}lens_light_{key}_{index}"
                    if isinstance(specification, (list, tuple)) and key in values[index] and site in initial:
                        initial[site] = jnp.asarray(values[index][key])
            source = inherited.get("kwargs_source", [{}])
            if source and isinstance(source[0], Mapping):
                source_values = source[0]
                for key in ("n_source_grid", "rho_source_grid", "sigma_source_grid"):
                    site = f"{prefix}{key}"
                    if key in source_values and site in initial:
                        # The Matérn implementation indexes these values as
                        # ``n[0]``, ``rho[0]``, and ``sigma[0]``.  JSON stores
                        # a scalar, so restore the required length-one shape.
                        initial[site] = jnp.atleast_1d(source_values[key])
                pixels_wn = source_values.get("pixels_wn")
                site = f"{prefix}pixels_wn_source_grid"
                if isinstance(pixels_wn, Mapping) and pixels_wn.get("file") and site in initial:
                    pixel_path = self.initialization_path / pixels_wn["file"]
                    if not pixel_path.is_file():
                        pixel_path = self.initialization_path / name / pixels_wn["file"]
                    if not pixel_path.is_file():
                        raise FileNotFoundError(f"Missing saved pixelated source coefficients: {pixel_path}")
                    from ..utils import load_array_file
                    initial[site] = jnp.asarray(load_array_file(pixel_path))
        self.initial_parameters = initial

    def _warmup_pixelated_source(
        self,
        kwargs_by_band: Mapping[str, Any],
        *,
        seed: int,
        num_iterations: int,
    ) -> None:
        """Fit only the pixelated source while inherited mass/light stay fixed."""
        from ..multiband import create_multiband_prob_model
        from ..samplers import run_svi

        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations_warmup must be a positive integer.")
        fixed_mass = {name: kwargs_by_band[name]["kwargs_lens"] for name in self.observations.band_names}
        fixed_light = {
            name: kwargs_by_band[name].get("kwargs_lens_light", [])
            for name in self.observations.band_names
        }
        warmup_model = create_multiband_prob_model(
            self.bands, self.prob_model.lens_mass_params_list,
            self.prob_model.lens_mass_type_list,
            SimpleNamespace(likelihood_scale=self.likelihood_scale),
            fixed_lens_mass_by_band=fixed_mass,
            fixed_lens_light_by_band=fixed_light,
        )
        warmup = SamplerConfig.svi(
            max_iterations=num_iterations, random_seed=seed,
        ).to_namespace()
        print(f"[svi-warmup] Starting {num_iterations} iteration pixelated-source image-match warmup...")
        warmup_parameters, _ = run_svi(warmup_model, None, warmup, self.initial_parameters)
        for key, value in warmup_parameters.items():
            if "source" in key or "pixels_wn" in key:
                self.initial_parameters[key] = value
        print("[svi-warmup] Pixelated-source image-match warmup complete.")

    def run(
        self, sampler: SamplerConfig, *, init_params: Mapping[str, Any] | None = None,
        save_path: str | Path | None = None,
    ) -> MultiBandFitResult:
        if sampler.name not in {"svi", "hmc"}:
            raise NotImplementedError("MultiBandModel currently supports SVI and HMC.")
        from ..samplers import run_hmc, run_svi
        initial = dict(init_params or self.initial_parameters or self.initialize(seed=sampler.random_seed))
        samples = None
        if sampler.name == "svi":
            parameters, details = run_svi(self.prob_model, None, sampler.to_namespace(), initial)
        else:
            if save_path is None:
                raise ValueError("HMC requires save_path for checkpoints and posterior samples.")
            if self.initialization_path is None:
                raise ValueError("HMC requires initialize(init_params_path=...) using a pixelated SVI result.")
            output = Path(save_path).expanduser()
            output.mkdir(parents=True, exist_ok=True)
            args = sampler.to_namespace()
            args.save_path = str(output)
            samples, parameters, details = run_hmc(
                self.prob_model, args, initial,
                init_params_path=str(self.initialization_path),
                batch_diagnostics_callback=lambda draws, batch_index, health: self._save_hmc_batch_diagnostics(
                    draws, batch_index, health, output, int(args.num_chains_hmc_numpyro),
                ),
            )
            details["num_chains_hmc_numpyro"] = int(args.num_chains_hmc_numpyro)
            print("[hmc] Computing final posterior-median component images in chunks...")
            details["component_medians_by_band"] = self._evaluate_hmc_component_medians(samples)
            details["sample_likelihood_summary"] = self._last_hmc_likelihood_summary
        initial_for_result = dict(initial)
        self.initial_parameters = parameters
        self.result = MultiBandFitResult(
            parameters, details, self, random_seed=sampler.random_seed,
            initial_parameters=initial_for_result, samples=samples,
        )
        return self.result

    def _save_hmc_batch_diagnostics(
        self,
        samples: Mapping[str, Any],
        batch_index: int,
        health: Mapping[str, Any],
        output: Path,
        num_chains: int,
    ) -> None:
        """Save cumulative multiband diagnostics after one HMC batch."""
        from ..samplers import save_hmc_diagnostics
        from ..visualizations import plot_multiband_composite

        directory = output / "diagnostics" / f"batch_{batch_index}"
        directory.mkdir(parents=True, exist_ok=True)
        save_hmc_diagnostics(
            samples, num_chains, str(directory), f"batch_{batch_index}",
            hmc_extra_fields=health,
        )
        medians = {key: np.median(np.asarray(value), axis=0) for key, value in samples.items()}
        kwargs_by_band = self.prob_model.params2kwargs_by_band(medians)
        components_by_band = self._evaluate_hmc_component_medians(samples)
        first = np.asarray(next(iter(samples.values())))
        if first.shape[0] % num_chains:
            raise ValueError("HMC samples cannot be divided into the configured number of chains.")
        draws_per_chain = first.shape[0] // num_chains
        chain_components = [
            self._evaluate_hmc_component_medians({
                key: np.asarray(value).reshape((num_chains, draws_per_chain) + np.asarray(value).shape[1:])[chain]
                for key, value in samples.items()
            })
            for chain in range(num_chains)
        ]
        plot_rows = []
        for band in self.bands:
            name = band["name"]
            kwargs = deepcopy(kwargs_by_band[name])
            source = components_by_band[name].get("source_plane")
            if source is not None and kwargs.get("kwargs_source"):
                kwargs["kwargs_source"][0]["pixels"] = source
            plot_rows.append({
                "name": name, "lens_image": band["lens_image"],
                "kwargs_result": kwargs,
                "image_data": band["image_data"], "noise_map": band["noise_map"],
                "pixel_scale": self.observations[name].pixel_scale,
                "model_total": components_by_band[name]["total"],
                "model_lensed_source": components_by_band[name]["source"],
                "model_lens_light": components_by_band[name]["lens_light"],
            })
            self._plot_hmc_chain_comparison(
                samples, band, directory, chain_components,
                batch_index=batch_index, num_chains=num_chains,
            )
        plot_multiband_composite(
            plot_rows, str(directory), output_filename=f"multiband_composite_batch_{batch_index}.png",
        )

    def _evaluate_hmc_component_medians(
        self, samples: Mapping[str, Any], *, batch_size: int = 1_000,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Forward every posterior draw in GPU-sized chunks, then take image medians.

        Samples remain archived on CPU.  Only one chunk is converted to JAX
        device arrays at a time, so this computes the statistically correct
        posterior image median without retaining the posterior on GPU.
        """
        import jax
        import jax.numpy as jnp

        keys = list(samples)
        if not keys:
            raise ValueError("Cannot evaluate HMC components without posterior samples.")
        n_samples = len(np.asarray(samples[keys[0]]))
        outputs: dict[str, dict[str, list[np.ndarray]]] = {
            band["name"]: {key: [] for key in ("total", "source", "lens_light", "no_lens_light", "point_source", "source_plane")}
            for band in self.bands
        }

        evaluators = {}
        band_likelihood = {}
        for band in self.bands:
            name, lens_image = band["name"], band["lens_image"]
            has_lens_light = bool(band["type_list"].get("lens_light_type_list"))
            has_point_source = bool(band["type_list"].get("point_source_type_list"))

            def evaluate_one(draw, *, name=name, lens_image=lens_image,
                             has_lens_light=has_lens_light, has_point_source=has_point_source):
                kwargs = self.prob_model.params2kwargs_by_band(draw)[name]
                total = jnp.squeeze(lens_image.model(**kwargs))
                source = jnp.squeeze(lens_image.model(
                    **kwargs, source_add=True, lens_light_add=False, point_source_add=False,
                ))
                lens_light = (
                    jnp.squeeze(lens_image.model(
                        **kwargs, source_add=False, lens_light_add=True, point_source_add=False,
                    )) if has_lens_light else jnp.zeros_like(total)
                )
                if has_point_source:
                    no_lens_light = jnp.squeeze(lens_image.model(
                        **kwargs, source_add=True, lens_light_add=False, point_source_add=True,
                    ))
                    point_source = jnp.squeeze(lens_image.model(
                        **kwargs, source_add=False, lens_light_add=False, point_source_add=True,
                    ))
                else:
                    no_lens_light, point_source = source, jnp.zeros_like(total)
                source_plane = kwargs.get("kwargs_source", [{}])[0].get("pixels")
                if source_plane is None:
                    source_plane = jnp.zeros((1, 1), dtype=total.dtype)
                return total, source, lens_light, no_lens_light, point_source, jnp.asarray(source_plane)

            evaluators[name] = jax.jit(jax.vmap(evaluate_one))
            data, noise = np.asarray(band["image_data"]), np.asarray(band["noise_map"])
            valid = np.isfinite(data) & np.isfinite(noise) & (noise > 0)
            if band["fit_mask_bool"] is not None:
                valid &= np.asarray(band["fit_mask_bool"], dtype=bool)
            band_likelihood[name] = (data, noise, valid, float(np.sum(np.log(2 * np.pi * noise[valid] ** 2))))

        best_loglike, best_chi2, best_index = -np.inf, None, None
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            device_draws = {
                key: jnp.asarray(np.asarray(value)[start:stop])
                for key, value in samples.items()
            }
            joint_chi2 = np.zeros(stop - start, dtype=float)
            joint_normalization = 0.0
            for name, evaluator in evaluators.items():
                values = evaluator(device_draws)
                for label, image_stack in zip(outputs[name], values):
                    outputs[name][label].append(np.asarray(image_stack))
                data, noise, valid, normalization = band_likelihood[name]
                total = np.asarray(values[0])
                residual = (total - data[None, ...]) / noise[None, ...]
                joint_chi2 += np.sum(np.square(residual[..., valid]), axis=1)
                joint_normalization += normalization
            joint_loglike = -0.5 * self.likelihood_scale * (joint_chi2 + joint_normalization)
            local_index = int(np.argmax(joint_loglike))
            if float(joint_loglike[local_index]) > best_loglike:
                best_loglike = float(joint_loglike[local_index])
                best_chi2 = float(joint_chi2[local_index])
                best_index = start + local_index

        self._last_hmc_likelihood_summary = {
            "max_log_likelihood": best_loglike,
            "chi2_max_loglike": best_chi2,
            "max_loglike_sample_index": best_index,
        }

        return {
            name: {
                label: np.median(np.concatenate(stacks, axis=0), axis=0)
                for label, stacks in values.items()
            }
            for name, values in outputs.items()
        }

    def _plot_hmc_chain_comparison(
        self,
        samples: Mapping[str, Any], band: Mapping[str, Any], directory: Path,
        chain_components: list[dict[str, dict[str, np.ndarray]]],
        *, batch_index: int, num_chains: int,
    ) -> None:
        """Save a six-panel composite row for every HMC chain in one band."""
        from ..visualizations import plot_multiband_composite
        first = np.asarray(next(iter(samples.values())))
        if first.shape[0] % num_chains:
            raise ValueError("HMC samples cannot be divided into the configured number of chains.")
        draws_per_chain = first.shape[0] // num_chains
        chain_rows = []
        for chain in range(num_chains):
            components = chain_components[chain][band["name"]]
            chain_parameters = {
                key: np.median(np.asarray(value).reshape((num_chains, draws_per_chain) + np.asarray(value).shape[1:])[chain], axis=0)
                for key, value in samples.items()
            }
            kwargs = self.prob_model.params2kwargs_by_band(chain_parameters)[band["name"]]
            if kwargs.get("kwargs_source"):
                kwargs = deepcopy(kwargs)
                kwargs["kwargs_source"][0]["pixels"] = components["source_plane"]
            chain_rows.append({
                "name": f"chain {chain}", "lens_image": band["lens_image"],
                "kwargs_result": kwargs, "image_data": band["image_data"],
                "noise_map": band["noise_map"],
                "pixel_scale": self.observations[band["name"]].pixel_scale,
                "model_total": components["total"],
                "model_lensed_source": components["source"],
                "model_lens_light": components["lens_light"],
            })
        plot_multiband_composite(
            chain_rows, str(directory),
            output_filename=f"hmc_chain_comparison_{band['name']}_batch_{batch_index}.png",
        )


@dataclass
class MultiBandResultsCombination:
    """Compare repeated joint multi-band SVI fits.

    A separate ``svi_run_comparison_<band>.png`` is written for each band,
    with one diagnostic row per repeated run.  This keeps F150W and F277W
    diagnostics visually distinct.
    """

    results: list[MultiBandFitResult]

    def __post_init__(self) -> None:
        self.results = list(self.results)
        if not self.results:
            raise ValueError("MultiBandResultsCombination requires at least one result.")
        if any(result._model is None for result in self.results):
            raise RuntimeError("Every multiband result must be attached to its MultiBandModel.")

    def output(
        self,
        save_path: str | Path,
        *,
        residual_vis_max: float = 0.0,
    ) -> dict[str, Path]:
        """Save metrics and a joint visual comparison for all repeated runs."""
        from ..utils import json_serializer
        from ..visualizations import plot_multiband_composite

        directory = Path(save_path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        # A previous API revision wrote one mixed-band comparison.  Remove
        # only that obsolete generated artifact when updating an existing run.
        (directory / "svi_run_comparison.png").unlink(missing_ok=True)
        comparison: dict[str, Any] = {}
        plot_rows_by_band: dict[str, list[dict[str, Any]]] = {}
        for index, result in enumerate(self.results):
            model, metrics = result._model, result.metrics()
            run_id = f"run_{index}"
            comparison[run_id] = {"seed": result.random_seed, "metrics": metrics}
            kwargs_by_band = result.kwargs_by_band()
            for band in model.bands:
                name = band["name"]
                plot_rows_by_band.setdefault(name, []).append({
                    "name": run_id,
                    "lens_image": band["lens_image"],
                    "kwargs_result": kwargs_by_band[name],
                    "image_data": band["image_data"],
                    "noise_map": band["noise_map"],
                    "pixel_scale": model.observations[name].pixel_scale,
                })

        comparison_path = directory / "comparison.json"
        with comparison_path.open("w") as stream:
            json.dump(comparison, stream, indent=2, default=json_serializer)
        files = {"comparison": comparison_path}
        for band_name, plot_rows in plot_rows_by_band.items():
            figure_path = directory / f"svi_run_comparison_{band_name}.png"
            plot_multiband_composite(
                plot_rows,
                str(directory),
                residual_vis_max=residual_vis_max,
                output_filename=figure_path.name,
            )
            files[f"svi_run_comparison_{band_name}"] = figure_path

        print("\n========================================")
        print("All runs completed.")
        print(f"Comparison summary saved to {comparison_path}")
        print("========================================")
        for index, result in enumerate(self.results):
            metrics = comparison[f"run_{index}"]["metrics"]
            print(
                f"run_{index} (seed={result.random_seed}): "
                f"log-likelihood (median)={metrics['log_likelihood_median']:.2f}, "
                f"chi2 (median)={metrics['chi2_median']:.2f}, "
                f"chi2/N_pix (median)={metrics['chi2_median'] / metrics['n_data_pixels']:.4f}, "
                f"reduced_chi2 (median)={metrics['reduced_chi2_median']:.4f}, "
                f"BIC_physical (median)={metrics['bic_physical_median']:.2f}"
            )
        print("========================================")
        return files
