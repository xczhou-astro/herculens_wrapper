"""Sampler configuration and in-memory fit results."""
from dataclasses import dataclass, field
from copy import deepcopy
import json
from pathlib import Path
import pickle
# import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import numpy as np
from .types import SamplerName

@dataclass
class SamplerConfig:
    name: SamplerName = "optax"
    random_seed: int = 42
    options: dict[str, Any] = field(default_factory=dict)
    @classmethod
    def svi(
        cls,
        *,
        max_iterations: int = 5_000,
        learning_rate: float = 1e-2,
        init_scale: float = 0.1,
        loss_kind: str = "trace_elbo",
        num_particles: int = 10,
        random_seed: int = 42,
    ) -> "SamplerConfig":
        """Configure NumPyro SVI for a parametric single-band model."""
        if loss_kind not in {"trace_elbo", "trace_meanfield_elbo"}:
            raise ValueError("loss_kind must be 'trace_elbo' or 'trace_meanfield_elbo'.")
        if num_particles < 1:
            raise ValueError("num_particles must be at least 1.")
        return cls("svi", random_seed=random_seed, options={
            "max_iterations_svi": max_iterations,
            "init_learning_rate_svi": learning_rate,
            "init_scale_svi": init_scale,
            "loss_kind_svi": loss_kind,
            "num_particles_svi": num_particles,
        })

    @classmethod
    def hmc(
        cls,
        *,
        num_warmup: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 1,
        checkpoint_interval: int = 250,
        chain_method: str = "auto",
        progress_bar: bool = True,
        init_max_retries: int = 100,
        random_seed: int = 42,
    ) -> "SamplerConfig":
        """Configure NumPyro NUTS/HMC sampling for an SVI-warm-started model."""
        positive_integers = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "checkpoint_interval": checkpoint_interval,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(init_max_retries, int) or isinstance(init_max_retries, bool) or init_max_retries < 0:
            raise ValueError("init_max_retries must be a non-negative integer.")
        if chain_method not in {"auto", "parallel", "vectorized", "sequential"}:
            raise ValueError("chain_method must be 'auto', 'parallel', 'vectorized', or 'sequential'.")
        return cls("hmc", random_seed=random_seed, options={
            "num_warmup_hmc_numpyro": num_warmup,
            "num_samples_hmc_numpyro": num_samples,
            "num_chains_hmc_numpyro": num_chains,
            "checkpoint_interval_hmc_numpyro": checkpoint_interval,
            "chain_method_hmc_numpyro": chain_method,
            "progress_bar_hmc_numpyro": progress_bar,
            "hmc_init_max_retries": init_max_retries,
        })
    def to_namespace(self) -> SimpleNamespace:
        defaults = {"sampler": self.name, "random_seed": self.random_seed, "algorithm_optax": "adabelief", "max_iterations_optax": 2000, "init_learning_rate_optax": 1e-2, "schedule_learning_rate_optax": True, "stop_at_loss_increase_optax": False, "progress_bar_optax": True, "max_iterations_svi": 10000, "init_learning_rate_svi": 1e-2, "init_scale_svi": 0.1, "loss_kind_svi": "trace_elbo", "num_particles_svi": 10, "num_warmup_hmc_numpyro": 1000, "num_samples_hmc_numpyro": 1000, "num_chains_hmc_numpyro": 1, "checkpoint_interval_hmc_numpyro": 250, "chain_method_hmc_numpyro": "auto", "progress_bar_hmc_numpyro": True, "hmc_init_max_retries": 100, "likelihood_scale": 1.0}
        defaults.update(self.options); return SimpleNamespace(**defaults)

@dataclass
class FitResult:
    parameters: Mapping[str, Any]
    details: Mapping[str, Any]
    samples: Mapping[str, Any] | None = None
    random_seed: int | None = None
    derived: Mapping[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, repr=False, compare=False)
    @property
    def loss_history(self) -> np.ndarray | None:
        history = self.details.get("loss_history")
        return None if history is None else np.asarray(history)

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("This FitResult is not attached to a SingleBandModel.")
        return self._model

    def metrics(self) -> dict[str, float | int | None]:
        """Return likelihood, probability, chi-square, and fit-summary metrics."""
        return self._require_model()._metrics(self.parameters)

    def plot_best_fit(self, *, scale: str = "linear", residual_vis_max: float = 0.0,
                      save_path: str | Path | None = None):
        """Plot data, best-fit model, and normalized residual."""
        return self._require_model().plot_fit(self.parameters, scale=scale, residual_vis_max=residual_vis_max, save_path=save_path)

    def plot_loss_curve(self, *, residual_vis_max: float = 0.0, save_path: str | Path | None = None):
        """Plot the full loss curve and its final 20 percent."""
        history = self.loss_history
        if history is None:
            raise RuntimeError("This result does not contain a loss history.")
        import matplotlib.pyplot as plt

        history = np.asarray(history)
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        iteration = np.arange(1, len(history) + 1)
        axes[0].plot(iteration, history, color="tab:blue")
        axes[0].set(xlabel="Iteration", ylabel="Loss", title="Loss curve")
        start = min(int(0.8 * len(history)), max(len(history) - 1, 0))
        axes[1].plot(iteration[start:], history[start:], color="tab:red")
        axes[1].set(xlabel="Iteration", ylabel="Loss", title="Loss curve (final 20%)")
        for axis in axes:
            axis.grid(alpha=0.3)
        if save_path is not None:
            output = Path(save_path).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=200, bbox_inches="tight")
        return figure, axes

    @staticmethod
    def _output_file(path: str | Path, default_name: str) -> Path:
        output = Path(path).expanduser()
        if output.is_dir() or not output.suffix:
            output = output / default_name
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    @staticmethod
    def _plot_destination(save_path: str | Path | None, default_name: str) -> tuple[Path, str, Path]:
        """Normalize a plot directory/file path for legacy wrapper plotters."""
        if save_path is None:
            directory = Path(tempfile.mkdtemp(prefix="herculens_plot_"))
            return directory, default_name, directory / default_name
        target = Path(save_path).expanduser()
        if target.suffix and not target.is_dir():
            directory, filename = target.parent, target.name
        else:
            directory, filename = target, default_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory, filename, directory / filename

    @staticmethod
    def _display_plot(path: Path) -> None:
        """Display a legacy file-based plot inline when running in IPython."""
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                return
            from IPython.display import Image, display
            display(Image(filename=str(path)))
        except Exception:
            pass

    def _kwargs_result(self) -> dict[str, Any]:
        model = self._require_model()
        if model.prob_model is None:
            raise RuntimeError("Call build() before creating result visualizations.")
        return model.prob_model.params2kwargs(self.parameters)

    def _legacy_plot_result(self, callback, save_path: str | Path | None, default_name: str, **kwargs: Any) -> Path:
        directory, filename, output = self._plot_destination(save_path, default_name)
        callback(directory, filename, **kwargs)
        if not output.is_file():
            raise RuntimeError(f"The visualization backend did not create {output}.")
        self._display_plot(output)
        return output

    def save_metrics(self, path: str | Path) -> Path:
        """Write :meth:`metrics` to ``metrics.json`` or an explicit JSON path."""
        output = self._output_file(path, "metrics.json")
        with output.open("w") as stream:
            json.dump(self.metrics(), stream, indent=2)
        return output

    def save_history(self, path: str | Path) -> Path:
        """Write the loss history to ``history.json`` or an explicit JSON path."""
        history = self.loss_history
        if history is None:
            raise RuntimeError("This result does not contain a loss history.")
        output = self._output_file(path, "history.json")
        with output.open("w") as stream:
            json.dump({"loss_history": history.tolist()}, stream, indent=2)
        return output

    def save_svi_guide(self, path: str | Path) -> Path:
        """Save raw NumPyro SVI guide parameters as a pickle file."""
        svi_result = self.details.get("result")
        if svi_result is None or "guide" not in self.details:
            raise RuntimeError("save_svi_guide() is only available for an SVI result.")
        output = self._output_file(path, "svi_guide_params.pkl")
        with output.open("wb") as stream:
            pickle.dump(svi_result.params, stream)
        return output

    def save_parameters(self, path: str | Path) -> Path:
        """Save constrained best-fit parameters as ``parameters.json``."""
        output = self._output_file(path, "parameters.json")
        with output.open("w") as stream:
            json.dump(self.parameters, stream, indent=2, default=_json_default)
        return output

    def plot_image_plane(self, *, save_path: str | Path | None = None, residual_vis_max: float = 0.0) -> Path:
        """Create the wrapper's 2×3 image-plane component diagnostic."""
        from ..visualizations import plot_image_plane
        model = self._require_model()
        return self._legacy_plot_result(
            lambda directory, filename: plot_image_plane(
                model.lens_image, self._kwargs_result(), model.data.pixel_scale,
                model.data.likelihood_image, model.data.likelihood_noise, str(directory),
                residual_vis_max=residual_vis_max, output_filename=filename,
            ), save_path, "image_plane.png",
        )

    def plot_composite(self, *, save_path: str | Path | None = None, residual_vis_max: float = 0.0) -> Path:
        """Create the wrapper's 2×3 composite/source-plane diagnostic."""
        from ..visualizations import plot_composite_2x3_panel
        model = self._require_model()
        return self._legacy_plot_result(
            lambda directory, filename: plot_composite_2x3_panel(
                model.lens_image, self._kwargs_result(), model.data.pixel_scale,
                model.data.likelihood_image, model.data.likelihood_noise, str(directory),
                residual_vis_max=residual_vis_max, output_filename=filename,
                source_arc_mask=model.data.source_arc_mask,
            ), save_path, "composite.png",
        )

    def plot_source_plane(self, *, save_path: str | Path | None = None, source_pixel_scale: float = 0.01,
                          num_pixel: int = 200, scale: str = "linear", residual_vis_max: float = 0.0) -> Path:
        """Plot the reconstructed source plane with caustics where available."""
        from ..visualizations import plot_source_plane
        model = self._require_model()
        return self._legacy_plot_result(
            lambda directory, filename: plot_source_plane(
                model.lens_image, self._kwargs_result(), str(directory),
                source_pixel_scale=source_pixel_scale, num_pixel=num_pixel,
                plot_scale=scale, output_filename=filename,
                source_arc_mask=model.data.source_arc_mask,
            ), save_path, "source_plane.png",
        )

    def plot_ring_model_comparison(self, *, save_path: str | Path | None = None, scale: str = "linear",
                                   residual_vis_max: float = 0.0) -> Path:
        """Compare the lens-light-subtracted ring image with its model."""
        from ..visualizations import plot_ring_model_comparison
        model = self._require_model()
        return self._legacy_plot_result(
            lambda directory, filename: plot_ring_model_comparison(
                model.lens_image, self._kwargs_result(), model.data.pixel_scale,
                model.data.likelihood_image, model.data.likelihood_noise, str(directory),
                plot_scale=scale, residual_vis_max=residual_vis_max, output_filename=filename,
            ), save_path, "ring_model_comparison.png",
        )

    def plot_lens_light_subtraction(self, *, save_path: str | Path | None = None,
                                    scale: str = "linear", residual_vis_max: float = 0.0) -> Path:
        """Plot data, lens-light model, and lens-light-subtracted image."""
        from ..visualizations import plot_lens_light_subtracted_image
        model = self._require_model()
        suffix = "_log" if scale == "log" else ""
        directory, _, output = self._plot_destination(save_path, f"lens_light_subtracted_image{suffix}.png")
        # The legacy function has a fixed filename, so an explicit filename is
        # not supported here; a file-style save_path is represented by moving it.
        plot_lens_light_subtracted_image(
            model.lens_image, self._kwargs_result(), model.data.pixel_scale,
            model.data.likelihood_image, model.data.likelihood_noise, str(directory),
            plot_scale=scale, residual_vis_max=residual_vis_max,
        )
        generated = directory / f"lens_light_subtracted_image{suffix}.png"
        if generated != output:
            generated.replace(output)
        self._display_plot(output)
        return output

    def plot_mass_profile_convergence(self, *, save_path: str | Path | None = None,
                                      residual_vis_max: float = 0.0) -> Path:
        """Plot convergence, magnification, and radial mass profile."""
        from ..visualizations import plot_mass_and_convergence
        model = self._require_model()
        directory, _, output = self._plot_destination(save_path, "mass_profile_convergence.png")
        plot_mass_and_convergence(model.lens_image, self._kwargs_result(), model.data.pixel_scale, str(directory))
        generated = directory / "mass_profile_convergence.png"
        if generated != output:
            generated.replace(output)
        self._display_plot(output)
        return output

    def mass_component_convergence(self) -> dict[str, np.ndarray]:
        """Return cached-parameter stellar, dark-matter, and total κ maps."""
        return self._require_model().mass_component_convergence(self.parameters)

    def enclosed_mass(self, geometry, *, radius_arcsec: float | None = None,
                      center: tuple[float, float] | None = None, grid_size: int = 400,
                      save_path: str | Path | None = None) -> dict[str, Any]:
        """Return the physical projected mass within a circular lens aperture."""
        from .physics import LensGeometry, enclosed_lensing_mass

        if not isinstance(geometry, LensGeometry):
            raise TypeError("geometry must be a LensGeometry instance.")
        return enclosed_lensing_mass(
            self._require_model(), self.parameters, geometry,
            radius_arcsec=radius_arcsec, center=center, grid_size=grid_size,
            save_path=save_path,
        )

    def plot_mass_decomposition(self, *, save_path: str | Path | None = None,
                                residual_vis_max: float = 0.0) -> Path:
        """Plot the stellar, dark-matter, and total convergence components."""
        import matplotlib.pyplot as plt
        from .visualization import _extent, _normalization

        maps = self.mass_component_convergence()
        required = {"stellar", "dark_matter"}
        if not required.issubset(maps):
            raise RuntimeError("plot_mass_decomposition() requires StellarMassMGE and GNFWHaloMGE.")
        model = self._require_model()
        output = self._output_file(save_path or Path(tempfile.mkdtemp()), "mass_decomposition.png")
        extent = _extent(model.data.image.shape, model.data.pixel_scale)
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, (name, image) in zip(axes, (
            ("Stellar convergence", maps["stellar"]),
            ("Dark-matter convergence", maps["dark_matter"]),
            ("Total convergence", maps["total"]),
        )):
            rendered = axis.imshow(
                image, origin="lower", extent=extent, cmap="twilight",
                norm=_normalization(image, "log", signed=False),
            )
            axis.set(title=name, xlabel="arcsec", ylabel="arcsec")
            figure.colorbar(rendered, ax=axis, shrink=0.86, label=r"Convergence $\kappa$")
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(figure)
        self._display_plot(output)
        return output

    def plot_corner(self, *, save_path: str | Path | None = None, num_samples: int = 2_000,
                    max_samples: int = 15_000, residual_vis_max: float = 0.0) -> Path:
        """Plot posterior parameter correlations from HMC samples or an SVI guide."""
        from ..visualizations import plot_corner_traced_params
        samples = self.samples
        if samples is None:
            guide, svi_result = self.details.get("guide"), self.details.get("result")
            if guide is None or svi_result is None:
                raise RuntimeError("plot_corner() requires HMC samples or an SVI guide.")
            import jax
            samples = guide.sample_posterior(jax.random.PRNGKey(42), svi_result.params, sample_shape=(num_samples,))
        model = self._require_model()
        _, parameter_lists = model.definition.as_dicts()
        return self._legacy_plot_result(
            lambda directory, filename: plot_corner_traced_params(
                samples, str(directory), max_samples=max_samples, filename=filename,
                param_list=parameter_lists,
            ), save_path, "corner.png",
        )

    def output(self, save_path: str | Path, *, scale: str = "log", residual_vis_max: float = 0.0,
               include_corner: bool = True) -> dict[str, Any]:
        """Write the standard single-band pipeline products to one run directory.

        The filenames deliberately match the existing config-driven pipeline,
        excluding its configuration snapshots.  This makes API and legacy runs
        interchangeable for downstream inspection tools.
        """
        directory = Path(save_path).expanduser()
        if directory.suffix and not directory.is_dir():
            raise ValueError("output(save_path) expects a directory, not a filename.")
        directory.mkdir(parents=True, exist_ok=True)
        from ..samplers import kwargs_with_deterministics, model_image_from_deterministics, save_metrics
        from ..utils import fit_dof_and_reduced_chi2, json_serializer, kwargs_best_to_json_pixelated_npy
        from ..visualizations import display_init, generate_run_plots

        model = self._require_model()
        if model.prob_model is None or model.lens_image is None:
            raise RuntimeError("Call build() before exporting a result.")
        type_list, parameter_lists = model.definition.as_dicts()
        kwargs_best = self.derived.get("kwargs")
        deterministics = self.derived.get("deterministics")
        if kwargs_best is None:
            kwargs_best, deterministics = kwargs_with_deterministics(
                model.prob_model, self.parameters,
            )
        best_fit_model = np.asarray(
            model_image_from_deterministics(model.prob_model, kwargs_best, deterministics)
        )
        components = self.derived.get("components") or self.derived.get("component_medians")
        if components is not None:
            best_fit_model = np.asarray(components["total"])
        kwargs_for_plots = kwargs_best
        source_plane = self.derived.get("source_plane") if self.samples is not None else None
        if source_plane is not None and kwargs_best.get("kwargs_source"):
            kwargs_for_plots = deepcopy(kwargs_best)
            kwargs_for_plots["kwargs_source"][0]["pixels"] = source_plane
        kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_for_plots, str(directory), type_list)
        kwargs_result_path = directory / "kwargs_result.json"
        with kwargs_result_path.open("w") as stream:
            json.dump(kwargs_json, stream, indent=4, default=json_serializer)

        metrics = self.metrics()
        save_metrics(
            str(directory), metrics["chi2"], model.data.likelihood_image,
            model.num_sampling_parameters, metrics["log_likelihood"],
            fit_dof_and_reduced_chi2,
            num_params_free=metrics["n_free_parameters"],
            num_params_physical=metrics["n_physical_parameters"],
            mask_bool=model.data.likelihood_mask,
        )
        files: dict[str, Path] = {
            "metrics": directory / "metrics.json",
            "kwargs_result": kwargs_result_path,
        }
        skipped: dict[str, str] = {}

        history = self.loss_history
        if history is not None:
            history_path = directory / "svi_loss_history.json"
            with history_path.open("w") as stream:
                json.dump({"loss_history": history.tolist()}, stream, indent=2)
            files["svi_loss_history"] = history_path
        if self.details.get("guide") is not None and self.details.get("result") is not None:
            files["svi_guide"] = self.save_svi_guide(directory)

            try:
                import jax
                cpu = jax.devices("cpu")[0]
                posterior = self.details["guide"].sample_posterior(
                    jax.random.PRNGKey(42),
                    jax.tree_util.tree_map(lambda value: jax.device_put(value, cpu), self.details["result"].params),
                    sample_shape=(2_000,),
                )
                sigma_parameters = jax.tree_util.tree_map(
                    lambda value: np.asarray(value).std(axis=0), posterior,
                )
                sigma_kwargs = model.prob_model.params2kwargs(sigma_parameters)
                sigma_json = kwargs_best_to_json_pixelated_npy(
                    sigma_kwargs, str(directory), type_list,
                    pixels_filename="kwargs_source_pixels_sigma.npy",
                    pixels_wn_filename="kwargs_source_pixels_wn_sigma.npy",
                    lens_light_pixels_prefix="kwargs_lens_light_pixels_sigma",
                )
                with (directory / "kwargs_sigma.json").open("w") as stream:
                    json.dump(sigma_json, stream, indent=4, default=json_serializer)
                files["kwargs_sigma"] = directory / "kwargs_sigma.json"
            except Exception as error:
                skipped["kwargs_sigma"] = str(error)

        initial = model.initial_parameters
        if initial is not None:
            try:
                display_init(
                    model.prob_model, initial, model.lens_image, model.data.likelihood_image,
                    model.data.likelihood_noise, model.data.pixel_scale, str(directory),
                    model.num_sampling_parameters, type_list=type_list,
                    residual_vis_max=residual_vis_max, fit_mask_bool=model.data.likelihood_mask,
                )
                files["kwargs_init"] = directory / "kwargs_init.json"
            except Exception as error:
                skipped["initial_guess"] = str(error)

        try:
            plot_details = dict(self.details)
            if not include_corner:
                plot_details.pop("guide", None)
                plot_details.pop("result", None)
            generate_run_plots(
                lens_image=model.lens_image, kwargs_best=kwargs_for_plots,
                image_data=model.data.likelihood_image, noise_map=model.data.likelihood_noise,
                psf_data=model.data.psf, pixel_scale=model.data.pixel_scale,
                save_path=str(directory), sampler=(
                    "hmc" if self.samples is not None
                    else "svi" if self.details.get("guide") is not None else "optax"
                ),
                best_fit_model=best_fit_model, chi2=metrics["chi2"],
                reduced_chi2=metrics["reduced_chi2"], extra=plot_details,
                mcmc_samples=self.samples, prob_model=model.prob_model,
                init_params=initial, param_list=parameter_lists,
                residual_vis_max=residual_vis_max,
                mcmc_component_medians=components,
                num_chains_hmc=(
                    int(self.details.get("num_chains_hmc_numpyro", 1))
                    if self.samples is not None else None
                ),
            )
        except Exception as error:
            skipped["diagnostic_plots"] = str(error)

        np.savez_compressed(
            directory / "modeling_result.npz", best_fit_model=best_fit_model,
            image_data=np.asarray(model.data.likelihood_image), noise_map=np.asarray(model.data.likelihood_noise),
            source_arc_mask=np.asarray(model.data.source_arc_mask) if model.data.source_arc_mask is not None else None,
            contaminate_mask=np.asarray(model.data.contaminate_mask) if model.data.contaminate_mask is not None else None,
            fit_mask_bool=np.asarray(model.data.likelihood_mask) if model.data.likelihood_mask is not None else None,
            image_unit=np.asarray("pixel_flux"), noise_unit=np.asarray("pixel_flux"),
        )
        files["modeling_result"] = directory / "modeling_result.npz"
        _write_parameter_shifts(directory, kwargs_best, type_list)
        files["parameter_shifts"] = directory / "parameter_shifts.txt"
        for filename in directory.iterdir():
            if filename.suffix == ".png":
                files[filename.stem] = filename
        return {"directory": directory, "files": files, "skipped": skipped}


def _write_parameter_shifts(directory: Path, kwargs: Mapping[str, Any], type_list: Mapping[str, list[str]]) -> None:
    """Write the legacy parameter summary for a self-contained API run."""
    categories = (
        ("kwargs_lens", "lens_mass", "lens_mass_type_list"),
        ("kwargs_lens_light", "lens_light", "lens_light_type_list"),
        ("kwargs_source", "source_light", "source_light_type_list"),
        ("kwargs_ps", "point_source", "point_source_type_list"),
    )
    lines: list[str] = []
    for key, label, type_key in categories:
        components = kwargs.get(key, [])
        if not components:
            continue
        lines.append(f"{label}:")
        types = type_list.get(type_key, [])
        for index, component in enumerate(components):
            name = types[index] if index < len(types) else "UNKNOWN"
            lines.append(f"     {name}:")
            suffix = f"_{index}" if len(components) > 1 else ""
            for parameter, value in component.items():
                array = np.asarray(value)
                if parameter == "pixels" or array.ndim != 0:
                    continue
                lines.append(f"            {parameter}{suffix}: null -> {float(array):.3f}")
    (directory / "parameter_shifts.txt").write_text("\n".join(lines) + "\n")


@dataclass
class SingleBandResultsCombination:
    """Comparison helper for repeated API fits of the same single-band model."""

    results: list[FitResult]

    def __post_init__(self) -> None:
        self.results = list(self.results)
        if not self.results:
            raise ValueError("ResultsCombination requires at least one FitResult.")

    def output(
        self,
        save_path: str | Path,
        *,
        residual_vis_max: float = 0.0,
        script_path: str | Path | None = None,
        copy_script: bool = True,
    ) -> dict[str, Path]:
        """Save the multi-run summary and copy the invoking Python script.

        If ``script_path`` is omitted, a normal ``python my_run.py`` execution
        is detected through ``__main__.__file__``.  Notebook sessions have no
        such file and are therefore left unchanged.
        """
        directory = Path(save_path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        from ..visualizations import plot_multiband_composite

        metrics: list[dict[str, Any]] = []
        bands: list[dict[str, Any]] = []
        comparison: dict[str, dict[str, Any]] = {}
        for index, result in enumerate(self.results):
            model = result._require_model()
            result_metrics = result.metrics()
            # Use the same metric keys and calculation as the config pipeline.
            legacy_metrics = {
                "BIC": result_metrics["bic"],
                "CHI2": result_metrics["chi2"],
                "CHI2_NPIX2": result_metrics["chi2"] / result_metrics["n_data_pixels"],
                "REDUCED_CHI2": result_metrics["reduced_chi2"],
                "CHI2_DOF": result_metrics["degrees_of_freedom"],
                "N_DATA_PIXELS": result_metrics["n_data_pixels"],
                "N_PARAMS_FITTED": model.num_sampling_parameters,
                "N_PARAMS_FREE": result_metrics["n_free_parameters"],
                "LOG_LIKELIHOOD": result_metrics["log_likelihood"],
            }
            metrics.append(legacy_metrics)
            comparison[f"run_{index}"] = {
                "seed": result.random_seed,
                "metrics": legacy_metrics,
            }
            bands.append({
                "name": f"Run {index}",
                "lens_image": model.lens_image,
                "kwargs_result": result._kwargs_result(),
                "image_data": model.data.likelihood_image,
                "noise_map": model.data.likelihood_noise,
                "pixel_scale": model.data.pixel_scale,
            })
        json_path = directory / "comparison.json"
        with json_path.open("w") as stream:
            json.dump(comparison, stream, indent=4, default=_json_default)
        plot_path = directory / "svi_run_comparison.png"
        plot_multiband_composite(
            bands, str(directory), residual_vis_max=residual_vis_max,
            output_filename=plot_path.name,
        )
        print("\n========================================")
        print("All runs completed.")
        print(f"Comparison summary saved to {json_path}")
        print("========================================")
        for index, entry in enumerate(metrics):
            seed = self.results[index].random_seed
            print(
                f"run_{index} (seed={seed}): "
                f"log-likelihood={entry['LOG_LIKELIHOOD']:.2f}, "
                f"chi2={entry['CHI2']:.2f}, "
                f"chi2/N_pix^2={entry['CHI2_NPIX2']:.4f}, "
                f"reduced_chi2={entry['REDUCED_CHI2']:.4f}, "
                f"BIC={entry['BIC']:.2f}"
            )
        print("========================================")
        files = {"comparison": json_path, "svi_run_comparison": plot_path}
        # if copy_script:
        #     if script_path is None:
        #         main_module = sys.modules.get("__main__")
        #         script_path = getattr(main_module, "__file__", None)
        #     if script_path is not None:
        #         source = Path(script_path).expanduser().resolve()
        #         if source.is_file() and source.suffix == ".py":
        #             destination = directory / source.name
        #             if source != destination.resolve():
        #                 shutil.copy2(source, destination)
        #             files["script"] = destination
        #             print(f"Copied pipeline script to {destination}")
        return files


def _json_default(value: Any):
    """Serialize NumPy/JAX scalar and array values in API result exports."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
