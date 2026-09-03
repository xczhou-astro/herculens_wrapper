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
from typing import Any, Mapping, Sequence
import numpy as np
from .models import SamplerName


_SVI_COMPLETION_FILES = (
    "kwargs_result.json",
    "metrics.json",
    "modeling_result.fits",
    "parameter_shifts.txt",
    "svi_loss_history.json",
    "svi_guide_params.pkl",
)


def is_completed_svi_run(save_path: str | Path) -> bool:
    """Return whether a standard API SVI run was exported successfully.

    Plot files are intentionally not required: they can be legitimately absent
    when a diagnostic does not apply.  A missing result, guide, or model FITS
    means the run was incomplete.  Incomplete SVI runs are restarted from
    their configured initialisation; unlike HMC, SVI has no saved optimizer
    state to continue an individual iteration sequence.
    """
    directory = Path(save_path).expanduser()
    return directory.is_dir() and all(
        (directory / filename).is_file() for filename in _SVI_COMPLETION_FILES
    )


_TRUTH_COMPONENTS = {
    "lens_mass": ("kwargs_lens", "lens_"),
    "lens_light": ("kwargs_lens_light", "lens_light_"),
    "source_light": ("kwargs_source", "source_"),
    "point_source": ("kwargs_point_source", "ps_"),
}
_TRUTH_COMPONENT_ALIASES = {
    "lens mass": "lens_mass",
    "lens light": "lens_light",
    "source light": "source_light",
    "point source": "point_source",
    "ps": "point_source",
}


def _normalise_truth_components(components: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("components must be a non-empty list, e.g. ['lens_mass', 'lens_light'].")
    result = []
    for component in components:
        canonical = _TRUTH_COMPONENT_ALIASES.get(str(component).strip().lower(), str(component).strip().lower())
        if canonical not in _TRUTH_COMPONENTS:
            allowed = ", ".join(_TRUTH_COMPONENTS)
            raise ValueError(f"Unknown truth-comparison component {component!r}; choose from {allowed}.")
        if canonical not in result:
            result.append(canonical)
    return result


def _truth_mapping(truth: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(truth, (str, Path)):
        path = Path(truth).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Truth JSON does not exist: {path}")
        with path.open() as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Truth JSON must contain a dictionary: {path}")
        return dict(loaded)
    if isinstance(truth, Mapping):
        return dict(truth)
    raise TypeError("truth must be a mapping or a path to a JSON file.")


def _parse_single_band_site(name: str) -> tuple[str, str, int] | None:
    """Map a single-band NumPyro site to component, parameter, profile index."""
    local = str(name).rsplit("/", 1)[-1]
    # Order matters because ``lens_light_`` begins with ``lens_``.
    for component, prefix in (
        ("lens_light", "lens_light_"),
        ("lens_mass", "lens_"),
        ("source_light", "source_"),
        ("point_source", "ps_"),
    ):
        if not local.startswith(prefix):
            continue
        remainder = local[len(prefix):]
        if "_" not in remainder:
            return None
        parameter, index = remainder.rsplit("_", 1)
        if parameter and index.isdigit():
            return component, parameter, int(index)
    return None


def _ordered_single_band_truth_sites(
    samples: Mapping[str, Any],
    components: Sequence[str],
    kwargs_result: Mapping[str, Any] | None,
) -> list[str]:
    """Order sampled scalar sites by declared component/profile/parameter.

    HDF5 enumerates dataset keys in storage order (normally lexical), which
    is unrelated to the physical model declaration.  ``kwargs_result.json``
    retains the profile and parameter insertion order used by the API output;
    use it whenever it is available, then append any unlisted sampled sites
    deterministically.
    """
    parsed = {
        site: parsed_site
        for site in samples
        if (parsed_site := _parse_single_band_site(site)) is not None
        and parsed_site[0] in components
    }
    ordered: list[str] = []
    seen: set[str] = set()
    for component in components:
        kwargs_key, _ = _TRUTH_COMPONENTS[component]
        profiles = [] if kwargs_result is None else kwargs_result.get(kwargs_key, [])
        if isinstance(profiles, list):
            for profile_index, profile in enumerate(profiles):
                if not isinstance(profile, Mapping):
                    continue
                for parameter in profile:
                    matches = [
                        site for site, item in parsed.items()
                        if item == (component, parameter, profile_index)
                    ]
                    for site in sorted(matches):
                        if site not in seen:
                            ordered.append(site)
                            seen.add(site)
        # Retain sampled values that are not serialised in kwargs_result
        # (for example a future scalar nuisance parameter), without allowing
        # them to disrupt the declared parameter order above.
        remainder = sorted(
            (site for site, item in parsed.items()
             if item[0] == component and site not in seen),
            key=lambda site: (parsed[site][2], parsed[site][1], site),
        )
        ordered.extend(remainder)
        seen.update(remainder)
    return ordered


def analyze_hmc_degeneracies(
    hmc_run: str | Path,
    *,
    components: Sequence[str] = ("lens_mass", "lens_light", "source_light", "point_source"),
    min_abs_correlation: float = 0.6,
    include_hyperparameters: bool = False,
    include_derived: bool = False,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load an HMC run and quantify pairwise posterior degeneracies.

    Parameters
    ----------
    hmc_run
        A directory containing both ``hmc_samples.h5`` and
        ``kwargs_result.json``, or the path to its ``hmc_samples.h5`` file.
    components
        Model components whose scalar NumPyro sites are included.  Pixelated
        fields are always excluded: a correlation matrix of source pixels is
        not a useful lens-parameter diagnostic.
    min_abs_correlation
        Threshold used to list ``strong_pairs``.  The complete Pearson and
        Spearman matrices are returned irrespective of this threshold.
    include_hyperparameters
        Include scalar sites that do not map to a standard model component.
    include_derived
        Also add ``q``/``phi_deg`` from each sampled ``e1``/``e2`` pair and
        ``gamma_ext``/``phi_ext_deg`` from ``gamma1``/``gamma2``.  This is
        disabled by default because derived quantities are deterministic
        functions of native parameters and therefore introduce trivial strong
        correlations.
    save_path
        Optional JSON summary path.  The large sample matrix is returned in
        memory but intentionally is not duplicated in the JSON file.

    Notes
    -----
    Pairwise correlations identify approximately *linear* posterior
    degeneracies.  Inspect a corner plot as well for curved or multimodal
    relations.  ``kwargs_result.json`` is loaded and returned so callers can
    associate profile indices with their fitted model configuration.
    """
    if not np.isfinite(min_abs_correlation) or not 0.0 <= min_abs_correlation <= 1.0:
        raise ValueError("min_abs_correlation must lie between 0 and 1.")
    selected_components = _normalise_truth_components(tuple(components))
    archive = Path(hmc_run).expanduser()
    if archive.is_dir():
        run_directory, archive = archive, archive / "hmc_samples.h5"
    else:
        run_directory = archive.parent
    if archive.name != "hmc_samples.h5" or not archive.is_file():
        raise FileNotFoundError("hmc_run must be a run directory or an existing hmc_samples.h5 file.")
    kwargs_path = run_directory / "kwargs_result.json"
    if not kwargs_path.is_file():
        raise FileNotFoundError(f"HMC run is missing kwargs_result.json: {kwargs_path}")
    with kwargs_path.open() as stream:
        kwargs_result = json.load(stream)
    if not isinstance(kwargs_result, Mapping):
        raise ValueError(f"kwargs_result.json must contain a dictionary: {kwargs_path}")

    from ..samplers import _load_hmc_samples_hdf5

    samples, _ = _load_hmc_samples_hdf5(str(archive))
    scalar_sites: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for site, raw_values in samples.items():
        parsed = _parse_single_band_site(site)
        if parsed is None:
            if not include_hyperparameters:
                continue
            label = str(site).rsplit("/", 1)[-1]
        else:
            component, parameter, profile_index = parsed
            if component not in selected_components:
                continue
            label = f"{component}[{profile_index}].{parameter}"
        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 1:
            pass
        elif values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        else:
            skipped.append(f"{site}: non-scalar shape {values.shape}")
            continue
        if values.size < 3:
            skipped.append(f"{site}: fewer than three posterior draws")
            continue
        if np.nanstd(values) == 0.0:
            skipped.append(f"{site}: zero posterior variance")
            continue
        scalar_sites[label] = values
    if len(scalar_sites) < 2:
        raise ValueError("Need at least two varying scalar HMC sites to calculate degeneracies.")

    if include_derived:
        native = dict(scalar_sites)
        profile_bases = {label.rsplit(".", 1)[0] for label in native if "." in label}
        for base in sorted(profile_bases):
            e1, e2 = native.get(f"{base}.e1"), native.get(f"{base}.e2")
            if e1 is not None and e2 is not None:
                ellipticity = np.hypot(e1, e2)
                scalar_sites[f"{base}.q"] = (1.0 - ellipticity) / (1.0 + ellipticity)
                scalar_sites[f"{base}.phi_deg"] = np.degrees(0.5 * np.arctan2(e2, e1))
            gamma1, gamma2 = native.get(f"{base}.gamma1"), native.get(f"{base}.gamma2")
            if gamma1 is not None and gamma2 is not None:
                scalar_sites[f"{base}.gamma_ext"] = np.hypot(gamma1, gamma2)
                scalar_sites[f"{base}.phi_ext_deg"] = np.degrees(0.5 * np.arctan2(gamma2, gamma1))

    labels = list(scalar_sites)
    lengths = {values.shape[0] for values in scalar_sites.values()}
    if len(lengths) != 1:
        raise ValueError(f"Scalar HMC sites have inconsistent draw counts: {sorted(lengths)}")
    matrix = np.column_stack([scalar_sites[label] for label in labels])
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[finite_rows]
    if matrix.shape[0] < 3:
        raise ValueError("Fewer than three finite joint posterior draws remain.")

    # Spearman catches monotonic but non-linear relationships without adding a
    # mandatory plotting dependency to this lightweight diagnostic.
    from scipy.stats import spearmanr

    pearson = np.corrcoef(matrix, rowvar=False)
    spearman = np.asarray(spearmanr(matrix, axis=0).statistic, dtype=float)
    # scipy returns a scalar rather than a 2x2 matrix for exactly two inputs.
    if spearman.ndim == 0:
        spearman = np.array([[1.0, float(spearman)], [float(spearman), 1.0]])
    strong_pairs: list[dict[str, Any]] = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            correlation = float(pearson[first, second])
            if abs(correlation) >= min_abs_correlation:
                strong_pairs.append({
                    "parameter_1": labels[first],
                    "parameter_2": labels[second],
                    "pearson_r": correlation,
                    "spearman_r": float(spearman[first, second]),
                })
    strong_pairs.sort(key=lambda item: abs(item["pearson_r"]), reverse=True)

    summary = {
        "hmc_samples": str(archive),
        "kwargs_result": str(kwargs_path),
        "components": selected_components,
        "parameter_labels": labels,
        "n_draws": int(matrix.shape[0]),
        "min_abs_correlation": float(min_abs_correlation),
        "include_hyperparameters": bool(include_hyperparameters),
        "include_derived": bool(include_derived),
        "pearson_correlation": pearson.tolist(),
        "spearman_correlation": spearman.tolist(),
        "strong_pairs": strong_pairs,
        "skipped": skipped,
    }
    summary_file: Path | None = None
    if save_path is not None:
        summary_file = Path(save_path).expanduser()
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "samples": matrix,
        "parameter_labels": labels,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "strong_pairs": strong_pairs,
        "kwargs_result": dict(kwargs_result),
        "skipped": skipped,
        "summary_file": summary_file,
    }


def compare_hmc_truth(
    hmc_samples: str | Path,
    truth: Mapping[str, Any] | str | Path,
    *,
    components: list[str] | tuple[str, ...],
    save_path: str | Path,
    max_samples: int = 15_000,
) -> dict[str, Any]:
    """Plot selected HMC posterior parameters against simulation truth.

    This standalone utility needs no declared data, profiles, or model.  Pass
    either a HMC run directory or its ``hmc_samples.h5`` file, plus a truth
    ``params.json``/mapping using the standard ``kwargs_*`` structure.
    """
    if not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer.")
    selected_components = _normalise_truth_components(components)
    archive = Path(hmc_samples).expanduser()
    if archive.is_dir():
        archive = archive / "hmc_samples.h5"
    if archive.name != "hmc_samples.h5" or not archive.is_file():
        raise FileNotFoundError("hmc_samples must be a run directory or an existing hmc_samples.h5 file.")
    from ..samplers import _load_hmc_samples_hdf5

    samples, _ = _load_hmc_samples_hdf5(str(archive))
    if not samples:
        raise ValueError(f"HMC archive contains no posterior samples: {archive}")
    truth_data = _truth_mapping(truth)
    directory = Path(save_path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    kwargs_result: Mapping[str, Any] | None = None
    kwargs_path = archive.parent / "kwargs_result.json"
    if kwargs_path.is_file():
        try:
            with kwargs_path.open() as stream:
                candidate = json.load(stream)
            if isinstance(candidate, Mapping):
                kwargs_result = candidate
        except (OSError, json.JSONDecodeError):
            # Truth comparison only needs HMC samples and truth; preserve its
            # original usefulness if an old/malformed output lacks kwargs.
            pass

    values: list[np.ndarray] = []
    true_values: list[float] = []
    labels: list[str] = []
    skipped: list[str] = []
    for site in _ordered_single_band_truth_sites(
        samples, selected_components, kwargs_result,
    ):
        site_values = samples[site]
        parsed = _parse_single_band_site(site)
        if parsed is None:
            continue
        component, parameter, profile_index = parsed
        if component not in selected_components:
            continue
        kwargs_key, _ = _TRUTH_COMPONENTS[component]
        truth_profiles = truth_data.get(kwargs_key)
        if truth_profiles is None and component == "point_source":
            truth_profiles = truth_data.get("kwargs_ps")
        if not isinstance(truth_profiles, list) or profile_index >= len(truth_profiles):
            skipped.append(f"{site}: missing {kwargs_key}[{profile_index}] in truth")
            continue
        truth_profile = truth_profiles[profile_index]
        if not isinstance(truth_profile, Mapping) or parameter not in truth_profile:
            skipped.append(f"{site}: missing truth parameter {parameter!r}")
            continue
        posterior = np.asarray(site_values)
        if posterior.ndim == 1:
            posterior = posterior[:, None]
        elif posterior.ndim != 2:
            # Pixel fields and other high-dimensional nuisance variables are
            # intentionally not suitable for a parameter recovery corner plot.
            continue
        truth_value = np.asarray(truth_profile[parameter]).reshape(-1)
        if posterior.shape[1] != truth_value.size:
            skipped.append(f"{site}: posterior/truth shape mismatch")
            continue
        for element in range(posterior.shape[1]):
            draws = np.asarray(posterior[:, element], dtype=float)
            true_value = float(truth_value[element])
            finite = np.isfinite(draws)
            if finite.sum() < 2 or not np.isfinite(true_value):
                skipped.append(f"{site}: non-finite posterior or truth")
                continue
            suffix = "" if posterior.shape[1] == 1 else f"[{element}]"
            labels.append(f"{component}[{profile_index}].{parameter}{suffix}")
            values.append(draws[finite])
            true_values.append(true_value)
    if not values:
        raise ValueError("No comparable scalar sites found for the requested components.")

    import matplotlib.pyplot as plt

    summary: dict[str, dict[str, float]] = {}
    ncols = min(3, len(values))
    nrows = int(np.ceil(len(values) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for index, (label, draws, true_value) in enumerate(zip(labels, values, true_values)):
        axis = axes.flat[index]
        median, p16, p84 = np.percentile(draws, [50.0, 16.0, 84.0])
        summary[label] = {
            "truth": true_value, "median": float(median),
            "lower_1sigma": float(median - p16), "upper_1sigma": float(p84 - median),
        }
        axis.hist(draws, bins="auto", density=True, histtype="stepfilled", color="tab:blue", alpha=0.55)
        axis.axvspan(p16, p84, color="tab:blue", alpha=0.16, label="posterior 1σ")
        axis.axvline(median, color="tab:blue", lw=1.7, label="posterior median")
        axis.axvline(true_value, color="tab:red", lw=1.8, ls="--", label="truth")
        axis.set(title=label, xlabel="parameter value", ylabel="posterior density")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(values):]:
        axis.remove()
    figure.suptitle("HMC posterior versus truth", y=1.01)
    figure.tight_layout()
    one_d_path = directory / "posterior_truth_1d.png"
    figure.savefig(one_d_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    corner_path: Path | None = None
    if len(values) >= 2:
        try:
            import corner
        except ImportError as error:
            raise ImportError("compare_hmc_truth() needs the optional 'corner' package for its 2D plot.") from error
        n_draws = min(len(draws) for draws in values)
        matrix = np.column_stack([draws[:n_draws] for draws in values])
        if n_draws > max_samples:
            matrix = matrix[np.random.default_rng(42).choice(n_draws, size=max_samples, replace=False)]
        # For a 2D Gaussian these enclosed-probability levels correspond to
        # the familiar nested 1σ, 2σ, and 3σ contours.
        figure = corner.corner(
            matrix, labels=labels, truths=true_values, truth_color="tab:red",
            quantiles=[0.16, 0.5, 0.84], levels=[0.393, 0.865, 0.989],
            show_titles=True, title_fmt=".4g",
        )
        corner_path = directory / "posterior_truth_corner.png"
        figure.savefig(corner_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    summary_path = directory / "posterior_truth_summary.json"
    summary_path.write_text(json.dumps({
        "hmc_samples": str(archive), "components": selected_components,
        "parameters": summary, "skipped": skipped,
    }, indent=2) + "\n")
    return {
        "summary": summary, "skipped": skipped, "one_dimensional": one_d_path,
        "corner": corner_path, "summary_file": summary_path,
    }


def _nested_difference(reference: Any, bound: Any, *, upper: bool) -> Any:
    """Return non-negative ``bound - reference`` or ``reference - bound``."""
    if isinstance(reference, Mapping) and isinstance(bound, Mapping):
        return {
            key: _nested_difference(reference[key], bound[key], upper=upper)
            for key in reference if key in bound
        }
    if isinstance(reference, list) and isinstance(bound, list):
        return [_nested_difference(a, b, upper=upper) for a, b in zip(reference, bound)]
    try:
        delta = np.asarray(bound) - np.asarray(reference) if upper else np.asarray(reference) - np.asarray(bound)
        return np.maximum(delta, 0.0)
    except (TypeError, ValueError):
        return reference


def _pack_asymmetric_one_sigma(lower: Any, upper: Any) -> Any:
    """Pack scalar errors as ``[lower, upper]`` and arrays as one RMS map."""
    if isinstance(lower, Mapping) and isinstance(upper, Mapping):
        return {
            key: _pack_asymmetric_one_sigma(lower[key], upper[key])
            for key in lower if key in upper
        }
    if isinstance(lower, list) and isinstance(upper, list):
        return [_pack_asymmetric_one_sigma(a, b) for a, b in zip(lower, upper)]
    try:
        lower_array, upper_array = np.asarray(lower), np.asarray(upper)
        if lower_array.ndim == 0 and upper_array.ndim == 0:
            return [float(lower_array), float(upper_array)]
        # FITS image stubs represent one array.  Use the RMS of the unequal
        # lower/upper credible errors, analogous to a conventional sigma map.
        return np.sqrt(0.5 * (np.square(lower_array) + np.square(upper_array)))
    except (TypeError, ValueError):
        return lower


def hmc_one_sigma_kwargs(
    prob_model: Any,
    samples: Mapping[str, Any],
    median_parameters: Mapping[str, Any],
    type_list: Mapping[str, list[str]],
    directory: str | Path,
    kwargs_from_params: Any | None = None,
) -> dict[str, Any]:
    """Build one-sigma HMC uncertainties in the legacy kwargs-sigma layout."""
    from ..utils import append_array_fits, kwargs_best_to_json_pixelated_npy

    root = Path(directory)
    kwargs_from_params = prob_model.params2kwargs if kwargs_from_params is None else kwargs_from_params
    reference_kwargs = kwargs_from_params(median_parameters)
    lower_params, upper_params = dict(median_parameters), dict(median_parameters)
    for name, values in samples.items():
        if name not in median_parameters:
            continue
        lower_params[name] = np.percentile(np.asarray(values), 15.865525, axis=0)
        upper_params[name] = np.percentile(np.asarray(values), 84.134475, axis=0)
    lower_kwargs = kwargs_from_params(lower_params)
    upper_kwargs = kwargs_from_params(upper_params)
    one_sigma = _pack_asymmetric_one_sigma(
        _nested_difference(reference_kwargs, lower_kwargs, upper=False),
        _nested_difference(reference_kwargs, upper_kwargs, upper=True),
    )
    # Physical source-pixel errors are intrinsically asymmetric.  Their
    # actual 16th/84th-percentile offsets are saved as LOWER/UPPER extensions
    # by ``evaluate_mcmc_source_pixels_summary``; do not replace them here by
    # a parameter-wise RMS approximation.
    source = (one_sigma.get("kwargs_source") or [{}])[0]
    if isinstance(source, Mapping):
        source.pop("pixels", None)
        # ``pixels_wn`` is an internal latent field, not a physical source
        # reconstruction.  Its conventional symmetric sigma map is retained.
        if source.get("pixels_wn") is not None:
            append_array_fits(
                root / "kwargs_source_pixels_wn.fits",
                source["pixels_wn"],
                extension_name="SIGMA",
            )
    sigma_json = kwargs_best_to_json_pixelated_npy(
        one_sigma, str(root), type_list,
        pixels_filename="kwargs_source_pixels.fits",
        pixels_wn_filename="kwargs_source_pixels_wn.fits",
        lens_light_pixels_prefix="kwargs_lens_light_pixels_sigma",
        save_pixel_arrays=False,
        references_already_saved=True,
        pixels_wn_hdu="SIGMA",
    )
    source_json = (sigma_json.get("kwargs_source") or [{}])[0]
    if isinstance(source_json, dict):
        source_json.pop("pixels", None)
        source_json["pixels_lower"] = {
            "_format": "pixelated_pixels_fits",
            "file": "kwargs_source_pixels.fits",
            "hdu": "LOWER",
            "_unit": "pixel_flux",
            "_pixel_area_reference": "image_data_pixel",
        }
        source_json["pixels_upper"] = {
            "_format": "pixelated_pixels_fits",
            "file": "kwargs_source_pixels.fits",
            "hdu": "UPPER",
            "_unit": "pixel_flux",
            "_pixel_area_reference": "image_data_pixel",
        }
    return sigma_json

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
        disable_gibbs: bool = False,
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
        if not isinstance(disable_gibbs, bool):
            raise TypeError("disable_gibbs must be a boolean.")
        return cls("hmc", random_seed=random_seed, options={
            "num_warmup_hmc_numpyro": num_warmup,
            "num_samples_hmc_numpyro": num_samples,
            "num_chains_hmc_numpyro": num_chains,
            "checkpoint_interval_hmc_numpyro": checkpoint_interval,
            "chain_method_hmc_numpyro": chain_method,
            "progress_bar_hmc_numpyro": progress_bar,
            "hmc_init_max_retries": init_max_retries,
            "disable_gibbs": disable_gibbs,
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
        """Return median and, for HMC, maximum-likelihood sample metrics."""
        metrics = self._require_model()._metrics(self.parameters)
        summary = self.derived.get("sample_likelihood_summary") if self.samples is not None else None
        if summary is None:
            metrics.update({
                "max_log_likelihood": None,
                "chi2_max_loglike": None,
                "reduced_chi2_max_loglike": None,
                "bic_physical_max_loglike": None,
                "max_loglike_sample_index": None,
            })
            return metrics
        max_loglike = float(summary["max_log_likelihood"])
        chi2 = float(summary["chi2_max_loglike"])
        n_data, n_physical = metrics["n_data_pixels"], metrics["n_physical_parameters"]
        metrics.update({
            "max_log_likelihood": max_loglike,
            "chi2_max_loglike": chi2,
            "reduced_chi2_max_loglike": chi2 / metrics["degrees_of_freedom"],
            "bic_physical_max_loglike": n_physical * np.log(max(n_data, 1)) - 2.0 * max_loglike,
            "max_loglike_sample_index": summary.get("max_loglike_sample_index"),
        })
        return metrics

    def compare_truth(
        self,
        truth: Mapping[str, Any] | str | Path,
        *,
        components: list[str] | tuple[str, ...],
        save_path: str | Path,
        max_samples: int = 15_000,
    ) -> dict[str, Any]:
        """Compare selected scalar HMC posteriors against simulation truth.

        ``truth`` is either a legacy-style truth dictionary or a JSON file
        containing ``kwargs_lens``, ``kwargs_lens_light``, ``kwargs_source``,
        and/or ``kwargs_point_source``.  ``components`` explicitly selects
        one or more of ``lens_mass``, ``lens_light``, ``source_light``, and
        ``point_source`` (space-separated aliases are accepted).  Fixed and
        linked parameters have no independent posterior site and are reported
        in ``skipped`` rather than plotted.

        The method writes ``posterior_truth_1d.png`` and, when two or more
        scalar parameters are selected, ``posterior_truth_corner.png``.  The
        latter marks truth in red and draws the 2D 1-sigma (39.3-percent)
        credible contour.
        """
        if self.samples is None:
            raise RuntimeError("compare_truth() requires HMC posterior samples.")
        if not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer.")
        selected_components = _normalise_truth_components(components)
        if isinstance(truth, (str, Path)):
            truth_path = Path(truth).expanduser()
            if not truth_path.is_file():
                raise FileNotFoundError(f"Truth JSON does not exist: {truth_path}")
            with truth_path.open() as stream:
                truth_data = json.load(stream)
        elif isinstance(truth, Mapping):
            truth_data = dict(truth)
        else:
            raise TypeError("truth must be a mapping or a path to a JSON file.")

        directory = Path(save_path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        model = self._require_model()
        _, parameter_lists = model.definition.as_dicts()
        samples = self.samples
        values: list[np.ndarray] = []
        truths: list[float] = []
        labels: list[str] = []
        skipped: list[str] = []

        for component in selected_components:
            kwargs_key, prefix = _TRUTH_COMPONENTS[component]
            truth_components = truth_data.get(kwargs_key)
            # Accept the historic short spelling as input only.
            if truth_components is None and component == "point_source":
                truth_components = truth_data.get("kwargs_ps")
            definitions = parameter_lists.get({
                "lens_mass": "lens_mass_params_list",
                "lens_light": "lens_light_params_list",
                "source_light": "source_light_params_list",
                "point_source": "point_source_params_list",
            }[component], [])
            if not isinstance(truth_components, list):
                skipped.append(f"{component}: missing {kwargs_key} in truth")
                continue
            for profile_index, definition in enumerate(definitions):
                if profile_index >= len(truth_components) or not isinstance(truth_components[profile_index], Mapping):
                    skipped.append(f"{component}[{profile_index}]: missing truth profile")
                    continue
                truth_profile = truth_components[profile_index]
                for parameter in definition:
                    site = f"{prefix}{parameter}_{profile_index}"
                    if site not in samples:
                        # Correlated and fixed parameters intentionally have
                        # no independent NumPyro posterior coordinate.
                        continue
                    if parameter not in truth_profile or truth_profile[parameter] is None:
                        skipped.append(f"{component}[{profile_index}].{parameter}: missing truth value")
                        continue
                    posterior = np.asarray(samples[site])
                    truth_value = np.asarray(truth_profile[parameter])
                    if posterior.ndim == 1:
                        posterior = posterior[:, None]
                    elif posterior.ndim > 2:
                        skipped.append(f"{component}[{profile_index}].{parameter}: non-scalar posterior")
                        continue
                    flat_truth = truth_value.reshape(-1)
                    if posterior.shape[1] != flat_truth.size:
                        skipped.append(
                            f"{component}[{profile_index}].{parameter}: posterior/truth shape mismatch"
                        )
                        continue
                    for element_index in range(posterior.shape[1]):
                        draws = np.asarray(posterior[:, element_index], dtype=float)
                        true_value = float(flat_truth[element_index])
                        finite = np.isfinite(draws)
                        if not np.isfinite(true_value) or finite.sum() < 2:
                            skipped.append(f"{component}[{profile_index}].{parameter}: non-finite values")
                            continue
                        suffix = "" if posterior.shape[1] == 1 else f"[{element_index}]"
                        labels.append(f"{component}[{profile_index}].{parameter}{suffix}")
                        values.append(draws[finite])
                        truths.append(true_value)

        if not values:
            raise ValueError(
                "No comparable scalar posterior sites were found. Check components, "
                "the declared profiles, and the truth kwargs dictionary."
            )

        import matplotlib.pyplot as plt

        summary: dict[str, dict[str, float]] = {}
        ncols = min(3, len(values))
        nrows = int(np.ceil(len(values) / ncols))
        figure, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
        for index, (label, draws, true_value) in enumerate(zip(labels, values, truths)):
            axis = axes.flat[index]
            median, lower, upper = np.percentile(draws, [50.0, 16.0, 84.0])
            summary[label] = {
                "truth": true_value,
                "median": float(median),
                "lower_1sigma": float(median - lower),
                "upper_1sigma": float(upper - median),
            }
            axis.hist(draws, bins="auto", density=True, histtype="stepfilled", color="tab:blue", alpha=0.55)
            axis.axvspan(lower, upper, color="tab:blue", alpha=0.16, label="posterior 1σ")
            axis.axvline(median, color="tab:blue", lw=1.7, label="posterior median")
            axis.axvline(true_value, color="tab:red", lw=1.8, ls="--", label="truth")
            axis.set(title=label, xlabel="parameter value", ylabel="posterior density")
            axis.legend(fontsize=8)
        for axis in axes.flat[len(values):]:
            axis.remove()
        figure.suptitle("HMC posterior versus truth", y=1.01)
        figure.tight_layout()
        one_d_path = directory / "posterior_truth_1d.png"
        figure.savefig(one_d_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        corner_path: Path | None = None
        if len(values) >= 2:
            try:
                import corner
            except ImportError as error:
                raise ImportError("compare_truth() needs the optional 'corner' package for its 2D plot.") from error
            common_count = min(len(item) for item in values)
            matrix = np.column_stack([item[:common_count] for item in values])
            if common_count > max_samples:
                rng = np.random.default_rng(42)
                matrix = matrix[rng.choice(common_count, size=max_samples, replace=False)]
            figure = corner.corner(
                matrix, labels=labels, truths=truths, truth_color="tab:red",
                quantiles=[0.16, 0.5, 0.84], levels=[0.393, 0.865, 0.989],
                show_titles=True, title_fmt=".4g",
            )
            corner_path = directory / "posterior_truth_corner.png"
            figure.savefig(corner_path, dpi=180, bbox_inches="tight")
            plt.close(figure)

        summary_path = directory / "posterior_truth_summary.json"
        summary_path.write_text(json.dumps({
            "components": selected_components,
            "parameters": summary,
            "skipped": skipped,
        }, indent=2) + "\n")
        return {
            "summary": summary,
            "skipped": skipped,
            "one_dimensional": one_d_path,
            "corner": corner_path,
            "summary_file": summary_path,
        }

    def fit_analytic_pixelated_source(
        self,
        *,
        profile: str = "SERSIC_ELLIPSE",
        n_samples: int = 200,
        crop_radius: float | None = None,
        truth: Mapping[str, Any] | str | Path | None = None,
        save_path: str | Path | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Strict per-HMC-draw analytic fit on a uniform pixelated source.

        Each draw ray-traces the declared source-arc mask using its own lens
        parameters before fitting, so source-plane physical coordinates and
        pixel scale are propagated with the lens posterior.  ``crop_radius``
        restricts each fit to a circle centred on the median source grid.
        """
        if self.samples is None:
            raise RuntimeError("fit_analytic_pixelated_source() requires HMC samples.")
        model = self._require_model()
        if model.initialization_path is None:
            raise RuntimeError("The HMC run directory is unavailable; load HMC from disk first.")
        from .utils import fit_analytic_pixelated_source
        return fit_analytic_pixelated_source(
            model.initialization_path, profile=profile, n_samples=n_samples,
            crop_radius=crop_radius, image_pixel_scale=model.data.pixel_scale,
            truth=truth, save_path=save_path, random_seed=random_seed,
            model=model, median_parameters=self.parameters,
        )

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

    def get_source_plane(self) -> dict[str, Any]:
        """Return reconstructed source pixels and their source-plane geometry.

        For an RTU source, ``x_corners`` and ``y_corners`` are physical arcsec
        cell corners of shape ``(ny + 1, nx + 1)``.  They can be passed to
        ``matplotlib.pyplot.pcolormesh`` together with ``pixels``.  Ordinary
        uniform grids instead return their regular ``x``/``y`` coordinates.
        """
        model = self._require_model()
        kwargs = self.derived.get("kwargs") or self._kwargs_result()
        sources = kwargs.get("kwargs_source", [])
        if not sources or "pixels" not in sources[0]:
            raise RuntimeError("This result does not contain a pixelated source reconstruction.")
        pixels = np.asarray(self.derived.get("source_plane", sources[0]["pixels"]))
        if getattr(model.lens_image, "_rtu_grid_source", False):
            x_corners, y_corners = model.lens_image.get_rtu_source_plane_grid(kwargs.get("kwargs_lens"))
            return {
                "grid_kind": "ray_transformed_uniform",
                "pixels": pixels,
                "x_corners": np.asarray(x_corners),
                "y_corners": np.asarray(y_corners),
            }
        x, y, extent = model.lens_image.get_source_coordinates(kwargs.get("kwargs_lens"))
        return {
            "grid_kind": "uniform",
            "pixels": pixels,
            "x": np.asarray(x),
            "y": np.asarray(y),
            "extent": np.asarray(extent),
        }

    def _uniform_pixelated_source_plane(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.get_source_plane()
        if source["grid_kind"] != "uniform":
            raise NotImplementedError("These source-plane diagnostics currently support uniform pixelated grids only.")
        x, y = np.meshgrid(source["x"], source["y"])
        return np.asarray(source["pixels"], dtype=float), x, y

    def plot_radial_profile(
        self,
        crop_radius: float | None = None,
        truth: Mapping[str, Any] | str | Path | None = None,
        save_path: str | Path | None = None,
        xscale: str = "linear",
        yscale: str = "linear",
    ) -> Path:
        """Plot median pixelated-source radial brightness and enclosed flux.

        ``crop_radius`` is in arcsec; both profiles terminate at that radius.
        When supplied, ``truth`` is a params JSON or mapping with an analytic
        ``kwargs_source`` component and is overplotted for comparison.
        """
        from .utils import plot_pixelated_source_radial_profile

        pixels, x, y = self._uniform_pixelated_source_plane()
        return plot_pixelated_source_radial_profile(
            pixels, x, y, crop_radius=crop_radius, truth=truth,
            image_pixel_scale=self._require_model().data.pixel_scale,
            save_path=save_path, xscale=xscale, yscale=yscale,
        )

    def plot_pixelated_source_construction(
        self,
        crop_radius: float | None = None,
        truth: Mapping[str, Any] | str | Path | None = None,
        save_path: str | Path | None = None,
    ) -> Path:
        """Plot the circularly cropped median pixelated source and truth residual."""
        from .utils import plot_pixelated_source_reconstruction

        pixels, x, y = self._uniform_pixelated_source_plane()
        return plot_pixelated_source_reconstruction(
            pixels, x, y, crop_radius=crop_radius, truth=truth,
            image_pixel_scale=self._require_model().data.pixel_scale,
            save_path=save_path,
        )

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
        from ..utils import (
            fit_dof_and_reduced_chi2, json_serializer, kwargs_best_to_json_pixelated_npy,
            save_rtu_source_fits,
        )
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
        skipped: dict[str, str] = {}
        source_plane = self.derived.get("source_plane") if self.samples is not None else None
        if source_plane is not None and kwargs_best.get("kwargs_source"):
            kwargs_for_plots = deepcopy(kwargs_best)
            kwargs_for_plots["kwargs_source"][0]["pixels"] = source_plane
        kwargs_json = kwargs_best_to_json_pixelated_npy(kwargs_for_plots, str(directory), type_list)
        kwargs_result_path = directory / "kwargs_result.json"
        with kwargs_result_path.open("w") as stream:
            json.dump(kwargs_json, stream, indent=4, default=json_serializer)

        # Keep the conventional source-pixel filename, but make RTU outputs
        # self-contained: its physical, non-uniform source-plane cell corners
        # live beside the primary brightness array in the same FITS file.
        if getattr(model.lens_image, "_rtu_grid_source", False) and kwargs_for_plots.get("kwargs_source"):
            try:
                x_corners, y_corners = model.lens_image.get_rtu_source_plane_grid(
                    kwargs_for_plots.get("kwargs_lens"),
                )
                save_rtu_source_fits(
                    directory / "kwargs_source_pixels.fits",
                    kwargs_for_plots["kwargs_source"][0]["pixels"], x_corners, y_corners,
                    polynomial_order=getattr(model.lens_image, "_rtu_polynomial_order", None),
                )
            except Exception as error:
                skipped["rtu_source_fits"] = str(error)

        # HMC evaluates every physical source reconstruction before taking its
        # pixel-wise posterior summary.  Preserve the asymmetric intervals in
        # the same FITS file as the median source instead of writing an RMS
        # ``SIGMA`` extension.
        if self.samples is not None:
            source_lower = self.derived.get("source_plane_lower")
            source_upper = self.derived.get("source_plane_upper")
            if source_lower is not None and source_upper is not None:
                try:
                    from ..utils import append_array_fits
                    source_path = directory / "kwargs_source_pixels.fits"
                    append_array_fits(source_path, source_lower, extension_name="LOWER")
                    append_array_fits(source_path, source_upper, extension_name="UPPER")
                except Exception as error:
                    skipped["hmc_source_intervals"] = str(error)

        metrics = self.metrics()
        save_metrics(
            str(directory), metrics["chi2_median"], model.data.likelihood_image,
            model.num_sampling_parameters, metrics["log_likelihood_median"],
            fit_dof_and_reduced_chi2,
            num_params_free=metrics["n_free_parameters"],
            num_params_physical=metrics["n_physical_parameters"],
            mask_bool=model.data.likelihood_mask,
            metric_summary=metrics,
        )
        files: dict[str, Path] = {
            "metrics": directory / "metrics.json",
            "kwargs_result": kwargs_result_path,
        }

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
                sigma_source = (sigma_kwargs.get("kwargs_source") or [{}])[0]
                from ..utils import append_array_fits
                if sigma_source.get("pixels") is not None:
                    append_array_fits(directory / "kwargs_source_pixels.fits", sigma_source["pixels"])
                if sigma_source.get("pixels_wn") is not None:
                    append_array_fits(directory / "kwargs_source_pixels_wn.fits", sigma_source["pixels_wn"])
                sigma_json = kwargs_best_to_json_pixelated_npy(
                    sigma_kwargs, str(directory), type_list,
                    pixels_filename="kwargs_source_pixels.fits",
                    pixels_wn_filename="kwargs_source_pixels_wn.fits",
                    lens_light_pixels_prefix="kwargs_lens_light_pixels_sigma",
                    save_pixel_arrays=False,
                    references_already_saved=True,
                    pixels_hdu="SIGMA",
                    pixels_wn_hdu="SIGMA",
                )
                with (directory / "kwargs_sigma.json").open("w") as stream:
                    json.dump(sigma_json, stream, indent=4, default=json_serializer)
                files["kwargs_sigma"] = directory / "kwargs_sigma.json"
            except Exception as error:
                skipped["kwargs_sigma"] = str(error)
        elif self.samples is not None:
            try:
                sigma_json = hmc_one_sigma_kwargs(
                    model.prob_model, self.samples, self.parameters,
                    type_list, directory,
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
                best_fit_model=best_fit_model, chi2=metrics["chi2_median"],
                reduced_chi2=metrics["reduced_chi2_median"], extra=plot_details,
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

        from ..utils import save_named_arrays_fits
        save_named_arrays_fits(directory / "modeling_result.fits", {
            "best_fit_model": best_fit_model,
            "image_data": model.data.likelihood_image,
            "noise_map": model.data.likelihood_noise,
            "source_arc_mask": model.data.source_arc_mask,
            "contaminate_mask": model.data.contaminate_mask,
            "fit_mask_bool": model.data.likelihood_mask,
        })
        files["modeling_result"] = directory / "modeling_result.fits"
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
                # Pixelated profiles keep some optional hyperparameters as
                # ``None`` (for example a fixed/unused Matérn n value).  They
                # are valid entries in kwargs_result, but are not numerical
                # parameter shifts and cannot be formatted as a float.
                if value is None:
                    continue
                array = np.asarray(value)
                if parameter == "pixels" or array.ndim != 0:
                    continue
                try:
                    numeric_value = float(array)
                except (TypeError, ValueError):
                    continue
                lines.append(f"            {parameter}{suffix}: null -> {numeric_value:.3f}")
    (directory / "parameter_shifts.txt").write_text("\n".join(lines) + "\n")


@dataclass
class SingleBandResultsCombination:
    """Comparison helper for repeated API fits of the same single-band model."""

    results: list[FitResult]

    def __post_init__(self) -> None:
        self.results = list(self.results)
        if not self.results:
            raise ValueError("SingleBandResultsCombination requires at least one FitResult.")

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
            summary_metrics = {
                "BIC_PHYSICAL_MEDIAN": result_metrics["bic_physical_median"],
                "BIC_PHYSICAL_MAX_LOGLIKE": result_metrics["bic_physical_max_loglike"],
                "CHI2_MEDIAN": result_metrics["chi2_median"],
                "CHI2_MAX_LOGLIKE": result_metrics["chi2_max_loglike"],
                "CHI2_PER_DATA_PIXEL_MEDIAN": result_metrics["chi2_median"] / result_metrics["n_data_pixels"],
                "REDUCED_CHI2_MEDIAN": result_metrics["reduced_chi2_median"],
                "REDUCED_CHI2_MAX_LOGLIKE": result_metrics["reduced_chi2_max_loglike"],
                "CHI2_DOF": result_metrics["degrees_of_freedom"],
                "N_DATA_PIXELS": result_metrics["n_data_pixels"],
                "N_PARAMS_FITTED": model.num_sampling_parameters,
                "N_PARAMS_FREE": result_metrics["n_free_parameters"],
                "LOG_LIKELIHOOD_MEDIAN": result_metrics["log_likelihood_median"],
                "MAX_LOG_LIKELIHOOD": result_metrics["max_log_likelihood"],
            }
            metrics.append(summary_metrics)
            comparison[f"run_{index}"] = {
                "seed": result.random_seed,
                "metrics": summary_metrics,
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
                f"log-likelihood (median)={entry['LOG_LIKELIHOOD_MEDIAN']:.2f}, "
                f"chi2 (median)={entry['CHI2_MEDIAN']:.2f}, "
                f"chi2/N_pix (median)={entry['CHI2_PER_DATA_PIXEL_MEDIAN']:.4f}, "
                f"reduced_chi2 (median)={entry['REDUCED_CHI2_MEDIAN']:.4f}, "
                f"BIC_physical (median)={entry['BIC_PHYSICAL_MEDIAN']:.2f}"
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
