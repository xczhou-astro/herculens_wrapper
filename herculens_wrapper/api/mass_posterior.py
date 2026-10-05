"""Lens shape and projected-mass summaries directly from archived HMC draws."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .physics import LensGeometry, _integrated_kappa_area


def posterior_interval(values, *, probability=0.682689492137, period=None):
    """Median and equal-tail interval; unwrap axial angles before quantiles.

    Vector parameters retain their trailing shape. Angle bounds are on the
    median's local branch and may cross 0/180, rather than implying a huge error.
    """
    values = np.asarray(values, dtype=float)
    if not 0 < probability < 1 or values.shape[0] == 0:
        raise ValueError("Require nonempty draws and 0 < probability < 1.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Posterior contains non-finite values.")
    concentration = None
    if period is not None:
        phase = values * (2 * np.pi / period)
        mean = np.mean(np.exp(1j * phase), axis=0)
        concentration = np.abs(mean)
        origin = np.angle(mean) * period / (2 * np.pi)
        values = origin + (values - origin + period / 2) % period - period / 2
    tail = (1 - probability) / 2
    lower, median, upper = np.quantile(values, [tail, .5, 1 - tail], axis=0)
    error_minus, error_plus = median - lower, upper - median
    if period is not None:
        canonical = median % period
        lower, upper, median = canonical - error_minus, canonical + error_plus, canonical
    result = dict(median=median, lower=lower, upper=upper,
                  error_minus=error_minus, error_plus=error_plus, probability=probability)
    if concentration is not None:
        result.update(period=period, circular_concentration=concentration)
    return {key: np.asarray(value).tolist() for key, value in result.items()}


def _configuration(run, config_path, band):
    candidates = [Path(config_path).expanduser()] if config_path else [
        parent / name for parent in (run, run.parent)
        for name in ("model_configuration.json", "config.json")
    ]
    for path in candidates:
        if not path.is_file():
            continue
        config = json.loads(path.read_text())
        if "backend_definitions" in config:
            definitions = config["backend_definitions"]
            if band is None:
                if len(definitions) != 1:
                    raise ValueError(f"Joint result: select --band from {list(definitions)}.")
                band = next(iter(definitions))
            definition = definitions[band]
        else:
            definition = config.get("backend_definition", {})
        types = definition.get("types", config.get("type_list", {}))
        params = definition.get("parameters", config.get("parameter_list", config.get("param_list", {})))
        mass_types = types.get("lens_mass_type_list", config.get("shared_lens_mass_type_list"))
        if mass_types:
            return list(mass_types), params, band, path
    raise FileNotFoundError("No saved mass-profile configuration found; supply --config.")


def _read_draws(archive, *, band, max_samples, seed):
    """Read only mass/analytic-light sites; never load pixelated source draws."""
    import h5py

    with h5py.File(archive, "r") as handle:
        group = handle.get("samples")
        if group is None or not group:
            raise ValueError("The HMC archive has no posterior samples group.")
        n_draws = next(iter(group.values())).shape[0]
        if n_draws == 0:
            raise ValueError("The HMC archive has zero draws.")
        if max_samples < 0:
            raise ValueError("max_samples must be nonnegative (0 means all).")
        indices = np.arange(n_draws)
        if 0 < max_samples < n_draws:
            indices = np.sort(np.random.default_rng(seed).choice(n_draws, max_samples, replace=False))
        datasets = {}
        for encoded, dataset in group.items():
            name = unquote(encoded)
            local = name.rsplit("/", 1)[-1]
            if not local.startswith("lens_") or "pixels" in local:
                continue
            if dataset.shape[0] != n_draws:
                raise ValueError(f"Inconsistent draw count for {name}.")
            if "/" in name and (band is None or not name.startswith(band + "/")):
                continue
            # Per-band sites override shared sites, especially mass centres.
            priority = int("/" in name)
            if local not in datasets or priority > datasets[local][0]:
                datasets[local] = priority, dataset
        draws = {name: np.asarray(dataset[indices]) for name, (_, dataset) in datasets.items()}
        n_chains = int(handle.attrs.get("num_chains", 1))
        divergent = None
        health = handle.get("sampler_health")
        if health is not None and "diverging" in health:
            divergent = int(np.count_nonzero(health["diverging"]))
    return draws, indices, n_draws, n_chains, divergent


def _kwargs_draw(saved, definition, mass_types, draws, row):
    """Restore sampled sites, correlated parameters and dynamic stellar MGEs."""
    from ..models import (
        _linkable_mass_parameters, _materialize_dynamic_stellar_mge,
        _materialize_mass_parameters, _normalize_link_spec, _resolve_link,
    )
    from ..priors import is_distribution_prior

    bank, pending = {"lens": [], "lens_light": []}, set()
    lists = {"lens": "lens_mass_params_list", "lens_light": "lens_light_params_list"}
    snapshots = {"lens": saved["kwargs_lens"], "lens_light": saved.get("kwargs_lens_light", [])}

    def build(component, index):
        array = bank[component]
        while len(array) <= index:
            array.append(None)
        if array[index] is not None:
            return array[index]
        token = (component, index)
        if token in pending:
            raise ValueError(f"Circular parameter link involving {token}.")
        pending.add(token)
        snapshot = snapshots[component][index]
        declarations = definition.get(lists[component], [])
        declaration = declarations[index] if index < len(declarations) else {}
        values = dict(snapshot)
        # Use API declarations to materialize angle units exactly once.
        for key, spec in declaration.items():
            if key.startswith("_"):
                continue
            site = f"{component}_{key}_{index}"
            link = _normalize_link_spec(spec)
            if link is not None:
                target, target_index, _ = link
                if target not in bank:
                    raise ValueError(f"Mass post-processing cannot resolve link to {target}.")
                if target_index == "flux_centroid":
                    for light_index in range(len(snapshots["lens_light"])):
                        build("lens_light", light_index)
                else:
                    build(target, target_index)
                values[key] = _resolve_link(bank, link, context="archived HMC mass")
            elif site in draws:
                values[key] = draws[site][row]
            elif is_distribution_prior(spec):
                raise ValueError(f"Missing sampled site {site}; cannot propagate its uncertainty.")
            elif isinstance(spec, (list, tuple)) and len(spec) in (2, 4) and (
                key not in snapshot or np.ndim(snapshot[key]) == 0
            ):
                raise ValueError(f"Missing sampled site {site}; cannot propagate its uncertainty.")
            elif not isinstance(spec, (list, tuple, dict)):
                values[key] = spec
            elif key not in snapshot:
                # q/phi aliases are absent in native saved kwargs.
                if key in ("q", "phi") and {"e1", "e2"} <= snapshot.keys():
                    e = np.hypot(snapshot["e1"], snapshot["e2"])
                    values[key] = (1 - e) / (1 + e) if key == "q" else np.rad2deg(.5 * np.arctan2(snapshot["e2"], snapshot["e1"]))
                else:
                    raise ValueError(f"Missing parameter {site} in posterior and saved kwargs.")
        # Older configs may have no declarations: update native sampled kwargs.
        for key in snapshot:
            site = f"{component}_{key}_{index}"
            if site in draws:
                values[key] = draws[site][row]
        if component == "lens":
            # Native snapshot e1/e2 must not shadow newly restored q/phi.
            if {"q", "phi"} <= values.keys():
                values.pop("e1", None); values.pop("e2", None)
            # Saved multipole angles are radians; only declarations/draws are API degrees.
            angle_keys = ("phi_m", "varphi_m", "phi_ref", "delta_varphi_m",
                          "delta_phi_m1", "delta_phi_m3", "delta_phi_m4")
            for key in angle_keys:
                if key in values and key not in declaration and f"lens_{key}_{index}" not in draws:
                    values[key] = np.rad2deg(values[key])
            native = _materialize_mass_parameters(mass_types[index], values)
            if "_stellar_lens_light_indices" in declaration:
                for light_index in declaration["_stellar_lens_light_indices"]:
                    build("lens_light", light_index)
                native = _materialize_dynamic_stellar_mge(native, declaration, bank["lens_light"])
            array[index] = _linkable_mass_parameters(mass_types[index], native)
            # Link aliases stay in the bank but never reach the profile backend.
            pending.remove(token)
            return native
        array[index] = values
        pending.remove(token)
        return values

    result = []
    for index in range(len(mass_types)):
        native = build("lens", index)
        # A component pre-built by a link returns the alias-enriched link bank.
        if mass_types[index] in ("EPL", "SIE") or mass_types[index].startswith("EPL_MULTIPOLE_"):
            native = {key: value for key, value in native.items() if key not in ("q", "phi")}
        result.append(native)
    return result


def _shape(profile, values):
    if "e1" in values:
        e1, e2 = np.asarray(values["e1"]), np.asarray(values["e2"])
    elif profile == "STELLAR_MGE":
        e1, e2 = np.asarray(values["light_e1"]), np.asarray(values["light_e2"])
    elif profile == "INCLINED_EXPONENTIAL_DISK":
        q = np.sqrt(np.cos(values["inclination"])**2 + values["q0"]**2 * np.sin(values["inclination"])**2)
        return q, np.rad2deg(values["phi"]), True
    elif profile in ("SIS", "NFW"):
        return np.asarray(1.), np.asarray(0.), False
    else:
        raise ValueError(f"Profile {profile} has no supported single ellipticity; select another component.")
    e = np.hypot(e1, e2)
    if np.any(e >= 1):
        raise ValueError("Mass ellipticity must have magnitude < 1.")
    return (1 - e) / (1 + e), np.rad2deg(.5 * np.arctan2(e2, e1)), bool(np.all(e > 1e-10))


def summarize_hmc_lens(
    run_directory, *, geometry=None, component_index=0, config_path=None, band=None,
    probability=0.682689492137, max_samples=2000, seed=42, grid_size=128,
    einstein_definition="auto", radius_bounds=(1e-3, 100.),
    pa_offset_deg=0., pa_sign=1, progress=None,
):
    """Summarize HMC lens shape and mass inside theta_E with joint-sample errors.

    ``auto`` uses the selected profile's fitted theta_E when available, else
    solves mean total convergence inside a circle = 1. These are distinct
    definitions for non-circular lenses; neither is a critical-curve area radius.
    q and PA refer to the selected component; vector MGEs retain one result
    per Gaussian. PA is counterclockwise from model +x, modulo 180 degrees.
    Optional PA sign/offset require a user-supplied sky-coordinate convention.
    Mass is the projected sum of all mass components inside each draw's
    theta_E circle. Redshifts and cosmology are held fixed.
    """
    import astropy.units as u
    import jax
    from scipy.optimize import brentq
    from ..profiles.registry import register_mass_profiles
    from herculens.MassModel.mass_model import MassModel

    jax.config.update("jax_enable_x64", True)
    if grid_size < 64 or not 0 < probability < 1:
        raise ValueError("Require grid_size >= 64 and 0 < probability < 1.")
    if einstein_definition not in ("auto", "parameter", "mean-kappa"):
        raise ValueError("Unknown Einstein radius definition.")
    if pa_sign not in (-1, 1):
        raise ValueError("pa_sign must be +1 or -1.")
    if not 0 < radius_bounds[0] < radius_bounds[1]:
        raise ValueError("Require 0 < lower < upper radius bounds.")
    geometry = geometry or LensGeometry(1.53, 3.417)
    run = Path(run_directory).expanduser().resolve()
    if run.is_file():
        if run.name != "hmc_samples.h5":
            raise ValueError("Supply an HMC directory or hmc_samples.h5.")
        run = run.parent
    if not (run / "hmc_samples.h5").is_file():
        raise FileNotFoundError(f"HMC archive is missing: {run / 'hmc_samples.h5'}")
    mass_types, definition, band, config_file = _configuration(run, config_path, band)
    if not 0 <= component_index < len(mass_types):
        raise ValueError("component_index is outside the mass-profile list.")
    saved = json.loads((run / "kwargs_result.json").read_text())
    if band and "kwargs_by_band" in saved:
        saved = saved["kwargs_by_band"][band]
    if len(saved["kwargs_lens"]) != len(mass_types):
        raise ValueError("Saved kwargs_lens and configured mass profiles do not match.")
    draws, indices, n_draws, n_chains, divergent = _read_draws(
        run / "hmc_samples.h5", band=band, max_samples=max_samples, seed=seed,
    )
    register_mass_profiles()
    mass_model = MassModel(mass_types)
    distances = geometry.distances_and_sigma_crit()
    kpc_per_arcsec = u.arcsec.to(u.rad) * distances["D_lens_mpc"] * 1000
    area_to_mass = distances["sigma_crit_msun_per_kpc2"] * kpc_per_arcsec**2
    collected = {key: [] for key in ("theta_E_arcsec", "q", "PA_deg", "mass_within_theta_E_msun")}
    pa_defined = True
    used_definition = None
    for row, archive_index in enumerate(indices):
        try:
            kwargs = _kwargs_draw(saved, definition, mass_types, draws, row)
            primary = kwargs[component_index]
            # Vector centres must coincide; do not silently use one Gaussian.
            centers = []
            for key in ("center_x", "center_y"):
                values = np.asarray(primary.get(key, primary.get("light_" + key, 0.)), dtype=float)
                if not np.allclose(values, values.flat[0], atol=1e-10, rtol=0):
                    raise ValueError("Selected MGE has different centres; a common aperture centre is undefined.")
                centers.append(float(values.flat[0]))
            center_x, center_y = centers

            def area(radius, index=None):
                return _integrated_kappa_area(
                    mass_model, kwargs, mass_types, radius_arcsec=radius,
                    center_x=center_x, center_y=center_y, grid_size=grid_size, component_index=index,
                )

            use_parameter = einstein_definition == "parameter" or (einstein_definition == "auto" and "theta_E" in primary)
            used_definition = "profile_theta_E_parameter" if use_parameter else "circular_mean_total_kappa_equals_one"
            if use_parameter:
                if "theta_E" not in primary:
                    raise ValueError("Selected profile has no theta_E parameter; use --einstein-definition mean-kappa.")
                theta = float(primary["theta_E"])
            else:
                theta = brentq(lambda radius: area(radius) / (np.pi * radius**2) - 1,
                               *radius_bounds, xtol=1e-8)
            q, phi, defined = _shape(mass_types[component_index], primary)
            pa_defined &= defined
            enclosed_area = area(theta)
            collected["theta_E_arcsec"].append(theta)
            collected["q"].append(q)
            collected["PA_deg"].append(pa_sign * phi + pa_offset_deg)
            collected["mass_within_theta_E_msun"].append(enclosed_area * area_to_mass)
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(f"Cannot evaluate archived draw {archive_index}: {error}") from error
        if progress and (row == 0 or (row + 1) % 50 == 0 or row + 1 == len(indices)):
            progress(row + 1, len(indices))

    summaries = {key: posterior_interval(value, probability=probability, period=180 if key == "PA_deg" else None)
                 for key, value in collected.items() if key != "PA_deg" or pa_defined}
    if not pa_defined:
        summaries["PA_deg"] = {"status": "undefined", "reason": "At least one draw is circular; a circular profile has no major-axis PA."}
    summaries["theta_E_kpc"] = posterior_interval(np.asarray(collected["theta_E_arcsec"]) * kpc_per_arcsec, probability=probability)
    result = {
        "run_directory": str(run), "configuration_file": str(config_file), "band": band,
        "mass_profiles": mass_types, "shape_component_index": component_index,
        "geometry": {"z_lens": geometry.z_lens, "z_source": geometry.z_source,
                     "cosmology": geometry.astropy_cosmology.name, **distances},
        "posterior": {"available_draws": n_draws, "used_draws": len(indices), "num_chains": n_chains,
                      "divergent_draws": divergent, "seed": seed, "archive_draw_indices": indices.tolist(),
                      "loaded_sites": sorted(draws)},
        "definitions": {"Einstein_radius": used_definition,
                        "mass_within_theta_E": "Sum of all mass components inside a circle, following each draw's radius and centre; projected M_2D.",
                        "q": "Minor/major axis ratio of selected profile; arrays are per Gaussian, not a global composite axis ratio.",
                        "PA_deg": f"({pa_sign} * phi_model + {pa_offset_deg}) modulo 180; phi_model is CCW from model +x. Sky PA requires a supplied WCS transformation.",
                        "errors": "Equal-tail posterior interval from joint draws; redshifts and cosmology held fixed."},
        "summaries": summaries, "grid_size": grid_size,
    }
    return result
