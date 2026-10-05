"""Physical-unit post-processing for fitted single-band lens models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LensGeometry:
    """Geometry needed to convert dimensionless lensing quantities to mass."""

    z_lens: float
    z_source: float
    cosmology: str | Any = "Planck18"

    def __post_init__(self) -> None:
        if self.z_lens <= 0 or self.z_source <= self.z_lens:
            raise ValueError("Require 0 < z_lens < z_source.")

    @property
    def astropy_cosmology(self):
        if not isinstance(self.cosmology, str):
            return self.cosmology
        from astropy.cosmology import Planck18, WMAP9

        choices = {"Planck18": Planck18, "WMAP9": WMAP9}
        try:
            return choices[self.cosmology]
        except KeyError as error:
            raise ValueError(f"Unknown cosmology {self.cosmology!r}; use Planck18, WMAP9, or an Astropy cosmology.") from error

    def distances_and_sigma_crit(self) -> dict[str, float]:
        """Return angular-diameter distances and critical density in standard units."""
        import astropy.units as u
        from astropy.constants import G, c

        cosmology = self.astropy_cosmology
        d_lens = cosmology.angular_diameter_distance(self.z_lens)
        d_source = cosmology.angular_diameter_distance(self.z_source)
        d_lens_source = cosmology.angular_diameter_distance_z1z2(self.z_lens, self.z_source)
        sigma_crit = (c**2 / (4 * math.pi * G) * d_source / (d_lens * d_lens_source)).to(u.Msun / u.kpc**2)
        return {
            "D_lens_mpc": float(d_lens.to_value(u.Mpc)),
            "D_source_mpc": float(d_source.to_value(u.Mpc)),
            "D_lens_source_mpc": float(d_lens_source.to_value(u.Mpc)),
            "sigma_crit_msun_per_kpc2": float(sigma_crit.value),
        }


def _integrated_kappa_area(mass_model, kwargs_lens, mass_types, *, radius_arcsec: float,
                           center_x: float, center_y: float, grid_size: int,
                           component_index: int | None = None) -> float:
    """Integrate κ dΩ with polar midpoint quadrature, avoiding singular centres."""
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive.")
    if grid_size < 64:
        raise ValueError("grid_size must be at least 64.")
    if component_index is None:
        # Each component gets its own cusp-adapted radial transform. A smooth
        # primary must not determine the quadrature for a singular secondary.
        return sum(_integrated_kappa_area(
            mass_model, kwargs_lens, mass_types, radius_arcsec=radius_arcsec,
            center_x=center_x, center_y=center_y, grid_size=grid_size,
            component_index=index,
        ) for index in range(len(mass_types)))
    if mass_types[component_index] in ("SHEAR", "SHEAR_GAMMA_PSI"):
        return 0.0
    gamma = None
    index = 0 if component_index is None else component_index
    if mass_types:
        if mass_types[index] == "EPL" or mass_types[index].startswith("EPL_MULTIPOLE_"):
            gamma = float(kwargs_lens[index].get("gamma", np.nan))
        elif mass_types[index] in ("SIS", "SIE"):
            gamma = 2.0
    if gamma is not None and (not np.isfinite(gamma) or not 1.0 < gamma < 3.0):
        raise ValueError("A singular power-law aperture requires 1 < gamma < 3.")
    profile = mass_types[index]
    component = kwargs_lens[index]
    if profile in ("MASS_MGE", "STELLAR_MGE"):
        prefix = "light_" if profile == "STELLAR_MGE" else ""
        gaussian_centers = [np.asarray(component.get(prefix + key, 0.0), dtype=float)
                            for key in ("center_x", "center_y")]
        if (np.all(gaussian_centers[0] == center_x)
                and np.all(gaussian_centers[1] == center_y)):
            sigma = np.asarray(component[prefix + "sigma"], dtype=float)
            amp = np.asarray(component[prefix + "amp"], dtype=float)
            if profile == "STELLAR_MGE":
                reference_sigma = np.exp(np.mean(np.log(sigma)))
                amp = (float(component["upsilon_kappa"]) * amp / np.sum(amp)
                       * (sigma / reference_sigma)**(-float(component["ml_gradient"])))
            amp, sigma, e1, e2 = np.broadcast_arrays(
                amp, sigma, np.asarray(component[prefix + "e1"], dtype=float),
                np.asarray(component[prefix + "e2"], dtype=float),
            )
            ellipticity = np.hypot(e1, e2)
            if np.any(sigma <= 0) or np.any(ellipticity >= 1):
                raise ValueError("Gaussian mass requires sigma > 0 and |e| < 1.")
            q = (1.0 - ellipticity) / (1.0 + ellipticity)
            phi = 0.5 * np.arctan2(e2, e1)
            angle = 2.0 * math.pi * np.arange(grid_size) / grid_size
            delta = angle[None, :] - phi.reshape(-1, 1)
            f = (q.reshape(-1, 1) * np.cos(delta)**2
                 + np.sin(delta)**2 / q.reshape(-1, 1))
            # kappa_i = amp_i/(2*pi*sigma_i^2) * exp(-r^2*f/(2*sigma_i^2)).
            # Integrating r*dr exactly resolves arbitrarily narrow Gaussians;
            # only the smooth angular integral needs numerical quadrature.
            fraction = -np.expm1(-radius_arcsec**2 * f / (2 * sigma.reshape(-1, 1)**2)) / f
            area = float(np.sum(amp.reshape(-1, 1) * fraction) / grid_size)
            if not np.isfinite(area):
                raise ValueError("The Gaussian model returned non-finite aperture mass.")
            return area
    if (profile == "EPL" or profile.startswith("EPL_MULTIPOLE_")) and (
        float(component.get("center_x", 0.0)) == center_x
        and float(component.get("center_y", 0.0)) == center_y
    ):
        # Centred EPL convergence is homogeneous: kappa(r, phi) is
        # proportional to r**(1-gamma). Integrate radius analytically to
        # avoid sub-machine-resolution rings when gamma approaches three.
        angle = 2.0 * math.pi * np.arange(grid_size) / grid_size
        kappa = np.asarray(mass_model.kappa(
            center_x + radius_arcsec * np.cos(angle),
            center_y + radius_arcsec * np.sin(angle), kwargs_lens, k=index,
        ))
        if not np.all(np.isfinite(kappa)):
            raise ValueError("The power-law model returned non-finite convergence.")
        return float(np.sum(kappa) * (2.0 * math.pi / grid_size) * radius_arcsec**2 / (3.0 - gamma))
    exponent = 3.0 - gamma if gamma is not None and 0.0 < 3.0 - gamma < 2.0 else 2.0
    coordinate = (np.arange(grid_size) + 0.5) / grid_size
    radius = radius_arcsec * coordinate ** (1.0 / exponent)
    radial_weights = radius_arcsec**2 / exponent * coordinate ** (2.0 / exponent - 1.0) / grid_size
    angle = 2.0 * math.pi * np.arange(grid_size) / grid_size
    x = center_x + radius[:, None] * np.cos(angle)[None, :]
    y = center_y + radius[:, None] * np.sin(angle)[None, :]
    kappa = np.asarray(mass_model.kappa(x, y, kwargs_lens, k=component_index))
    if not np.all(np.isfinite(kappa)):
        raise ValueError("The mass model returned non-finite convergence inside this aperture.")
    return float(np.sum(kappa * radial_weights[:, None]) * (2.0 * math.pi / grid_size))


def enclosed_lensing_mass(
    model: Any,
    parameters: Mapping[str, Any],
    geometry: LensGeometry,
    *,
    radius_arcsec: float | None = None,
    center: tuple[float, float] | None = None,
    grid_size: int = 400,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate projected aperture masses from a fitted lens model.

    This is a finite 2-D lensing mass, not an extrapolated halo mass.  The
    returned component decomposition is meaningful for explicit mass profiles
    such as ``STELLAR_MGE``, ``GNFW_MGE``, and ``NFW_ELLIPSE_KAPPA``.
    """
    if model.prob_model is None or model.lens_image is None:
        raise RuntimeError("The backend model is unavailable.")
    type_list, _ = model.definition.as_dicts()
    mass_types = type_list["lens_mass_type_list"]
    kwargs_lens = model.prob_model.params2kwargs(parameters)["kwargs_lens"]
    return enclosed_mass_from_kwargs(
        model.lens_image.MassModel, kwargs_lens, mass_types, geometry,
        radius_arcsec=radius_arcsec, center=center, grid_size=grid_size, save_path=save_path,
    )


def enclosed_mass_from_kwargs(
    mass_model: Any, kwargs_lens: list[dict[str, Any]], mass_types: list[str],
    geometry: LensGeometry, *, radius_arcsec: float | None = None,
    center: tuple[float, float] | None = None, grid_size: int = 400,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Projected aperture mass from native kwargs, without rebuilding image data.

    Suitable for archived posterior samples. Uses the same calculation as
    ``FitResult.enclosed_mass``; all profiles contribute to the aperture.
    """
    import astropy.units as u

    if not mass_types:
        raise ValueError("The model has no lens-mass profiles.")
    if len(mass_types) != len(kwargs_lens):
        raise ValueError("mass_types and kwargs_lens must have matching lengths.")
    primary = kwargs_lens[0]
    if radius_arcsec is None:
        if "theta_E" not in primary:
            raise ValueError("radius_arcsec is required when mass component 0 has no theta_E.")
        radius_arcsec = float(primary["theta_E"])
    if center is None:
        center = (float(primary.get("center_x", 0.0)), float(primary.get("center_y", 0.0)))
    center_x, center_y = map(float, center)

    distances = geometry.distances_and_sigma_crit()
    d_lens = geometry.astropy_cosmology.angular_diameter_distance(geometry.z_lens)
    radius_kpc = (float(radius_arcsec) * u.arcsec * d_lens).to_value(u.kpc, u.dimensionless_angles())
    area_to_mass = distances["sigma_crit_msun_per_kpc2"] * (u.arcsec.to(u.rad) * d_lens.to_value(u.kpc)) ** 2

    total_area = _integrated_kappa_area(
        mass_model, kwargs_lens, mass_types, radius_arcsec=float(radius_arcsec),
        center_x=center_x, center_y=center_y, grid_size=grid_size,
    )
    labels = {
        "STELLAR_MGE": "stellar",
        "GNFW_MGE": "dark_matter",
        "NFW_ELLIPSE_KAPPA": "dark_matter",
    }
    components: dict[str, dict[str, float | str]] = {}
    for index, profile_type in enumerate(mass_types):
        area = _integrated_kappa_area(
            mass_model, kwargs_lens, mass_types, radius_arcsec=float(radius_arcsec),
            center_x=center_x, center_y=center_y, grid_size=grid_size, component_index=index,
        )
        label = labels.get(profile_type, f"{profile_type.lower()}_{index}")
        if label in components:
            label = f"{label}_{index}"
        components[label] = {
            "profile_type": profile_type,
            "integrated_kappa_area_arcsec2": area,
            "projected_enclosed_mass_msun": area * area_to_mass,
        }
    result: dict[str, Any] = {
        "quantity": "projected_enclosed_lensing_mass_M_2D",
        "aperture": {
            "shape": "circle", "radius_arcsec": float(radius_arcsec),
            "radius_kpc": float(radius_kpc), "center_arcsec": {"x": center_x, "y": center_y},
        },
        "geometry": {
            "z_lens": geometry.z_lens, "z_source": geometry.z_source,
            "cosmology": geometry.cosmology if isinstance(geometry.cosmology, str) else geometry.astropy_cosmology.name,
            **distances,
        },
        "total": {
            "integrated_kappa_area_arcsec2": total_area,
            "mean_kappa_inside_aperture": total_area / (math.pi * float(radius_arcsec) ** 2),
            "projected_enclosed_mass_msun": total_area * area_to_mass,
        },
        "components": components,
        "caveat": "This is a projected finite-aperture lensing mass, not a 3-D or halo-total mass.",
    }
    if save_path is not None:
        output = Path(save_path).expanduser()
        if output.is_dir() or not output.suffix:
            output = output / "enclosed_mass.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        result["saved_to"] = str(output)
    return result
