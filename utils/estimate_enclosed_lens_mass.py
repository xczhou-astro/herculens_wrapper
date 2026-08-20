#!/usr/bin/env python3
"""Estimate projected lensing mass inside the fitted Einstein-radius aperture.

The script reconstructs the configured Herculens mass model from a run's
``kwargs_result.json`` and integrates its convergence, kappa, inside a circle.
By default the circle is centred on the first mass component and has radius
equal to that component's fitted ``theta_E``.

This is an aperture mass, M_2D(<R_E), rather than the total mass of the lens
halo.  A finite total mass would require a separately chosen outer aperture or
a finite-mass halo model.

Example
-------
python utils/estimate_enclosed_lens_mass.py results
python utils/estimate_enclosed_lens_mass.py /path/to/run_0 --zl 1.53 --zs 3.417
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18, WMAP9
import matplotlib.pyplot as plt


COSMOLOGIES = {
    'Planck18': Planck18,
    'WMAP9': WMAP9,
}


def _load_json(path):
    with path.open() as handle:
        return json.load(handle)


def _mass_types(config):
    """Extract the shared/single-band mass profile list from saved config."""
    type_list = config.get('type_list', {})
    types = (
        type_list.get('lens_mass_type_list')
        or config.get('shared_lens_mass_type_list')
    )
    if not types:
        raise ValueError('Could not find lens_mass_type_list in config.json.')
    return list(types)


def _kwargs_lens(result):
    """Extract kwargs_lens from either a single-band or joint result file."""
    kwargs = result.get('kwargs_lens')
    if not kwargs:
        raise ValueError('Could not find kwargs_lens in kwargs_result.json.')
    return kwargs


def _scalar_uncertainty(value):
    """Return (lower, upper) errors, accepting symmetric or HMC asymmetric data."""
    if isinstance(value, (int, float)):
        error = abs(float(value))
        return error, error
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return abs(float(value[0])), abs(float(value[1]))
    return None


def _make_kappa_integrator(mass_types, grid_size):
    """Create a numerical disk integrator in angular (arcsec) coordinates."""
    if grid_size < 64:
        raise ValueError('--grid-size must be at least 64.')

    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        from herculens.MassModel.mass_model import MassModel
    except ImportError as error:
        raise ImportError(
            'Herculens is required to evaluate the configured mass model. '
            'Run this utility in the same environment used for modeling.'
        ) from error

    mass_model = MassModel(mass_types)

    def enclosed_kappa_area(kwargs_lens, radius_arcsec, center_x, center_y):
        if radius_arcsec <= 0:
            raise ValueError('The aperture radius must be positive.')

        # Polar midpoint quadrature never samples an EPL's singular centre. For a
        # primary EPL, the radial transform resolves its kappa ~ R**(1-gamma)
        # behaviour near the origin instead of undersampling the central mass.
        radial_coordinate = (np.arange(grid_size) + 0.5) / grid_size
        gamma = None
        if mass_types[0] == 'EPL':
            gamma = float(kwargs_lens[0].get('gamma', np.nan))
        exponent = 3.0 - gamma if gamma is not None and 0.0 < 3.0 - gamma < 2.0 else 2.0
        radial = radius_arcsec * radial_coordinate**(1.0 / exponent)
        radial_weights = (
            radius_arcsec**2 / exponent
            * radial_coordinate**(2.0 / exponent - 1.0)
            / grid_size
        )
        phi = 2.0 * math.pi * np.arange(grid_size) / grid_size
        evaluation_radial = radial.copy()
        resolution_floor = 16.0 * np.finfo(float).eps * max(
            1.0, abs(center_x), abs(center_y), radius_arcsec
        )
        unresolved = radial < resolution_floor
        if np.any(unresolved):
            compatible_profiles = (
                mass_types.count('EPL') == 1
                and all(profile in ('EPL', 'SHEAR') for profile in mass_types)
                and gamma is not None
            )
            if not compatible_profiles:
                raise ValueError(
                    'The innermost aperture ring is below floating-point resolution. '
                    'Use a smaller --grid-size for this mass-profile combination.'
                )
            # EPL convergence is homogeneous: kappa(lambda R) =
            # lambda**(1-gamma) kappa(R). Evaluate away from the singular point
            # and rescale the affected ring after evaluation.
            evaluation_radial[unresolved] = resolution_floor
        x_offset = evaluation_radial[:, None] * np.cos(phi)[None, :]
        y_offset = evaluation_radial[:, None] * np.sin(phi)[None, :]
        kappa = np.array(
            mass_model.kappa(center_x + x_offset, center_y + y_offset, kwargs_lens),
            copy=True,
        )
        if np.any(unresolved):
            kappa[unresolved] *= (
                radial[unresolved] / evaluation_radial[unresolved]
            )[:, None]**(1.0 - gamma)
        if not np.all(np.isfinite(kappa)):
            raise ValueError('The mass model returned non-finite kappa values in the aperture.')
        return float(np.sum(kappa * radial_weights[:, None]) * (2.0 * math.pi / grid_size))

    return enclosed_kappa_area


def _critical_surface_density(z_lens, z_source, cosmology):
    if z_lens <= 0 or z_source <= 0:
        raise ValueError('Redshifts must be positive.')
    if z_source <= z_lens:
        raise ValueError('The source redshift must be greater than the lens redshift.')

    d_l = cosmology.angular_diameter_distance(z_lens)
    d_s = cosmology.angular_diameter_distance(z_source)
    d_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)
    sigma_crit = (c**2 / (4.0 * math.pi * G) * d_s / (d_l * d_ls)).to(
        u.Msun / u.kpc**2
    )
    return d_l, d_s, d_ls, sigma_crit


def _mass_from_kappa_area(kappa_area_arcsec2, d_lens, sigma_crit):
    angular_area = kappa_area_arcsec2 * u.arcsec**2
    physical_area = (angular_area * d_lens**2).to(u.kpc**2, u.dimensionless_angles())
    return (sigma_crit * physical_area).to(u.Msun)


def _estimate_independent_uncertainty(
    kwargs_lens,
    kwargs_sigma_lens,
    integrator,
    radius_arcsec,
    center_x,
    center_y,
    d_lens,
    sigma_crit,
):
    """Finite-difference uncertainty estimate without parameter covariances."""
    if not kwargs_sigma_lens:
        return None

    lower_squared = 0.0
    upper_squared = 0.0
    contributions = []
    nominal_kappa_area = integrator(kwargs_lens, radius_arcsec, center_x, center_y)
    nominal_mass = _mass_from_kappa_area(nominal_kappa_area, d_lens, sigma_crit).value

    for component_index, component_sigma in enumerate(kwargs_sigma_lens):
        if component_index >= len(kwargs_lens) or not isinstance(component_sigma, dict):
            continue
        for name, sigma_value in component_sigma.items():
            errors = _scalar_uncertainty(sigma_value)
            if errors is None or name not in kwargs_lens[component_index]:
                continue
            base_value = kwargs_lens[component_index][name]
            if not isinstance(base_value, (int, float)):
                continue
            lower_error, upper_error = errors
            if lower_error == 0.0 and upper_error == 0.0:
                continue

            def evaluate(offset):
                trial = copy.deepcopy(kwargs_lens)
                trial[component_index][name] = float(base_value) + offset
                trial_radius = radius_arcsec
                trial_center_x = center_x
                trial_center_y = center_y
                if component_index == 0 and name == 'theta_E':
                    trial_radius = max(float(trial[0]['theta_E']), 1e-8)
                if component_index == 0 and name == 'center_x':
                    trial_center_x = float(trial[0]['center_x'])
                if component_index == 0 and name == 'center_y':
                    trial_center_y = float(trial[0]['center_y'])
                kappa_area = integrator(trial, trial_radius, trial_center_x, trial_center_y)
                return _mass_from_kappa_area(kappa_area, d_lens, sigma_crit).value

            try:
                mass_minus = evaluate(-lower_error)
                mass_plus = evaluate(upper_error)
            except (TypeError, ValueError) as error:
                print(f'[uncertainty] Skipping {name}_{component_index}: {error}')
                continue

            lower_contribution = abs(nominal_mass - mass_minus)
            upper_contribution = abs(mass_plus - nominal_mass)
            lower_squared += lower_contribution**2
            upper_squared += upper_contribution**2
            contributions.append({
                'component_index': component_index,
                'parameter': name,
                'mass_error_lower_msun': lower_contribution,
                'mass_error_upper_msun': upper_contribution,
            })

    return {
        'method': 'one_at_a_time_finite_difference_independent_quadrature',
        'mass_error_lower_msun': math.sqrt(lower_squared),
        'mass_error_upper_msun': math.sqrt(upper_squared),
        'parameter_contributions_msun': contributions,
        'caveat': (
            'This approximation ignores posterior correlations. Use posterior samples '
            '(for example hmc_samples.npz) for a correlated credible interval.'
        ),
    }


def _estimate_critical_circle_uncertainty(
    radius_arcsec, theta_e_uncertainty, d_lens, sigma_crit
):
    """Propagate theta_E errors for M_E = pi * theta_E**2 * Sigma_crit."""
    if theta_e_uncertainty is None:
        return None
    lower_error, upper_error = theta_e_uncertainty
    nominal = _mass_from_kappa_area(math.pi * radius_arcsec**2, d_lens, sigma_crit).value
    lower_radius = max(radius_arcsec - lower_error, 0.0)
    lower_mass = _mass_from_kappa_area(
        math.pi * lower_radius**2, d_lens, sigma_crit
    ).value
    upper_mass = _mass_from_kappa_area(
        math.pi * (radius_arcsec + upper_error)**2, d_lens, sigma_crit
    ).value
    return {
        'method': 'analytic_theta_E_propagation_for_critical_circle',
        'mass_error_lower_msun': nominal - lower_mass,
        'mass_error_upper_msun': upper_mass - nominal,
        'parameter_contributions_msun': [{
            'component_index': 0,
            'parameter': 'theta_E',
            'mass_error_lower_msun': nominal - lower_mass,
            'mass_error_upper_msun': upper_mass - nominal,
        }],
        'caveat': (
            'This uncertainty includes theta_E only. It is appropriate for the '
            'standard critical-circle Einstein mass and does not include mass-model '
            'shape or cosmological-redshift uncertainty.'
        ),
    }


def _total_mass_to_infinity_assessment(mass_types, kwargs_lens):
    """State whether the configured singular profiles have a finite all-space mass.

    An EPL has kappa proportional to R**(1-gamma).  Its projected mass at
    large radius is therefore proportional to R**(3-gamma).  For gamma <= 3,
    it diverges at infinity.  For gamma > 3, the singular centre diverges.
    Thus a singular EPL never has a finite all-space projected mass.
    """
    assessments = []
    for index, profile_type in enumerate(mass_types):
        kwargs = kwargs_lens[index]
        if profile_type == 'EPL':
            gamma = float(kwargs.get('gamma', np.nan))
            if not np.isfinite(gamma):
                assessments.append({
                    'component_index': index,
                    'profile_type': profile_type,
                    'status': 'undetermined',
                    'reason': 'Missing finite gamma value.',
                })
            elif gamma < 3.0:
                assessments.append({
                    'component_index': index,
                    'profile_type': profile_type,
                    'gamma': gamma,
                    'status': 'diverges_at_infinity',
                    'asymptotic_enclosed_mass_scaling': f'R^{3.0 - gamma:.6f}',
                    'reason': 'For an EPL, M_2D(<R) is proportional to R^(3-gamma).',
                })
            elif gamma == 3.0:
                assessments.append({
                    'component_index': index,
                    'profile_type': profile_type,
                    'gamma': gamma,
                    'status': 'diverges_logarithmically_at_infinity_and_center',
                    'reason': 'The singular EPL has logarithmic divergence at both limits.',
                })
            else:
                assessments.append({
                    'component_index': index,
                    'profile_type': profile_type,
                    'gamma': gamma,
                    'status': 'diverges_at_singular_center',
                    'reason': 'The outer integral converges, but the singular central integral does not.',
                })
        elif profile_type == 'SHEAR':
            assessments.append({
                'component_index': index,
                'profile_type': profile_type,
                'status': 'zero_convergence',
                'reason': 'External shear changes deflections but has zero convergence and mass.',
            })
        else:
            assessments.append({
                'component_index': index,
                'profile_type': profile_type,
                'status': 'not_assessed',
                'reason': 'No general analytic all-space mass test is implemented for this profile.',
            })

    divergent = any(item['status'].startswith('diverges') for item in assessments)
    not_assessed = any(item['status'] == 'not_assessed' for item in assessments)
    if divergent:
        status = 'not_finite'
        mass_msun = None
    elif not_assessed:
        status = 'undetermined'
        mass_msun = None
    else:
        status = 'finite_or_zero'
        mass_msun = 0.0
    return {
        'quantity': 'all_space_projected_mass_M_2D',
        'integration_limit': 'R -> infinity',
        'status': status,
        'mass_msun': mass_msun,
        'components': assessments,
        'caveat': (
            'A lensing power-law model is an effective local model and generally '
            'should not be extrapolated to infinity. Use a finite aperture or a '
            'truncated/finite-mass halo model for a physical halo-total mass.'
        ),
    }


def _spherical_epl_m200(theta_e_arcsec, gamma, d_lens, sigma_crit, cosmology, z_lens):
    """Return the spherical-EPL extrapolated M_200c and r_200c.

    This assumes rho(r) proportional to r**(-gamma) and uses the circularized
    EPL Einstein-radius convention. It is an extrapolation of the lensing model,
    not a finite halo-total mass.
    """
    if not 1.0 < gamma < 3.0:
        raise ValueError('Spherical EPL M_200c requires 1 < gamma < 3.')
    theta_e_rad = (theta_e_arcsec * u.arcsec).to_value(u.rad)
    d_lens_kpc = d_lens.to_value(u.kpc)
    rho_crit = cosmology.critical_density(z_lens).to_value(u.Msun / u.kpc**3)
    mass_normalization = (
        2.0
        * math.sqrt(math.pi)
        * math.gamma(gamma / 2.0)
        / math.gamma((gamma - 1.0) / 2.0)
        * sigma_crit.to_value(u.Msun / u.kpc**2)
        * (d_lens_kpc * theta_e_rad) ** (gamma - 1.0)
    )
    r200_kpc = (
        3.0 * mass_normalization / (4.0 * math.pi * 200.0 * rho_crit)
    ) ** (1.0 / gamma)
    m200_msun = 4.0 * math.pi / 3.0 * 200.0 * rho_crit * r200_kpc**3
    r200_arcsec = (r200_kpc * u.kpc / d_lens).to_value(u.arcsec, u.dimensionless_angles())
    return {
        'm200c_msun': m200_msun,
        'r200c_kpc': r200_kpc,
        'r200c_arcsec': r200_arcsec,
        'rho_critical_msun_per_kpc3': rho_crit,
    }


def _spherical_epl_m200_uncertainty(primary, primary_sigma, d_lens, sigma_crit, cosmology, z_lens):
    """Independent finite-difference M_200c errors from theta_E and gamma only."""
    if not isinstance(primary_sigma, dict):
        return None
    nominal = _spherical_epl_m200(
        float(primary['theta_E']), float(primary['gamma']), d_lens, sigma_crit, cosmology, z_lens
    )
    lower_squared = {'m200c_msun': 0.0, 'r200c_kpc': 0.0}
    upper_squared = {'m200c_msun': 0.0, 'r200c_kpc': 0.0}
    contributions = []
    for parameter in ('theta_E', 'gamma'):
        errors = _scalar_uncertainty(primary_sigma.get(parameter))
        if errors is None:
            continue
        lower_error, upper_error = errors
        if lower_error == 0.0 and upper_error == 0.0:
            continue
        try:
            lower_primary = dict(primary)
            upper_primary = dict(primary)
            lower_primary[parameter] = float(primary[parameter]) - lower_error
            upper_primary[parameter] = float(primary[parameter]) + upper_error
            lower = _spherical_epl_m200(
                float(lower_primary['theta_E']), float(lower_primary['gamma']),
                d_lens, sigma_crit, cosmology, z_lens,
            )
            upper = _spherical_epl_m200(
                float(upper_primary['theta_E']), float(upper_primary['gamma']),
                d_lens, sigma_crit, cosmology, z_lens,
            )
        except ValueError as error:
            print(f'[m200] Skipping {parameter} uncertainty: {error}')
            continue
        contribution = {'parameter': parameter}
        for key in lower_squared:
            lower_delta = abs(nominal[key] - lower[key])
            upper_delta = abs(upper[key] - nominal[key])
            lower_squared[key] += lower_delta**2
            upper_squared[key] += upper_delta**2
            contribution[f'{key}_error_lower'] = lower_delta
            contribution[f'{key}_error_upper'] = upper_delta
        contributions.append(contribution)
    return {
        'method': 'spherical_EPL_independent_theta_E_gamma_finite_difference',
        'm200c_error_lower_msun': math.sqrt(lower_squared['m200c_msun']),
        'm200c_error_upper_msun': math.sqrt(upper_squared['m200c_msun']),
        'r200c_error_lower_kpc': math.sqrt(lower_squared['r200c_kpc']),
        'r200c_error_upper_kpc': math.sqrt(upper_squared['r200c_kpc']),
        'parameter_contributions': contributions,
        'caveat': 'This ignores posterior covariance, ellipticity, and line-of-sight geometry.',
    }


def _spherical_epl_kappa_area(radius_arcsec, theta_e, gamma):
    """Analytic integral of a spherical EPL inside a circular aperture."""
    if gamma >= 3.0:
        raise ValueError('A singular spherical EPL is divergent at its centre for gamma >= 3.')
    return math.pi * theta_e**(gamma - 1.0) * radius_arcsec**(3.0 - gamma)


def _aperture_mass_profile(
    multipliers, theta_e, kwargs_lens, center_x, center_y, integrator,
    d_lens, sigma_crit, mass_types,
):
    """Compute M_2D within several circular apertures expressed in theta_E."""
    profile = []
    fallback_gamma = None
    if integrator is None:
        if mass_types[0] != 'EPL':
            raise ValueError('The no-Herculens fallback is implemented only for a primary EPL.')
        fallback_gamma = float(kwargs_lens[0]['gamma'])

    for multiplier in multipliers:
        radius = multiplier * theta_e
        if integrator is None:
            kappa_area = _spherical_epl_kappa_area(radius, theta_e, fallback_gamma)
            method = 'analytic_spherical_EPL_primary_only_fallback'
        else:
            kappa_area = integrator(kwargs_lens, radius, center_x, center_y)
            method = 'numerical_configured_kappa_integral'
        mass = _mass_from_kappa_area(kappa_area, d_lens, sigma_crit)
        radius_kpc = (radius * u.arcsec * d_lens).to(u.kpc, u.dimensionless_angles())
        profile.append({
            'radius_theta_E': multiplier,
            'radius_arcsec': radius,
            'radius_kpc': radius_kpc.value,
            'projected_enclosed_mass_msun': mass.value,
            'mean_kappa_inside_aperture': kappa_area / (math.pi * radius**2),
            'method': method,
        })
    return profile


def _save_enclosed_mass_plot(curve_profile, aperture_profile, output_path):
    """Plot finite projected mass as a function of radius in units of theta_E."""
    curve_radius = np.array([point['radius_theta_E'] for point in curve_profile])
    curve_mass = np.array([point['projected_enclosed_mass_msun'] for point in curve_profile])
    marker_radius = np.array([point['radius_theta_E'] for point in aperture_profile])
    marker_mass = np.array([point['projected_enclosed_mass_msun'] for point in aperture_profile])

    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(curve_radius, curve_mass / 1e10, color='tab:blue', linewidth=2.0)
    axis.scatter(marker_radius, marker_mass / 1e10, color='tab:orange', zorder=3)
    axis.set_xlabel(r'$R / \theta_E$')
    axis.set_ylabel(r'$M_{\rm 2D}(<R)$ [$10^{10}\,M_\odot$]')
    axis.set_xlim(left=float(np.min(curve_radius)))
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def estimate_mass(
    run_dir, z_lens, z_source, cosmology_name, radius_arcsec, grid_size,
    aperture_multipliers=(1.0, 3.0, 5.0, 10.0), plot_points=50,
):
    run_dir = Path(run_dir).expanduser().resolve()
    
    # Fallback logic for joint/multiband runs: search for any band subdirectory containing kwargs_result.json
    band_dir = None
    for item in run_dir.iterdir():
        if item.is_dir() and (item / 'kwargs_result.json').is_file():
            band_dir = item
            break
            
    if band_dir is not None:
        print(f'[estimate_mass] Found band-specific subdirectory: {band_dir.name}')
        result_path = band_dir / 'kwargs_result.json'
        sigma_path = band_dir / 'kwargs_sigma.json'
    else:
        result_path = run_dir / 'kwargs_result.json'
        sigma_path = run_dir / 'kwargs_sigma.json'
        
    config_path = run_dir / 'config.json'
    
    for path in (result_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')

    result = _load_json(result_path)
    config = _load_json(config_path)
    mass_types = _mass_types(config)
    kwargs_lens = _kwargs_lens(result)
    if len(mass_types) != len(kwargs_lens):
        raise ValueError(
            f'config.json defines {len(mass_types)} mass profiles but '
            f'kwargs_result.json contains {len(kwargs_lens)} kwargs_lens entries.'
        )

    primary = kwargs_lens[0]
    if radius_arcsec is None:
        if 'theta_E' not in primary:
            raise ValueError(
                'The first mass component has no theta_E. Supply --radius-arcsec explicitly.'
            )
        radius_arcsec = float(primary['theta_E'])
    center_x = float(primary.get('center_x', 0.0))
    center_y = float(primary.get('center_y', 0.0))
    cosmology = COSMOLOGIES[cosmology_name]
    d_lens, d_source, d_ls, sigma_crit = _critical_surface_density(
        z_lens, z_source, cosmology
    )

    physical_radius = (radius_arcsec * u.arcsec * d_lens).to(
        u.kpc, u.dimensionless_angles()
    )
    critical_circle_mass = _mass_from_kappa_area(
        math.pi * radius_arcsec**2, d_lens, sigma_crit
    )

    integration_method = 'numerical_configured_kappa_integral'
    try:
        integrator = _make_kappa_integrator(mass_types, grid_size)
        kappa_area = integrator(kwargs_lens, radius_arcsec, center_x, center_y)
        enclosed_mass = _mass_from_kappa_area(kappa_area, d_lens, sigma_crit)
    except ImportError as error:
        # The standard Einstein mass remains useful on systems without Herculens.
        print(f'[mass] {error}')
        print('[mass] Falling back to the critical-circle Einstein-mass relation.')
        integrator = None
        integration_method = 'analytic_critical_circle_fallback'
        kappa_area = math.pi * radius_arcsec**2
        enclosed_mass = critical_circle_mass

    uncertainty = None
    sigma_result = None
    if sigma_path.is_file():
        sigma_result = _load_json(sigma_path)
        if integrator is None:
            theta_e_error = None
            sigma_lens = sigma_result.get('kwargs_lens', [])
            if sigma_lens:
                theta_e_error = _scalar_uncertainty(sigma_lens[0].get('theta_E'))
            uncertainty = _estimate_critical_circle_uncertainty(
                radius_arcsec, theta_e_error, d_lens, sigma_crit
            )
        else:
            uncertainty = _estimate_independent_uncertainty(
                kwargs_lens,
                sigma_result.get('kwargs_lens'),
                integrator,
                radius_arcsec,
                center_x,
                center_y,
                d_lens,
                sigma_crit,
            )

    aperture_profile = _aperture_mass_profile(
        aperture_multipliers,
        float(primary['theta_E']),
        kwargs_lens,
        center_x,
        center_y,
        integrator,
        d_lens,
        sigma_crit,
        mass_types,
    )
    plot_multipliers = np.linspace(
        min(aperture_multipliers), max(aperture_multipliers), plot_points
    )
    plot_multipliers = np.unique(np.concatenate((plot_multipliers, aperture_multipliers)))
    plot_profile = _aperture_mass_profile(
        plot_multipliers,
        float(primary['theta_E']),
        kwargs_lens,
        center_x,
        center_y,
        integrator,
        d_lens,
        sigma_crit,
        mass_types,
    )
    m200c = None
    if mass_types[0] == 'EPL':
        try:
            m200c = _spherical_epl_m200(
                float(primary['theta_E']), float(primary['gamma']),
                d_lens, sigma_crit, cosmology, z_lens,
            )
            m200c['method'] = 'spherical_EPL_extrapolation'
            m200c['caveat'] = (
                'This is not a directly constrained halo mass. It assumes a spherical '
                '3D deprojection of the EPL and extrapolates it to r_200c; the fitted '
                'ellipticity and external shear are excluded.'
            )
            if sigma_result and sigma_result.get('kwargs_lens'):
                m200c['uncertainty'] = _spherical_epl_m200_uncertainty(
                    primary,
                    sigma_result['kwargs_lens'][0],
                    d_lens,
                    sigma_crit,
                    cosmology,
                    z_lens,
                )
        except ValueError as error:
            m200c = {'status': 'not_available', 'reason': str(error)}

    return {
        'quantity': 'projected_enclosed_lensing_mass_M_2D',
        'aperture': {
            'shape': 'circle',
            'radius_arcsec': radius_arcsec,
            'radius_kpc': physical_radius.value,
            'center_arcsec': {'x': center_x, 'y': center_y},
            'default_radius_note': 'Default aperture radius is theta_E of mass component 0.',
        },
        'mass_model': {'profile_types': mass_types, 'kwargs_lens': kwargs_lens},
        'redshifts': {'z_lens': z_lens, 'z_source': z_source},
        'cosmology': cosmology_name,
        'distances_mpc': {
            'D_lens': d_lens.to_value(u.Mpc),
            'D_source': d_source.to_value(u.Mpc),
            'D_lens_source': d_ls.to_value(u.Mpc),
        },
        'critical_surface_density_msun_per_kpc2': sigma_crit.value,
        'integration_method': integration_method,
        'integrated_kappa_area_arcsec2': kappa_area,
        'mean_kappa_inside_aperture': kappa_area / (math.pi * radius_arcsec**2),
        'projected_enclosed_mass_msun': enclosed_mass.value,
        'critical_circle_mass_msun': critical_circle_mass.value,
        'critical_circle_mass_note': (
            'pi * R_E^2 * Sigma_crit, shown for comparison. It equals the enclosed '
            'mass only when the aperture has mean kappa = 1.'
        ),
        'aperture_mass_profile': aperture_profile,
        'mass_profile_plot_samples': plot_profile,
        'm200c_spherical_EPL_extrapolation': m200c,
        'total_mass_to_infinity': _total_mass_to_infinity_assessment(
            mass_types, kwargs_lens
        ),
        'kwargs_sigma': sigma_result.get('kwargs_lens') if sigma_result else None,
        'uncertainty': uncertainty,
        'caveat': (
            'This is a projected lensing mass inside a chosen aperture, not a 3D or '
            'halo-total mass. The numerical kappa integral includes every configured '
            'mass component; a SHEAR component contributes zero convergence.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='?', default='results', help='Directory containing the saved JSON results.')
    parser.add_argument('--zl', type=float, default=1.53, help='Lens redshift (default: 1.53).')
    parser.add_argument('--zs', type=float, default=3.417, help='Source redshift (default: 3.417).')
    parser.add_argument('--cosmology', choices=sorted(COSMOLOGIES), default='Planck18')
    parser.add_argument('--radius-arcsec', type=float, default=None, help='Override the default theta_E aperture radius.')
    parser.add_argument(
        '--grid-size', type=int, default=800,
        help='Radial and azimuthal integration samples (default: 800).',
    )
    parser.add_argument(
        '--plot-points', type=int, default=50,
        help='Number of directly evaluated points in the enclosed-mass plot (default: 50).',
    )
    parser.add_argument(
        '--aperture-multipliers', default='1,3,5,10',
        help='Comma-separated finite apertures in units of theta_E (default: 1,3,5,10).',
    )
    parser.add_argument('--output', default=None, help='Output JSON path (default: <run_dir>/enclosed_mass_theta_E.json).')
    args = parser.parse_args()

    try:
        aperture_multipliers = tuple(
            float(value.strip()) for value in args.aperture_multipliers.split(',')
        )
    except ValueError as error:
        parser.error(f'--aperture-multipliers must be comma-separated numbers: {error}')
    if not aperture_multipliers or any(value <= 0.0 for value in aperture_multipliers):
        parser.error('--aperture-multipliers must contain one or more positive values.')
    if args.plot_points < 2:
        parser.error('--plot-points must be at least 2.')

    estimate = estimate_mass(
        args.run_dir, args.zl, args.zs, args.cosmology, args.radius_arcsec, args.grid_size,
        aperture_multipliers, args.plot_points,
    )
    output_path = Path(args.output).expanduser() if args.output else Path(args.run_dir) / 'enclosed_mass_theta_E.json'
    output_path = output_path.resolve()
    
    # Construct simplified clean output JSON
    clean_profiles = [t for t in estimate['mass_model']['profile_types'] if t != 'SHEAR']
    
    def propagate_q_pa(e1, e2, e1_sigmas=None, e2_sigmas=None, n_samples=10000):
        # Best-fit values
        eta = np.sqrt(e1**2 + e2**2)
        eta = np.clip(eta, 0.0, 0.999)
        q = (1.0 - eta) / (1.0 + eta)
        
        pa = 0.5 * np.arctan2(e2, e1) * 180.0 / np.pi
        pa = pa % 180.0
        
        if e1_sigmas is None or e2_sigmas is None:
            return f"{q:.5f}", f"{pa:.5f}"
            
        # Generate split-normal samples
        s1 = np.random.randn(n_samples)
        e1_samples = np.where(s1 < 0, e1 + s1 * e1_sigmas[0], e1 + s1 * e1_sigmas[1])
        
        s2 = np.random.randn(n_samples)
        e2_samples = np.where(s2 < 0, e2 + s2 * e2_sigmas[0], e2 + s2 * e2_sigmas[1])
        
        # Calculate q samples
        eta_samples = np.sqrt(e1_samples**2 + e2_samples**2)
        eta_samples = np.clip(eta_samples, 0.0, 0.999)
        q_samples = (1.0 - eta_samples) / (1.0 + eta_samples)
        
        q_lower = q - np.percentile(q_samples, 16)
        q_upper = np.percentile(q_samples, 84) - q
        
        # Calculate PA samples and center them around best PA to avoid boundary wrapping issues
        pa_samples = 0.5 * np.arctan2(e2_samples, e1_samples) * 180.0 / np.pi
        d_pa = (pa_samples - pa + 90.0) % 180.0 - 90.0
        pa_samples_centered = pa + d_pa
        
        pa_lower = pa - np.percentile(pa_samples_centered, 16)
        pa_upper = np.percentile(pa_samples_centered, 84) - pa
        
        return f"{q:.5f}, -{q_lower:.5f}, +{q_upper:.5f}", f"{pa:.5f}, -{pa_lower:.5f}, +{pa_upper:.5f}"

    clean_lens_params = []
    clean_axis_ratio_and_pa = []
    for component_index, (kw, t) in enumerate(zip(estimate['mass_model']['kwargs_lens'], estimate['mass_model']['profile_types'])):
        if t == 'SHEAR':
            continue
        comp_dict = {}
        for name, best_fit in kw.items():
            # Check if we have sigmas for this parameter
            sigmas = None
            if estimate.get('kwargs_sigma') and component_index < len(estimate['kwargs_sigma']):
                sigmas = estimate['kwargs_sigma'][component_index].get(name)
            
            # Format value
            if sigmas is not None and isinstance(sigmas, (list, tuple)) and len(sigmas) == 2:
                lower, upper = sigmas
                if lower > 0.0 or upper > 0.0:
                    comp_dict[name] = f"{best_fit:.5f}, -{lower:.5f}, +{upper:.5f}"
                else:
                    comp_dict[name] = f"{best_fit:.5f}"
            else:
                comp_dict[name] = f"{best_fit:.5f}"
        clean_lens_params.append(comp_dict)
                
        # Calculate physical axis ratio and PA (with error propagation)
        if 'e1' in kw and 'e2' in kw:
            e1 = kw['e1']
            e2 = kw['e2']
            e1_sigmas = None
            e2_sigmas = None
            if estimate.get('kwargs_sigma') and component_index < len(estimate['kwargs_sigma']):
                e1_sigmas = estimate['kwargs_sigma'][component_index].get('e1')
                e2_sigmas = estimate['kwargs_sigma'][component_index].get('e2')
            
            q_str, pa_str = propagate_q_pa(e1, e2, e1_sigmas, e2_sigmas)
            clean_axis_ratio_and_pa.append({
                'axis_ratio': q_str,
                'PA': pa_str
            })
        
    def to_sci(val):
        if val is None:
            return None
        return f"{val:.6e}"
        
    simplified = {
        'profile_types': clean_profiles,
        'lens_parameters': clean_lens_params,
        'axis_ratio_and_PA': clean_axis_ratio_and_pa,
        'redshifts': estimate['redshifts'],
        'cosmology': estimate['cosmology'],
        'physical_distances_mpc': estimate['distances_mpc'],
        'enclosed_mass': {
            'mass_type': 'projected_2D_surface_mass',
            'aperture_radius_arcsec': estimate['aperture']['radius_arcsec'],
            'aperture_radius_kpc': estimate['aperture']['radius_kpc'],
            'mass_msun': to_sci(estimate['projected_enclosed_mass_msun']),
            'uncertainty_lower_msun': to_sci(estimate['uncertainty']['mass_error_lower_msun']) if estimate['uncertainty'] else None,
            'uncertainty_upper_msun': to_sci(estimate['uncertainty']['mass_error_upper_msun']) if estimate['uncertainty'] else None,
        },
        'total_mass': {
            'mass_type': 'spherical_3D_deprojected_mass_m200c',
            'm200c_msun': to_sci(estimate['m200c_spherical_EPL_extrapolation']['m200c_msun']) if (estimate['m200c_spherical_EPL_extrapolation'] and 'm200c_msun' in estimate['m200c_spherical_EPL_extrapolation']) else None,
            'r200c_kpc': estimate['m200c_spherical_EPL_extrapolation']['r200c_kpc'] if (estimate['m200c_spherical_EPL_extrapolation'] and 'r200c_kpc' in estimate['m200c_spherical_EPL_extrapolation']) else None,
            'uncertainty_lower_msun': to_sci(estimate['m200c_spherical_EPL_extrapolation']['uncertainty']['m200c_error_lower_msun']) if (estimate['m200c_spherical_EPL_extrapolation'] and estimate['m200c_spherical_EPL_extrapolation'].get('uncertainty')) else None,
            'uncertainty_upper_msun': to_sci(estimate['m200c_spherical_EPL_extrapolation']['uncertainty']['m200c_error_upper_msun']) if (estimate['m200c_spherical_EPL_extrapolation'] and estimate['m200c_spherical_EPL_extrapolation'].get('uncertainty')) else None,
        }
    }
    
    with output_path.open('w') as handle:
        json.dump(simplified, handle, indent=2)
    plot_path = output_path.with_name('enclosed_mass_vs_radius_theta_E.png')
    _save_enclosed_mass_plot(
        estimate['mass_profile_plot_samples'], estimate['aperture_mass_profile'], plot_path
    )

    aperture = estimate['aperture']
    print(f"Mass profiles: {estimate['mass_model']['profile_types']}")
    print(
        'Aperture: R = '
        f"{aperture['radius_arcsec']:.6f} arcsec = {aperture['radius_kpc']:.4f} kpc, "
        f"center = ({aperture['center_arcsec']['x']:.6f}, {aperture['center_arcsec']['y']:.6f}) arcsec"
    )
    print(f"M_2D(<R_E) = {estimate['projected_enclosed_mass_msun']:.6e} Msun")
    if estimate['uncertainty']:
        uncertainty = estimate['uncertainty']
        print(
            'Approximate independent uncertainty: '
            f"-{uncertainty['mass_error_lower_msun']:.3e} / "
            f"+{uncertainty['mass_error_upper_msun']:.3e} Msun"
        )
    else:
        print('No kwargs_sigma.json found; no parameter uncertainty was estimated.')
    print('Finite-aperture mass profile:')
    for point in estimate['aperture_mass_profile']:
        print(
            f"  {point['radius_theta_E']:g} theta_E "
            f"({point['radius_arcsec']:.6f} arcsec; {point['radius_kpc']:.4f} kpc): "
            f"{point['projected_enclosed_mass_msun']:.6e} Msun "
            f"[{point['method']}]"
        )
    m200c = estimate['m200c_spherical_EPL_extrapolation']
    if m200c and 'm200c_msun' in m200c:
        print(
            f"Spherical-EPL extrapolated M_200c = {m200c['m200c_msun']:.6e} Msun; "
            f"r_200c = {m200c['r200c_kpc']:.4f} kpc ({m200c['r200c_arcsec']:.6f} arcsec)"
        )
        if m200c.get('uncertainty'):
            m200_error = m200c['uncertainty']
            print(
                'Approximate M_200c uncertainty: '
                f"-{m200_error['m200c_error_lower_msun']:.3e} / "
                f"+{m200_error['m200c_error_upper_msun']:.3e} Msun"
            )
    print(f'Saved {plot_path}')
    print(f'Saved {output_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
