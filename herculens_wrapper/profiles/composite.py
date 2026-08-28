"""JAX mass profiles used by the API composite stellar--halo model.

The implementations delegate the elliptical-Gaussian and gNFW MGE machinery
to :mod:`jax_lensing_profiles`.  The two small adapters below expose a stable
wrapper-facing parameter convention.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _mge_backend():
    try:
        from jax_lensing_profiles.MassModel.Profiles.CuspyHalo_ellipse_kappa import (
            CuspyHaloEllipseKappa,
        )
        from jax_lensing_profiles.MassModel.Profiles.multi_gaussian_ellipse_kappa import (
            MultiGaussianEllipseKappa,
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The stellar + gNFW MGE API requires the 'jax-lensing-profiles' package."
        ) from error
    return CuspyHaloEllipseKappa, MultiGaussianEllipseKappa


class StellarMGE:
    """Light-tracing stellar convergence with global scale and M/L gradient.

    ``light_*`` vectors are fixed properties of an already fitted lens-light
    MGE.  The sampled ``upsilon_kappa`` and ``ml_gradient`` transform those
    light amplitudes into integrated Gaussian convergence amplitudes.
    """

    param_names = [
        "upsilon_kappa", "ml_gradient", "light_amp", "light_sigma",
        "light_e1", "light_e2", "light_center_x", "light_center_y",
    ]
    lower_limit_default = {name: -1e6 for name in param_names}
    upper_limit_default = {name: 1e6 for name in param_names}
    fixed_default = {name: False for name in param_names}

    def __init__(self) -> None:
        _, multi_gaussian = _mge_backend()
        self._mge = multi_gaussian()

    @staticmethod
    def _amplitudes(upsilon_kappa, ml_gradient, light_amp, light_sigma):
        light_amp = jnp.asarray(light_amp)
        light_sigma = jnp.asarray(light_sigma)
        total_light = jnp.sum(light_amp)
        # The API validates positive light amplitudes before construction.
        relative = light_amp / total_light
        reference_sigma = jnp.exp(jnp.mean(jnp.log(light_sigma)))
        return upsilon_kappa * relative * (light_sigma / reference_sigma) ** (-ml_gradient)

    def function(self, x, y, upsilon_kappa, ml_gradient, light_amp,
                 light_sigma, light_e1, light_e2, light_center_x,
                 light_center_y):
        return self._mge.function(
            x, y,
            self._amplitudes(upsilon_kappa, ml_gradient, light_amp, light_sigma),
            jnp.asarray(light_sigma), jnp.asarray(light_e1), jnp.asarray(light_e2),
            jnp.asarray(light_center_x), jnp.asarray(light_center_y),
        )

    def derivatives(self, x, y, upsilon_kappa, ml_gradient, light_amp,
                    light_sigma, light_e1, light_e2, light_center_x,
                    light_center_y):
        return self._mge.derivatives(
            x, y,
            self._amplitudes(upsilon_kappa, ml_gradient, light_amp, light_sigma),
            jnp.asarray(light_sigma), jnp.asarray(light_e1), jnp.asarray(light_e2),
            jnp.asarray(light_center_x), jnp.asarray(light_center_y),
        )

    def hessian(self, x, y, upsilon_kappa, ml_gradient, light_amp,
                light_sigma, light_e1, light_e2, light_center_x,
                light_center_y):
        return self._mge.hessian(
            x, y,
            self._amplitudes(upsilon_kappa, ml_gradient, light_amp, light_sigma),
            jnp.asarray(light_sigma), jnp.asarray(light_e1), jnp.asarray(light_e2),
            jnp.asarray(light_center_x), jnp.asarray(light_center_y),
        )


class GNFWMGE:
    """Elliptical gNFW halo approximated by a differentiable 3-D MGE."""

    param_names = ["kappa_s", "r_s", "gamma_inner", "e1", "e2", "center_x", "center_y", "n_outer"]
    lower_limit_default = {name: -1e6 for name in param_names}
    upper_limit_default = {name: 1e6 for name in param_names}
    fixed_default = {name: False for name in param_names}

    def __init__(self) -> None:
        cuspy_halo, _ = _mge_backend()
        self._mge = cuspy_halo()

    def function(self, x, y, kappa_s, r_s, gamma_inner, e1, e2,
                 center_x=0.0, center_y=0.0, n_outer=3.0):
        return self._mge.function(
            x, y, kappa_s=kappa_s, R_s=r_s, gamma=gamma_inner,
            n=n_outer, e1=e1, e2=e2, center_x=center_x, center_y=center_y,
        )

    def derivatives(self, x, y, kappa_s, r_s, gamma_inner, e1, e2,
                    center_x=0.0, center_y=0.0, n_outer=3.0):
        return self._mge.derivatives(
            x, y, kappa_s=kappa_s, R_s=r_s, gamma=gamma_inner,
            n=n_outer, e1=e1, e2=e2, center_x=center_x, center_y=center_y,
        )

    def hessian(self, x, y, kappa_s, r_s, gamma_inner, e1, e2,
                center_x=0.0, center_y=0.0, n_outer=3.0):
        return self._mge.hessian(
            x, y, kappa_s=kappa_s, R_s=r_s, gamma=gamma_inner,
            n=n_outer, e1=e1, e2=e2, center_x=center_x, center_y=center_y,
        )


class InclinedExponentialDiskMGE:
    """Projected exponential disk represented by a fixed-size MGE basis.

    This is an *effective projected-mass* model.  Its intrinsic geometry is a
    thick axisymmetric disk with radial scale ``r_scale``, intrinsic thickness
    ``q0``, and inclination ``inclination`` (0 is face-on and pi/2 is
    edge-on).  The apparent axis ratio follows

    ``q_proj**2 = cos(inclination)**2 + q0**2 * sin(inclination)**2``.

    The projected convergence is approximated by an elliptical exponential,
    ``kappa(R_ell) = kappa_0 exp(-R_ell / r_scale)``, expanded as co-centred,
    co-aligned elliptical Gaussians. ``n_gaussians`` is a fixed scalar model
    setting, not a sampling parameter; supported choices are 5, 7, and 9.
    Internally all choices are zero-padded to nine Gaussians so JAX always
    receives arrays with the same shape.  It is intentionally not an
    exact line-of-sight projection of an exponential--sech-squared density;
    that calculation would require numerical quadrature and a Poisson solve
    during every likelihood evaluation.  The MGE approximation instead keeps
    the potential, deflections, and Hessian differentiable and fast.

    ``kappa_0`` is the central convergence of the approximating exponential,
    and ``r_scale`` is in the same angular unit as the lens-plane coordinates
    (normally arcsec).  The total convergence area is approximately
    ``2 pi q_proj kappa_0 r_scale**2``.
    """

    # Individually fitted MGE bases for exp(-R/r_scale), each padded to the
    # maximum supported length.  Each row is [5, 7, 9] Gaussians respectively;
    # weights are integrated-convergence fractions and sum to one per row.
    _BASIS_COUNTS = jnp.asarray([5, 7, 9])
    _SIGMA_OVER_SCALE = jnp.asarray([
        [0.1060049752, 0.3396955994, 0.7541927912, 1.40246892, 2.396619889,
         2.396619889, 2.396619889, 2.396619889, 2.396619889],
        [0.0391847299, 0.1329749183, 0.3093018093, 0.6032081015, 1.050816427,
         1.702812385, 2.679427269, 2.679427269, 2.679427269],
        [0.01556513452, 0.05502380744, 0.1329144173, 0.2681616776,
         0.4850863567, 0.8108727348, 1.275418931, 1.93311404, 2.90034822],
    ])
    _INTEGRATED_WEIGHTS = jnp.asarray([
        [0.001470540764, 0.02734190423, 0.1790125798, 0.4649008966,
         0.3272740786, 0.0, 0.0, 0.0, 0.0],
        [7.743731531e-05, 0.001817027977, 0.01665692429, 0.0883926984,
         0.2730140956, 0.4258427001, 0.1941991164, 0.0, 0.0],
        [5.025796987e-06, 0.0001346417112, 0.001440805688, 0.009490748104,
         0.04460122497, 0.1468847676, 0.3159570343, 0.3606920186,
         0.1207937333],
    ])

    param_names = [
        "kappa_0", "r_scale", "inclination", "q0", "phi", "center_x", "center_y",
        "n_gaussians",
    ]
    lower_limit_default = {
        "kappa_0": 0.0,
        "r_scale": 1e-4,
        "inclination": 0.0,
        # A small but non-zero thickness avoids a singular projected disk at
        # exactly edge-on inclination.
        "q0": 0.05,
        "phi": -np.pi / 2.0,
        "center_x": -100.0,
        "center_y": -100.0,
        "n_gaussians": 5,
    }
    upper_limit_default = {
        "kappa_0": 1e4,
        "r_scale": 100.0,
        "inclination": np.pi / 2.0,
        "q0": 1.0,
        "phi": np.pi / 2.0,
        "center_x": 100.0,
        "center_y": 100.0,
        "n_gaussians": 9,
    }
    fixed_default = {name: False for name in param_names}
    fixed_default["n_gaussians"] = True

    def __init__(self) -> None:
        _, multi_gaussian = _mge_backend()
        self._mge = multi_gaussian()

    @classmethod
    def _mge_parameters(cls, kappa_0, r_scale, inclination, q0, phi, center_x, center_y,
                        n_gaussians):
        """Convert disk parameters to the MGE backend's vector convention."""
        q_proj = jnp.sqrt(
            jnp.cos(inclination) ** 2 + q0 ** 2 * jnp.sin(inclination) ** 2
        )
        # The jax-lensing-profiles elliptical-Gaussian formula has a 1-q**2
        # denominator and no circular branch.  Its q -> 1 limit is regular,
        # but evaluating it at q == 1 produces NaNs.  A 0.1% flattening is
        # well below any meaningful mass-shape precision and keeps face-on
        # disks numerically stable in 32-bit JAX as well.
        q_backend = jnp.minimum(q_proj, 0.999)
        ellipticity = (1.0 - q_backend) / (1.0 + q_backend)
        e1 = ellipticity * jnp.cos(2.0 * phi)
        e2 = ellipticity * jnp.sin(2.0 * phi)
        # Use JAX selection rather than Python slicing: n_gaussians arrives as
        # a traced scalar even when fixed by the sampler, whereas all three
        # alternatives retain the shared, padded (9,) shape.
        basis_index = jnp.where(
            n_gaussians == 5,
            0,
            jnp.where(n_gaussians == 7, 1, 2),
        )
        sigma = cls._SIGMA_OVER_SCALE[basis_index] * r_scale
        weights = cls._INTEGRATED_WEIGHTS[basis_index]

        # MultiGaussianEllipseKappa uses integrated convergence amplitudes.
        total_convergence_area = 2.0 * jnp.pi * q_backend * kappa_0 * r_scale ** 2
        amplitude = total_convergence_area * weights
        return (
            amplitude,
            sigma,
            jnp.full_like(sigma, e1),
            jnp.full_like(sigma, e2),
            jnp.full_like(sigma, center_x),
            jnp.full_like(sigma, center_y),
        )

    def function(self, x, y, kappa_0, r_scale, inclination, q0, phi,
                 center_x=0.0, center_y=0.0, n_gaussians=7):
        return self._mge.function(
            x, y,
            *self._mge_parameters(
                kappa_0, r_scale, inclination, q0, phi, center_x, center_y, n_gaussians,
            ),
        )

    def derivatives(self, x, y, kappa_0, r_scale, inclination, q0, phi,
                    center_x=0.0, center_y=0.0, n_gaussians=7):
        return self._mge.derivatives(
            x, y,
            *self._mge_parameters(
                kappa_0, r_scale, inclination, q0, phi, center_x, center_y, n_gaussians,
            ),
        )

    def hessian(self, x, y, kappa_0, r_scale, inclination, q0, phi,
                center_x=0.0, center_y=0.0, n_gaussians=7):
        return self._mge.hessian(
            x, y,
            *self._mge_parameters(
                kappa_0, r_scale, inclination, q0, phi, center_x, center_y, n_gaussians,
            ),
        )
