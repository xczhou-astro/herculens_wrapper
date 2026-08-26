"""JAX mass profiles used by the API composite stellar--halo model.

The implementations delegate the elliptical-Gaussian and gNFW MGE machinery
to :mod:`jax_lensing_profiles`.  The two small adapters below expose a stable
wrapper-facing parameter convention.
"""

from __future__ import annotations

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
