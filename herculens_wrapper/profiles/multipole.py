"""EPL-slope-dependent multipole mass profile.

This is deliberately exposed as ``MPPL`` instead of replacing Herculens's
standard ``MULTIPOLE`` profile.  The two profiles use different potential
normalizations and must remain distinguishable in saved configurations.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _normalization(three_minus_gamma, m, radius):
    """Return the MPPL radial normalization, including its analytic limit."""
    coincident = jnp.abs(three_minus_gamma - m) < 1e-12
    safe_m = jnp.where(coincident, m + 1e-5, m)
    regular = three_minus_gamma**2 / (
        (three_minus_gamma - safe_m) * (three_minus_gamma + safe_m)
    )
    limiting = three_minus_gamma * jnp.log(radius) / 2.0
    return jnp.where(coincident, limiting, regular)


class MPPL:
    """Multipole perturbation whose radial scaling follows an EPL slope.

    Parameters
    ----------
    m : int
        Multipole order. It must be fixed to a positive integer; it is not
        meaningful to sample it as a continuous parameter.
    a_m, phi_m : float, optional
        Direct perturbation amplitude and orientation in radians.  Supply
        these together, or supply ``e_x`` and ``e_y`` together, but never
        both parameterizations.
    e_x, e_y : float, optional
        Paper-style multipole ellipticity coordinates.  When supplied, the
        profile derives ``a_m = 2 e / (1 + e)`` and
        ``phi_m = atan2(e_y, e_x) / m``, where
        ``e = sqrt(e_x**2 + e_y**2)``.
    gamma : float
        EPL three-dimensional density slope. Normally link this to the EPL.
    center_x, center_y : float
        Perturbation centre. Normally link these to the EPL centre.
    b : float
        Reference scale radius.  To use the Enzi et al. (2025) convention,
        link this fixed parameter to the Einstein radius of the companion EPL
        component.  Then ``a_m`` is their dimensionless ``A_Mn``.
    """

    param_names = [
        "m", "a_m", "phi_m", "e_x", "e_y", "gamma", "center_x", "center_y", "b",
    ]
    lower_limit_default = {
        "m": 1,
        "a_m": 0,
        "phi_m": -np.pi,
        "e_x": -1,
        "e_y": -1,
        "gamma": 1.0,
        "center_x": -100,
        "center_y": -100,
        "b": 1e-6,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": np.pi,
        "e_x": 1,
        "e_y": 1,
        "gamma": 3.0,
        "center_x": 100,
        "center_y": 100,
        "b": 100,
    }
    fixed_default = {
        "m": True,
        "a_m": False,
        "phi_m": False,
        "e_x": False,
        "e_y": False,
        "gamma": False,
        "center_x": False,
        "center_y": False,
        # This is a reference scale, not an additional physical degree of
        # freedom.  It should normally be linked to the EPL Einstein radius.
        "b": True,
    }

    @staticmethod
    def _polar_coordinates(x, y, center_x, center_y):
        x_shifted = x - center_x
        y_shifted = y - center_y
        # Keeping the epsilon inside sqrt gives finite JAX derivatives at the
        # exact centre without affecting resolved lens-plane pixels.
        radius = jnp.sqrt(x_shifted**2 + y_shifted**2 + 1e-10)
        angle = jnp.arctan2(y_shifted, x_shifted)
        return radius, angle

    @staticmethod
    def function(
        x,
        y,
        m,
        a_m=None,
        phi_m=None,
        e_x=None,
        e_y=None,
        gamma=2.0,
        center_x=0.0,
        center_y=0.0,
        b=1.0,
    ):
        """Return the MPPL lensing potential in arcsec squared."""
        using_ellipticity = e_x is not None or e_y is not None
        using_amplitude_phase = a_m is not None or phi_m is not None
        if using_ellipticity:
            if e_x is None or e_y is None:
                raise ValueError("MPPL requires both 'e_x' and 'e_y' when using ellipticity coordinates.")
            if using_amplitude_phase:
                raise ValueError(
                    "MPPL accepts either ('a_m', 'phi_m') or ('e_x', 'e_y'), not both."
                )
            ellipticity = jnp.sqrt(e_x**2 + e_y**2)
            a_m = 2.0 * ellipticity / (1.0 + ellipticity)
            phi_m = jnp.arctan2(e_y, e_x) / m
        elif a_m is None or phi_m is None:
            raise ValueError("MPPL requires both 'a_m' and 'phi_m' when ellipticity coordinates are absent.")

        radius, angle = MPPL._polar_coordinates(x, y, center_x, center_y)
        three_minus_gamma = 3.0 - gamma
        amplitude = _normalization(three_minus_gamma, m, radius)
        return (
            radius**three_minus_gamma
            * amplitude
            * jnp.cos(m * (angle - phi_m))
            / three_minus_gamma
            * b ** (gamma - 1.0)
            * a_m
        )

    @staticmethod
    def _gradient_at_point(x, y, **kwargs):
        return jnp.array(jax.grad(MPPL.function, argnums=(0, 1))(x, y, **kwargs))

    @staticmethod
    def derivatives(x, y, **kwargs):
        """Return the x and y deflection angles."""
        gradient = jnp.vectorize(
            lambda x_value, y_value: MPPL._gradient_at_point(x_value, y_value, **kwargs),
            signature="(),()->(i)",
        )(x, y)
        return gradient[..., 0], gradient[..., 1]

    @staticmethod
    def _hessian_at_point(x, y, **kwargs):
        return jnp.array(jax.hessian(MPPL.function, argnums=(0, 1))(x, y, **kwargs))

    @staticmethod
    def hessian(x, y, **kwargs):
        """Return ``f_xx, f_yy, f_xy`` as required by Herculens."""
        hessian = jnp.vectorize(
            lambda x_value, y_value: MPPL._hessian_at_point(x_value, y_value, **kwargs),
            signature="(),()->(i,i)",
        )(x, y)
        return hessian[..., 0, 0], hessian[..., 1, 1], hessian[..., 0, 1]
