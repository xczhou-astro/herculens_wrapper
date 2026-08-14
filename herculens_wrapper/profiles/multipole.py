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
        Multipole order. It must be fixed to an integer greater than or equal
        to two; it is not meaningful to sample it as a continuous parameter.
    a_m, phi_m : float
        Perturbation amplitude and orientation in radians.
    gamma : float
        EPL three-dimensional density slope. Normally link this to the EPL.
    center_x, center_y : float
        Perturbation centre. Normally link these to the EPL centre.
    b : float, optional
        Scale radius used by the profile. It defaults to one arcsec and can be
        supplied as an additional fixed/sampled config parameter when needed.
    """

    param_names = ["m", "a_m", "phi_m", "gamma", "center_x", "center_y"]
    lower_limit_default = {
        "m": 2,
        "a_m": 0,
        "phi_m": -np.pi,
        "gamma": 1.0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": np.pi,
        "gamma": 3.0,
        "center_x": 100,
        "center_y": 100,
    }
    fixed_default = {key: False for key in param_names}

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
        a_m,
        phi_m,
        gamma=2.0,
        center_x=0.0,
        center_y=0.0,
        b=1.0,
    ):
        """Return the MPPL lensing potential in arcsec squared."""
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
