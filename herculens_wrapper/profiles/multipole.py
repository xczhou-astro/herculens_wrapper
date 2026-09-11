"""EPL-slope-dependent multipole mass profile.

This is deliberately exposed as ``MPPL`` instead of replacing Herculens's
standard ``MULTIPOLE`` profile.  The two profiles use different potential
normalizations and must remain distinguishable in saved configurations.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .jaxtronomy_multipole import EllipticalMultipole as _JAXtronomyEllipticalMultipole


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


class MPPLOffset:
    """MPPL whose phase is an offset from a reference position angle.

    ``phi_ref`` and ``delta_phi_m`` use degrees in the wrapper API.  The
    reference angle is normally linked to an EPL ``phi`` parameter; only the
    relative offset is sampled.  Internally,
    ``phi_m = deg2rad(phi_ref + delta_phi_m)``.
    """

    param_names = [
        "m", "a_m", "phi_ref", "delta_phi_m", "gamma", "center_x", "center_y", "b",
    ]
    lower_limit_default = {
        "m": 1, "a_m": 0, "phi_ref": -360, "delta_phi_m": -180,
        "gamma": 1.0, "center_x": -100, "center_y": -100, "b": 1e-6,
    }
    upper_limit_default = {
        "m": 100, "a_m": 100, "phi_ref": 360, "delta_phi_m": 180,
        "gamma": 3.0, "center_x": 100, "center_y": 100, "b": 100,
    }
    fixed_default = {
        "m": True, "a_m": False, "phi_ref": False, "delta_phi_m": False,
        "gamma": False, "center_x": False, "center_y": False, "b": True,
    }

    @staticmethod
    def function(
        x, y, m, a_m, phi_ref, delta_phi_m, gamma=2.0,
        center_x=0.0, center_y=0.0, b=1.0,
    ):
        """Return the lensing potential with a dynamically linked phase."""
        return MPPL.function(
            x, y, m=m, a_m=a_m,
            phi_m=jnp.deg2rad(phi_ref + delta_phi_m), gamma=gamma,
            center_x=center_x, center_y=center_y, b=b,
        )

    @staticmethod
    def _gradient_at_point(x, y, **kwargs):
        return jnp.array(jax.grad(MPPLOffset.function, argnums=(0, 1))(x, y, **kwargs))

    @staticmethod
    def derivatives(x, y, **kwargs):
        """Return the x and y deflection angles."""
        gradient = jnp.vectorize(
            lambda x_value, y_value: MPPLOffset._gradient_at_point(x_value, y_value, **kwargs),
            signature="(),()->(i)",
        )(x, y)
        return gradient[..., 0], gradient[..., 1]

    @staticmethod
    def _hessian_at_point(x, y, **kwargs):
        return jnp.array(jax.hessian(MPPLOffset.function, argnums=(0, 1))(x, y, **kwargs))

    @staticmethod
    def hessian(x, y, **kwargs):
        """Return ``f_xx, f_yy, f_xy`` as required by Herculens."""
        hessian = jnp.vectorize(
            lambda x_value, y_value: MPPLOffset._hessian_at_point(x_value, y_value, **kwargs),
            signature="(),()->(i,i)",
        )(x, y)
        return hessian[..., 0, 0], hessian[..., 1, 1], hessian[..., 0, 1]


def _elliptical_angle(phi, q):
    """Eccentric anomaly of a point with polar angle ``phi``."""
    return jnp.arctan2(jnp.sin(phi), q * jnp.cos(phi))


def _phi_ell(phi, q):
    return phi - jnp.arctan2(jnp.sin(phi), jnp.cos(phi)) + _elliptical_angle(phi, q)


def _f_m1(phi, q):
    log_term = jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    constant = jnp.log(2) * (1 + q) / 2 - (1 - q**2) * (1 + jnp.log(2) / 4)
    return -(
        jnp.cos(phi) * (q * log_term - constant)
        + 2 * jnp.sin(phi) * (phi - _phi_ell(phi, q))
    ) / (2 * (1 - q**2))


def _f_m3(phi, q):
    log_term = jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    constant = (
        jnp.log(2) * (1 + q) ** 2
        - 2 * (1 - q) * (1 + q) ** 2 * (1 + jnp.log(2) / 4)
        + (1 - q**2) ** 2 / 4
    )
    return (
        jnp.cos(phi) * (q * (3 + q**2) * log_term - constant)
        + 2 * jnp.sin(phi) * (1 + 3 * q**2) * (phi - _phi_ell(phi, q))
    ) / (2 * (1 - q**2) ** 2)


def _f_m4_1(phi, q):
    denominator = jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    root = jnp.sqrt(1 - q**2)
    prefactor = (1 + 6 * q**2 + q**4) / (1 - q**2) ** (5 / 2)
    atan_term = jnp.arctan(jnp.sqrt(2) * root * jnp.cos(phi) / denominator)
    log_term = jnp.log(root * jnp.sin(phi) / q + jnp.sqrt(1 + (1 - q**2) * jnp.sin(phi) ** 2 / q**2))
    return (
        -4 * jnp.sqrt(2) * (1 + 4 * q**2 + q**4 + (q**4 - 1) * jnp.cos(2 * phi))
        / (3 * (1 - q**2) ** 2 * denominator)
        + prefactor * jnp.cos(phi) * atan_term
        + prefactor * jnp.sin(phi) * log_term
    )


def _f_m4_2(phi, q):
    denominator = jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    root = jnp.sqrt(1 - q**2)
    prefactor = 4 * q * (1 + q**2) / (1 - q**2) ** (5 / 2)
    atan_term = jnp.arctan(jnp.sqrt(2) * root * jnp.cos(phi) / denominator)
    log_term = jnp.log(root * jnp.sin(phi) / q + jnp.sqrt(1 + (1 - q**2) * jnp.sin(phi) ** 2 / q**2))
    return (
        -4 * jnp.sqrt(2) * q * jnp.sin(2 * phi) / (3 * (1 - q**2) * denominator)
        - prefactor * jnp.sin(phi) * atan_term
        + prefactor * jnp.cos(phi) * log_term
    )


class ELLMPPL:
    """Exact elliptical multipole for SIE-like reference isodensity contours.

    This is a standalone implementation of the ``m=1,3,4`` solutions of
    Paugnat & Gilman (2025), following JAXtronomy's ``EllipticalMultipole``
    convention.  It deliberately does *not* combine an EPL with the
    multipole.  Link ``q``, ``phi_ref``, ``center_x``, ``center_y``, and
    normally ``r_E`` to the companion EPL in the public API configuration.

    ``varphi_m`` is the eccentric anomaly from the reference ellipse's
    semi-major axis, not a polar angle.  The native profile receives radians;
    the wrapper converts the public degree-valued API inputs before calling
    this class.

    Supply exactly one amplitude convention:

    * ``a_m`` is the physical amplitude in arcsec, matching JAXtronomy;
    * ``a_m_frac`` is the paper's dimensionless fractional elliptical-radius
      perturbation.  It is converted internally as ``a_m = a_m_frac * r_E``.

    The fractional convention can be signed.  In particular, with an aligned
    ``m=4`` phase, positive/negative values represent disky/boxy contours.
    """

    param_names = [
        "m", "a_m", "a_m_frac", "varphi_m", "q", "phi_ref", "center_x", "center_y", "r_E",
    ]
    lower_limit_default = {
        "m": 1, "a_m": 0, "a_m_frac": -1, "varphi_m": -np.pi, "q": 0.001,
        "phi_ref": -np.pi, "center_x": -100, "center_y": -100, "r_E": 1e-6,
    }
    upper_limit_default = {
        "m": 4, "a_m": 100, "a_m_frac": 1, "varphi_m": np.pi, "q": 1.0,
        "phi_ref": np.pi, "center_x": 100, "center_y": 100, "r_E": 100,
    }
    fixed_default = {
        "m": True, "a_m": False, "a_m_frac": False, "varphi_m": False, "q": False,
        "phi_ref": False, "center_x": False, "center_y": False, "r_E": True,
    }

    @staticmethod
    def _circular_potential(radius, angle, m, a_m, varphi_m, r_E):
        radius = jnp.maximum(radius, 1e-6)
        m_one = radius * jnp.log(radius / r_E) * a_m / 2 * jnp.cos(angle - varphi_m)
        m_other = radius * a_m / (1 - m**2) * jnp.cos(m * (angle - varphi_m))
        return jnp.where(m == 1, m_one, m_other)

    @staticmethod
    def function(
        x, y, m, a_m=None, a_m_frac=None, varphi_m=0.0, q=1.0,
        phi_ref=0.0, center_x=0.0, center_y=0.0, r_E=1.0,
    ):
        """Return the exact elliptical-multipole potential in arcsec squared."""
        if (a_m is None) == (a_m_frac is None):
            raise ValueError(
                "ELL_MPPL requires exactly one of 'a_m' (arcsec) or "
                "'a_m_frac' (dimensionless)."
            )
        if a_m_frac is not None:
            a_m = a_m_frac * r_E
        x_shifted, y_shifted = x - center_x, y - center_y
        radius = jnp.maximum(jnp.hypot(x_shifted, y_shifted), 1e-6)
        angle = jnp.arctan2(y_shifted, x_shifted)
        local_angle = angle - phi_ref
        near_circular = jnp.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8

        def m_one(_):
            phase_cos, phase_sin = jnp.cos(m * varphi_m), jnp.sin(m * varphi_m)
            lambda_m = 2 / (1 + q)
            first = radius * _f_m1(local_angle, q) + lambda_m / 2 * radius * jnp.log(radius / r_E) * jnp.cos(local_angle)
            second_angle = local_angle + jnp.pi / 2
            second = radius * _f_m1(second_angle, 1 / q) + (2 / (1 + 1 / q)) / 2 * radius * jnp.log(radius / r_E) * jnp.cos(second_angle)
            return a_m * jnp.sqrt(q) * (phase_cos * first - phase_sin * second / q)

        def m_three(_):
            phase_cos, phase_sin = jnp.cos(m * varphi_m), jnp.sin(m * varphi_m)
            lambda_m = -2 * (1 - q) / (1 + q) ** 2
            first = radius * _f_m3(local_angle, q) + lambda_m / 2 * radius * jnp.log(radius / r_E) * jnp.cos(local_angle)
            inverse_q = 1 / q
            lambda_inverse = -2 * (1 - inverse_q) / (1 + inverse_q) ** 2
            second_angle = local_angle + jnp.pi / 2
            second = radius * _f_m3(second_angle, inverse_q) + lambda_inverse / 2 * radius * jnp.log(radius / r_E) * jnp.cos(second_angle)
            return a_m * jnp.sqrt(q) * (phase_cos * first + phase_sin * second / q)

        def m_four(_):
            phase_cos, phase_sin = jnp.cos(m * varphi_m), jnp.sin(m * varphi_m)
            return a_m * jnp.sqrt(q) * radius * (
                _f_m4_1(local_angle, q) * phase_cos + _f_m4_2(local_angle, q) * phase_sin
            )

        def invalid(_):
            return jnp.full_like(radius, jnp.nan)

        case = jnp.where(m == 1, 0, jnp.where(m == 3, 1, jnp.where(m == 4, 2, 3)))

        return jax.lax.cond(
            near_circular,
            # At q=1 the reference ellipse has no intrinsic PA.  Preserve
            # the continuous q->1 limit by converting the relative eccentric
            # anomaly into its global polar phase before using the circular
            # solution.
            lambda _: ELLMPPL._circular_potential(
                radius, angle, m, a_m, varphi_m + phi_ref, r_E
            ),
            lambda _: jax.lax.switch(case, (m_one, m_three, m_four, invalid), operand=None),
            operand=None,
        )

    @staticmethod
    def _gradient_at_point(x, y, **kwargs):
        return jnp.array(jax.grad(ELLMPPL.function, argnums=(0, 1))(x, y, **kwargs))

    @staticmethod
    def derivatives(x, y, **kwargs):
        gradient = jnp.vectorize(
            lambda x_value, y_value: ELLMPPL._gradient_at_point(x_value, y_value, **kwargs),
            signature="(),()->(i)",
        )(x, y)
        return gradient[..., 0], gradient[..., 1]

    @staticmethod
    def _hessian_at_point(x, y, **kwargs):
        return jnp.array(jax.hessian(ELLMPPL.function, argnums=(0, 1))(x, y, **kwargs))

    @staticmethod
    def hessian(x, y, **kwargs):
        hessian = jnp.vectorize(
            lambda x_value, y_value: ELLMPPL._hessian_at_point(x_value, y_value, **kwargs),
            signature="(),()->(i,i)",
        )(x, y)
        return hessian[..., 0, 0], hessian[..., 1, 1], hessian[..., 0, 1]


class EPLM1M3M4:
    """JAXtronomy's EPL plus elliptical ``m=1,3,4`` multipoles.

    This is the wrapper equivalent of JAXtronomy's
    ``EPL_MULTIPOLE_M1M3M4_ELL``.  The amplitudes ``a1_a``, ``a3_a``, and
    ``a4_a`` are dimensionless and each becomes the physical elliptical-
    multipole amplitude ``a_m = a*_a * theta_E``.  The delta phases are
    eccentric anomalies relative to the EPL major axis, rather than sky PA.
    """

    param_names = [
        "theta_E", "gamma", "e1", "e2", "center_x", "center_y",
        "a1_a", "delta_phi_m1", "a3_a", "delta_phi_m3", "a4_a", "delta_phi_m4",
    ]
    lower_limit_default = {
        "theta_E": 0.0, "gamma": 1.5, "e1": -0.5, "e2": -0.5,
        "center_x": -100, "center_y": -100, "a1_a": -0.2,
        "delta_phi_m1": -np.pi, "a3_a": -0.2, "delta_phi_m3": -np.pi / 6,
        "a4_a": -0.2, "delta_phi_m4": -np.pi / 8,
    }
    upper_limit_default = {
        "theta_E": 100.0, "gamma": 2.5, "e1": 0.5, "e2": 0.5,
        "center_x": 100, "center_y": 100, "a1_a": 0.2,
        "delta_phi_m1": np.pi, "a3_a": 0.2, "delta_phi_m3": np.pi / 6,
        "a4_a": 0.2, "delta_phi_m4": np.pi / 8,
    }
    fixed_default = {
        "theta_E": False, "gamma": False, "e1": False, "e2": False,
        "center_x": False, "center_y": False, "a1_a": False,
        "delta_phi_m1": False, "a3_a": False, "delta_phi_m3": False,
        "a4_a": False, "delta_phi_m4": False,
    }

    @staticmethod
    @jax.jit
    def _multipole_kwargs(theta_E, e1, e2, center_x, center_y, a1_a, delta_phi_m1,
                          a3_a, delta_phi_m3, a4_a, delta_phi_m4):
        ellipticity = jnp.sqrt(e1**2 + e2**2)
        q = (1.0 - ellipticity) / (1.0 + ellipticity)
        phi_ref = 0.5 * jnp.arctan2(e2, e1)
        shared = dict(q=q, phi_ref=phi_ref, center_x=center_x, center_y=center_y, r_E=theta_E)
        return (
            dict(m=1, a_m=a1_a * theta_E, varphi_m=delta_phi_m1, **shared),
            dict(m=3, a_m=a3_a * theta_E, varphi_m=delta_phi_m3, **shared),
            dict(m=4, a_m=a4_a * theta_E, varphi_m=delta_phi_m4, **shared),
        )

    @staticmethod
    @jax.jit
    def function(x, y, theta_E, gamma, e1, e2, a1_a, delta_phi_m1, a3_a,
                 delta_phi_m3, a4_a, delta_phi_m4, center_x=0.0, center_y=0.0):
        from herculens.MassModel.Profiles.epl import EPL
        epl = EPL().function(x, y, theta_E, e1, e2, gamma, center_x, center_y)
        m1, m3, m4 = EPLM1M3M4._multipole_kwargs(
            theta_E, e1, e2, center_x, center_y, a1_a, delta_phi_m1,
            a3_a, delta_phi_m3, a4_a, delta_phi_m4,
        )
        multipole_potential = jnp.zeros_like(epl)
        for multipole in (m1, m3, m4):
            value = _JAXtronomyEllipticalMultipole.function(x, y, **multipole)
            multipole_potential = multipole_potential + jnp.where(
                multipole["a_m"] == 0, jnp.zeros_like(value), value
            )
        return epl + multipole_potential

    @staticmethod
    @jax.jit
    def derivatives(x, y, theta_E, gamma, e1, e2, a1_a, delta_phi_m1, a3_a,
                    delta_phi_m3, a4_a, delta_phi_m4, center_x=0.0, center_y=0.0):
        from herculens.MassModel.Profiles.epl import EPL
        alpha_x, alpha_y = EPL().derivatives(x, y, theta_E, e1, e2, gamma, center_x, center_y)
        m1, m3, m4 = EPLM1M3M4._multipole_kwargs(
            theta_E, e1, e2, center_x, center_y, a1_a, delta_phi_m1,
            a3_a, delta_phi_m3, a4_a, delta_phi_m4,
        )
        for multipole in (m1, m3, m4):
            dx, dy = _JAXtronomyEllipticalMultipole.derivatives(x, y, **multipole)
            dx = jnp.where(multipole["a_m"] == 0, jnp.zeros_like(dx), dx)
            dy = jnp.where(multipole["a_m"] == 0, jnp.zeros_like(dy), dy)
            alpha_x, alpha_y = alpha_x + dx, alpha_y + dy
        return alpha_x, alpha_y

    @staticmethod
    @jax.jit
    def hessian(x, y, theta_E, gamma, e1, e2, a1_a, delta_phi_m1, a3_a,
                delta_phi_m3, a4_a, delta_phi_m4, center_x=0.0, center_y=0.0):
        from herculens.MassModel.Profiles.epl import EPL
        f_xx, f_yy, f_xy = EPL().hessian(x, y, theta_E, e1, e2, gamma, center_x, center_y)
        m1, m3, m4 = EPLM1M3M4._multipole_kwargs(
            theta_E, e1, e2, center_x, center_y, a1_a, delta_phi_m1,
            a3_a, delta_phi_m3, a4_a, delta_phi_m4,
        )
        for multipole in (m1, m3, m4):
            dx_x, dx_y, _, dy_y = _JAXtronomyEllipticalMultipole.hessian(x, y, **multipole)
            dx_x = jnp.where(multipole["a_m"] == 0, jnp.zeros_like(dx_x), dx_x)
            dy_y = jnp.where(multipole["a_m"] == 0, jnp.zeros_like(dy_y), dy_y)
            dx_y = jnp.where(multipole["a_m"] == 0, jnp.zeros_like(dx_y), dx_y)
            f_xx, f_yy, f_xy = f_xx + dx_x, f_yy + dy_y, f_xy + dx_y
        return f_xx, f_yy, f_xy
