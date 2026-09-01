"""Ray-guided transformed-uniform (RTU) source-grid coordinates.

This is a compact JAX implementation of the transform in Enzi et al. (2026):
source-plane rays inside an image-plane arc mask define two smooth marginal
CDFs.  Pixelated sources remain uniform on ``[0, 1]²`` and are evaluated at
the transformed ray positions, giving adaptive physical source pixels while
preserving FFT Gaussian-process priors.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _interp1d(x, xp, fp):
    """Differentiable piecewise-linear interpolation with bounded indices."""
    index = jnp.clip(
        jnp.searchsorted(xp, x, side="right", method="scan_unrolled"),
        1,
        xp.shape[0] - 1,
    )
    left = index - 1
    dx = xp[index] - xp[left]
    fraction = (x - xp[left]) / jnp.where(jnp.abs(dx) > 1e-14, dx, 1.0)
    return fp[left] + fraction * (fp[index] - fp[left])


def _cubic_hermite(x_knots, y_knots, slopes, x):
    """Evaluate the explicit inverse-CDF cubic spline used by the RTU grid."""
    index = jnp.clip(
        jnp.searchsorted(x_knots, x, side="right", method="scan_unrolled"),
        1,
        x_knots.shape[0] - 1,
    )
    left = index - 1
    width = x_knots[index] - x_knots[left]
    safe_width = jnp.where(jnp.abs(width) > 1e-14, width, 1.0)
    t = (x - x_knots[left]) / safe_width
    t2, t3 = t * t, t * t * t
    h00, h10 = 2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t
    h01, h11 = -2.0 * t3 + 3.0 * t2, t3 - t2
    return (
        y_knots[left] * h00 + y_knots[index] * h01
        + safe_width * (slopes[left] * h10 + slopes[index] * h11)
    )


def _fit_smooth_inverse_cdf(standardized_rays, degree: int):
    """Return polynomial/spline data for both marginal RTU transforms."""
    n_rays = standardized_rays.shape[0]
    fractions = jnp.arange(1, n_rays + 1, dtype=standardized_rays.dtype) / (n_rays + 1)
    fractions = jnp.stack((fractions, fractions), axis=1)
    sorted_rays = jnp.sort(standardized_rays, axis=0)

    node_count = degree + 1
    nodes = (
        jnp.cos((2 * jnp.arange(node_count) + 1) * jnp.pi / (2 * node_count))[::-1] + 1.0
    ) / 2.0
    nodes = jax.lax.stop_gradient(nodes)
    node_matrix = jnp.stack((nodes, nodes), axis=1)
    quantiles = jax.vmap(_interp1d, in_axes=(None, 1, 1), out_axes=1)(
        nodes, fractions, sorted_rays,
    )
    weights = jax.vmap(jnp.gradient, in_axes=(1, 1), out_axes=1)(node_matrix, quantiles)
    weights = jnp.maximum(jnp.abs(weights), 1e-10)
    coefficients = jax.vmap(
        jnp.polyfit, in_axes=(1, 1, None, None, None, 1), out_axes=1,
    )(node_matrix, quantiles, degree, None, False, weights)

    # The polynomial describes the quantile function Q(u).  Evaluate it at
    # fixed probability knots and use an explicit cubic-Hermite approximation
    # to its inverse CDF, exactly to avoid an iterative inverse in reverse AD.
    probability_knots = jnp.vstack((jnp.zeros(2), node_matrix, jnp.ones(2)))
    x_knots = jax.vmap(jnp.polyval, in_axes=(1, 1), out_axes=1)(
        coefficients, probability_knots,
    )
    derivative_coefficients = jax.vmap(jnp.polyder, in_axes=1, out_axes=1)(coefficients)
    derivative_values = jax.vmap(jnp.polyval, in_axes=(1, 1), out_axes=1)(
        derivative_coefficients, probability_knots,
    )
    slopes = 1.0 / jnp.where(jnp.abs(derivative_values) > 1e-10, derivative_values, 1e-10)
    return (
        x_knots, probability_knots, slopes, derivative_values,
        sorted_rays[0], sorted_rays[-1],
    )


def ray_transformed_uniform_coordinates(
    mask_rays_x,
    mask_rays_y,
    evaluate_x,
    evaluate_y,
    *,
    polynomial_order: int = 11,
):
    """Map physical source coordinates to RTU coordinates in ``[0, 1]²``.

    ``mask_rays_*`` are ray-shooting positions of *native image-grid* pixels
    selected by the source-arc mask.  ``evaluate_*`` may be supersampled rays.
    The standardization makes the mapping invariant to translation and global
    source-plane scaling, as in Enzi et al. (2026, Appendix C).
    """
    rays = jnp.stack((jnp.ravel(mask_rays_x), jnp.ravel(mask_rays_y)), axis=1)
    points = jnp.stack((jnp.ravel(evaluate_x), jnp.ravel(evaluate_y)), axis=1)
    mean = jnp.mean(rays, axis=0)
    scale = jnp.maximum(jnp.min(jnp.std(rays, axis=0)), 1e-12)
    rays_standardized = (rays - mean) / scale
    points_standardized = (points - mean) / scale
    x_knots, probability_knots, slopes, _derivatives, lower, upper = _fit_smooth_inverse_cdf(
        rays_standardized, int(polynomial_order),
    )
    transformed = jax.vmap(
        _cubic_hermite, in_axes=(1, 1, 1, 1), out_axes=1,
    )(x_knots, probability_knots, slopes, points_standardized)
    transformed = jnp.where(points_standardized <= lower, 0.0, transformed)
    transformed = jnp.where(points_standardized >= upper, 1.0, transformed)
    transformed = jnp.clip(transformed, 0.0, 1.0)
    return transformed[:, 0].reshape(jnp.shape(evaluate_x)), transformed[:, 1].reshape(jnp.shape(evaluate_y))


def ray_transformed_uniform_physical_grid(
    mask_rays_x,
    mask_rays_y,
    *,
    nx: int,
    ny: int,
    polynomial_order: int = 11,
):
    """Return physical RTU source-cell corners with shape ``(ny+1, nx+1)``.

    It inverts the fitted marginal RTU CDFs at fixed uniform-coordinate cell
    edges.  It is intended for result storage and visualization, rather than
    the differentiable likelihood evaluation.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must both be positive.")
    rays = jnp.stack((jnp.ravel(mask_rays_x), jnp.ravel(mask_rays_y)), axis=1)
    mean = jnp.mean(rays, axis=0)
    scale = jnp.maximum(jnp.min(jnp.std(rays, axis=0)), 1e-12)
    standardized = (rays - mean) / scale
    x_knots, probability_knots, _slopes, derivatives, _lower, _upper = _fit_smooth_inverse_cdf(
        standardized, int(polynomial_order),
    )
    u = jnp.linspace(0.0, 1.0, int(nx) + 1, dtype=rays.dtype)
    v = jnp.linspace(0.0, 1.0, int(ny) + 1, dtype=rays.dtype)
    x_axis = _cubic_hermite(probability_knots[:, 0], x_knots[:, 0], derivatives[:, 0], u)
    y_axis = _cubic_hermite(probability_knots[:, 1], x_knots[:, 1], derivatives[:, 1], v)
    x_axis = x_axis * scale + mean[0]
    y_axis = y_axis * scale + mean[1]
    return jnp.meshgrid(x_axis, y_axis, indexing="xy")
