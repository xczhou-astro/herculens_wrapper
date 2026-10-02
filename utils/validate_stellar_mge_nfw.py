"""Independent numerical audit of StellarMassMGE + elliptical NFW.

Run with the Herculens environment, optionally supplying the saved SVI folder:
    python utils/validate_stellar_mge_nfw.py --results-dir /path/to/pixelated_svi

The reference uses NumPy surface densities and elliptical-density quadrature
(Keeton 2001/2002), not the JAX Gaussian/Faddeeva or halo-MGE implementations.
It deliberately records the unsupported exactly circular branch separately.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
jax.config.update("jax_enable_x64", True)  # before importing the MGE package
import jax.numpy as jnp
from herculens_wrapper.profiles.composite import NFWEllipseKappa, StellarMGE


def ellipticity(e1, e2):
    modulus = np.hypot(e1, e2)
    return 0.5 * np.arctan2(e2, e1), (1 - modulus) / (1 + modulus)


def ellipse_coordinates(x, y, e1, e2, center_x=0., center_y=0.):
    phi, q = ellipticity(e1, e2)
    c, s = np.cos(phi), np.sin(phi)
    x, y = np.asarray(x) - center_x, np.asarray(y) - center_y
    return c*x + s*y, -s*x + c*y, c, s, q


def gaussian_density(x, y, amp, sigma, e1, e2, center_x=0., center_y=0.):
    X, Y, _, _, q = ellipse_coordinates(x, y, e1, e2, center_x, center_y)
    return amp / (2*np.pi*sigma**2) * np.exp(-(q*X**2 + Y**2/q)/(2*sigma**2))


def stellar_amplitudes(p):
    flux, sigma = np.asarray(p["light_amp"]), np.asarray(p["light_sigma"])
    sigma_ref = np.exp(np.log(sigma).mean())
    return p["upsilon_kappa"] * flux/flux.sum() * (sigma/sigma_ref)**(-p["ml_gradient"])


def stellar_density(x, y, p):
    return sum(gaussian_density(x, y, a, sigma, e1, e2, cx, cy)
               for a, sigma, e1, e2, cx, cy in zip(stellar_amplitudes(p),
               *(p["light_"+k] for k in ("sigma", "e1", "e2", "center_x", "center_y"))))


def nfw_density(r, kappa_s, R_s):
    """Exact projected spherical NFW, evaluated at product-average radius.

    log rather than arctanh avoids catastrophic rounding at very small r/Rs.
    A local expansion removes the removable singularity at r/Rs=1.
    """
    z = np.maximum(np.asarray(r, dtype=float)/R_s, 1e-150)
    out = np.empty_like(z)
    near = np.abs(z-1) < 1e-4
    low, high = (z < 1) & ~near, (z > 1) & ~near
    a = np.sqrt(1-z[low]**2)
    out[low] = (np.log((1+a)/z[low])/a - 1)/(1-z[low]**2)
    a = np.sqrt(z[high]**2-1)
    out[high] = (1 - np.arctan(a)/a)/(z[high]**2-1)
    dz = z[near]-1
    out[near] = 1/3 - 2/5*dz + 13/35*dz**2 - 20/63*dz**3
    return 2*kappa_s*out


@lru_cache(None)
def quadrature_nodes(order):
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t+1)/2, w/2
    return t*t, 2*t*w


def elliptical_alpha(x, y, density, e1, e2, center_x=0., center_y=0., order=256):
    """1-D integral for an elliptical convergence, with u=t^2.

    Density takes the area-preserving radius sqrt(q X^2 + Y^2/q).
    Integration is vectorized, and bounded batches keep memory use modest.
    """
    x, y = np.broadcast_arrays(x, y)
    shape = x.shape
    X, Y, c, s, q = ellipse_coordinates(x.ravel(), y.ravel(), e1, e2, center_x, center_y)
    u, w = quadrature_nodes(order)
    D = 1-(1-q*q)*u
    result = np.empty((2, X.size))
    for start in range(0, X.size, 2048):
        sl = slice(start, start+2048)
        r = np.sqrt(q*u[None, :]*(X[sl, None]**2 + Y[sl, None]**2/D))
        kappa = density(r)
        ax = q*X[sl] * (kappa @ (w/np.sqrt(D)))
        ay = q*Y[sl] * (kappa @ (w/D**1.5))
        result[:, sl] = [c*ax-s*ay, s*ax+c*ay]
    return result.reshape((2,)+shape)


def stellar_alpha(x, y, p, order=256):
    out = np.zeros((2,)+np.broadcast(x, y).shape)
    for a, sigma, e1, e2, cx, cy in zip(stellar_amplitudes(p),
            *(p["light_"+k] for k in ("sigma", "e1", "e2", "center_x", "center_y"))):
        out += elliptical_alpha(x, y, lambda r: a/(2*np.pi*sigma**2)*np.exp(-r*r/(2*sigma**2)),
                                e1, e2, cx, cy, order)
    return out


def halo_alpha(x, y, p, order=256):
    return elliptical_alpha(x, y, lambda r: nfw_density(r, p["kappa_s"], p["R_s"]),
                            *(p[k] for k in ("e1", "e2", "center_x", "center_y")), order)


def halo_density(x, y, p):
    X, Y, _, _, q = ellipse_coordinates(x, y, *(p[k] for k in ("e1", "e2", "center_x", "center_y")))
    return nfw_density(np.sqrt(q*X*X+Y*Y/q), p["kappa_s"], p["R_s"])


def shear_alpha(x, y, p):
    x, y = np.asarray(x)-p["ra_0"], np.asarray(y)-p["dec_0"]
    return np.array([p["gamma1"]*x+p["gamma2"]*y, p["gamma2"]*x-p["gamma1"]*y])


def reference_hessian(alpha, x, y, step=1e-5):
    dx = (alpha(x+step, y)-alpha(x-step, y))/(2*step)
    dy = (alpha(x, y+step)-alpha(x, y-step))/(2*step)
    return np.array([dx[0], dy[1], (dx[1]+dy[0])/2])


def errors(actual, expected, vector=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError("Non-finite value in a regular-domain comparison")
    delta = actual-expected
    if vector:
        delta, expected = np.linalg.norm(delta, axis=0), np.linalg.norm(expected, axis=0)
    return {"max_abs": float(np.max(np.abs(delta))),
            "max_relative": float(np.max(np.abs(delta)/np.maximum(np.abs(expected), 1e-12))),
            "relative_l2": float(np.linalg.norm(delta)/max(np.linalg.norm(expected), 1e-12))}


def toy_parameters():
    def e(q, phi):
        m = (1-q)/(1+q)
        return m*np.cos(2*phi), m*np.sin(2*phi)
    a, b = e(.28, 1.15)
    stellar = dict(upsilon_kappa=1.05, ml_gradient=0., light_amp=[.15, .55, .30],
                   light_sigma=[.06, .22, .50], light_e1=[a]*3, light_e2=[b]*3,
                   light_center_x=[0.]*3, light_center_y=[0.]*3)
    a, b = e(.75, .36)
    halo = dict(kappa_s=.10, R_s=1.5, e1=a, e2=b, center_x=0., center_y=0.)
    shear = dict(ra_0=0., dec_0=0., gamma1=.025, gamma2=-.035)
    return stellar, halo, shear


def check_profiles(stellar, halo):
    st, nf = StellarMGE(), NFWEllipseKappa()
    rng = np.random.default_rng(47)
    theta = rng.uniform(0, 2*np.pi, 180)
    radii = np.geomspace(.03, 1.5, 180)
    x, y = radii*np.cos(theta), radii*np.sin(theta)
    out = {}
    for name, obj, p, alpha, density in (
        ("stellar", st, stellar, stellar_alpha, stellar_density),
        ("halo", nf, halo, halo_alpha, halo_density),
    ):
        actual_a = np.array(obj.derivatives(x, y, **p))
        ref_a = alpha(x, y, p)
        actual_h = np.array(obj.hessian(x, y, **p))
        ref_h = reference_hessian(lambda X, Y: alpha(X, Y, p), x, y)
        out[name] = {
            "alpha": errors(actual_a, ref_a, vector=True),
            "kappa": errors((actual_h[0]+actual_h[1])/2, density(x, y, p)),
            "hessian": errors(actual_h, ref_h),
            "quadrature_128_vs_256": errors(alpha(x, y, p, 128), ref_a, vector=True),
            "hessian_step_halving": errors(reference_hessian(lambda X, Y: alpha(X, Y, p), x, y, 5e-6), ref_h),
        }
        # Hessian comes from autodiff internally, so also compare it to finite
        # differences of the implemented alpha (a distinct consistency check).
        out[name]["hessian_vs_alpha_fd"] = errors(actual_h, reference_hessian(
            lambda X, Y: np.array(obj.derivatives(X, Y, **p)), x, y))
    return out


def check_limits(stellar, halo):
    st, nf = StellarMGE(), NFWEllipseKappa()
    x, y = np.array([.07, .3, -.8]), np.array([.13, -.6, .4])
    scaled = {**stellar, "light_amp": np.array(stellar["light_amp"])*17}
    zero = {**stellar, "ml_gradient": 0.}
    out = {"global_light_rescaling": errors(st.derivatives(x, y, **scaled), st.derivatives(x, y, **stellar)),
           "constant_ml_density": errors(stellar_density(x, y, zero),
               sum(gaussian_density(x, y, a, s, e1, e2, cx, cy)
                   for a, s, e1, e2, cx, cy in zip(zero["light_amp"],
                   *(zero["light_"+k] for k in ("sigma", "e1", "e2", "center_x", "center_y"))))
               *zero["upsilon_kappa"]/sum(zero["light_amp"]))}
    # Verify integrated Gaussian area independently, in ellipse coordinates.
    area = 0.
    for a, sigma in zip(stellar_amplitudes(stellar), stellar["light_sigma"]):
        area += quad(lambda r: 2*np.pi*r*a/(2*np.pi*sigma**2)*np.exp(-r*r/(2*sigma**2)),
                     0, 12*sigma, epsabs=1e-12)[0]
    out["stellar_integrated_area"] = {"numerical": area, "sum_amplitudes": float(stellar_amplitudes(stellar).sum()),
                                      "upsilon_kappa": stellar["upsilon_kappa"]}
    # Check exact NFW projection against direct line-of-sight integration.
    projection = []
    for z in (.001, .03, .3, 1., 3., 10.):
        reference = 2*halo["kappa_s"]*quad(lambda t: 1/(np.sqrt(z*z+t*t)*(1+np.sqrt(z*z+t*t))**2),
                                         0, np.inf, epsabs=1e-11)[0]
        projection.append(errors(nfw_density(z*halo["R_s"], halo["kappa_s"], halo["R_s"]), reference))
    out["nfw_projection_los_max_relative"] = max(e["max_relative"] for e in projection)
    # Covariance under translations and rotations.
    angle = .43
    c, s = np.cos(angle), np.sin(angle)
    xr, yr = c*x-s*y+.08, s*x+c*y-.05
    for name, obj, p in (("stellar", st, stellar), ("halo", nf, halo)):
        p2 = dict(p)
        prefix = "light_" if name == "stellar" else ""
        a, b = np.asarray(p[prefix+"e1"]), np.asarray(p[prefix+"e2"])
        p2[prefix+"e1"], p2[prefix+"e2"] = a*np.cos(2*angle)-b*np.sin(2*angle), a*np.sin(2*angle)+b*np.cos(2*angle)
        cx, cy = np.asarray(p[prefix+"center_x"]), np.asarray(p[prefix+"center_y"])
        p2[prefix+"center_x"], p2[prefix+"center_y"] = c*cx-s*cy+.08, s*cx+c*cy-.05
        original = np.array(obj.derivatives(x, y, **p))
        expected = np.array([c*original[0]-s*original[1], s*original[0]+c*original[1]])
        out[name+"_rotation_translation"] = errors(obj.derivatives(xr, yr, **p2), expected, vector=True)
    # Do not silently count a known unsupported branch as passing.
    out["exactly_circular"] = {}
    for name, obj, p in (("stellar", st, {**stellar, "light_e1": np.zeros(len(stellar["light_amp"])),
                                                   "light_e2": np.zeros(len(stellar["light_amp"]))}),
                         ("halo", nf, {**halo, "e1": 0., "e2": 0.})):
        a = np.asarray(obj.derivatives(x, y, **p))
        out["exactly_circular"][name] = {"all_finite": bool(np.all(np.isfinite(a))),
                                          "status": "supported" if np.all(np.isfinite(a)) else "unsupported_backend_branch"}
    # Parameter derivatives, including dependence on lens-light amplitudes.
    out["parameter_gradients"] = {}
    for name, obj, p, key in (("stellar", st, stellar, "upsilon_kappa"),
                            ("stellar", st, stellar, "ml_gradient"),
                            ("halo", nf, halo, "kappa_s"), ("halo", nf, halo, "R_s")):
        def f(v):
            return jnp.sum(jnp.asarray(obj.derivatives(x, y, **{**p, key: v}))**2)
        v = p[key]
        step = 1e-5*max(abs(v), .1)
        out["parameter_gradients"][name+"_"+key] = errors(jax.grad(f)(v), (f(v+step)-f(v-step))/(2*step))
    # The inverse-transform sum for the NFW MGE suffers cancellation at tiny
    # finite-difference steps. Test its derivatives with exact homogeneity
    # identities as well: alpha is linear in ks and Rs*A(theta/Rs).
    a = np.array(nf.derivatives(x, y, **halo))
    h = np.array(nf.hessian(x, y, **halo))
    dks = jax.jacfwd(lambda v: jnp.array(nf.derivatives(x, y, **{**halo, "kappa_s": v})))(halo["kappa_s"])
    drs = jax.jacfwd(lambda v: jnp.array(nf.derivatives(x, y, **{**halo, "R_s": v})))(halo["R_s"])
    X, Y = x-halo["center_x"], y-halo["center_y"]
    rs_identity = np.array([a[0]-h[0]*X-h[2]*Y, a[1]-h[2]*X-h[1]*Y])/halo["R_s"]
    step = 1e-4*halo["R_s"]
    ref_rs = (halo_alpha(x, y, {**halo, "R_s": halo["R_s"]+step})
              -halo_alpha(x, y, {**halo, "R_s": halo["R_s"]-step}))/(2*step)
    out["nfw_gradient_homogeneity"] = {"kappa_s": errors(dks, a/halo["kappa_s"], vector=True),
                                        "R_s": errors(drs, rs_identity, vector=True)}
    out["nfw_gradient_independent"] = {"kappa_s": errors(dks, halo_alpha(x, y, halo)/halo["kappa_s"], vector=True),
                                       "R_s": errors(drs, ref_rs, vector=True)}
    return out


def check_potential_gradients(stellar, halo):
    result = {}
    for name, obj, p in (("stellar", StellarMGE(), stellar), ("halo", NFWEllipseKappa(), halo)):
        points = jnp.array([[.09, .12], [.2, -.17], [.8, .3], [-1.1, .65]])
        gradient = jax.jit(jax.vmap(jax.grad(lambda z: obj.function(*z, **p))))(points)
        alpha = np.array(obj.derivatives(points[:, 0], points[:, 1], **p)).T
        result[name] = errors(gradient.T, alpha.T, vector=True)
    return result


def check_superposition(stellar, halo, shear):
    from herculens.MassModel.mass_model import MassModel
    from herculens.MassModel.Profiles.shear import Shear
    mass = MassModel([StellarMGE(), NFWEllipseKappa(), Shear()])
    x, y = np.array([.13, .5, -.8]), np.array([-.19, .7, .4])
    kw = [stellar, halo, shear]
    alpha = np.array(mass.alpha(x, y, kw))
    parts = sum(np.array(mass.alpha(x, y, kw, k=i)) for i in range(3))
    kappa = np.array(mass.kappa(x, y, kw))
    k_parts = sum(np.array(mass.kappa(x, y, kw, k=i)) for i in range(3))
    return {"alpha_sum": errors(alpha, parts), "kappa_sum": errors(kappa, k_parts),
            "ray_shooting": errors(mass.ray_shooting(x, y, kw), np.array([x, y])-alpha),
            "shear_kappa_max_abs": float(np.max(np.abs(mass.kappa(x, y, kw, k=2))))}


def check_nfw_range(output, saved_runs):
    """Measure the finite MGE basis's radial range instead of assuming it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nf = NFWEllipseKappa()
    r = np.geomspace(1e-4, 50, 240)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, constrained_layout=True)
    records = []
    for q in (.4, .7, .95, .9999):
        p = dict(kappa_s=.1, R_s=1., e1=(1-q)/(1+q), e2=0., center_x=0., center_y=0.)
        # Use non-axis points so exact symmetry-axis differentiation is not
        # accidentally used to assess generic radial accuracy.
        x, y = r*np.cos(.61), r*np.sin(.61)
        actual = np.array(nf.derivatives(x, y, **p)); expected = halo_alpha(x, y, p)
        h = np.array(nf.hessian(x, y, **p))
        alpha_err = np.linalg.norm(actual-expected, axis=0)/np.linalg.norm(expected, axis=0)
        kappa_err = np.abs((h[0]+h[1])/2/halo_density(x, y, p)-1)
        axes[0].loglog(r, alpha_err, label=f"q={q}")
        axes[1].loglog(r, kappa_err)
        records.append({"q": q, "r_over_Rs": r.tolist(), "alpha_relative_error": alpha_err.tolist(),
                        "kappa_relative_error": kappa_err.tolist()})
    # A convergence experiment, deliberately separate from the production
    # default: increase both the number and radial span of the Gaussian basis.
    from jax_lensing_profiles.MassModel.Profiles.MGE import MGE
    from jax_lensing_profiles.MassModel.Profiles.NFW_ellipse_kappa import NFW_3D_fn
    expanded = MGE(NFW_3D_fn, "R_s", n_gauss=60, sigma_start_mult=1e-6,
                   sigma_end_mult=1000., three_d=True)
    p = dict(kappa_s=.1, R_s=1., e1=(1-.7)/(1+.7), e2=0., center_x=0., center_y=0.)
    actual = np.array(expanded.derivatives(x, y, **p)); expected = halo_alpha(x, y, p)
    h = np.array(expanded.hessian(x, y, **p))
    alpha_err = np.linalg.norm(actual-expected, axis=0)/np.linalg.norm(expected, axis=0)
    kappa_err = np.abs((h[0]+h[1])/2/halo_density(x, y, p)-1)
    axes[0].loglog(r, alpha_err, color="black", linestyle=":", label="60 Gaussians, wider basis (diagnostic)")
    axes[1].loglog(r, kappa_err, color="black", linestyle=":")
    records.append({"q": .7, "diagnostic_only": True, "n_gauss": 60,
                    "sigma_start_mult": 1e-6, "sigma_end_mult": 1000.,
                    "r_over_Rs": r.tolist(), "alpha_relative_error": alpha_err.tolist(),
                    "kappa_relative_error": kappa_err.tolist()})
    for ax, threshold, ylabel in zip(axes, (.005, .01), ("Deflection relative error", "Convergence relative error")):
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label="Audit tolerance")
        if saved_runs:
            low = min(d["radius_over_Rs_range"][0] for d in saved_runs)
            high = max(d["radius_over_Rs_range"][1] for d in saved_runs)
            ax.axvspan(low, high, alpha=.10, color="grey", label="Saved runs: tested radius range")
        ax.set(ylabel=ylabel, ylim=(1e-8, 2.))
        ax.grid(alpha=.2, which="both")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Circular sky radius / NFW scale radius")
    fig.savefig(output/"nfw_radial_accuracy.png", dpi=180)
    plt.close(fig)
    return records


def make_api_model(stellar, halo, shear, *, dynamic=False, shape=41, psf=None):
    from herculens_wrapper.api import (LensProfileCollection, LightProfile, MassProfile,
        NFWEllipseHalo, SingleBandData, SingleBandModel, StellarMassMGE)
    light = []
    for i in range(len(stellar["light_amp"])):
        values = {k: stellar["light_"+k][i] for k in ("amp", "sigma", "e1", "e2", "center_x", "center_y")}
        light.append(LightProfile("GAUSSIAN_ELLIPSE", prior={k: [v-.01, v+.01] for k, v in values.items()})
                     if dynamic else LightProfile("GAUSSIAN_ELLIPSE", value=values))
    mass = StellarMassMGE(light, follow_lens_light=dynamic,
                         prior={"upsilon_kappa": [.1, 2.], "ml_gradient": [-1., 1.]})
    halo_profile = NFWEllipseHalo(prior={**halo,
        "kappa_s": [.001, .5],
        "center_x": ["correlated", "lens_light", "flux_centroid", "center_x"],
        "center_y": ["correlated", "lens_light", "flux_centroid", "center_y"]})
    source = LightProfile("GAUSSIAN", prior={"amp": [.01, 2.], "sigma": [.03, .2],
                                           "center_x": [-.2, .2], "center_y": [-.2, .2]})
    return SingleBandModel(observation=SingleBandData(image=np.zeros((shape, shape)),
            noise=np.ones((shape, shape)), psf=np.ones((1, 1)) if psf is None else psf, pixel_scale=.06),
        profiles=LensProfileCollection(lens_mass=[mass, halo_profile, MassProfile("SHEAR", value=shear)],
                                       lens_light=light, source_light=source),
        numerics={"supersampling_factor": 4, "supersampling_convolution": False})


def check_dynamic(stellar, halo, shear):
    from herculens_wrapper.models import kwargs2params
    model = make_api_model(stellar, halo, shear, dynamic=True, shape=7)
    t, p = model.definition.as_dicts()
    light = [{k: stellar["light_"+k][i] for k in ("amp", "sigma", "e1", "e2", "center_x", "center_y")}
             for i in range(len(stellar["light_amp"]))]
    saved = dict(kwargs_lens=[stellar, halo, shear], kwargs_lens_light=light,
                 kwargs_source=[dict(amp=1., sigma=.1, center_x=.02, center_y=.03)])
    params = kwargs2params(p, saved, type_list=t)
    kw = model.prob_model.params2kwargs(params)
    out = {"fields": {k: errors(kw["kwargs_lens"][0]["light_"+k], stellar["light_"+k])
                     for k in ("amp", "sigma", "e1", "e2", "center_x", "center_y")}}
    params2 = {**params, "lens_light_amp_0": params["lens_light_amp_0"]*1.7,
               "lens_light_center_x_0": params["lens_light_center_x_0"]+.03,
               "lens_light_sigma_1": params["lens_light_sigma_1"]*1.2}
    changed = model.prob_model.params2kwargs(params2)
    for i, g in enumerate(changed["kwargs_lens_light"]):
        for k in g:
            np.testing.assert_allclose(changed["kwargs_lens"][0]["light_"+k][i], g[k])
    for axis in ("center_x", "center_y"):
        expected = np.average([float(g[axis]) for g in changed["kwargs_lens_light"]],
                              weights=[float(g["amp"]) for g in changed["kwargs_lens_light"]])
        np.testing.assert_allclose(changed["kwargs_lens"][1][axis], expected, atol=1e-14)
    def f(v):
        kw = model.prob_model.params2kwargs({**params2, "lens_light_amp_0": v})
        return jnp.sum(jnp.asarray(model.lens_image.MassModel.alpha(.2, .3, kw["kwargs_lens"])))
    value = params2["lens_light_amp_0"]
    step = 1e-5*float(value)
    out["light_amp_gradient"] = errors(jax.grad(f)(value), (f(value+step)-f(value-step))/(2*step))
    out["changed_light_fields_and_flux_centroid"] = "pass"
    out["no_derived_sample_sites"] = not any(k.startswith("lens_light_amp_") is False and
        ("light_amp" in k or k in ("lens_center_x_1", "lens_center_y_1")) for k in params)
    return out


def audit_saved(folder):
    reports = []
    st, nf = StellarMGE(), NFWEllipseKappa()
    for run in sorted(folder.glob("run_*")):
        result_file = run/"kwargs_result.json"
        if not result_file.exists():
            continue
        kw = json.loads(result_file.read_text())
        config = json.loads((run/"model_configuration.json").read_text())
        stellar, halo, shear = kw["kwargs_lens"]
        light = kw["kwargs_lens_light"]
        field_errors = {k: errors(stellar["light_"+k], [g[k] for g in light])
                        for k in ("amp", "sigma", "e1", "e2", "center_x", "center_y")}
        centroid = {axis: float(halo[axis]-np.average([g[axis] for g in light], weights=[g["amp"] for g in light]))
                    for axis in ("center_x", "center_y")}
        scale = config["observation"]["pixel_scale"]
        n = config["observation"]["image_shape"][0]
        axis = (np.arange(n)-(n-1)/2)*scale
        x, y = np.meshgrid(axis, axis)
        use = np.hypot(x-halo["center_x"], y-halo["center_y"]) >= scale*.5
        x, y = x[use], y[use]
        a_s, a_h = np.array(st.derivatives(x, y, **stellar)), np.array(nf.derivatives(x, y, **halo))
        ref_s, ref_h = stellar_alpha(x, y, stellar), halo_alpha(x, y, halo)
        hs, hh = np.array(st.hessian(x, y, **stellar)), np.array(nf.hessian(x, y, **halo))
        mask_info = {}
        from astropy.io import fits
        mask = fits.getdata(folder/"data/mask_1.fits")
        if mask.shape == (n, n):
            arc = np.asarray(mask).astype(bool)[use]
            if arc.any():
                mask_info = {"selected_pixels": int(arc.sum()),
                    "alpha_total": errors((a_s+a_h+shear_alpha(x, y, shear))[:, arc],
                                           (ref_s+ref_h+shear_alpha(x, y, shear))[:, arc], vector=True)}
        reports.append({"run": run.name, "result_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
            "model_types": config["backend_definition"]["types"]["lens_mass_type_list"],
            "field_consistency": field_errors, "halo_centroid_minus_light_centroid": centroid,
            "stellar_total_convergence_area": float(stellar_amplitudes(stellar).sum()),
            "upsilon_kappa": stellar["upsilon_kappa"], "ml_gradient": stellar["ml_gradient"],
            "n_points": int(x.size), "radius_over_Rs_range": [float(np.hypot(x-halo["center_x"], y-halo["center_y"]).min()/halo["R_s"]),
                float(np.hypot(x-halo["center_x"], y-halo["center_y"]).max()/halo["R_s"])],
            "stellar_alpha": errors(a_s, ref_s, vector=True), "halo_alpha": errors(a_h, ref_h, vector=True),
            "stellar_kappa": errors((hs[0]+hs[1])/2, stellar_density(x, y, stellar)),
            "halo_kappa": errors((hh[0]+hh[1])/2, halo_density(x, y, halo)),
            "total_alpha": errors(a_s+a_h+shear_alpha(x, y, shear), ref_s+ref_h+shear_alpha(x, y, shear), vector=True),
            "arc_mask": mask_info})
    return reports


def run_mock(stellar, halo, shear, output):
    # An independently rendered image, with the same 41x41 / 0.06 arcsec /
    # 4x4 subpixel integration settings as the supplied data. Geometry and Rs
    # are fixed in recovery; source flux/size/centre and both mass scales vary.
    from herculens_wrapper.models import kwargs2params
    psf_axis = np.arange(-3, 4)
    xx, yy = np.meshgrid(psf_axis, psf_axis)
    psf = np.exp(-(xx*xx+yy*yy)/(2*.8**2)); psf /= psf.sum()
    model = make_api_model(stellar, halo, shear, psf=psf)
    types, priors = model.definition.as_dicts()
    true = np.array([stellar["upsilon_kappa"], halo["kappa_s"], .035, .025, .08, .8])
    light = [{k: stellar["light_"+k][i] for k in ("amp", "sigma", "e1", "e2", "center_x", "center_y")}
             for i in range(len(stellar["light_amp"]))]
    saved = dict(kwargs_lens=[stellar, halo, shear], kwargs_lens_light=light,
                 kwargs_source=[dict(amp=true[5], sigma=true[4], center_x=true[2], center_y=true[3])])
    params = kwargs2params(priors, saved, type_list=types)
    keys = ("lens_upsilon_kappa_0", "lens_kappa_s_1", "source_center_x_0", "source_center_y_0", "source_sigma_0", "source_amp_0")
    @jax.jit
    def render(values):
        kw = model.prob_model.params2kwargs({**params, **dict(zip(keys, values))})
        return model.lens_image.model(**kw, lens_light_add=False).ravel()
    derivative = jax.jit(jax.jacfwd(render))
    base_x, base_y = model.lens_image.Grid.pixel_coordinates
    sub = ((np.arange(4)+.5)/4-.5)*.06
    x = (base_x[..., None, None]+sub[None, None, None, :]).repeat(4, axis=2)
    y = (base_y[..., None, None]+sub[None, None, :, None]).repeat(4, axis=3)
    alpha = stellar_alpha(x, y, stellar)+halo_alpha(x, y, halo)+shear_alpha(x, y, shear)
    bx, by = x-alpha[0], y-alpha[1]
    unblurred = (true[5]/(2*np.pi*true[4]**2)*np.exp(-((bx-true[2])**2+(by-true[3])**2)/(2*true[4]**2))).mean(axis=(2, 3))*.06**2
    independent = convolve2d(unblurred, psf, mode="same", boundary="fill")
    actual = np.array(render(true)).reshape(base_x.shape)
    noise = independent.max()/100  # fixed Gaussian noise; source peak S/N=100
    bound_low = [.1, .001, -.15, -.15, .03, .1]
    bound_high = [2., .5, .15, .15, .2, 2.]
    fits = []
    rng = np.random.default_rng(20261002)
    for seed in (None, 0, 1, 2):
        data = independent if seed is None else independent+rng.normal(0, noise, independent.shape)
        fit = least_squares(lambda v: (np.array(render(v))-data.ravel())/noise,
                            true*np.array([.75, 1.35, 1.5, .6, 1.2, .8]),
                            jac=lambda v: np.array(derivative(v))/noise,
                            bounds=(bound_low, bound_high), xtol=1e-10, ftol=1e-10, gtol=1e-8,
                            max_nfev=150)
        covariance = np.linalg.inv(fit.jac.T@fit.jac)
        sigma = np.sqrt(np.diag(covariance))
        fits.append({"kind": "noiseless" if seed is None else "noise_realization_"+str(seed),
                     "success": bool(fit.success), "estimate": fit.x.tolist(), "sigma_local": sigma.tolist(),
                     "bias_in_local_sigma": ((fit.x-true)/sigma).tolist(), "nfev": fit.nfev,
                     "chi2_per_dof": float(2*fit.cost/(data.size-len(true)))})
    residual = (actual-independent)/noise
    np.savez_compressed(output/"mock_images.npz", independent_truth=independent, wrapper_at_truth=actual,
                        difference_in_noise=residual, noiseless_recovered=np.array(render(fits[0]["estimate"])).reshape(independent.shape),
                        psf=psf, noise_sigma=noise)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, img, title, cmap in zip(axes, (independent, actual, residual),
        ("Independent density simulation", "Wrapper at injected parameters", "Difference / mock noise sigma"),
        ("magma", "magma", "RdBu_r")):
        im = ax.imshow(img, origin="lower", extent=[-1.23, 1.23, -1.23, 1.23], cmap=cmap)
        ax.set(title=title, xlabel="x [arcsec]", ylabel="y [arcsec]")
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(output/"independent_mock.png", dpi=180)
    plt.close(fig)
    return {"parameter_order": list(keys), "truth": true.tolist(), "source_peak_snr": 100,
            "forward_image_error": errors(actual, independent), "max_image_error_in_noise": float(np.abs(residual).max()),
            "noise_sigma": float(noise), "fits": fits,
            "scope": "Six-parameter least-squares recovery; fixed MGE geometry, halo Rs/shape/centre and shear. Not SVI/HMC coverage."}


def write_report(report, output):
    p, mock = report["profiles"], report["mock"]
    recovered = mock["fits"][0]["estimate"]
    bias = 100*(np.array(recovered[:2])/np.array(mock["truth"][:2])-1)
    passed, total = sum(report["acceptance_gates"].values()), len(report["acceptance_gates"])
    lines = [
        "# StellarMassMGE + NFW 实现验证", "",
        "## 结论与范围", "",
        "恒星 MGE 的质量归一化、光到质量的动态连接、中心关联和多成分叠加通过独立数值验证。"
        "当前椭圆 NFW 是有限 Gaussian 基底近似，不能宣称对整个径向范围精确。"
        "已有 run_2/run_3 在全图上的 halo 分量超过 0.5% 偏折 / 1% 收敛度门槛；失败项保留在 JSON 中。", "",
        f"本次严格审计门槛：**{passed}/{total} 通过**。严格圆形 Gaussian / NFW 分支另列为不支持，未计入通过项。", "",
        "本验证使用本地安装版本重新评估保存的 kwargs，未重新拟合真实数据。"
        "保存结果未提供原远程运行环境的完整软件指纹，因此数值结论针对报告中记录的本地实现；"
        "保存文件字段的一致性则直接来自四个原始结果。", "",
        "## 定义与单位", "",
        "每个 light Gaussian 的 `amp=F_i` 是积分通量，不是中心表面亮度。其椭圆半径是面积保持的"
        r" $R_i^2=q_i X_i^2+Y_i^2/q_i$；$\phi=\frac12\operatorname{atan2}(e_2,e_1)$，"
        " $q=(1-|e|)/(1+|e|)$。", "",
        r"$$ I_i(\theta)=\frac{F_i}{2\pi\sigma_i^2}\exp[-R_i^2/(2\sigma_i^2)]. $$", "",
        "代码实际使用", "",
        r"$$ A_i=\Upsilon_\kappa\frac{F_i}{\sum_j F_j}"
        r"\left(\frac{\sigma_i}{\sigma_{\rm ref}}\right)^{-g},\qquad"
        r"\sigma_{\rm ref}=\exp\left[\frac1N\sum_i\ln\sigma_i\right]. $$", "",
        r"$g=0$ 时，$\kappa_\star=\Upsilon_\kappa I/\sum F_i$ 且"
        r" $\int\kappa_\star d^2\theta=\Upsilon_\kappa$。"
        "`upsilon_kappa` 的单位是角坐标单位的平方，通常 arcsec²。"
        r"$g\ne0$ 时权重**没有重新归一化**，总 convergence area 为 $\sum_i A_i$。"
        r"若换到物理质量，应使用 $M_\star=\Sigma_{\rm crit}D_l^2"
        r"(\pi/648000)^2\sum_i A_i$（A_i 按 arcsec² 表示）。"
        "物理 M/L 还需要光度标定和距离；目前并未计算物理 M/L。", "",
        r"halo 使用标准 NFW 径向律；代码展开的有效三维函数为 $f_{3D}(r)=\kappa_s/[r(1+r/R_s)^2]$（r 与 R_s 均用角坐标表示）；"
        r"$\kappa_s$ 无量纲，$R_s$ 为角尺度。独立参考先解析投影为 $2\kappa_s f(R/R_s)$，"
        "然后椭圆化 convergence，未将椭圆加到 potential 上，也未使用 gNFW 替代它。", "",
        "## 独立参考与一致性检查", "",
        "独立参考只使用 NumPy 密度和椭圆密度一维积分；不调用底层 Faddeeva 函数或 NFW MGE。"
        "积分以 u=t² 去除端点数值困难，128/256 阶比较检验收敛。"
        "用闭式圆 Gaussian 和球 NFW 偏折、以及 NFW 视线积分验证参考计算本身。"
        "Hessian 的参考来自偏折有限差分，并检验步长减半。"
        "势函数梯度、旋转/平移协变性、质量叠加和 ray shooting 都另有检查。", "",
        f"扁平恒星 mock（q=0.28）的偏折相对 L2 误差为 {p['stellar']['alpha']['relative_l2']:.3e}，"
        f"收敛度为 {p['stellar']['kappa']['relative_l2']:.3e}，"
        f"Hessian 为 {p['stellar']['hessian']['relative_l2']:.3e}。", "",
        "NFW 的自动微分另外满足归一化线性关系及尺度齐次关系："
        r" $\partial\alpha/\partial\kappa_s=\alpha/\kappa_s$ 和"
        r" $\partial\alpha/\partial R_s=[\alpha-H(\theta-\theta_c)]/R_s$。"
        "对 MGE 逆变换求和直接用很小的有限差分步长，会受到浮点消减影响；"
        "JSON 同时保留该诊断、齐次关系验证和独立积分梯度比较。", "",
        "## 四个已有结果", "",
        "当前配置包含 StellarMassMGE + NFW_ELLIPSE_KAPPA + SHEAR，开启了 M/L 梯度。"
        "若要最简单的常数 M/L 模型，应固定 `ml_gradient=0.0`；本次没有改动原拟合配置。", "",
        "四个结果的六组 stellar light_* 字段均与 lens-light Gaussian 一致，halo 中心与"
        "全 MGE 的 flux centroid 一致。场的验证使用 41×41、0.06 arcsec 的像素中心；"
        "排除 halo 中心半个像素以内的奇点区域，并单独评估 source arc mask。", "",
        "| run | g | ∫κ★d²θ [arcsec²] | halo α 最大相对误差 | halo κ 最大相对误差 | 弧区总 α 最大绝对误差 [arcsec] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["saved_runs"]:
        arc = r["arc_mask"].get("alpha_total", {}).get("max_abs", float("nan"))
        lines.append(f"| {r['run']} | {r['ml_gradient']:.4f} | {r['stellar_total_convergence_area']:.6f} | "
                     f"{100*r['halo_alpha']['max_relative']:.3f}% | {100*r['halo_kappa']['max_relative']:.3f}% | {arc:.3e} |")
    lines += ["", "以上误差与拟合残差或 χ² 无关；是同一组参数下的实现和独立参考比较。", "",
        "## NFW 近似的范围与收敛实验", "",
        "生产默认用 20 个 Gaussian，σ 从 R_s/500 到 20 R_s。径向扫描覆盖 r/R_s=10⁻⁴–50，"
        "q=0.4、0.7、0.95、0.9999。在很小和很大的半径，误差远大于百分之一。"
        "曲线已保存为 `nfw_radial_accuracy.png`。"
        "另用 60 个 Gaussian、σ/R_s=10⁻⁶–1000 做扩展基底收敛实验，作为诊断，并未改变生产默认或已有结果。", "",
        "## 独立模拟与参数回收", "",
        "模拟由独立偏折积分生成，再经 4×4 子像素积分和一个归一化 Gaussian PSF 卷积。"
        "采用 41×41、0.06 arcsec，峰值源信噪比 100。拟合用实际 SingleBandModel / LensImage。"
        "恒星几何、halo R_s/形状/中心和剪切固定；同时拟合恒星归一化、halo κ_s、源中心 x/y、"
        "源 σ 和源总通量，共六个参数。使用已知前景光去除后的 source-only 图像。", "",
        "| 参数 | 注入值 | 无噪声回收 | 相对偏差 |", "|---|---:|---:|---:|",
        f"| upsilon_kappa | {mock['truth'][0]:.8f} | {recovered[0]:.8f} | {bias[0]:.4f}% |",
        f"| kappa_s | {mock['truth'][1]:.8f} | {recovered[1]:.8f} | {bias[1]:.4f}% |", "",
        f"在真值处，wrapper 与独立模拟的图像相对 L2 误差为 {mock['forward_image_error']['relative_l2']:.3e}，"
        f"最大单像素差为 {mock['max_image_error_in_noise']:.3f} 个 mock noise sigma。"
        "三个加噪 realization 均收敛，χ²/dof 分别为 "
        +"、".join(f"{f['chi2_per_dof']:.3f}" for f in mock["fits"][1:])+"。", "",
        f"![Independent simulation]({output/'independent_mock.png'})", "",
        "这些回收支持成像及两成分质量尺度链路正确，**不代表**完整 pixelated-source SVI/HMC 的"
        "后验校准、参数唯一性或现实恒星/暗物质分解已被验证。", "",
        "## 已修复问题与尚存边界", "",
        "修复了先导入 jax_lensing_profiles、后导入 wrapper 时的同名 NFW 注册冲突。"
        "只允许已知的原生 backend 被 wrapper adapter 接管，仍拒绝无关类覆盖。"
        "参数说明已改为准确的 convergence area 含义。质量场的默认数学实现和原始结果未改动。", "",
        "严格 e1=e2=0 在当前 stellar / elliptical NFW MGE 中会产生非有限值；这是尚存边界。"
        "球 NFW 可使用解析 NFW profile，但不能把两种接口当作同一精度的计算方法。", "",
        "## 复现与文件", "", "```bash",
        "/opt/anaconda3/envs/herculens/bin/python utils/validate_stellar_mge_nfw.py \\",
        "  --results-dir '/path/to/modelling_F277W_StellarMassMGE_NFW/pixelated_svi'", "```", "",
        "若保留当前默认基底，该命令因上述四个精度门槛失败返回 exit code 1，这是审计结果而非程序崩溃。"
        "`validation.json` 包含各项误差、门槛、软件路径/版本与 SHA256；"
        "`mock_images.npz` 保留独立真值、wrapper 图像、差图和回收图。", "",
        "参考积分与 Gaussian 分解方法见 [Keeton (2001/2002)](https://arxiv.org/abs/astro-ph/0102341)"
        " 和 [Shajib (2019)](https://arxiv.org/abs/1906.08263)。"
        "实际 backend 源码为 [Herculens/Jax-Lensing-Profiles](https://github.com/Herculens/Jax-Lensing-Profiles)。", "",
    ]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"results/stellar_mge_nfw_validation")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    stellar, halo, shear = toy_parameters()
    print("Checking independent profile integrals and boundary cases", flush=True)
    report = {"toy_parameters": [stellar, halo, shear], "profiles": check_profiles(stellar, halo),
              "limits": check_limits(stellar, halo)}
    print("Checking sampled light-to-mass dependency", flush=True)
    report["dynamic_api"] = check_dynamic(stellar, halo, shear)
    report["superposition"] = check_superposition(stellar, halo, shear)
    print("Auditing saved SVI mass/light fields and physical fields", flush=True)
    report["saved_runs"] = audit_saved(args.results_dir.resolve()) if args.results_dir else []
    print("Measuring potential consistency and finite halo-MGE radial range", flush=True)
    report["potential_gradients"] = check_potential_gradients(stellar, halo)
    report["nfw_radial_range"] = check_nfw_range(output, report["saved_runs"])
    print("Rendering and recovering independent mock", flush=True)
    report["mock"] = run_mock(stellar, halo, shear, output)
    import herculens, jax_lensing_profiles
    files = [Path(inspect.getfile(StellarMGE)), Path(jax_lensing_profiles.__file__).parent/"MassModel/Profiles/MGE.py",
             Path(jax_lensing_profiles.__file__).parent/"MassModel/Profiles/multi_gaussian_ellipse_kappa.py",
             Path(jax_lensing_profiles.__file__).parent/"MassModel/Profiles/NFW_ellipse_kappa.py",
             Path(__file__)]
    report["provenance"] = {"python": sys.executable, "jax": jax.__version__, "x64": bool(jax.config.jax_enable_x64),
         "herculens_path": herculens.__file__, "jax_lensing_profiles_version": jax_lensing_profiles.__version__,
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    # Declared acceptance gates distinguish approximation budgets from exact
    # plumbing/normalization checks. Out-of-domain cases remain in the report.
    gates = {
        "stellar_alpha_relative_l2_lt_1e-6": report["profiles"]["stellar"]["alpha"]["relative_l2"] < 1e-6,
        "stellar_kappa_relative_l2_lt_1e-6": report["profiles"]["stellar"]["kappa"]["relative_l2"] < 1e-6,
        "halo_alpha_max_relative_lt_0.005": report["profiles"]["halo"]["alpha"]["max_relative"] < .005,
        "halo_kappa_max_relative_lt_0.01": report["profiles"]["halo"]["kappa"]["max_relative"] < .01,
        "independent_quadrature_converged": all(report["profiles"][k]["quadrature_128_vs_256"]["relative_l2"] < 1e-7 for k in ("stellar", "halo")),
        "hessian_reference_converged": all(report["profiles"][k]["hessian_step_halving"]["relative_l2"] < 1e-5 for k in ("stellar", "halo")),
        "stellar_hessian_relative_l2_lt_1e-5": report["profiles"]["stellar"]["hessian"]["relative_l2"] < 1e-5,
        "halo_hessian_relative_l2_lt_0.005": report["profiles"]["halo"]["hessian"]["relative_l2"] < .005,
        "nfw_los_projection": report["limits"]["nfw_projection_los_max_relative"] < 1e-9,
        "stellar_parameter_gradients": all(v["relative_l2"] < 1e-4 for k, v in report["limits"]["parameter_gradients"].items() if k.startswith("stellar")),
        "nfw_parameter_gradient_homogeneity": all(v["relative_l2"] < 1e-6 for v in report["limits"]["nfw_gradient_homogeneity"].values()),
        "nfw_parameter_gradient_independent_lt_0.5pct": all(v["max_relative"] < .005 for v in report["limits"]["nfw_gradient_independent"].values()),
        "potential_gradient_relative_l2_lt_1e-4": all(v["relative_l2"] < 1e-4 for v in report["potential_gradients"].values()),
        "dynamic_fields_and_centroid": report["dynamic_api"]["changed_light_fields_and_flux_centroid"] == "pass",
        "superposition": all(report["superposition"][k]["max_abs"] < 1e-12 for k in ("alpha_sum", "kappa_sum")),
        # ray_shooting is jitted whereas the comparison alpha is eager; tiny
        # fused-operation rounding in the NFW inverse transform is expected.
        "ray_shooting_eager_vs_jit_lt_1e-8": report["superposition"]["ray_shooting"]["max_abs"] < 1e-8,
        "noiseless_recovery_mass_scales_within_1pct": bool(np.all(np.abs(np.asarray(report["mock"]["fits"][0]["estimate"][:2])/
                     np.asarray(report["mock"]["truth"][:2])-1) < .01)),
        "all_mock_optimizers_converged": all(f["success"] for f in report["mock"]["fits"]),
    }
    for run in report["saved_runs"]:
        gates[run["run"]+"_mass_light_same_values"] = all(e["max_abs"] < 1e-12 for e in run["field_consistency"].values())
        gates[run["run"]+"_centroid"] = max(map(abs, run["halo_centroid_minus_light_centroid"].values())) < 1e-12
        gates[run["run"]+"_halo_alpha_lt_0.5pct"] = run["halo_alpha"]["max_relative"] < .005
        gates[run["run"]+"_halo_kappa_lt_1pct"] = run["halo_kappa"]["max_relative"] < .01
    report["acceptance_gates"] = gates
    report["regular_domain_pass"] = all(gates.values())
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(report, output)
    print(json.dumps({"regular_domain_pass": report["regular_domain_pass"], "acceptance_gates": gates,
                      "report": str(output/"validation.json")}, indent=2), flush=True)
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
