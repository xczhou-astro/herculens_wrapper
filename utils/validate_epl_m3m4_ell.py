"""Check the joint elliptical profile against a local lenstronomy checkout.

Run with the Herculens environment:
    python utils/validate_epl_m3m4_ell.py --reference-root ../lenstronomy

The circular fallback is checked against its mathematical limit and autodiff,
because the reference checkout also omits the reference-frame conversion there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--reference-root", type=Path, default=root.parent / "lenstronomy")
    parser.add_argument("--output", type=Path,
                        default=root / "results/epl_m3m4_validation/validation.json")
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    if not (reference_root / "lenstronomy/LensModel/Profiles/epl_multipole_m3m4.py").is_file():
        parser.error("--reference-root must point to a lenstronomy source checkout")
    # The installed release may use an older phase convention. Select explicitly.
    sys.path[:0] = [str(reference_root), str(root)]

    import jax
    import jax.numpy as jnp
    import lenstronomy
    from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import EPL_MULTIPOLE_M3M4_ELL
    from herculens.MassModel.Profiles.epl import EPL
    from herculens_wrapper.api import (
        LensProfileCollection, LightProfile, MassProfile, SingleBandData, SingleBandModel,
    )
    from herculens_wrapper.models import _materialize_mass_parameters
    from herculens_wrapper.profiles.multipole import EPLM3M4
    from herculens_wrapper.profiles.jaxtronomy_multipole import EllipticalMultipole, Multipole

    if not Path(lenstronomy.__file__).resolve().is_relative_to(reference_root):
        raise RuntimeError("The reference import did not use the requested checkout")
    reference = EPL_MULTIPOLE_M3M4_ELL()
    x = np.array([0.12, 0.8, -1.1, 0.37, -0.6, 1.6])
    y = np.array([0.91, -0.32, 0.65, -1.4, -0.78, 0.4])
    public = dict(theta_E=1.31, gamma=2.0, q=0.7, phi=28.0,
                  center_x=0.04, center_y=-0.03,
                  a3_a=0.018, delta_phi_m3=13.0, a4_a=-0.024, delta_phi_m4=-8.0)

    def native(**updates):
        return _materialize_mass_parameters("EPL_MULTIPOLE_M3M4_ELL", {**public, **updates})

    errors = dict(function=0.0, derivatives=0.0, hessian=0.0)
    for q in (0.35, 0.7, 0.95):
        for gamma in (1.7, 1.99999, 2.0, 2.00001, 2.3):
            kwargs = {key: float(value) for key, value in native(q=q, gamma=gamma).items()}
            for method in errors:
                actual = np.asarray(getattr(EPLM3M4, method)(x, y, **kwargs))
                expected = getattr(reference, method)(x, y, **kwargs)
                if method == "hessian":
                    expected = (expected[0], expected[3], expected[1])
                expected = np.asarray(expected)
                np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)
                errors[method] = max(errors[method], float(np.max(np.abs(actual - expected))))

    # Rotational covariance and spatial derivatives of the circular fallback.
    fallback_error = 0.0
    for m in (3, 4):
        for q in (0.99999, 1.0):
            kwargs = dict(m=m, a_m=0.03, varphi_m=0.17, q=q, phi_ref=0.5,
                          center_x=0.04, center_y=-0.03, r_E=1.3)
            circular = {key: value for key, value in kwargs.items()
                        if key not in {"q", "phi_ref", "varphi_m"}}
            circular["phi_m"] = kwargs["phi_ref"] + kwargs["varphi_m"]
            for method in errors:
                actual = np.asarray(getattr(EllipticalMultipole, method)(x, y, **kwargs))
                expected = np.asarray(getattr(Multipole, method)(x, y, **circular))
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                fallback_error = max(fallback_error, float(np.max(np.abs(actual - expected))))
            point = jnp.array([0.8, 0.3])
            potential = lambda z: EllipticalMultipole.function(*z, **kwargs)
            np.testing.assert_allclose(EllipticalMultipole.derivatives(*point, **kwargs),
                                       jax.grad(potential)(point), atol=1e-12)
            hxx, hxy, _, hyy = EllipticalMultipole.hessian(*point, **kwargs)
            np.testing.assert_allclose([[hxx, hxy], [hxy, hyy]],
                                       jax.hessian(potential)(point), atol=1e-12)

    # Gamma has a finite derivative through two, from EPL alone.
    for q in (0.35, 0.7, 0.99999):
        def potential(gamma, zero=False):
            kwargs = native(q=q, gamma=gamma)
            if zero:
                kwargs.update(a3_a=0.0, a4_a=0.0)
            return EPLM3M4.function(0.8, 0.3, **kwargs)
        for gamma in (1.99999, 2.0, 2.00001):
            derivative = jax.grad(potential)(gamma)
            step = 1e-5
            finite_difference = (potential(gamma + step) - potential(gamma - step)) / (2 * step)
            assert np.isfinite(derivative)
            np.testing.assert_allclose(derivative, finite_difference, rtol=2e-6, atol=1e-8)
            np.testing.assert_allclose(derivative, jax.grad(lambda g: potential(g, True))(gamma),
                                       atol=1e-12)

    # Build a real public API model and sample gamma, geometry, and phase offsets.
    mass = MassProfile("EPL_MULTIPOLE_M3M4_ELL", prior={
        **public, "gamma": [1.7, 2.3], "q": [0.5, 0.9], "phi": [-40.0, 40.0],
        "delta_phi_m3": [-15.0, 15.0], "delta_phi_m4": [-15.0, 15.0],
    })
    source = LightProfile("SERSIC_ELLIPSE", value=dict(
        amp=1.0, R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    ))
    model = SingleBandModel(
        profiles=LensProfileCollection(lens_mass=[mass], source_light=[source]),
        observation=SingleBandData(image=np.zeros((7, 7)), noise=np.ones((7, 7)),
                                   psf=np.ones((1, 1)), pixel_scale=0.1),
    )
    sample = model.prob_model.get_sample(jax.random.PRNGKey(2))
    kwargs = model.prob_model.params2kwargs(sample)["kwargs_lens"][0]
    assert 1.7 <= float(sample["lens_gamma_0"]) <= 2.3
    assert "q" not in kwargs and "phi" not in kwargs
    assert not any("a1_a" in key or "delta_phi_m1" in key for key in sample)
    for phase in ("delta_phi_m3", "delta_phi_m4"):
        np.testing.assert_allclose(kwargs[phase], np.deg2rad(sample[f"lens_{phase}_0"]), atol=1e-12)
    assert np.all(np.isfinite(np.asarray(EPLM3M4.derivatives(x, y, **kwargs))))
    ep = {key: kwargs[key] for key in ("theta_E", "gamma", "e1", "e2", "center_x", "center_y")}
    np.testing.assert_allclose(EPLM3M4.function(x, y, **{**kwargs, "a3_a": 0.0, "a4_a": 0.0}),
                               EPL().function(x, y, **ep), atol=1e-12)

    report = dict(passed=True, reference=str(Path(lenstronomy.__file__).resolve()),
                  reference_cases=15, points_per_case=6, max_absolute_errors=errors,
                  circular_fallback_max_absolute_error=fallback_error,
                  circular_fallback_derivatives="passed", free_gamma_derivatives="passed",
                  public_api_sampling="passed", zero_amplitude_epl_limit="passed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
