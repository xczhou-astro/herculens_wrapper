"""Flux-weighted MGE centroids used as linked halo centres."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from herculens_wrapper.api import (
    LensProfileCollection,
    LightProfile,
    NFWEllipseHalo,
    SingleBandData,
    SingleBandModel,
    StellarMassMGE,
)
from herculens_wrapper.models import (
    _normalize_link_spec,
    _resolve_link,
    param_list_to_init_kwargs,
    validate_param_list,
)


def test_flux_centroid_follows_sampled_light_in_halo_and_initial_kwargs():
    light = LightProfile.mge(
        n_gauss=2,
        sigma_lims=(0.03, 0.3),
        amp_prior=(1.0, 0.1),
        ellipticity_prior=(0.1, 0.02, 0.03, 0.2),
        center_prior=(0.0, 0.02, -0.1, 0.1),
    )
    stellar = StellarMassMGE(light, follow_lens_light=True, value={"upsilon_kappa": 0.2})
    halo = NFWEllipseHalo(
        value={"kappa_s": 0.03, "R_s": 1.0, "e1": 0.1, "e2": 0.02},
        prior={
            "center_x": ["correlated", "lens_light", "flux_centroid", "center_x"],
            "center_y": ["correlated", "lens_light", "flux_centroid", "center_y"],
        },
    )
    source = LightProfile("GAUSSIAN", value={
        "amp": 1.0, "sigma": 0.2, "center_x": 0.0, "center_y": 0.0,
    })
    data = SingleBandData(
        image=np.ones((3, 3)), noise=np.ones((3, 3)),
        psf=np.eye(3), pixel_scale=0.1,
    )
    model = SingleBandModel(
        observation=data,
        profiles=LensProfileCollection(
            lens_mass=[stellar, halo], lens_light=light, source_light=source,
        ),
    )
    params = model.prob_model.get_sample(jax.random.PRNGKey(4))
    assert "lens_center_x_1" not in params
    assert "lens_center_y_1" not in params

    for axis in ("center_x", "center_y"):
        kwargs = model.prob_model.params2kwargs(params)
        gaussians = kwargs["kwargs_lens_light"]
        expected = np.average(
            [float(gaussian[axis]) for gaussian in gaussians],
            weights=[float(gaussian["amp"]) for gaussian in gaussians],
        )
        assert float(kwargs["kwargs_lens"][1][axis]) == pytest.approx(expected)

    first_center = float(model.prob_model.params2kwargs(params)["kwargs_lens"][1]["center_x"])
    params["lens_light_amp_0"] *= 2.0
    changed_center = float(model.prob_model.params2kwargs(params)["kwargs_lens"][1]["center_x"])
    assert changed_center != pytest.approx(first_center)

    type_list, param_list = model.definition.as_dicts()
    initial_kwargs = param_list_to_init_kwargs(param_list, type_list, model.lens_image)
    assert np.isfinite(float(initial_kwargs["kwargs_lens"][1]["center_x"]))


def test_flux_centroid_link_is_differentiable_and_requires_gaussian_light():
    spec = _normalize_link_spec(["correlated", "lens_light", "flux_centroid", "center_x"])
    light = [{"amp": jnp.array(1.0), "center_x": jnp.array(-0.2)},
             {"amp": jnp.array(3.0), "center_x": jnp.array(0.4)}]
    assert float(_resolve_link({"lens_light": light}, spec)) == pytest.approx(0.25)
    derivative = jax.grad(lambda amplitude: _resolve_link({
        "lens_light": [{"amp": amplitude, "center_x": light[0]["center_x"]}, light[1]],
    }, spec))(jnp.array(1.0))
    assert float(derivative) == pytest.approx(-0.1125)

    with pytest.raises(ValueError, match="exclusively GAUSSIAN_ELLIPSE"):
        validate_param_list(
            {"lens_mass_type_list": ["NFW"], "lens_light_type_list": ["SERSIC_ELLIPSE"]},
            {"lens_mass_params_list": [{"center_x": ["correlated", "lens_light", "flux_centroid", "center_x"]}],
             "lens_light_params_list": [{"amp": 1.0, "center_x": 0.0, "center_y": 0.0}]},
        )
