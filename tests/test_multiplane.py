"""Unit tests for multi-plane ray tracing in herculens_wrapper."""

import numpy as np
import jax.numpy as jnp
import pytest

from herculens_wrapper.api import (
    SingleBandData, SingleBandModel, LensProfileCollection,
    EPL, Shear, NFWHalo, SIS, Sersic, MassProfile, LightProfile,
    SamplerConfig,
)


def _make_mock_data(npix=30, pix_scale=0.08):
    image = np.zeros((npix, npix))
    noise = np.ones((npix, npix)) * 0.01
    psf = np.zeros((5, 5))
    psf[2, 2] = 1.0
    return SingleBandData(
        image=image,
        noise=noise,
        psf=psf,
        pixel_scale=pix_scale,
    )


def test_profile_redshift_declarations():
    """Profiles properly accept and retain redshift attribute."""
    epl = EPL(theta_E=1.2, redshift=0.5)
    assert epl.redshift == 0.5

    nfw1 = NFWHalo(kappa_s=0.1, R_s=8.0, redshift=0.5)
    assert nfw1.redshift == 0.5

    nfw2 = NFWHalo(kappa_s=0.15, R_s=12.0, redshift=0.8)
    assert nfw2.redshift == 0.8

    sis = SIS(theta_E=0.2, redshift=0.2)
    assert sis.redshift == 0.2

    src = Sersic(amp=10.0, redshift=1.5)
    assert src.redshift == 1.5

    # Test freeze retains redshift
    frozen_epl = epl.freeze()
    assert frozen_epl.redshift == 0.5


def test_collection_perturbers_kwarg():
    """LensProfileCollection supports perturbers kwarg and appends to lens_mass."""
    epl = EPL(theta_E=1.2, redshift=0.5)
    nfw_los = NFWHalo(kappa_s=0.15, R_s=12.0, redshift=0.8)
    src = Sersic(amp=10.0, redshift=1.5)

    coll = LensProfileCollection(
        lens_mass=[epl],
        perturbers=[nfw_los],
        source_light=[src],
    )
    assert len(coll.lens_mass) == 2
    assert coll.lens_mass[0].profile_type == "EPL"
    assert coll.lens_mass[1].profile_type == "NFW"
    assert coll.lens_mass[1].redshift == 0.8


def test_multiplane_ray_tracing_model():
    """Verify multi-plane model creation, eta matrix calculation, and image forward evaluation."""
    data = _make_mock_data(npix=40, pix_scale=0.08)
    z_lens = 0.5
    z_source = 1.5

    # Target scenario:
    # 1. Image plane: EPL + Shear + NFW at z=0.5
    # 2. LOS intermediate plane: NFW at z=0.8 (angular distance ~15'')
    # 3. Foreground plane: SIS at z=0.2
    # 4. Source plane: Sersic at z=1.5
    profiles = LensProfileCollection(
        lens_mass=[
            SIS(theta_E=0.1, center_x=-2.0, center_y=1.0, redshift=0.2),
            EPL(theta_E=1.1, gamma=2.0, e1=0.05, e2=0.0, center_x=0.0, center_y=0.0, redshift=z_lens),
            Shear(gamma1=0.03, gamma2=0.0, redshift=z_lens),
            NFWHalo(kappa_s=0.08, R_s=6.0, center_x=0.3, center_y=-0.3, redshift=z_lens),
            NFWHalo(kappa_s=0.12, R_s=10.0, center_x=15.0, center_y=0.0, redshift=0.8),
        ],
        lens_light=[
            Sersic(amp=50.0, R_sersic=0.8, n_sersic=4.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, redshift=z_lens),
        ],
        source_light=[
            Sersic(amp=8.0, R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.05, center_y=0.05, redshift=z_source),
        ],
    )

    model = SingleBandModel(
        observation=data,
        profiles=profiles,
        z_lens=z_lens,
        z_source=z_source,
        cosmology="Planck18",
    )

    # Validate multi-plane structure
    assert type(model.lens_image).__name__ == "MPLensImageExtension"
    assert model.lens_image.MPMassModel.number_mass_planes == 3  # z=0.2, z=0.5, z=0.8
    assert model.lens_image.MPLightModel.number_light_planes == 4  # + source plane z=1.5
    assert len(model.lens_image.eta_flat) == 3

    # Validate forward evaluation
    init_params = model.initialize(seed=123)
    kwargs = model.prob_model.params2kwargs(init_params)
    rendered = model.lens_image.model(**kwargs)

    assert rendered.shape == (40, 40)
    assert np.all(np.isfinite(rendered))
    assert float(jnp.sum(rendered)) > 0


def test_single_plane_backward_compatibility():
    """Verify single-plane models without explicit multiplane redshifts use standard single-plane path."""
    data = _make_mock_data(npix=30, pix_scale=0.08)

    profiles = LensProfileCollection(
        lens_mass=[EPL(theta_E=1.0, gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)],
        source_light=[Sersic(amp=10.0, R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)],
    )

    model = SingleBandModel(
        observation=data,
        profiles=profiles,
        z_lens=0.5,
        z_source=1.5,
    )

    # Must be standard LensImageExtension, not multiplane
    assert type(model.lens_image).__name__ == "LensImageExtension"
    init_params = model.initialize(seed=123)
    kwargs = model.prob_model.params2kwargs(init_params)
    rendered = model.lens_image.model(**kwargs)
    assert rendered.shape == (30, 30)
    assert np.all(np.isfinite(rendered))


def test_user_scenario_two_nfws_and_fit():
    """Verify end-to-end SVI on user scenario: EPL + NFW on lens plane, NFW between lens and source."""
    data = _make_mock_data(npix=36, pix_scale=0.08)
    z_lens = 0.5
    z_source = 1.5

    # Main lens: EPL + Shear at z_lens
    # Perturber 1: NFW on lens plane (z=0.5)
    # Perturber 2: NFW between lens and source (z=0.8, separation ~15'')
    profiles = LensProfileCollection(
        lens_mass=[
            EPL(prior={"theta_E": [0.5, 2.0]}, gamma=2.0, e1=0.05, e2=0.0, center_x=0.0, center_y=0.0, redshift=z_lens),
            Shear(gamma1=0.03, gamma2=0.0, redshift=z_lens),
            NFWHalo(prior={"kappa_s": [0.01, 0.5]}, R_s=5.0, center_x=0.5, center_y=0.5, redshift=z_lens),
        ],
        perturbers=[
            NFWHalo(prior={"kappa_s": [0.01, 0.5]}, R_s=12.0, center_x=15.0, center_y=0.0, redshift=0.8),
        ],
        source_light=[
            Sersic(amp=15.0, R_sersic=0.25, n_sersic=1.2, e1=0.0, e2=0.0, center_x=0.02, center_y=0.02, redshift=z_source),
        ],
    )

    model = SingleBandModel(
        observation=data,
        profiles=profiles,
        z_lens=z_lens,
        z_source=z_source,
        cosmology="Planck18",
    )

    # Multi-plane checks
    assert model.lens_image.MPMassModel.number_mass_planes == 2  # z=0.5 and z=0.8
    assert model.lens_image.MPLightModel.number_light_planes == 3  # z=0.5, z=0.8 (None), z=1.5 (source)

    # Initialize and run a few SVI iterations
    model.initialize(seed=42)
    fit_res = model.run(SamplerConfig.svi(max_iterations=10, random_seed=42))
    assert fit_res is not None
    assert len(fit_res.loss_history) == 10
    assert np.all(np.isfinite(fit_res.loss_history))
