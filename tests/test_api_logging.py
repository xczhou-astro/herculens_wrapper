import multiprocessing as mp
import os
from pathlib import Path
import sys

import pytest
import numpy as np
from types import SimpleNamespace

from herculens_wrapper.api import (
    LensProfileCollection,
    LightProfile,
    MassProfile,
    PixelatedLensLight,
    PixelatedSource,
    SingleBandData,
    SingleBandModel,
)
from herculens_wrapper.api._logging import (
    RunContext,
    logged_model_run,
    logged_result_output,
)


class _Sampler:
    name = "svi"


class _Model:
    _run_context = None

    def configuration(self, *, sampler=None):
        return {"model": "test", "sampler": sampler.name}

    def describe(self, *, sampler=None):
        return "test model description"

    @logged_model_run
    def run(self, sampler, *, save_path=None):
        print("sampler body")
        return _Result(self)


class _Result:
    def __init__(self, model):
        self._model = model

    @logged_result_output
    def output(self, save_path=None):
        print("result body")
        return Path(save_path)


def test_uniform_exposure_poisson_noise_uses_the_model_prediction():
    """The Poisson correction is active in the actual likelihood LensImage."""
    data = SingleBandData(
        image=np.zeros((7, 7)), noise=None, psf=np.eye(3), pixel_scale=0.1,
        background_rms=2.0, exposure_time=10.0,
    )
    source = LightProfile(
        "SERSIC_ELLIPSE",
        prior={"amp": [1.0, 0.1], "R_sersic": [0.1, 0.2], "n_sersic": [1.0, 0.2],
               "e1": [-0.1, 0.1], "e2": [-0.1, 0.1], "center_x": [-0.1, 0.1],
               "center_y": [-0.1, 0.1]},
    )
    model = SingleBandModel(
        profiles=LensProfileCollection(source_light=source), observation=data,
    )
    variance = np.asarray(model.lens_image.Noise.C_D_model(np.full((7, 7), 5.0)))
    assert data.uses_poisson_noise
    assert np.allclose(variance, 4.5)  # 2**2 + 5 e-/s / (10 s)
    assert np.allclose(data.noise_from_model(np.full((7, 7), 5.0)), np.sqrt(4.5))


def test_noise_map_and_poisson_noise_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either noise"):
        SingleBandData(
            image=np.zeros((3, 3)), noise=np.ones((3, 3)), psf=np.eye(3), pixel_scale=0.1,
            background_rms=1.0, exposure_time=10.0,
        )


def test_nonfinite_pixels_are_excluded_before_building_the_noise_model(tmp_path):
    from astropy.io import fits

    image = np.ones((3, 3))
    noise = np.full((3, 3), 0.2)
    image[0, 1] = np.nan
    noise[1, 1] = np.nan
    noise[2, 1] = 0.0
    data = SingleBandData(image=image, noise=noise, psf=np.eye(3), pixel_scale=0.1)

    expected_mask = np.ones((3, 3), dtype=bool)
    expected_mask[0, 1] = expected_mask[1, 1] = expected_mask[2, 1] = False
    assert np.array_equal(data.likelihood_mask, expected_mask)
    assert np.all(np.isfinite(data.likelihood_image))
    assert np.all(data.likelihood_noise > 0)
    assert np.all(data.likelihood_image[~expected_mask] == 0)
    assert np.all(data.likelihood_noise[~expected_mask] == 1e10)
    saved = data.save(tmp_path)
    restored = SingleBandData.from_fits(
        saved["image"], saved["noise"], saved["psf"], pixel_scale=0.1,
    )
    assert np.array_equal(restored.likelihood_mask, expected_mask)
    contaminant = np.zeros((3, 3), dtype=np.uint8)
    contaminant[0, 0] = 1
    mask_path = tmp_path / "contaminant.fits"
    fits.writeto(mask_path, contaminant)
    data.contaminate_mask_path = str(mask_path)
    expected_mask[0, 0] = False
    assert np.array_equal(data.likelihood_mask, expected_mask)


def test_no_usable_pixels_fail_with_clear_error():
    with pytest.raises(ValueError, match="no pixels"):
        SingleBandData(
            image=np.full((3, 3), np.nan), noise=np.ones((3, 3)),
            psf=np.eye(3), pixel_scale=0.1,
        )


def test_hmc_fits_source_uses_pixelwise_median_instead_of_parameter_pixels():
    from herculens_wrapper.api.samplers import _source_plane_fits_arrays

    posterior_median = np.full((3, 3), 5.0)
    lens_image = SimpleNamespace(
        _rtu_grid_source=False,
        get_source_coordinates=lambda *_args, **_kwargs: (
            np.arange(3), np.arange(3), [-1, 1, -1, 1],
        ),
    )
    model = SimpleNamespace(lens_image=lens_image, source_grid_scale=1.0)
    result = SimpleNamespace(samples={"pixels": np.ones((2, 3, 3))}, derived={
        "source_plane": posterior_median,
    })
    arrays = _source_plane_fits_arrays(
        result, model,
        {"kwargs_source": [{"pixels": np.full((3, 3), 99.0)}], "kwargs_lens": []},
        {"source_light_type_list": ["PIXELATED"]}, None,
    )
    assert np.array_equal(arrays["source_plane"], posterior_median)
    assert arrays["source_x"].shape == posterior_median.shape
    assert arrays["source_y"].shape == posterior_median.shape


def test_hmc_parametric_source_plane_is_median_of_rendered_draws(monkeypatch):
    import jax.numpy as jnp
    from herculens_wrapper import samplers

    monkeypatch.setattr(samplers, "get_active_sample_sites", lambda _model: ["amp"])

    class _SourceModel:
        def surface_brightness(self, x, _y, kwargs_source):
            return jnp.ones_like(x) * kwargs_source[0]["amp"] ** 2

    prob_model = SimpleNamespace(
        lens_image=SimpleNamespace(
            Grid=SimpleNamespace(pixel_area=2.0), SourceModel=_SourceModel(),
        ),
        params2kwargs=lambda sample: {"kwargs_source": [{"amp": sample["amp"]}]},
    )
    grid = np.zeros((2, 3))
    median = samplers.evaluate_mcmc_parametric_source_plane_median(
        prob_model, {"amp": np.array([1.0, 3.0])}, grid, grid,
        batch_size=1, tile_size=1,
    )
    assert np.array_equal(median, np.full((2, 3), 10.0))


@pytest.mark.parametrize("mode, model_name, summary, source_value", [
    ("hmc", "MEDIAN_MODEL", "PIXMED", 3.0),
    ("svi", "MEDIAN_MODEL", "GUIDEMED", 99.0),
    ("optax", "BEST_FIT_MODEL", "PARAMSET", 99.0),
])
def test_modeling_result_fits_writes_source_lens_light_and_psf(
    tmp_path, monkeypatch, mode, model_name, summary, source_value,
):
    from astropy.io import fits
    from herculens_wrapper.api.samplers import FitResult
    from herculens_wrapper import samplers as sampler_backend
    from herculens_wrapper import utils, visualizations

    model = SimpleNamespace(
        definition=SimpleNamespace(as_dicts=lambda: (
            {"source_light_type_list": ["PIXELATED"]}, {},
        )),
        prob_model=object(),
        lens_image=SimpleNamespace(
            _rtu_grid_source=False,
            get_source_coordinates=lambda *_args, **_kwargs: (
                np.arange(4), np.arange(4), None,
            ),
        ),
        data=SimpleNamespace(
            likelihood_image=np.zeros((3, 3)), likelihood_mask=np.ones((3, 3), bool),
            source_arc_mask=None, contaminate_mask=None, psf=np.eye(3),
            psf_supersampling_factor=4, pixel_scale=0.1,
            samples_background_rms=False,
        ),
        noise_from_model=lambda *_args: np.ones((3, 3)),
        num_sampling_parameters=1, initial_parameters=None,
        source_grid_scale=1.0,
        _metrics=lambda _parameters: {
            "chi2_median": 1.0, "log_likelihood_median": -1.0,
            "n_free_parameters": 1, "n_physical_parameters": 1,
            "reduced_chi2_median": 1.0,
        },
    )
    monkeypatch.setattr(utils, "kwargs_best_to_json_pixelated_npy", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sampler_backend, "save_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualizations, "generate_run_plots", lambda **_kwargs: None)
    result = FitResult(
        parameters={}, details={"guide": object()} if mode == "svi" else {},
        samples={"amp": np.array([1.0, 3.0])} if mode == "hmc" else None,
        derived={
            "kwargs": {"kwargs_source": [{"pixels": np.full((4, 4), 99.0)}], "kwargs_lens": []},
            "source_plane": np.full((4, 4), 3.0),
            "component_medians": {
                "total": np.ones((3, 3)), "lens_light": np.full((3, 3), 2.0),
            },
            "components": {
                "total": np.ones((3, 3)), "lens_light": np.full((3, 3), 2.0),
            },
        }, _model=model,
    )

    result.output(tmp_path, include_corner=False)
    with fits.open(tmp_path / "modeling_result.fits") as hdus:
        assert np.array_equal(hdus[model_name].data, np.ones((3, 3)))
        assert np.array_equal(hdus["LENS_LIGHT"].data, np.full((3, 3), 2.0))
        assert np.array_equal(hdus["SOURCE_PLANE"].data, np.full((4, 4), source_value))
        assert np.array_equal(hdus["PSF"].data, np.eye(3))
        assert hdus[model_name].header["SUMMARY"] == summary
        assert hdus["SOURCE_PLANE"].header["SUMMARY"] == summary
        assert hdus["PSF"].header["PSFSSAMP"] == 4


@pytest.mark.parametrize("model_key", ["median_model", "best_fit_model"])
def test_recreate_model_plots_accepts_current_and_legacy_fits_names(
    tmp_path, monkeypatch, model_key,
):
    from herculens_wrapper import visualizations
    from herculens_wrapper.utils import save_named_arrays_fits

    save_named_arrays_fits(tmp_path / "modeling_result.fits", {
        model_key: np.full((3, 3), 2.0),
        "image_data": np.ones((3, 3)),
        "noise_map": np.ones((3, 3)),
    })
    plotted = []
    monkeypatch.setattr(
        visualizations, "display",
        lambda images, **_kwargs: plotted.append(images),
    )
    assert visualizations.recreate_best_fit_plots_for_run(tmp_path)
    assert len(plotted) == 2
    assert np.array_equal(plotted[0][0], np.full((3, 3), 2.0))


def test_sampled_background_rms_is_a_single_likelihood_latent():
    import jax
    from numpyro import handlers

    data = SingleBandData(
        image=np.zeros((7, 7)), noise=None, psf=np.eye(3), pixel_scale=0.1,
        background_rms_prior=[1.0, 4.0], exposure_time=10.0,
    )
    source = LightProfile(
        "SERSIC_ELLIPSE",
        prior={"amp": [1.0, 0.1], "R_sersic": [0.1, 0.2], "n_sersic": [1.0, 0.2],
               "e1": [-0.1, 0.1], "e2": [-0.1, 0.1], "center_x": [-0.1, 0.1],
               "center_y": [-0.1, 0.1]},
    )
    model = SingleBandModel(
        profiles=LensProfileCollection(source_light=source), observation=data,
    )
    trace = handlers.trace(handlers.seed(model.prob_model.model, jax.random.PRNGKey(0))).get_trace()
    assert "background_rms" in trace
    assert data.samples_background_rms
    assert np.allclose(data.noise_from_model(np.full((7, 7), 5.0), background_rms=2.0), np.sqrt(4.5))


def test_hmc_component_diagnostics_stream_through_hdf5(tmp_path):
    """Exact component medians do not require retaining posterior draws in RAM."""
    import jax
    from numpyro import handlers
    from herculens_wrapper.samplers import (
        _append_hmc_samples_hdf5,
        _stream_hmc_component_medians_hdf5,
    )

    data = SingleBandData(
        image=np.zeros((7, 7)), noise=np.ones((7, 7)), psf=np.eye(3), pixel_scale=0.1,
    )
    source = LightProfile(
        "SERSIC_ELLIPSE",
        prior={"amp": [1.0, 0.1], "R_sersic": [0.1, 0.2], "n_sersic": [1.0, 0.2],
               "e1": [-0.1, 0.1], "e2": [-0.1, 0.1], "center_x": [-0.1, 0.1],
               "center_y": [-0.1, 0.1]},
    )
    model = SingleBandModel(
        profiles=LensProfileCollection(source_light=source), observation=data,
    )
    trace = handlers.trace(handlers.seed(model.prob_model.model, jax.random.PRNGKey(0))).get_trace()
    samples = {
        name: np.stack([np.asarray(site["value"]), np.asarray(site["value"])])
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    posterior_path = tmp_path / "hmc_samples.h5"
    component_path = tmp_path / "components.h5"
    _append_hmc_samples_hdf5(posterior_path, samples, {}, num_chains=1)
    medians = _stream_hmc_component_medians_hdf5(
        model.prob_model, posterior_path, component_path,
        sample_chunk_size=1, tile_size=3,
    )
    assert medians["total"].shape == (7, 7)
    assert np.allclose(medians["total"], medians["source"])


def test_init_restores_saved_likelihood_background_rms(tmp_path):
    """A profile-agnostic likelihood latent survives SVI-to-HMC handoff."""
    import json
    import jax.numpy as jnp
    from herculens_wrapper.models import get_init_params

    class _ProbabilityModel:
        def get_sample(self, _key):
            return {"background_rms": jnp.asarray(1.0)}

    (tmp_path / "kwargs_result.json").write_text(json.dumps({
        "kwargs_lens": [],
        "kwargs_source": [],
        "likelihood_parameters": {"background_rms": 2.5},
    }))
    initial = get_init_params(
        _ProbabilityModel(),
        {"lens_mass_params_list": [], "source_light_params_list": []},
        {"lens_mass_type_list": [], "source_light_type_list": []},
        init_params_path=tmp_path,
    )
    assert float(initial["background_rms"]) == pytest.approx(2.5)


def test_new_point_source_starts_at_supplied_image_positions(tmp_path):
    """Adding points to a no-point-source run uses the declared image centers."""
    import json
    import jax.numpy as jnp
    from herculens_wrapper.models import get_init_params

    class _ProbabilityModel:
        def get_sample(self, _key):
            return {
                "ps_ra_0": jnp.full(4, 0.5),
                "ps_dec_0": jnp.full(4, -0.5),
                "ps_amp_0": jnp.full(4, 1.0),
            }

    (tmp_path / "kwargs_result.json").write_text(json.dumps({
        "kwargs_lens": [], "kwargs_source": [],
    }))
    initial = get_init_params(
        _ProbabilityModel(),
        {
            "lens_mass_params_list": [],
            "source_light_params_list": [],
            "point_source_params_list": [{
                "ra": [-0.3, -0.1, 0.1, 0.3],
                "dec": [0.2, 0.4, -0.4, -0.2],
                "n_images": 4,
                "sigma_image": 0.03,
                "amp": [0.2, 2.0],
            }],
        },
        {
            "lens_mass_type_list": [],
            "source_light_type_list": [],
            "point_source_type_list": ["IMAGE_POSITIONS"],
        },
        init_params_path=tmp_path,
    )
    np.testing.assert_array_equal(initial["ps_ra_0"], [-0.3, -0.1, 0.1, 0.3])
    np.testing.assert_array_equal(initial["ps_dec_0"], [0.2, 0.4, -0.4, -0.2])
    np.testing.assert_array_equal(initial["ps_amp_0"], [1.0] * 4)


def _worker_log(directory, run_id):
    context = RunContext(Path(directory) / f"run_{run_id}", console=False, run_id=run_id)
    with context.capture(f"worker {run_id}"):
        print(f"python-output-{run_id}")
        print(f"python-error-{run_id}", file=sys.stderr)
        os.write(2, f"native-error-{run_id}\n".encode())


def test_capture_appends_and_nested_context_does_not_duplicate(tmp_path, capsys):
    context = RunContext(tmp_path)
    with context.capture("outer"):
        print("first")
        with context.capture("inner"):
            print("second")
    with context.capture("later"):
        print("third")

    text = context.log_path.read_text()
    assert text.count("outer started") == 1
    assert "inner started" not in text
    assert text.count("later started") == 1
    assert all(word in text for word in ("first", "second", "third"))
    assert "first" in capsys.readouterr().out


def test_model_binds_result_to_the_same_log(tmp_path):
    model = _Model()
    result = model.run(_Sampler(), save_path=tmp_path)

    assert model._run_context.directory == tmp_path
    assert result._run_context.directory == tmp_path
    assert result.output() == tmp_path
    text = (tmp_path / "log.txt").read_text()
    assert "test model description" not in text
    assert "Model configuration:" not in text
    assert "sampler body" in text
    assert "result body" in text
    assert (tmp_path / "model_configuration.json").is_file()


def test_explicit_result_output_path_preserves_the_old_interface(tmp_path):
    first = tmp_path / "sampling"
    products = tmp_path / "products"
    model = _Model()
    result = model.run(_Sampler(), save_path=first)

    assert result.output(products) == products
    assert result._run_context.directory == products
    assert model._run_context.directory == first
    assert "result body" in (products / "log.txt").read_text()


def test_reusing_model_does_not_move_an_older_result_log(tmp_path):
    model = _Model()
    first = model.run(_Sampler(), save_path=tmp_path / "run_0")
    second = model.run(_Sampler(), save_path=tmp_path / "run_1")

    assert first._run_context.directory == tmp_path / "run_0"
    assert second._run_context.directory == tmp_path / "run_1"
    assert first.output() == tmp_path / "run_0"


def test_capture_records_traceback_and_failed_status(tmp_path):
    context = RunContext(tmp_path)
    with pytest.raises(RuntimeError, match="broken operation"):
        with context.capture("failing operation"):
            raise RuntimeError("broken operation")

    text = context.log_path.read_text()
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: broken operation" in text
    assert "failing operation failed" in text


def test_spawned_workers_write_isolated_python_and_native_logs(tmp_path):
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=_worker_log, args=(str(tmp_path), run_id))
        for run_id in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0

    for run_id in range(2):
        text = (tmp_path / f"run_{run_id}" / "log.txt").read_text()
        assert f"python-output-{run_id}" in text
        assert f"python-error-{run_id}" in text
        assert f"native-error-{run_id}" in text
        assert f"python-output-{1 - run_id}" not in text


def test_profile_configuration_does_not_create_phantom_parameters():
    sie = MassProfile("SIE", prior={"theta_E": [1.0, 0.1, 0.2, 2.0]})
    profiles = LensProfileCollection(lens_mass=sie)

    configuration = profiles.configuration

    assert set(sie._parameters) == {"theta_E"}
    assert "pixel_grid" not in configuration["lens_mass"][0]
    assert profiles.as_definition().as_dicts()[1]["lens_mass_params_list"] == [
        {"theta_E": [1.0, 0.1, 0.2, 2.0]}
    ]


def test_profile_configuration_keeps_real_pixelated_settings():
    source = PixelatedSource(
        pixel_grid={"pixel_grid_shape": 24},
        pixelated_prior={"prior_type": "matern"},
    )
    lens_light = PixelatedLensLight(scale_factor=0.75)
    profiles = LensProfileCollection(
        source_light=source,
        lens_light=lens_light,
    )

    configuration = profiles.configuration

    assert configuration["source_light"][0]["pixel_grid"]["pixel_grid_shape"] == 24
    assert configuration["source_light"][0]["pixelated_prior"]["prior_type"] == "matern"
    assert configuration["lens_light"][0]["pixel_grid"]["pixel_scale_factor"] == 0.75


def test_mge_collection_can_be_combined_with_pixelated_lens_light():
    mge = LightProfile(
        ["GAUSSIAN_ELLIPSE"] * 3,
        prior=[{
            "amp": [2.0, 0.2], "sigma": [0.03, 1.0],
            "e1": [-0.3, 0.3], "e2": [-0.3, 0.3],
            "center_x": [-0.2, 0.2], "center_y": [-0.2, 0.2],
        } for _ in range(3)],
    )

    profiles = LensProfileCollection(
        lens_light=[mge, PixelatedLensLight()],
    )

    assert [profile.profile_type for profile in profiles.lens_light] == [
        "GAUSSIAN_ELLIPSE", "GAUSSIAN_ELLIPSE", "GAUSSIAN_ELLIPSE", "PIXELATED",
    ]


def test_mass_light_warm_start_leaves_source_and_point_source_new(tmp_path):
    """A no-point-source HMC must not supply the new source pixels."""
    import json
    import jax.numpy as jnp
    from herculens_wrapper.models import get_init_params

    class ProbabilityModel:
        def get_sample(self, _key):
            return {
                "lens_theta_E_0": jnp.asarray(0.2),
                "lens_light_amp_0": jnp.asarray(1.0),
                "pixels_wn_source_grid": jnp.ones((2, 2)),
                "n_source_grid": jnp.asarray([2.0]),
                "rho_source_grid": jnp.asarray([3.0]),
                "sigma_source_grid": jnp.asarray([4.0]),
                "ps_ra_0": jnp.ones(4),
                "ps_dec_0": jnp.ones(4),
                "ps_amp_0": jnp.full(4, 5.0),
            }

    # A partial restore must not try to read these deliberately absent FITS.
    (tmp_path / "kwargs_result.json").write_text(json.dumps({
        "kwargs_lens": [{"theta_E": 0.4}],
        "kwargs_lens_light": [{"amp": 10.0}],
        "kwargs_source": [{
            "pixels": {"_format": "pixelated_pixels_fits", "file": "missing.fits"},
            "pixels_wn": {"_format": "pixelated_pixels_fits", "file": "missing_wn.fits"},
            "n_source_grid": 50.0,
            "rho_source_grid": 60.0,
            "sigma_source_grid": 70.0,
        }],
    }))
    params = {
        "lens_mass_params_list": [{"theta_E": [0.05, 0.6]}],
        "lens_light_params_list": [{"amp": [2.0, 0.1]}],
        "source_light_params_list": [{"pixels": None}],
        "point_source_params_list": [{
            "ra": [-0.3, -0.1, 0.1, 0.3],
            "dec": [0.2, 0.4, -0.4, -0.2],
            "n_images": 4,
            "sigma_image": 0.003,
            "amp": [0.2, 2.0],
        }],
    }
    types = {
        "lens_mass_type_list": ["SIE"],
        "lens_light_type_list": ["GAUSSIAN"],
        "source_light_type_list": ["PIXELATED"],
        "point_source_type_list": ["IMAGE_POSITIONS"],
    }

    initial = get_init_params(
        ProbabilityModel(), params, types, init_params_path=tmp_path,
        restore_components=("lens_mass", "lens_light"),
    )

    assert float(initial["lens_theta_E_0"]) == 0.4
    assert float(initial["lens_light_amp_0"]) == 10.0
    np.testing.assert_array_equal(initial["pixels_wn_source_grid"], np.ones((2, 2)))
    assert float(initial["n_source_grid"][0]) == 2.0
    assert float(initial["rho_source_grid"][0]) == 3.0
    assert float(initial["sigma_source_grid"][0]) == 4.0
    np.testing.assert_allclose(initial["ps_ra_0"], params["point_source_params_list"][0]["ra"])
    np.testing.assert_allclose(initial["ps_dec_0"], params["point_source_params_list"][0]["dec"])
    np.testing.assert_array_equal(initial["ps_amp_0"], np.full(4, 5.0))


def test_image_match_warmup_keeps_mass_light_and_copies_point_source(monkeypatch, tmp_path):
    """Warmup updates Matérn and point-source sites before the full SVI."""
    import json
    import jax.numpy as jnp
    from types import SimpleNamespace
    import herculens_wrapper.api.session as session

    (tmp_path / "kwargs_result.json").write_text(json.dumps({}))
    types = {
        "source_light_type_list": ["PIXELATED"],
        "lens_light_type_list": ["GAUSSIAN"],
    }
    params = {"source_light_params_list": [{"pixelated_prior": {}}]}
    recorded = {}
    initial = {
        "lens_theta_E_0": jnp.asarray(0.4),
        "lens_light_amp_0": jnp.asarray(10.0),
        "pixels_wn_source_grid": jnp.ones((2, 2)),
        "n_source_grid": jnp.asarray([2.0]),
        "rho_source_grid": jnp.asarray([3.0]),
        "sigma_source_grid": jnp.asarray([4.0]),
        "ps_ra_0": jnp.zeros(4),
        "ps_amp_0": jnp.ones(4),
    }

    def get_init_params(_prob_model, _params, _types, **kwargs):
        recorded["restore_components"] = kwargs["restore_components"]
        return dict(initial)

    def create_prob_model(*_args, **kwargs):
        recorded["warmup_kwargs"] = kwargs
        return object()

    def run_svi(_prob_model, _image, _sampler, _initial, **_kwargs):
        return {
            "lens_theta_E_0": jnp.asarray(0.9),
            "lens_light_amp_0": jnp.asarray(20.0),
            "pixels_wn_source_grid": jnp.full((2, 2), 7.0),
            "n_source_grid": jnp.asarray([8.0]),
            "rho_source_grid": jnp.asarray([9.0]),
            "sigma_source_grid": jnp.asarray([10.0]),
            "ps_ra_0": jnp.full(4, 0.02),
            "ps_amp_0": jnp.full(4, 6.0),
        }, {}

    monkeypatch.setattr(
        session, "_model_backend",
        lambda: (None, create_prob_model, get_init_params, None),
    )
    monkeypatch.setattr(session, "_sampler_backend", lambda: (None, None, run_svi))
    model = SingleBandModel.__new__(SingleBandModel)
    model.profiles = SimpleNamespace(
        apply_initializations=lambda: False, lens_light=None,
        warm_start_declarations=lambda: {},
    )
    model.definition = SimpleNamespace(
        as_dicts=lambda: (types, params), update_values=lambda _values: None,
    )
    model.prob_model = SimpleNamespace(params2kwargs=lambda _values: {
        "kwargs_lens": [{"theta_E": 0.4}],
        "kwargs_lens_light": [{"amp": 10.0}],
    })
    model.lens_image = object()
    model.data = SimpleNamespace(
        likelihood_image=np.zeros((2, 2)),
        likelihood_noise=np.ones((2, 2)), likelihood_mask=None,
        exposure_time=None, background_rms=None, background_rms_prior=None,
    )
    model.likelihood_scale = 1.0

    result = model.initialize(
        init_lens_mass_light_path=tmp_path,
        pixelated_init_match="image",
        num_iterations_warmup=2,
    )

    assert recorded["restore_components"] == ("lens_mass", "lens_light")
    assert recorded["warmup_kwargs"]["fix_lens_mass"]
    assert recorded["warmup_kwargs"]["fix_lens_light"]
    assert recorded["warmup_kwargs"]["init_params_path"] is None
    assert float(result["lens_theta_E_0"]) == 0.4
    assert float(result["lens_light_amp_0"]) == 10.0
    np.testing.assert_array_equal(result["pixels_wn_source_grid"], np.full((2, 2), 7.0))
    assert float(result["n_source_grid"][0]) == 8.0
    assert float(result["rho_source_grid"][0]) == 9.0
    assert float(result["sigma_source_grid"][0]) == 10.0
    np.testing.assert_array_equal(result["ps_ra_0"], np.full(4, 0.02))
    np.testing.assert_array_equal(result["ps_amp_0"], np.full(4, 6.0))


def _mock_plot_lens_image():
    grid_x, grid_y = np.meshgrid(np.linspace(-0.3, 0.3, 3), np.linspace(-0.3, 0.3, 3))
    pixel_grid = SimpleNamespace(
        extent=(-0.45, 0.45, -0.45, 0.45),
        pixel_coordinates=(grid_x, grid_y),
    )

    class LensImage:
        SourceModel = SimpleNamespace(pixel_grid=pixel_grid)
        source_arc_mask = None

        def model(self, **kwargs):
            source = 2.0 if kwargs.get("source_add", True) else 0.0
            lens = 3.0 if kwargs.get("lens_light_add", True) else 0.0
            point = 5.0 if kwargs.get("point_source_add", True) else 0.0
            return np.full((3, 3), source + lens + point)

    return LensImage()


def _mock_plot_geometry(monkeypatch, plots):
    positions = {
        "image": [(np.array([0.1]), np.array([0.2]))],
        "source": [(np.array([0.02]), np.array([0.03]))],
    }
    monkeypatch.setattr(plots, "_point_source_positions", lambda *_: positions)
    monkeypatch.setattr(
        plots.model_util,
        "critical_lines_caustics",
        lambda *_args, **_kwargs: ([], [(np.array([-0.2, 0.2]), np.array([0.0, 0.1]))]),
    )


@pytest.mark.parametrize("plot_scale", ["linear", "log"])
def test_source_plane_shows_three_components_and_caustics(monkeypatch, tmp_path, plot_scale):
    from herculens_wrapper import visualizations as plots

    _mock_plot_geometry(monkeypatch, plots)
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda *_args, **_kwargs: None)
    kwargs = {
        "kwargs_lens": [],
        "kwargs_source": [{"pixels": np.ones((3, 3))}],
        "kwargs_point_source": [{"amp": [1.0]}],
    }
    try:
        plots.plot_source_plane(
            _mock_plot_lens_image(), kwargs, str(tmp_path),
            plot_scale=plot_scale, output_filename=f"source_{plot_scale}.png",
        )
        figure = plots.plt.gcf()
        extended, points, reconstruction = figure.axes[:3]
        assert len(extended.images) == 1
        assert len(points.images) == 0
        assert len(reconstruction.images) == 1
        assert len(extended.collections) == 0
        assert len(points.collections) == 1
        assert len(reconstruction.collections) == 1
        assert all(any(line.get_color() == "lime" for line in axis.lines)
                   for axis in (extended, points, reconstruction))
        assert "positions only" in points.get_title()
        assert (tmp_path / f"source_{plot_scale}.png").is_file()
    finally:
        original_close(plots.plt.gcf())


@pytest.mark.parametrize("use_override", [False, True])
def test_composite_lensed_source_includes_point_flux(monkeypatch, tmp_path, use_override):
    from herculens_wrapper import visualizations as plots

    _mock_plot_geometry(monkeypatch, plots)
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda *_args, **_kwargs: None)
    kwargs = {
        "kwargs_lens": [],
        "kwargs_source": [{"pixels": np.ones((3, 3))}],
        "kwargs_point_source": [{"amp": [1.0]}],
    }
    overrides = (
        {"model_extended_override": np.full((3, 3), 2.0),
         "model_no_lens_light_override": np.full((3, 3), 11.0)}
        if use_override else {}
    )
    try:
        plots.plot_composite_2x3_panel(
            _mock_plot_lens_image(), kwargs, 0.1,
            np.full((3, 3), 10.0), np.ones((3, 3)), str(tmp_path),
            output_filename="composite.png", **overrides,
        )
        figure = plots.plt.gcf()
        lensed_source = next(axis for axis in figure.axes if axis.get_title() == "Lensed Source + Point Sources")
        np.testing.assert_allclose(
            lensed_source.images[0].get_array(),
            11.0 if use_override else 7.0,
        )
        assert (tmp_path / "composite.png").is_file()
    finally:
        original_close(plots.plt.gcf())


@pytest.mark.parametrize("use_override", [False, True])
def test_svi_run_comparison_lensed_source_includes_point_flux(monkeypatch, tmp_path, use_override):
    from herculens_wrapper import visualizations as plots

    _mock_plot_geometry(monkeypatch, plots)
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda *_args, **_kwargs: None)
    band = {
        "name": "Run 0",
        "lens_image": _mock_plot_lens_image(),
        "kwargs_result": {
            "kwargs_lens": [],
            "kwargs_source": [{"pixels": np.ones((3, 3))}],
            "kwargs_point_source": [{"amp": [1.0]}],
        },
        "image_data": np.full((3, 3), 10.0),
        "noise_map": np.ones((3, 3)),
        "pixel_scale": 0.1,
    }
    if use_override:
        band["model_lensed_source"] = np.full((3, 3), 11.0)
    try:
        plots.plot_multiband_composite(
            [band], str(tmp_path), output_filename="svi_run_comparison.png",
        )
        figure = plots.plt.gcf()
        lensed_source = next(axis for axis in figure.axes if axis.get_title() == "Lensed Source + Point Sources")
        np.testing.assert_allclose(
            lensed_source.images[0].get_array(),
            11.0 if use_override else 7.0,
        )
        assert (tmp_path / "svi_run_comparison.png").is_file()
    finally:
        original_close(plots.plt.gcf())


def test_stellar_nfw_einstein_radius_written_to_mass_json(monkeypatch, tmp_path):
    import json
    from herculens_wrapper import visualizations as plots

    class Grid:
        num_pixel_axes = (101, 101)

        def map_coord2pix(self, x, y):
            return 50 + np.asarray(x) * 40, 50 + np.asarray(y) * 40

    class MassModel:
        profile_type_list = ['STELLAR_MGE', 'NFW_ELLIPSE_KAPPA', 'SHEAR']

        def kappa(self, x, y, kwargs_lens):
            return np.where(np.hypot(x, y) < 0.25, 1.2, 0.5)

        def gamma(self, x, y, kwargs_lens):
            radius = np.hypot(x, y)
            return np.where(radius < 0.25, 0.2, 0.5), np.zeros_like(radius)

    def circle(radius):
        phi = np.linspace(0, 2 * np.pi, 200)
        return radius * np.cos(phi), radius * np.sin(phi)

    lens = SimpleNamespace(
        Grid=SimpleNamespace(create_model_grid=lambda **_: Grid()),
        MassModel=MassModel(),
    )
    kwargs = {'kwargs_lens': [{}, {'center_x': 0.0, 'center_y': 0.0}, {}]}
    monkeypatch.setattr(
        plots.model_util, 'critical_lines_caustics',
        lambda *args, **kw: ([circle(0.15), circle(0.4)], []),
    )
    summary = plots.save_lens_mass_ellipticity_summary(lens, kwargs, tmp_path)
    assert summary['einstein_radius']['status'] == 'ok'
    assert summary['einstein_radius']['theta_E_eff_arcsec'] == pytest.approx(0.4, rel=2e-4)
    saved = json.loads((tmp_path / 'lens_mass_parameters.json').read_text())
    assert saved['einstein_radius'] == summary['einstein_radius']

    x, y = circle(0.4)
    monkeypatch.setattr(
        plots.model_util, 'critical_lines_caustics',
        lambda *args, **kw: ([(x[:100], y[:100])], []),
    )
    unavailable = plots.lens_mass_ellipticity_summary(lens, kwargs)['einstein_radius']
    assert unavailable['status'] == 'not_found'
    assert unavailable['theta_E_eff_arcsec'] is None
