import multiprocessing as mp
import os
from pathlib import Path
import sys

import pytest
import numpy as np

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
