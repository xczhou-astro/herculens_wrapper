import multiprocessing as mp
import os
from pathlib import Path
import sys

import pytest

from herculens_wrapper.api import (
    LensProfileCollection,
    MassProfile,
    PixelatedLensLight,
    PixelatedSource,
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
