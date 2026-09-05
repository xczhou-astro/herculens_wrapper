"""Internal, process-safe logging for the public modelling API."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


_ACTIVE_LOG_PATH: ContextVar[str | None] = ContextVar(
    "herculens_wrapper_active_log_path", default=None,
)


class _Tee:
    """Mirror text to the existing stream and one run-local log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _json_safe(value: Any) -> Any:
    """Keep configuration snapshots readable without embedding image arrays."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        array = np.asarray(value)
        if array.ndim == 0:
            return array.item()
        if array.size <= 16:
            return array.tolist()
        summary: dict[str, Any] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if np.issubdtype(array.dtype, np.number) and array.size:
            finite = array[np.isfinite(array)]
            if finite.size:
                summary.update({"min": float(finite.min()), "max": float(finite.max())})
        return summary
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def format_configuration(configuration: dict[str, Any]) -> str:
    """Return a stable human-readable form shared by ``describe`` and logs."""
    return json.dumps(_json_safe(configuration), indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class RunContext:
    """Lightweight run identity; no open streams are retained or pickled."""

    directory: Path
    log_name: str = "log.txt"
    console: bool = True
    run_id: int | str | None = None

    def __post_init__(self):
        object.__setattr__(self, "directory", Path(self.directory).expanduser())

    @property
    def log_path(self) -> Path:
        return self.directory / self.log_name

    def save_configuration(self, configuration: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        output = self.directory / "model_configuration.json"
        output.write_text(format_configuration(configuration) + "\n", encoding="utf-8")
        return output

    @contextmanager
    def capture(self, operation: str):
        """Capture Python output, and worker-native output when console is disabled."""
        self.directory.mkdir(parents=True, exist_ok=True)
        resolved = str(self.log_path.resolve())
        if _ACTIVE_LOG_PATH.get() == resolved:
            yield self
            return

        token = _ACTIVE_LOG_PATH.set(resolved)
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(f"\n--- {operation} started {timestamp} pid={os.getpid()} ---\n")
            original_stdout, original_stderr = sys.stdout, sys.stderr
            saved_fds: tuple[int, int] | None = None
            status = "completed"
            try:
                if self.console:
                    stdout, stderr = _Tee(original_stdout, stream), _Tee(original_stderr, stream)
                else:
                    stdout = stderr = stream
                    try:
                        original_stdout.flush()
                        original_stderr.flush()
                        saved_fds = (os.dup(1), os.dup(2))
                        os.dup2(stream.fileno(), 1)
                        os.dup2(stream.fileno(), 2)
                    except (AttributeError, OSError):
                        if saved_fds is not None:
                            os.close(saved_fds[0])
                            os.close(saved_fds[1])
                        saved_fds = None
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    yield self
            except BaseException:
                status = "failed"
                traceback.print_exc(file=stream)
                raise
            finally:
                try:
                    stdout.flush()
                    stderr.flush()
                except Exception:
                    pass
                if saved_fds is not None:
                    os.dup2(saved_fds[0], 1)
                    os.dup2(saved_fds[1], 2)
                    os.close(saved_fds[0])
                    os.close(saved_fds[1])
                finished = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                stream.write(f"--- {operation} {status} {finished} ---\n")
                _ACTIVE_LOG_PATH.reset(token)


def logged_model_run(method):
    """Bind a model run to ``save_path`` without changing the public call."""
    @wraps(method)
    def wrapped(self, sampler, *args, **kwargs):
        save_path = kwargs.get("save_path")
        context = getattr(self, "_run_context", None)
        if save_path is not None:
            directory = Path(save_path).expanduser()
            if context is None or context.directory != directory:
                context = RunContext(directory)
                self._run_context = context
        if context is None:
            return method(self, sampler, *args, **kwargs)
        with context.capture(f"{type(self).__name__}.run[{sampler.name}]"):
            configuration = self.configuration(sampler=sampler)
            context.save_configuration(configuration)
            result = method(self, sampler, *args, **kwargs)
            # A model may be reused for multiple runs.  Pinning the context to
            # each returned result prevents an older result from following the
            # model to a later run directory.
            try:
                result._run_context = context
            except (AttributeError, TypeError):
                pass
            return result
    return wrapped


def logged_result_output(method):
    """Continue result output in the model's run log, or bind on first output."""
    @wraps(method)
    def wrapped(self, save_path=None, *args, **kwargs):
        model = getattr(self, "_model", None)
        context = getattr(self, "_run_context", None)
        if context is None and model is not None:
            context = getattr(model, "_run_context", None)
        explicit_save_path = save_path is not None
        if save_path is None:
            if context is None:
                raise ValueError("output() needs save_path when the result has no run directory.")
            save_path = context.directory
        directory = Path(save_path).expanduser()
        if context is None or (explicit_save_path and context.directory != directory):
            context = RunContext(directory)
            self._run_context = context
            if model is not None and getattr(model, "_run_context", None) is None:
                model._run_context = context
        with context.capture(f"{type(self).__name__}.output"):
            print(f"[api] Writing result products to: {directory}")
            return method(self, directory, *args, **kwargs)
    return wrapped
