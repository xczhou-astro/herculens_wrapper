"""Text progress for hybrid HMC without modifying its sampler state."""
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.mcmc import MCMCKernel

from herculens_wrapper.hmc_chains import DEVICE_AXIS, LOCAL_CHAIN_AXIS


def iteration_number(state):
    """Read the transition counter from joint NUTS or the first Gibbs block."""
    if hasattr(state, "hmc_states"):
        return state.hmc_states[0].i
    return state.i


class HMCTextProgress:
    """Aggregate device updates, with a host heartbeat while JAX is busy."""

    def __init__(self, *, num_devices, warmup, batch_samples, completed_samples,
                 total_samples, interval=50, refresh_seconds=30):
        self.num_devices = num_devices
        self.warmup = warmup
        self.batch_samples = batch_samples
        self.completed_samples = completed_samples
        self.total_samples = total_samples
        self.interval = interval
        self.refresh_seconds = refresh_seconds
        self._steps = [0] * num_devices
        self._reported = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._active = False

    def __enter__(self):
        self._start = time.monotonic()
        self._active = True
        print(f"[hmc] Text progress: {self.warmup} warmup iterations + "
              f"{self.batch_samples} draws/chain in this batch; "
              "waiting for initialization/JIT compilation and the first iteration...", flush=True)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _message(self, step):
        elapsed = time.monotonic() - self._start
        if step == 0:
            return ("waiting for initialization/JIT compilation or the first iteration; "
                    f"elapsed={elapsed:.0f}s")
        if self.warmup and step <= self.warmup:
            return (f"Warmup {step}/{self.warmup} iterations/chain "
                    f"({100 * step / self.warmup:.1f}%); batch elapsed={elapsed:.0f}s")
        draws = min(max(step - self.warmup, 0), self.batch_samples)
        total = self.completed_samples + draws
        return (f"Sampling {total}/{self.total_samples} draws/chain "
                f"({100 * total / self.total_samples:.1f}%); "
                f"batch {draws}/{self.batch_samples}; batch elapsed={elapsed:.0f}s")

    def update(self, device, step):
        """Callbacks may arrive out of order; never regress reported progress."""
        with self._lock:
            if not self._active:
                return
            index = int(device)
            self._steps[index] = max(self._steps[index], int(step))
            common_step = min(self._steps)
            self._report(common_step)

    def _report(self, step):
        """Called under the lock; preserve phase completion if callbacks skip ahead."""
        if step <= self._reported:
            return
        if self._reported < self.warmup < step:
            print(f"[hmc] {self._message(self.warmup)}", flush=True)
        self._reported = step
        print(f"[hmc] {self._message(step)}", flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.refresh_seconds):
            with self._lock:
                print(f"[hmc] {self._message(min(self._steps))}; "
                      "waiting for next progress update", flush=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._active = False
            if exc_type is None:
                final = self.warmup + self.batch_samples
                self._report(final)
            else:
                print(f"[hmc] Batch interrupted; last reported "
                      f"{self._message(min(self._steps))}", flush=True)


class ProgressLoggingKernel(MCMCKernel):
    """Delegate transitions unchanged; log only counters, without RNG use."""

    def __init__(self, kernel, progress, *, start_state=None, mapped=True):
        self.kernel = kernel
        self.progress = progress
        self.mapped = mapped
        self.offset = (0 if start_state is None else
                       int(np.min(np.asarray(iteration_number(start_state)))))

    @property
    def sample_field(self):
        return self.kernel.sample_field

    @property
    def default_fields(self):
        return self.kernel.default_fields

    @property
    def is_ensemble_kernel(self):
        return self.kernel.is_ensemble_kernel

    @property
    def _sample_fn(self):
        return getattr(self.kernel, "_sample_fn", None)

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return self.kernel.init(rng_key, num_warmup, init_params, model_args, model_kwargs)

    def postprocess_fn(self, model_args, model_kwargs):
        return self.kernel.postprocess_fn(model_args, model_kwargs)

    def get_diagnostics_str(self, state):
        return self.kernel.get_diagnostics_str(state)

    def sample(self, state, model_args, model_kwargs):
        state = self.kernel.sample(state, model_args, model_kwargs)
        step = iteration_number(state) - self.offset
        if self.mapped:
            # Reduce the vmap axis BEFORE cond: its predicate and callback are
            # then unbatched, giving one occasional callback per device rather
            # than a callback for every chain at every transition.
            step = jax.lax.pmin(step, LOCAL_CHAIN_AXIS)
            device = jax.lax.axis_index(DEVICE_AXIS)
        else:
            device = jnp.int32(0)
        warmup = self.progress.warmup
        due = ((step == 1) | (step == warmup)
               | (step == warmup + self.progress.batch_samples)
               | ((step <= warmup) & (step % self.progress.interval == 0))
               | ((step > warmup) & ((step - warmup) % self.progress.interval == 0)))

        def report(_):
            # Unordered callbacks work under pmap. The host aggregator handles
            # duplicates/reordering and reports the slowest device's counter.
            jax.debug.callback(self.progress.update, device, step)
            return jnp.int32(0)

        jax.lax.cond(due, report, lambda _: jnp.int32(0), operand=None)
        return state
