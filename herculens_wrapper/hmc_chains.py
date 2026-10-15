"""NumPyro chain transform: pmap across devices, vmap within each device."""
from functools import wraps


class HybridChainMap:
    """Keep a flat public chain axis for samples, diagnostics and checkpoints.

    Only inside the transform is the axis split into devices and local chains.
    NumPyro receives a single-chain function; vmap maps that complete function,
    including its postprocessing, without changing the MCMC kernel interface.
    """

    def __init__(self, num_chains, chains_per_device=1):
        import jax

        for name, value in (("num_chains", num_chains), ("chains_per_device", chains_per_device)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if num_chains % chains_per_device:
            raise ValueError("Hybrid num_chains must be divisible by chains_per_device.")
        self.num_chains = num_chains
        self.chains_per_device = chains_per_device
        self.num_devices = num_chains // chains_per_device
        available = jax.local_devices()
        if self.num_devices > len(available):
            raise ValueError(
                f"Hybrid HMC needs {self.num_devices} JAX devices for {num_chains} chains "
                f"with {chains_per_device} chains per device, but only {len(available)} are visible. "
                "Select more GPUs or increase chains_per_device (--chains-per-gpu)."
            )
        self.devices = tuple(available[:self.num_devices])

    def __call__(self, function):
        import jax
        import jax.numpy as jnp

        mapped = jax.pmap(jax.vmap(function), devices=self.devices, axis_name="hmc_devices")

        @wraps(function)
        def apply(arguments):
            def group(value):
                value = jnp.asarray(value)
                if value.ndim == 0 or value.shape[0] != self.num_chains:
                    raise ValueError("Hybrid HMC input must have one leading entry per chain.")
                return value.reshape((self.num_devices, self.chains_per_device) + value.shape[1:])

            grouped = jax.tree_util.tree_map(group, arguments)
            result = mapped(grouped)
            return jax.tree_util.tree_map(
                lambda value: value.reshape((self.num_chains,) + value.shape[2:]), result,
            )

        return apply
