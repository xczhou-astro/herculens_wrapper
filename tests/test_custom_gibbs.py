import jax
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from herculens_wrapper.custom_gibbs import MultiHMCGibbs


def _model():
    x = numpyro.sample("x", dist.Normal())
    numpyro.sample("y", dist.Normal(x, 1.0))


def _kernel():
    return MultiHMCGibbs(
        [NUTS(_model, max_tree_depth=3), NUTS(_model, max_tree_depth=3)],
        [["x"], ["y"]],
    )


def test_single_chain_interface_is_unchanged():
    mcmc = MCMC(
        _kernel(),
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0))

    assert jax.random.key_data(mcmc.last_state.rng_key).shape == (2,)
    assert all(value.shape == (2,) for value in mcmc.get_samples().values())


@pytest.mark.parametrize("chain_method", ["sequential", "vectorized", "parallel"])
def test_multiple_chains_keep_one_outer_rng_key_per_chain(chain_method):
    if chain_method == "parallel" and jax.local_device_count() < 2:
        pytest.skip("parallel test needs two JAX devices")

    mcmc = MCMC(
        _kernel(),
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0))

    assert jax.random.key_data(mcmc.last_state.rng_key).shape == (2, 2)
    assert {name: value.shape for name, value in mcmc.get_samples(True).items()} == {
        "x": (2, 2),
        "y": (2, 2),
    }


def test_parallel_chains_resume_without_rng_shape_change():
    if jax.local_device_count() < 2:
        pytest.skip("parallel test needs two JAX devices")

    kernel = _kernel()
    first = MCMC(
        kernel,
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        chain_method="parallel",
        progress_bar=False,
    )
    first.run(jax.random.PRNGKey(1))

    resumed = MCMC(
        kernel,
        num_warmup=0,
        num_samples=2,
        num_chains=2,
        chain_method="parallel",
        progress_bar=False,
    )
    resumed.post_warmup_state = first.last_state
    resumed.run(first.last_state.rng_key)

    assert jax.random.key_data(resumed.last_state.rng_key).shape == (2, 2)
    assert all(value.shape == (2, 2) for value in resumed.get_samples(True).values())
