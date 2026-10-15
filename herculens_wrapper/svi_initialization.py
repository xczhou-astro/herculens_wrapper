"""SVI initialization helpers, keeping the best finite observed ELBO state."""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.svi import SVIRunResult


def run_svi_best(svi, rng_key, num_steps, *args, progress_bar=True, init_params=None):
    """Retain the parameter state at the lowest finite update loss.

    ``SVI.stable_update`` evaluates the loss BEFORE the optimizer update, so
    the corresponding best state is the incoming state, not the returned one.
    Only one best state is retained, rather than every high-dimensional draw.
    This is selection by the observed stochastic negative ELBO, not a claim
    that the global optimum has been found.
    """
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    state = svi.init(rng_key, *args, init_params=init_params)
    initial_carry = (state, state, jnp.asarray(jnp.inf), jnp.int32(0), jnp.int32(0))

    def step(carry, _):
        current, best, best_loss, best_iteration, iteration = carry
        updated, loss = svi.stable_update(current, *args)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best = jax.lax.cond(improved, lambda _: current, lambda _: best, operand=None)
        return (updated, best, jnp.where(improved, loss, best_loss),
                jnp.where(improved, iteration, best_iteration), iteration + 1), loss

    if progress_bar:
        from tqdm.auto import trange

        update = jax.jit(step)
        carry = initial_carry
        losses = []
        with trange(num_steps, desc="Pixelated initialization SVI") as progress:
            for _ in progress:
                carry, loss = update(carry, None)
                losses.append(jax.device_get(loss))
                progress.set_postfix(best_loss=float(carry[2]), refresh=False)
        losses = jnp.asarray(losses)
    else:
        carry, losses = jax.lax.scan(step, initial_carry, None, length=num_steps)
    _, best, best_loss, best_iteration, _ = carry
    if not np.isfinite(float(best_loss)):
        raise RuntimeError("Pixelated initialization SVI produced no finite ELBO loss.")
    print(f"[pixelated-init] Selected best finite SVI loss={float(best_loss):.6g} "
          f"at evaluated state {int(best_iteration)} of {num_steps}.", flush=True)
    return SVIRunResult(svi.get_params(best), best, losses)
