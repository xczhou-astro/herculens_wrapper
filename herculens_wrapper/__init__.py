import os

# Prevent XLA CPU YNN/Slinky recursion stack overflow (Bus error: 10) on macOS ARM64
_flags = os.environ.get("XLA_FLAGS", "")
if "dot-library-rewriter" not in _flags:
    os.environ["XLA_FLAGS"] = f"{_flags} --xla_disable_hlo_passes=dot-library-rewriter,ynn-fusion".strip()


def _patch_jax_clip():
    """Ensure jax.numpy.clip accepts both (a_min, a_max) and (min, max) seamlessly across JAX versions."""
    try:
        import functools
        import jax.numpy as jnp
    except ImportError:
        return

    orig_clip = jnp.clip
    if getattr(orig_clip, "_is_compatible_patched", False):
        return

    @functools.wraps(orig_clip)
    def compatible_clip(*args, **kwargs):
        if "a_min" in kwargs or "a_max" in kwargs or "a" in kwargs:
            try:
                return orig_clip(*args, **kwargs)
            except TypeError:
                k = dict(kwargs)
                if "a_min" in k:
                    k["min"] = k.pop("a_min")
                if "a_max" in k:
                    k["max"] = k.pop("a_max")
                if "a" in k:
                    k["arr"] = k.pop("a")
                return orig_clip(*args, **k)
        elif "min" in kwargs or "max" in kwargs or "arr" in kwargs:
            try:
                return orig_clip(*args, **kwargs)
            except TypeError:
                k = dict(kwargs)
                if "min" in k:
                    k["a_min"] = k.pop("min")
                if "max" in k:
                    k["a_max"] = k.pop("max")
                if "arr" in k:
                    k["a"] = k.pop("arr")
                return orig_clip(*args, **k)
        return orig_clip(*args, **kwargs)

    compatible_clip._is_compatible_patched = True

    import jax
    jax.numpy.clip = compatible_clip
    jnp.clip = compatible_clip
    try:
        import jax._src.numpy.lax_numpy as lax_numpy
        lax_numpy.clip = compatible_clip
    except (ImportError, AttributeError):
        pass


_patch_jax_clip()

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WRAPPER_DIR)
HERCULENS_PKG = os.path.join(PROJECT_ROOT, 'herculens')
