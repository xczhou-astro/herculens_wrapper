import os

# Prevent XLA CPU YNN/Slinky recursion stack overflow (Bus error: 10) on macOS ARM64
_flags = os.environ.get("XLA_FLAGS", "")
if "dot-library-rewriter" not in _flags:
    os.environ["XLA_FLAGS"] = f"{_flags} --xla_disable_hlo_passes=dot-library-rewriter,ynn-fusion".strip()

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WRAPPER_DIR)
HERCULENS_PKG = os.path.join(PROJECT_ROOT, 'herculens')
