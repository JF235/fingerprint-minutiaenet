"""
MinutiaeNet - Robust Minutiae Extractor (PyTorch)

Lightweight package exports with lazy-loading.
Heavy submodules are imported only when their symbols are actually used.
"""
from typing import Any


class _LazyImport:
    """Proxy that imports the real object on first access/call."""

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._obj: Any | None = None

    def _load(self) -> Any:
        if self._obj is None:
            mod = __import__(f"{__package__}.{self._module}", fromlist=[self._name])
            self._obj = getattr(mod, self._name)
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __repr__(self) -> str:
        return f"<LazyImport {self._module}.{self._name}>"


# Public API - lazy proxies
run_inference = _LazyImport("api", "run_inference")

# CoarseNet
get_coarsenet_core = _LazyImport("coarsenet_model", "get_coarsenet_core")
CoarseNet = _LazyImport("coarsenet_model", "CoarseNet")

# FineNet
get_finenet_core = _LazyImport("finenet_model", "get_finenet_core")
FineNet = _LazyImport("finenet_model", "FineNet")

# Wrapper (CoarseNet + optional FineNet)
get_minutiaenet = _LazyImport("wrapper", "get_minutiaenet")
MinutiaeNetWrapper = _LazyImport("wrapper", "MinutiaeNetWrapper")

# Visualization
plot_output = _LazyImport("plot", "plot_output")
plot_raw_output = _LazyImport("plot", "plot_raw_output")

# Utilities
get_minutiaenet_logger = _LazyImport("mnet_utils", "get_minutiaenet_logger")
MnetTimer = _LazyImport("mnet_utils", "MnetTimer")
