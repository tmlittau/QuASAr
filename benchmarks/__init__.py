"""Compatibility shims for relocated benchmark utilities.

The benchmark suite now stores most helper modules under
:mod:`benchmarks.bench_utils`.  Importing submodules through the legacy
``benchmarks`` package path continues to work via the extended ``__path__``
configured here.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

_BENCH_UTILS_PATH = Path(__file__).resolve().parent / "bench_utils"
if str(_BENCH_UTILS_PATH) not in __path__:
    __path__.append(str(_BENCH_UTILS_PATH))


def __getattr__(name: str) -> Any:
    """Dynamically resolve submodules from :mod:`benchmarks.bench_utils`."""

    return import_module(f"{__name__}.bench_utils.{name}")


def __dir__() -> Iterable[str]:  # pragma: no cover - convenience
    names = set(globals())
    try:
        pkg = import_module(f"{__name__}.bench_utils")
    except ModuleNotFoundError:  # pragma: no cover - defensive
        return sorted(names)
    return sorted(names | set(getattr(pkg, "__all__", [])))


__all__ = ["bench_utils"]
