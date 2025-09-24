"""Simulation backend adapters for QuASAr."""

from importlib.util import find_spec

from .base import Backend


def _require(package: str, backend: str) -> None:
    """Ensure that *package* is available.

    Parameters
    ----------
    package:
        Name of the Python package providing the backend.
    backend:
        Name of the backend class that depends on ``package``.

    Raises
    ------
    ImportError
        If the requested ``package`` cannot be found.
    """

    if find_spec(package) is None:  # pragma: no cover - environment specific
        raise ImportError(
            f"{backend} requires the '{package}' package. Install it to use this backend."
        )


# Core backends -----------------------------------------------------------
_require("qiskit_aer", "StatevectorBackend")
from .statevector import (
    StatevectorBackend,
    AerStatevectorBackend,
    ExtendedStabilizerBackend,
)

_require("qiskit_aer", "MPSBackend")
from .mps import MPSBackend, AerMPSBackend

# Optional backends -------------------------------------------------------
_require("stim", "StimBackend")
from .stim_backend import StimBackend

_require("mqt.core", "DecisionDiagramBackend")
from .mqt_dd import DecisionDiagramBackend

__all__ = [
    "Backend",
    "StatevectorBackend",
    "AerStatevectorBackend",
    "ExtendedStabilizerBackend",
    "MPSBackend",
    "AerMPSBackend",
    "StimBackend",
    "DecisionDiagramBackend",
]

