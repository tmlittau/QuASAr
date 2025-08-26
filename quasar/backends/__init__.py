"""Simulation backend adapters for QuASAr."""

from .base import Backend
from .statevector import StatevectorBackend
from .mps import MPSBackend
from ..cost import Backend as BackendType

# Optional backends -------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from .stim_backend import StimBackend
except ImportError as exc:  # pragma: no cover - executed when stim missing
    class StimBackend(Backend):
        """Stub used when the optional ``stim`` dependency is missing."""

        backend: BackendType = BackendType.TABLEAU

        def _unavailable(self, *_, **__):
            raise ImportError(
                "Stim backend requires the 'stim' package. "
                "Install 'stim' to use this backend."
            ) from exc

        load = ingest = apply_gate = extract_ssd = _unavailable

try:  # pragma: no cover - optional dependency
    from .mqt_dd import DecisionDiagramBackend
except ImportError as exc:  # pragma: no cover - executed when MQT libraries missing
    class DecisionDiagramBackend(Backend):
        """Stub for the decision diagram backend when MQT packages are missing."""

        backend: BackendType = BackendType.DECISION_DIAGRAM

        def _unavailable(self, *_, **__):
            raise ImportError(
                "DecisionDiagramBackend requires the 'mqt.core' and 'mqt.ddsim' "
                "packages. Install them to use this backend."
            ) from exc

        load = ingest = apply_gate = extract_ssd = _unavailable

__all__ = [
    "Backend",
    "StatevectorBackend",
    "MPSBackend",
    "StimBackend",
    "DecisionDiagramBackend",
]
