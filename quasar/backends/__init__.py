"""Simulation backend adapters for QuASAr."""

from .base import Backend
from ..cost import Backend as BackendType

# Core backends -----------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from .statevector import StatevectorBackend, AerStatevectorBackend
except ImportError as exc:  # pragma: no cover - executed when qiskit is missing
    class StatevectorBackend(Backend):
        """Stub when Qiskit Aer is not installed."""

        backend: BackendType = BackendType.STATEVECTOR

        def _unavailable(self, *_, **__):
            raise ImportError(
                "StatevectorBackend requires the 'qiskit-aer' package. "
                "Install it to use this backend."
            ) from exc

        load = ingest = apply_gate = extract_ssd = statevector = _unavailable

    class AerStatevectorBackend(StatevectorBackend):
        pass

try:  # pragma: no cover - optional dependency
    from .mps import MPSBackend, AerMPSBackend
except ImportError as exc:  # pragma: no cover - executed when qiskit is missing
    class MPSBackend(Backend):
        """Stub when Qiskit Aer is not installed."""

        backend: BackendType = BackendType.MPS

        def _unavailable(self, *_, **__):
            raise ImportError(
                "MPSBackend requires the 'qiskit-aer' package. "
                "Install it to use this backend."
            ) from exc

        load = ingest = apply_gate = extract_ssd = statevector = _unavailable

    class AerMPSBackend(MPSBackend):
        pass

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
                "DecisionDiagramBackend requires the 'mqt.core' package. "
                "Install it to use this backend."
            ) from exc

        load = ingest = apply_gate = extract_ssd = _unavailable

__all__ = [
    "Backend",
    "StatevectorBackend",
    "AerStatevectorBackend",
    "MPSBackend",
    "AerMPSBackend",
    "StimBackend",
    "DecisionDiagramBackend",
]
