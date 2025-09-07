"""Convenience wrappers exposing QuASAr backends for benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Dict

from quasar.backends import (
    StatevectorBackend,
    MPSBackend,
    DecisionDiagramBackend,
    StimBackend,
)


@dataclass
class _Adapter:
    """Thin wrapper around a QuASAr backend with an optional state return."""

    backend: Any
    name: str

    def load(self, num_qubits: int, **kwargs: Any) -> None:  # pragma: no cover - simple passthrough
        self.backend.load(num_qubits, **kwargs)

    def prepare_benchmark(self, circuit: Any | None = None) -> None:  # pragma: no cover
        self.backend.prepare_benchmark(circuit)

    def apply_gate(self, gate: str, qubits: Sequence[int], params: Dict[str, float] | None = None) -> None:  # pragma: no cover
        self.backend.apply_gate(gate, qubits, params)

    def run_benchmark(self, *, return_state: bool = False) -> Any | None:
        result = self.backend.run_benchmark(return_state=return_state)
        if return_state:
            if result is None and hasattr(self.backend, "statevector"):
                try:
                    return self.backend.statevector()
                except Exception:  # pragma: no cover - best effort
                    return result
            return result
        return None

    # Convenience to mimic backend interface ---------------------------------
    def extract_ssd(self) -> Any:  # pragma: no cover - passthrough
        return self.backend.extract_ssd()

    def statevector(self) -> Any:  # pragma: no cover - passthrough
        return self.backend.statevector()

    # Convenience run method used in some contexts ---------------------------
    def run(self, circuit: Any, **kwargs: Any) -> Any | None:
        self.load(getattr(circuit, "num_qubits", 0))
        self.prepare_benchmark(circuit)
        for g in getattr(circuit, "gates", []):
            self.apply_gate(g.gate, g.qubits, g.params)
        return self.run_benchmark(**kwargs)


class StatevectorAdapter(_Adapter):
    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(StatevectorBackend(), "statevector")


class DecisionDiagramAdapter(_Adapter):
    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(DecisionDiagramBackend(), "mqt_dd")


class MPSAdapter(_Adapter):
    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(MPSBackend(), "mps")


class StimAdapter(_Adapter):
    def __init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(StimBackend(), "stim")


__all__ = [
    "StatevectorAdapter",
    "DecisionDiagramAdapter",
    "MPSAdapter",
    "StimAdapter",
]
