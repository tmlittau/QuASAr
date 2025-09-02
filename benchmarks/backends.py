from __future__ import annotations

"""Native backend adapters for benchmarking."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from quasar.circuit import Circuit
from quasar.backends import (
    StatevectorBackend,
    StimBackend,
    MPSBackend,
    DecisionDiagramBackend,
)


@dataclass
class BackendAdapter:
    """Generic adapter implementing the :class:`BenchmarkRunner` protocol."""

    name: str
    backend_cls: Any
    defer_build: bool = True

    # ------------------------------------------------------------------
    def prepare(self, circuit: Circuit) -> Any:
        """Prepare ``circuit`` for execution on the native backend.

        When ``defer_build`` is ``True`` the circuit is converted into a
        lightweight gate list that is later consumed during :meth:`run`.  If
        ``defer_build`` is ``False`` all gates are applied to a fresh backend
        instance and the fully prepared backend object is returned.  This
        allows expensive translation or transpilation steps to be accounted for
        in ``prepare_time`` while keeping ``run_time`` focused on the actual
        simulation.
        """

        if self.defer_build:
            ops = [(g.gate, g.qubits, g.params) for g in circuit.gates]
            return circuit.num_qubits, ops

        backend = self.backend_cls()
        backend.load(circuit.num_qubits)
        for g in circuit.gates:
            backend.apply_gate(g.gate, g.qubits, g.params)
        return backend

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Any,
        *,
        return_state: bool = True,
    ) -> Any:
        """Execute ``circuit`` on the native backend."""

        if self.defer_build:
            if isinstance(circuit, Circuit):
                # Fallback path: preparation happens inside ``run`` which means
                # the translation cost will be included in ``run_time``.
                num_qubits, ops = self.prepare(circuit)
            else:
                num_qubits, ops = circuit  # type: ignore[misc]

            backend = self.backend_cls()
            backend.load(num_qubits)
            for name, qubits, params in ops:
                backend.apply_gate(name, qubits, params)

            if not return_state:
                try:
                    backend.statevector()  # type: ignore[call-arg]
                except Exception:
                    try:
                        backend.extract_ssd()
                    except Exception:
                        pass
                return backend

            try:
                return backend.statevector()  # type: ignore[call-arg]
            except Exception:
                try:
                    return backend.extract_ssd()
                except Exception:
                    return None

        else:
            backend = circuit if not isinstance(circuit, Circuit) else self.prepare(circuit)

            if not return_state:
                try:
                    backend.statevector()  # type: ignore[call-arg]
                except Exception:
                    try:
                        backend.extract_ssd()
                    except Exception:
                        pass
                return backend

            try:
                return backend.statevector()  # type: ignore[call-arg]
            except Exception:
                try:
                    return backend.extract_ssd()
                except Exception:
                    return None


class StatevectorAdapter(BackendAdapter):
    """Adapter executing circuits using the dense statevector backend."""

    def __init__(self) -> None:
        super().__init__(
            name="statevector", backend_cls=StatevectorBackend, defer_build=False
        )


class StimAdapter(BackendAdapter):
    """Adapter executing circuits using the Stim tableau simulator."""

    def __init__(self) -> None:
        super().__init__(name="stim", backend_cls=StimBackend)


class MPSAdapter(BackendAdapter):
    """Adapter executing circuits using the MPS simulator."""

    def __init__(self) -> None:
        super().__init__(name="mps", backend_cls=MPSBackend, defer_build=False)


class DecisionDiagramAdapter(BackendAdapter):
    """Adapter executing circuits using the decision diagram simulator."""

    def __init__(self) -> None:
        super().__init__(
            name="mqt_dd", backend_cls=DecisionDiagramBackend, defer_build=False
        )


__all__ = [
    "BackendAdapter",
    "StatevectorAdapter",
    "StimAdapter",
    "MPSAdapter",
    "DecisionDiagramAdapter",
]
