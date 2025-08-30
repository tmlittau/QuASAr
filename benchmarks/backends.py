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

    # ------------------------------------------------------------------
    def prepare(self, circuit: Circuit) -> Tuple[int, List[Tuple[str, Sequence[int], Dict[str, Any]]]]:
        """Convert a :class:`Circuit` into a lightweight gate list.

        The returned tuple ``(n, ops)`` contains the number of qubits and a
        list of operations where each operation is represented as
        ``(gate, qubits, params)``.  This step can be performed prior to
        invoking :meth:`run` so that conversion overhead is excluded from the
        measured runtime.  Any substantial translation work must happen here,
        ensuring that ``run_time`` reflects only the actual gate execution.
        """

        ops = [(g.gate, g.qubits, g.params) for g in circuit.gates]
        return circuit.num_qubits, ops

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Union[
            Circuit, Tuple[int, Iterable[Tuple[str, Sequence[int], Dict[str, Any]]]]
        ],
        *,
        return_state: bool = True,
    ) -> Any:
        """Execute ``circuit`` on the native backend."""

        if isinstance(circuit, Circuit):
            # Fallback path: preparation happens inside ``run`` which means the
            # translation cost will be included in ``run_time``.
            num_qubits, ops = self.prepare(circuit)
        else:
            num_qubits, ops = circuit

        backend = self.backend_cls()
        backend.load(num_qubits)
        for name, qubits, params in ops:
            backend.apply_gate(name, qubits, params)

        if not return_state:
            return backend

        # Return whatever state representation the backend exposes.  The
        # return value is not interpreted by :class:`BenchmarkRunner` but may
        # be useful for sanity checks.
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
        super().__init__(name="statevector", backend_cls=StatevectorBackend)


class StimAdapter(BackendAdapter):
    """Adapter executing circuits using the Stim tableau simulator."""

    def __init__(self) -> None:
        super().__init__(name="stim", backend_cls=StimBackend)


class MPSAdapter(BackendAdapter):
    """Adapter executing circuits using the MPS simulator."""

    def __init__(self) -> None:
        super().__init__(name="mps", backend_cls=MPSBackend)


class DecisionDiagramAdapter(BackendAdapter):
    """Adapter executing circuits using the decision diagram simulator."""

    def __init__(self) -> None:
        super().__init__(name="mqt_dd", backend_cls=DecisionDiagramBackend)


__all__ = [
    "BackendAdapter",
    "StatevectorAdapter",
    "StimAdapter",
    "MPSAdapter",
    "DecisionDiagramAdapter",
]
