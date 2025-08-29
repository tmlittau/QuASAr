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
class _BaseAdapter:
    """Common helper implementing the BenchmarkRunner interface."""

    name: str
    backend_cls: Any

    # ------------------------------------------------------------------
    def prepare(self, circuit: Circuit) -> Tuple[int, List[Tuple[str, Sequence[int], Dict[str, Any]]]]:
        """Convert a :class:`Circuit` into a lightweight gate list.

        The returned tuple ``(n, ops)`` contains the number of qubits and a
        list of operations where each operation is represented as
        ``(gate, qubits, params)``.  This step can be performed prior to
        invoking :meth:`run` so that conversion overhead is excluded from the
        measured runtime.
        """

        ops = [(g.gate, g.qubits, g.params) for g in circuit.gates]
        return circuit.num_qubits, ops

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Union[
            Circuit, Tuple[int, Iterable[Tuple[str, Sequence[int], Dict[str, Any]]]]
        ],
    ) -> Any:
        """Execute ``circuit`` on the native backend.

        ``circuit`` may either be a :class:`Circuit` instance or the prepared
        ``(num_qubits, ops)`` tuple returned by :meth:`prepare`.  Passing a
        precompiled circuit allows benchmarks to measure only the actual
        simulation time, excluding conversion costs.
        """

        if isinstance(circuit, Circuit):
            num_qubits, ops = self.prepare(circuit)
        else:
            num_qubits, ops = circuit

        backend = self.backend_cls()
        backend.load(num_qubits)
        for name, qubits, params in ops:
            backend.apply_gate(name, qubits, params)

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


class StatevectorAdapter(_BaseAdapter):
    """Adapter executing circuits using the dense statevector backend."""

    def __init__(self) -> None:
        super().__init__(name="statevector", backend_cls=StatevectorBackend)


class StimAdapter(_BaseAdapter):
    """Adapter executing circuits using the Stim tableau simulator."""

    def __init__(self) -> None:
        super().__init__(name="stim", backend_cls=StimBackend)


class MPSAdapter(_BaseAdapter):
    """Adapter executing circuits using the MPS simulator."""

    def __init__(self) -> None:
        super().__init__(name="mps", backend_cls=MPSBackend)


class DecisionDiagramAdapter(_BaseAdapter):
    """Adapter executing circuits using the decision diagram simulator."""

    def __init__(self) -> None:
        super().__init__(name="mqt_dd", backend_cls=DecisionDiagramBackend)


class _AerAdapter(_BaseAdapter):
    """Common helper for Qiskit Aer based backends."""

    method: str

    def __init__(self, name: str, method: str) -> None:
        super().__init__(name=name, backend_cls=None)
        self.method = method

    def run(
        self,
        circuit: Union[
            Circuit, Tuple[int, Iterable[Tuple[str, Sequence[int], Dict[str, Any]]]]
        ],
    ) -> Any:  # pragma: no cover - optional dependency
        if isinstance(circuit, Circuit):
            num_qubits, ops = self.prepare(circuit)
        else:
            num_qubits, ops = circuit

        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(num_qubits)
        for name, qubits, params in ops:
            func = getattr(qc, name.lower(), None)
            if func is None:
                raise NotImplementedError(f"Unsupported Qiskit gate {name}")
            args = [float(v) for v in params.values()] if params else []
            func(*args, *qubits)

        qc.save_statevector()
        sim = AerSimulator(method=self.method)
        result = sim.run(qc).result()
        try:
            return result.get_statevector()
        except Exception:
            return None


class AerStatevectorAdapter(_AerAdapter):
    """Adapter for the Qiskit Aer dense statevector simulator."""

    def __init__(self) -> None:  # pragma: no cover - optional dependency
        super().__init__(name="aer_statevector", method="statevector")


class AerMPSAdapter(_AerAdapter):
    """Adapter for the Qiskit Aer matrix product state simulator."""

    def __init__(self) -> None:  # pragma: no cover - optional dependency
        super().__init__(name="aer_mps", method="matrix_product_state")


class MQTDDAdapter(_BaseAdapter):
    """Adapter executing circuits using the MQT decision diagram simulator."""

    _ALIASES = {"SDG": "sdg", "U1": "p"}

    def __init__(self) -> None:
        super().__init__(name="mqt_ddsim", backend_cls=None)

    def run(
        self,
        circuit: Union[
            Circuit, Tuple[int, Iterable[Tuple[str, Sequence[int], Dict[str, Any]]]]
        ],
    ) -> Any:  # pragma: no cover - optional dependency
        if isinstance(circuit, Circuit):
            num_qubits, ops = self.prepare(circuit)
        else:
            num_qubits, ops = circuit

        from mqt.core.ir import QuantumComputation
        import mqt.ddsim as ddsim

        qc = QuantumComputation(num_qubits)
        for name, qubits, params in ops:
            lname = self._ALIASES.get(name.upper(), name.lower())
            func = getattr(qc, lname, None)
            if func is None:
                raise NotImplementedError(f"Unsupported MQT DD gate {name}")
            args = [float(v) for v in params.values()] if params else []
            func(*args, *qubits)

        simulator = ddsim.CircuitSimulator(qc)
        return simulator.get_constructed_dd()


__all__ = [
    "StatevectorAdapter",
    "StimAdapter",
    "MPSAdapter",
    "DecisionDiagramAdapter",
    "AerStatevectorAdapter",
    "AerMPSAdapter",
    "MQTDDAdapter",
]
