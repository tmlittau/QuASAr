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
        """Execute ``circuit`` on the native backend.

        ``circuit`` may either be a :class:`Circuit` instance or the prepared
        ``(num_qubits, ops)`` tuple returned by :meth:`prepare`.  Passing a
        precompiled circuit allows benchmarks to measure only the actual
        simulation time, excluding conversion costs.  The ``return_state`` flag
        controls whether the simulator's state representation is extracted and
        returned or whether the backend instance is handed back to the caller
        for later inspection.
        """
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


class StatevectorAdapter(_BaseAdapter):
    """Adapter executing circuits using the dense statevector backend."""

    def __init__(self) -> None:
        super().__init__(name="statevector", backend_cls=StatevectorBackend)


class StimAdapter(_BaseAdapter):
    """Adapter executing circuits using the Stim tableau simulator."""

    def __init__(self) -> None:
        super().__init__(name="stim", backend_cls=StimBackend)

    def prepare(
        self, circuit: Circuit
    ) -> Tuple[int, Any]:  # pragma: no cover - trivial wrapper
        import stim

        num_qubits, ops = super().prepare(circuit)
        # Translate to Stim's bulk circuit representation outside the timed
        # simulation.  Building this circuit can be expensive and therefore
        # belongs to ``prepare_time``.
        stim_circuit = stim.Circuit()
        aliases = {k.upper(): v.upper() for k, v in StimBackend()._ALIASES.items()}
        for name, qubits, params in ops:
            lname = aliases.get(name.upper(), name.upper())
            if lname in {"I", "ID"}:
                continue
            if lname == "CSWAP":
                c, a, b = qubits
                stim_circuit.append("CX", [c, b])
                stim_circuit.append("CX", [a, b])
                stim_circuit.append("CX", [c, a])
                stim_circuit.append("CX", [a, b])
                stim_circuit.append("CX", [c, b])
                continue
            stim_circuit.append(lname, qubits)
        return num_qubits, stim_circuit

    def run(
        self,
        circuit: Union[Circuit, Tuple[int, Any]],
        *,
        return_state: bool = True,
    ) -> Any:
        import stim

        if isinstance(circuit, Circuit):
            num_qubits, stim_circuit = self.prepare(circuit)
        else:
            num_qubits, stim_circuit = circuit

        sim = stim.TableauSimulator()
        # Only the execution of the pre-built Stim circuit is considered part
        # of the timed ``run`` phase.
        sim.do_circuit(stim_circuit)

        backend = self.backend_cls()
        # ``backend.ingest`` merely converts the Stim simulator into the
        # backend's native representation and is performed after the simulation
        # has finished.
        backend.ingest(sim)
        if not return_state:
            return backend
        try:
            return sim.current_inverse_tableau()
        except Exception:
            return None


class MPSAdapter(_BaseAdapter):
    """Adapter executing circuits using the MPS simulator."""

    def __init__(self) -> None:
        super().__init__(name="mps", backend_cls=MPSBackend)


class DecisionDiagramAdapter(_BaseAdapter):
    """Adapter executing circuits using the decision diagram simulator."""

    def __init__(self) -> None:
        super().__init__(name="mqt_dd", backend_cls=DecisionDiagramBackend)

    def run(
        self,
        circuit: Union[
            Circuit, Tuple[int, Iterable[Tuple[str, Sequence[int], Dict[str, Any]]]]
        ],
        *,
        return_state: bool = True,
    ) -> Any:
        # ``_BaseAdapter.run`` performs the actual simulation; any state
        # extraction below therefore happens after the timed section.
        backend = super().run(circuit, return_state=False)
        if not return_state:
            return backend
        try:
            ssd = backend.extract_ssd()
            part = next(iter(ssd.partitions), None)
            state = getattr(part, "state", None)
            if isinstance(state, tuple) and len(state) == 2:
                return state[1]
            return state
        except Exception:
            return None


class _AerAdapter(_BaseAdapter):
    """Common helper for Qiskit Aer based backends."""

    method: str

    def __init__(self, name: str, method: str) -> None:
        super().__init__(name=name, backend_cls=None)
        self.method = method

    def prepare(
        self, circuit: Circuit
    ) -> Tuple[int, Any]:  # pragma: no cover - optional dependency
        """Translate ``circuit`` into a :class:`qiskit.QuantumCircuit`.

        Compilation (via :func:`qiskit.transpile`) is performed here so that
        ``run_time`` only measures the actual execution of the Aer simulator.
        """
        from qiskit import QuantumCircuit, transpile

        num_qubits, ops = super().prepare(circuit)
        qc = QuantumCircuit(num_qubits)
        for name, qubits, params in ops:
            func = getattr(qc, name.lower(), None)
            if func is None:
                raise NotImplementedError(f"Unsupported Qiskit gate {name}")
            args = [float(v) for v in params.values()] if params else []
            func(*args, *qubits)

        if self.method == "statevector":
            qc.save_statevector()
        else:
            qc.save_matrix_product_state()  # type: ignore[attr-defined]

        # Transpile with the lowest optimization level to ensure fair timing.
        qc = transpile(qc, optimization_level=0)

        return num_qubits, qc

    def run(
        self,
        circuit: Union[Circuit, Tuple[int, Any]],
        *,
        return_state: bool = True,
    ) -> Any:  # pragma: no cover - optional dependency
        if isinstance(circuit, Circuit):
            # Fallback: compilation will occur inside ``run`` and therefore
            # contribute to ``run_time``.
            num_qubits, qc = self.prepare(circuit)
        else:
            num_qubits, qc = circuit

        from qiskit_aer import AerSimulator

        sim = AerSimulator(method=self.method)
        # Only the simulator execution is timed; ``prepare`` performs the heavy
        # QuantumCircuit construction and transpilation.
        result = sim.run(qc).result()
        if not return_state:
            return result
        try:
            if self.method == "statevector":
                return result.get_statevector()
            return result.data(0)["matrix_product_state"]
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

    def prepare(
        self, circuit: Circuit
    ) -> Tuple[int, Any]:  # pragma: no cover - optional dependency
        from mqt.core.ir import QuantumComputation

        num_qubits, ops = super().prepare(circuit)
        qc = QuantumComputation(num_qubits)
        for name, qubits, params in ops:
            lname = self._ALIASES.get(name.upper(), name.lower())
            func = getattr(qc, lname, None)
            if func is None:
                raise NotImplementedError(f"Unsupported MQT DD gate {name}")
            args = [float(v) for v in params.values()] if params else []
            func(*args, *qubits)
        return num_qubits, qc

    def run(
        self,
        circuit: Union[Circuit, Tuple[int, Any]],
        *,
        return_state: bool = True,
    ) -> Any:  # pragma: no cover - optional dependency
        if isinstance(circuit, Circuit):
            # Fallback: compilation inside ``run`` counts towards ``run_time``.
            num_qubits, qc = self.prepare(circuit)
        else:
            num_qubits, qc = circuit

        import mqt.ddsim as ddsim

        # Only the actual simulation is timed; ``prepare`` performed the heavy
        # translation into ``QuantumComputation``.
        simulator = ddsim.CircuitSimulator(qc)
        if not return_state:
            return simulator
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
