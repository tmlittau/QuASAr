"""Validate runtime and peak memory measurements.

The test executes small circuits with QuASAr and baseline simulators and
asserts that the statevector backend uses significantly more runtime and
memory than the others. This avoids brittle absolute baselines that vary
across hardware.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Dict, Tuple

import pytest

from benchmarks.circuits import ghz_circuit, clifford_ec_circuit
from quasar.circuit import Circuit
from quasar.simulation_engine import SimulationEngine
from quasar.backends import AerStatevectorBackend, StimBackend
from quasar.analyzer import CircuitAnalyzer


def measure_runtime_memory(circuit: Circuit) -> Dict[str, Tuple[float, float]]:
    """Return runtime and peak memory for the given circuit."""

    engine = SimulationEngine()
    analyzer = CircuitAnalyzer(circuit, estimator=engine.planner.estimator)
    analysis = analyzer.analyze()
    plan = engine.planner.plan(circuit, analysis=analysis)
    _, metrics = engine.scheduler.run(
        circuit, plan, analysis=analysis, instrument=True
    )
    results: Dict[str, Tuple[float, float]] = {
        "quasar": (metrics.cost.time, float(metrics.cost.memory))
    }

    def run_backend(backend) -> Tuple[float, float]:
        backend.load(circuit.num_qubits)
        for gate in circuit.gates:
            backend.apply_gate(gate.gate, gate.qubits, gate.params)
        tracemalloc.start()
        tracemalloc.reset_peak()
        start = time.perf_counter()
        if isinstance(backend, AerStatevectorBackend):
            backend.statevector()
        else:
            backend.extract_ssd()
        runtime = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return runtime, float(peak)

    results["statevector"] = run_backend(AerStatevectorBackend())
    results["stim"] = run_backend(StimBackend())
    return results


def circuits() -> Dict[str, Circuit]:
    return {
        "ghz3": ghz_circuit(3, use_classical_simplification=False),
        "clifford_ec": clifford_ec_circuit(),
    }


@pytest.mark.parametrize("name,circuit", circuits().items())
def test_runtime_memory_summary(name: str, circuit: Circuit) -> None:
    metrics = measure_runtime_memory(circuit)
    sv_runtime, sv_memory = metrics["statevector"]
    assert sv_runtime > 0 and sv_memory > 0
    for backend, (runtime, memory) in metrics.items():
        assert runtime > 0 and memory > 0
        if backend == "statevector":
            continue
        # The dense statevector backend should be noticeably heavier than
        # stabilizer or hybrid approaches.
        assert sv_runtime > runtime * 2
        assert sv_memory > memory * 2
