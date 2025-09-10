"""Validate runtime and peak memory measurements.

The test executes small circuits with QuASAr and baseline simulators
and compares runtime and peak memory usage against reference values.
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


BASELINES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ghz3": {
        "quasar": (5.2176999815856107e-05, 2536.0),
        "statevector": (0.007553487999757635, 84063.0),
        "stim": (7.521899897255935e-05, 1208.0),
    },
    "clifford_ec": {
        "quasar": (7.368299884547014e-05, 2792.0),
        "statevector": (0.002430655000352999, 17418.0),
        "stim": (7.140500019886531e-05, 1104.0),
    },
}


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
    for backend, (runtime, memory) in metrics.items():
        expected_runtime, expected_memory = BASELINES[name][backend]
        assert runtime == pytest.approx(expected_runtime, rel=1.0, abs=1e-3)
        assert memory == pytest.approx(expected_memory, rel=0.5, abs=1024)
