from __future__ import annotations

import networkx as nx

from benchmarks import large_scale_circuits as lsc
from benchmarks.runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend


def test_large_scale_benchmark_runs_execute() -> None:
    """Ensure benchmark circuits from the notebook execute on all backends.

    The notebook ``benchmarks/notebooks/large_scale_circuits.ipynb``
    benchmarks several circuit families across multiple backends.  This test
    exercises the same builders with small configurations and verifies that the
    benchmark runner returns a result dictionary for each backend and for
    QuASAr's planner without raising exceptions.
    """

    circuit_builders = {
        "ripple_carry": (
            lsc.ripple_carry_modular_circuit,
            dict(bit_width=4, modulus=None, arithmetic="cdkm"),
        ),
        "surface_code": (
            lsc.surface_code_cycle,
            dict(distance=3, rounds=1),
        ),
        "grover": (
            lsc.grover_with_oracle_circuit,
            dict(n_qubits=4, oracle_depth=2, iterations=1),
        ),
        "qaoa": (
            lsc.deep_qaoa_circuit,
            dict(graph=nx.cycle_graph(4), p_layers=2),
        ),
        "phase_est": (
            lsc.phase_estimation_classical_unitary,
            dict(eigen_qubits=2, precision_qubits=2, classical_depth=1),
        ),
    }

    backends = [
        Backend.STATEVECTOR,
        Backend.TABLEAU,
        Backend.MPS,
        Backend.DECISION_DIAGRAM,
    ]

    runner = BenchmarkRunner()
    engine = SimulationEngine()

    for name, (builder, params) in circuit_builders.items():
        circuit = builder(**params)
        # Ensure the circuit executes on each individual backend
        for backend in backends:
            record = runner.run_quasar_multiple(
                circuit, engine, backend=backend, repetitions=1
            )
            assert "repetitions" in record
            if backend is Backend.STATEVECTOR:
                # The reference backend should always succeed
                assert not record.get("failed")
        # And via QuASAr's backend selection
        record = runner.run_quasar_multiple(circuit, engine, repetitions=1)
        assert "repetitions" in record
        assert not record.get("failed")
