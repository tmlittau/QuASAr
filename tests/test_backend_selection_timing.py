from benchmarks.runner import BenchmarkRunner
from quasar import Backend, SimulationEngine
from quasar.circuit import Circuit
from quasar.planner import Planner
import pytest


def test_backend_selection_timing():
    circuit_auto = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "RZ", "qubits": [0], "params": {"param0": 0.3}},
    ])
    engine_auto = SimulationEngine(planner=Planner(perf_prio="time"))

    runner = BenchmarkRunner()
    auto = runner.run_quasar_multiple(circuit_auto, engine_auto, repetitions=3)

    circuit_forced = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "RZ", "qubits": [0], "params": {"param0": 0.3}},
    ])
    engine_forced = SimulationEngine(planner=Planner(perf_prio="time"))
    forced = runner.run_quasar_multiple(
        circuit_forced, engine_forced, backend=Backend.STATEVECTOR, repetitions=3
    )

    assert auto["backend"] == "STATEVECTOR"
    assert forced["backend"] == "STATEVECTOR"
    assert auto["run_time_mean"] == pytest.approx(forced["run_time_mean"], rel=0.5)
    assert auto["run_peak_memory_mean"] == pytest.approx(
        forced["run_peak_memory_mean"], rel=0.5
    )

