from benchmarks.runner import BenchmarkRunner
from quasar.backends import DecisionDiagramBackend
from quasar.circuit import Circuit, Gate


def test_decision_diagram_benchmark_reports_runtime():
    gates = [
        Gate("H", [0]),
        Gate("CX", [0, 1]),
        Gate("H", [1]),
    ]
    circuit = Circuit(gates)
    backend = DecisionDiagramBackend()
    runner = BenchmarkRunner()
    record = runner.run(circuit, backend)
    assert record["run_time"] > 0
