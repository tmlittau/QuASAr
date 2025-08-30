from unittest.mock import patch

from benchmarks.runner import BenchmarkRunner
from quasar.circuit import Circuit


class MockBackend:
    """Backend that exposes ``prepare`` and ``run`` without delays."""

    name = "mock"

    def prepare(self, circuit: Circuit) -> Circuit:
        return circuit

    def run(self, circuit: Circuit, **_):
        return None


def test_total_time_sums_prepare_and_run():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    backend = MockBackend()
    runner = BenchmarkRunner()
    # Simulate timings: prepare takes 1s, run takes 2s
    with patch("benchmarks.runner.time.perf_counter", side_effect=[0.0, 1.0, 1.0, 3.0]):
        record = runner.run(circuit, backend)
    assert record["prepare_time"] == 1.0
    assert record["run_time"] == 2.0
    assert record["total_time"] == 3.0
    assert record["total_time"] == record["prepare_time"] + record["run_time"]
