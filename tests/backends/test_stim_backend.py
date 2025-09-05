"""Tests for the Stim backend initialisation."""

from quasar.backends.stim_backend import StimBackend
from quasar.circuit import Circuit
from benchmarks.runner import BenchmarkRunner


def test_load_and_apply_highest_qubit() -> None:
    """Loading three qubits allows operations on the highest index."""
    backend = StimBackend()
    backend.load(3)
    backend.apply_gate("X", [2])
    assert backend.simulator is not None
    assert backend.simulator.num_qubits == 3


def test_run_benchmark_reports_time() -> None:
    """Queued gates trigger runtime measurement via run_benchmark."""
    backend = StimBackend()
    runner = BenchmarkRunner()
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    rec = runner.run(circuit, backend)
    assert rec["run_time"] > 0

