import pytest
from benchmarks.runner import BenchmarkRunner
from benchmarks import circuits
from quasar import SimulationEngine


def test_run_wide_circuit_no_index_error():
    """Ensure wide circuits ingest into Stim without index errors."""
    runner = BenchmarkRunner()
    engine = SimulationEngine()
    circuit = circuits.w_state_circuit(13)
    try:
        runner.run_quasar_multiple(circuit, engine, repetitions=1)
    except IndexError as exc:  # pragma: no cover - regression guard
        pytest.fail(f"Unexpected index error: {exc}")
