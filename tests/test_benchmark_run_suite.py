import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark_cli

from quasar.circuit import Circuit


def dummy_run_quasar_multiple(self, circuit, engine, backend, repetitions):
    return {
        "backend": backend.name if hasattr(backend, "name") else backend,
        "repetitions": repetitions,
        "run_time_mean": 0.0,
        "run_time_std": 0.0,
        "total_time_mean": 0.0,
        "total_time_std": 0.0,
        "prepare_peak_memory_mean": 0.0,
        "prepare_peak_memory_std": 0.0,
        "run_peak_memory_mean": 0.0,
        "run_peak_memory_std": 0.0,
        "result": object(),
    }


def test_run_suite_passes_classical_flag(monkeypatch):
    flags = []

    def circuit_fn(n, *, use_classical_simplification):
        flags.append(use_classical_simplification)
        return Circuit([], use_classical_simplification=use_classical_simplification)

    monkeypatch.setattr(benchmark_cli, "SimulationEngine", lambda: object())
    monkeypatch.setattr(
        benchmark_cli.BenchmarkRunner, "run_quasar_multiple", dummy_run_quasar_multiple
    )
    results = benchmark_cli.run_suite(
        circuit_fn, [1], 1, use_classical_simplification=False
    )
    assert flags == [False]
    assert results[0]["qubits"] == 1
