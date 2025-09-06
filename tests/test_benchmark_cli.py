import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmarks import benchmark_cli


def test_benchmark_cli_respects_disable_flag(monkeypatch, tmp_path):
    def fake_run_suite(circuit_fn, qubits, repetitions, *, use_classical_simplification=True):
        assert not use_classical_simplification
        return []

    def fake_save_results(results, output):
        pass

    monkeypatch.setattr(benchmark_cli, "run_suite", fake_run_suite)
    monkeypatch.setattr(benchmark_cli, "save_results", fake_save_results)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--circuit",
            "ghz",
            "--qubits",
            "1:1",
            "--output",
            str(tmp_path / "out"),
            "--disable-classical-simplify",
        ],
    )

    benchmark_cli.main()
