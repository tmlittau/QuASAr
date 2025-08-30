from __future__ import annotations

"""Command line interface for running simple benchmark suites."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, List, Callable

# Allow importing the project modules when executed as a script.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from runner import BenchmarkRunner
from backends import BackendAdapter
from quasar.backends import StatevectorBackend
import circuits as circuit_lib


def parse_qubit_range(spec: str) -> List[int]:
    """Parse a ``start:end[:step]`` range specification."""
    parts = [int(p) for p in spec.split(":")]
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError("range must be start:end[:step]")
    start, stop = parts[0], parts[1] if len(parts) > 1 else parts[0]
    step = parts[2] if len(parts) > 2 else 1
    if step <= 0:
        raise argparse.ArgumentTypeError("step must be positive")
    if stop < start:
        raise argparse.ArgumentTypeError("end must be >= start")
    return list(range(start, stop + 1, step))


def resolve_circuit(name: str) -> Callable[[int], object]:
    """Return circuit constructor for ``name``."""
    func_name = f"{name}_circuit"
    try:
        return getattr(circuit_lib, func_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"unknown circuit family '{name}'") from exc


def run_suite(circuit_fn: Callable[[int], object], qubits: Iterable[int], repetitions: int) -> List[dict]:
    backend = BackendAdapter(name="statevector", backend_cls=StatevectorBackend)
    results = []
    for n in qubits:
        circuit = circuit_fn(n)
        runner = BenchmarkRunner()
        for _ in range(repetitions):
            runner.run(circuit, backend, return_state=False)
        times = [r["run_time"] for r in runner.results]
        memories = [r["run_memory"] for r in runner.results]
        record = {
            "circuit": circuit_fn.__name__,
            "qubits": n,
            "framework": backend.name,
            "repetitions": repetitions,
            "avg_time": statistics.mean(times),
            "time_variance": statistics.pvariance(times) if repetitions > 1 else 0.0,
            "avg_memory": statistics.mean(memories),
            "memory_variance": statistics.pvariance(memories) if repetitions > 1 else 0.0,
        }
        results.append(record)
    return results


def save_results(results: List[dict], output: Path) -> None:
    base = output.with_suffix("")
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    fields = [
        "circuit",
        "qubits",
        "framework",
        "repetitions",
        "avg_time",
        "time_variance",
        "avg_memory",
        "memory_variance",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute benchmark circuits and record timings")
    parser.add_argument("--circuit", required=True, help="Circuit family name (e.g. ghz, qft)")
    parser.add_argument(
        "--qubits",
        required=True,
        type=parse_qubit_range,
        help="Qubit range as start:end[:step]",
    )
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per configuration")
    parser.add_argument("--output", required=True, type=Path, help="Output file path without extension")
    args = parser.parse_args()

    circuit_fn = resolve_circuit(args.circuit)
    results = run_suite(circuit_fn, args.qubits, args.repetitions)
    save_results(results, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
