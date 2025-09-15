from __future__ import annotations

"""Command line interface for running simple benchmark suites."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Callable

# Allow importing the project modules when executed as a script.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend
import circuits as circuit_lib
from circuits import is_clifford


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


def run_suite(
    circuit_fn: Callable[[int], object],
    qubits: Iterable[int],
    repetitions: int,
    *,
    use_classical_simplification: bool = True,
) -> List[dict]:
    """Execute ``circuit_fn`` for each qubit count using a fixed backend.

    Circuits consisting solely of Clifford gates are ignored.  The helper runs
    QuASAr's scheduler multiple times, forcing the
    :class:`~quasar.cost.Backend.STATEVECTOR` backend so that results are
    comparable to single-method simulators.  Each summary record returned by
    :meth:`BenchmarkRunner.run_quasar_multiple` is expected to include the
    final simulation state under the ``"result"`` key.  The state is retained in
    memory for potential downstream analysis but is intentionally omitted from
    serialised output.
    """

    engine = SimulationEngine()
    results = []
    for n in qubits:
        circuit = circuit_fn(n)
        if use_classical_simplification:
            enable = getattr(circuit, "enable_classical_simplification", None)
            if callable(enable):
                enable()
            else:
                circuit.use_classical_simplification = True
        else:
            circuit.use_classical_simplification = False
        if is_clifford(circuit):
            continue
        runner = BenchmarkRunner()
        rec = runner.run_quasar_multiple(
            circuit,
            engine,
            backend=Backend.STATEVECTOR,
            repetitions=repetitions,
            quick=True,
        )
        state = rec.get("result")
        if state is None:
            raise RuntimeError("benchmark run did not return a state")
        _ = state  # retain state for potential downstream use
        record = {
            "circuit": circuit_fn.__name__,
            "qubits": n,
            "framework": rec["backend"],
            "repetitions": rec["repetitions"],
            "avg_time": rec["run_time_mean"],
            "time_variance": rec["run_time_std"] ** 2,
            "avg_total_time": rec["total_time_mean"],
            "total_time_variance": rec["total_time_std"] ** 2,
            "avg_prepare_peak_memory": rec["prepare_peak_memory_mean"],
            "prepare_peak_memory_variance": rec["prepare_peak_memory_std"] ** 2,
            "avg_run_peak_memory": rec["run_peak_memory_mean"],
            "run_peak_memory_variance": rec["run_peak_memory_std"] ** 2,
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
        "avg_total_time",
        "total_time_variance",
        "avg_prepare_peak_memory",
        "prepare_peak_memory_variance",
        "avg_run_peak_memory",
        "run_peak_memory_variance",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute benchmark circuits and record timings"
    )
    parser.add_argument(
        "--circuit", required=True, help="Circuit family name (e.g. ghz, qft)"
    )
    parser.add_argument(
        "--qubits",
        required=True,
        type=parse_qubit_range,
        help="Qubit range as start:end[:step]",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per configuration",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output file path without extension"
    )
    parser.add_argument(
        "--disable-classical-simplify",
        action="store_true",
        help="Disable classical control simplification",
    )
    args = parser.parse_args()

    circuit_fn = resolve_circuit(args.circuit)
    results = run_suite(
        circuit_fn,
        args.qubits,
        args.repetitions,
        use_classical_simplification=not args.disable_classical_simplify,
    )
    save_results(results, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
