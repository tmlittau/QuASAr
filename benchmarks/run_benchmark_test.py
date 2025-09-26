from __future__ import annotations

"""Quick smoke test runner for the benchmarking pipeline.

The script executes a tiny benchmark configuration to confirm that the
infrastructure works end-to-end without requiring the full benchmark sweep.
"""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:  # package execution
    from .run_benchmark import (
        _configure_logging,
        run_all,
        run_scenarios,
        save_results,
    )
    from .bench_utils.benchmark_cli import parse_qubit_range, resolve_circuit
    from .bench_utils.partitioning_workloads import iter_scenario
except ImportError:  # pragma: no cover - script execution fallback
    from run_benchmark import (  # type: ignore
        _configure_logging,
        run_all,
        run_scenarios,
        save_results,
    )
    from bench_utils.benchmark_cli import parse_qubit_range, resolve_circuit  # type: ignore
    from bench_utils.partitioning_workloads import iter_scenario  # type: ignore


DEFAULT_OUTPUT = Path("benchmarks/results/smoke_test")
DEFAULT_CIRCUIT = "ghz"
DEFAULT_QUBITS = (4, 6)
def _limit(iterable: Iterable, count: int) -> list:
    items = []
    for idx, value in enumerate(iterable):
        if idx >= count:
            break
        items.append(value)
    return items


def run_smoke_test(
    *,
    output: Path,
    circuit: str = DEFAULT_CIRCUIT,
    qubits: Iterable[int] = DEFAULT_QUBITS,
    scenario: str | None = None,
    repetitions: int = 1,
    memory_bytes: int | None = None,
    max_workers: int | None = 1,
    disable_classical_simplify: bool = False,
) -> pd.DataFrame:
    """Execute a minimal benchmark run and persist the results."""

    if scenario:
        instances = _limit(iter_scenario(scenario), 1)
        df = run_scenarios(
            instances,
            repetitions,
            memory_bytes=memory_bytes,
            max_workers=max_workers,
        )
    else:
        circuit_fn = resolve_circuit(circuit)
        df = run_all(
            circuit_fn,
            qubits,
            repetitions,
            use_classical_simplification=not disable_classical_simplify,
            memory_bytes=memory_bytes,
            max_workers=max_workers,
        )
    save_results(df, output)
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Run a minimal benchmark smoke test")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension (default: benchmarks/results/smoke_test)",
    )
    parser.add_argument(
        "--circuit",
        default=DEFAULT_CIRCUIT,
        help="Circuit family name to benchmark (default: ghz)",
    )
    parser.add_argument(
        "--qubits",
        type=parse_qubit_range,
        default=DEFAULT_QUBITS,
        help="Qubit range as start:end[:step] (default: 4,6)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Partitioning scenario to benchmark (default: none)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per configuration (default: 1)",
    )
    parser.add_argument(
        "--memory-bytes",
        type=int,
        help="Approximate peak memory budget per backend (default: unlimited)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of worker threads to use (default: 1)",
    )
    parser.add_argument(
        "--disable-classical-simplify",
        action="store_true",
        help="Disable classical control simplification",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for debug output)",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be a positive integer")

    run_smoke_test(
        output=args.output,
        circuit=args.circuit,
        qubits=args.qubits,
        scenario=args.scenario,
        repetitions=args.repetitions,
        memory_bytes=args.memory_bytes,
        max_workers=args.workers,
        disable_classical_simplify=args.disable_classical_simplify,
    )


if __name__ == "__main__":
    main()
