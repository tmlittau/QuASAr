"""Smoke test utilities for the streamlined benchmark runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:  # package execution
    from .run_benchmark import (
        _configure_logging,
        generate_theoretical_estimates,
        run_showcase_suite,
    )
    from .bench_utils import showcase_benchmarks
    from .bench_utils.benchmark_cli import parse_qubit_range
except ImportError:  # pragma: no cover - script execution fallback
    from run_benchmark import (  # type: ignore
        _configure_logging,
        generate_theoretical_estimates,
        run_showcase_suite,
    )
    from bench_utils import showcase_benchmarks  # type: ignore
    from bench_utils.benchmark_cli import parse_qubit_range  # type: ignore


DEFAULT_OUTPUT = Path("benchmarks/results/smoke_test")
DEFAULT_CIRCUIT = "classical_controlled"
DEFAULT_WIDTHS = (2,)


# Legacy CI pipelines invoked the original smoke test with circuit families such
# as ``ghz`` or ``w_state`` that are no longer part of the showcase suite.  Map
# those names to their closest showcase counterparts so that the lightweight
# regression check keeps running without requiring changes to the pipeline
# configuration.
LEGACY_CIRCUIT_ALIASES: dict[str, str] = {
    "ghz": "clustered_ghz_random",
    "grover": "dynamic_classical_control",
    "qft": "clustered_ghz_qft",
    "w_state": "clustered_w_random",
}


def _resolve_circuit_name(name: str) -> str:
    """Return a showcase circuit name, applying legacy aliases when needed."""

    normalised = name.strip().lower()
    canonical = LEGACY_CIRCUIT_ALIASES.get(normalised, normalised)
    if canonical not in showcase_benchmarks.SHOWCASE_CIRCUITS:
        available = ", ".join(sorted(showcase_benchmarks.SHOWCASE_CIRCUITS))
        raise ValueError(
            f"unknown showcase circuit '{name}'. Available circuits: {available}"
        )
    return canonical


def run_smoke_test(
    *,
    output: Path,
    circuit: str = DEFAULT_CIRCUIT,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    repetitions: int = 1,
    workers: int | None = 1,
    run_timeout: float | None = 0.0,
    enable_classical_simplification: bool = False,
    estimate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Execute a minimal showcase benchmark and optionally estimate resources."""

    showcase_circuit = _resolve_circuit_name(circuit)
    df = run_showcase_suite(
        showcase_circuit,
        widths,
        repetitions=repetitions,
        run_timeout=run_timeout,
        classical_simplification=enable_classical_simplification,
        workers=workers,
        include_baselines=False,
        quick=True,
    )

    base = output.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    df.to_json(base.with_suffix(".json"), orient="records", indent=2)

    summary: pd.DataFrame | None = None
    if estimate:
        detail, summary, _ = generate_theoretical_estimates(workers=workers)
        detail.to_csv(base.with_name(base.name + "_estimate_detail.csv"), index=False)
        summary.to_csv(base.with_name(base.name + "_estimate_summary.csv"), index=False)
    return df, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal showcase benchmark smoke test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension (default: benchmarks/results/smoke_test)",
    )
    parser.add_argument(
        "--circuit",
        default=DEFAULT_CIRCUIT,
        help="Showcase circuit name to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--widths",
        "--qubits",
        dest="widths",
        type=parse_qubit_range,
        default=DEFAULT_WIDTHS,
        help="Qubit widths as start:end[:step] (default: %(default)s)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of worker threads to use (default: %(default)s)",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=0.0,
        help="Per-run timeout in seconds (<= 0 disables the timeout)",
    )
    parser.add_argument(
        "--enable-classical-simplification",
        action="store_true",
        help="Enable classical control simplification on the generated circuit",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Generate theoretical estimates after the benchmark run",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use twice for debug output)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be a positive integer")

    _configure_logging(args.verbose)

    run_smoke_test(
        output=args.output,
        circuit=args.circuit,
        widths=args.widths,
        repetitions=args.repetitions,
        workers=args.workers,
        run_timeout=args.run_timeout,
        enable_classical_simplification=args.enable_classical_simplification,
        estimate=args.estimate,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

