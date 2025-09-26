"""Unified CLI for showcase benchmarks and theoretical estimates.

The script exposes the streamlined benchmarking workflow used throughout the
project.  It delegates the heavy lifting to :mod:`benchmarks.bench_utils` so
that the same helper functions power both programmatic usage and the command
line interface.  The key entry points are:

* Showcase benchmarks from :mod:`benchmarks.bench_utils.showcase_benchmarks`.
  Users can benchmark a single circuit, selected groups or the entire suite.
* Theoretical resource estimation via
  :mod:`benchmarks.bench_utils.theoretical_estimation_runner`.

Both paths honour the ``--workers`` flag to keep the previous multithreading
behaviour.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

if __package__ in {None, ""}:  # pragma: no cover - script execution
    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

try:  # package execution
    from .bench_utils import paper_figures
    from .bench_utils import showcase_benchmarks
    from .bench_utils.showcase_benchmarks import RUN_TIMEOUT_DEFAULT_SECONDS
    from .bench_utils.theoretical_estimation_runner import collect_estimates
    from .bench_utils.theoretical_estimation_utils import (
        OPS_PER_SECOND_DEFAULT,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from bench_utils import paper_figures  # type: ignore
    from bench_utils import showcase_benchmarks  # type: ignore
    from bench_utils.showcase_benchmarks import (  # type: ignore
        RUN_TIMEOUT_DEFAULT_SECONDS,
    )
    from bench_utils.theoretical_estimation_runner import (  # type: ignore
        collect_estimates,
    )
    from bench_utils.theoretical_estimation_utils import (  # type: ignore
        OPS_PER_SECOND_DEFAULT,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )


LOGGER = logging.getLogger(__name__)

__all__ = [
    "generate_theoretical_estimates",
    "run_showcase_suite",
]


def _configure_logging(verbosity: int) -> None:
    """Initialise logging for CLI usage."""

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _list_circuits() -> str:
    """Return a formatted list of available showcase circuits."""

    lines = ["Available circuits:"]
    for name, spec in sorted(showcase_benchmarks.SHOWCASE_CIRCUITS.items()):
        lines.append(f"  - {name}: {spec.display_name}")
    return "\n".join(lines)


def _list_groups() -> str:
    """Return a formatted list of available showcase circuit groups."""

    lines = ["Available groups:"]
    for name, members in sorted(showcase_benchmarks.SHOWCASE_GROUPS.items()):
        circuits = ", ".join(members)
        lines.append(f"  - {name}: {circuits}")
    return "\n".join(lines)


def run_showcase_suite(
    circuit: str,
    widths: Iterable[int],
    *,
    repetitions: int = 1,
    run_timeout: float | None = None,
    memory_bytes: int | None = None,
    classical_simplification: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    """Execute a subset of the showcase suite programmatically.

    This helper is primarily intended for tests and automation where callers
    require direct access to the raw measurements instead of the CSV/Markdown
    artefacts produced by the CLI entry point.
    """

    if circuit not in showcase_benchmarks.SHOWCASE_CIRCUITS:
        raise ValueError(f"unknown showcase circuit '{circuit}'")
    spec = showcase_benchmarks.SHOWCASE_CIRCUITS[circuit]
    timeout = run_timeout
    if timeout is None:
        timeout = RUN_TIMEOUT_DEFAULT_SECONDS
    return showcase_benchmarks._run_backend_suite(  # type: ignore[attr-defined]
        spec,
        widths,
        repetitions=repetitions,
        run_timeout=None if timeout <= 0 else timeout,
        memory_bytes=memory_bytes,
        classical_simplification=classical_simplification,
        max_workers=workers,
    )


def generate_theoretical_estimates(
    *,
    ops_per_second: float | None = OPS_PER_SECOND_DEFAULT,
    calibration: Path | None = None,
    workers: int | None = None,
):
    """Return detailed and summary DataFrames for theoretical estimates."""

    throughput = ops_per_second if ops_per_second and ops_per_second > 0 else None
    estimator = load_estimator(calibration)
    records = collect_estimates(
        paper_figures.CIRCUITS,
        paper_figures.BACKENDS,
        estimator,
        max_workers=workers,
    )
    detail = build_dataframe(records, throughput)
    summary = build_summary(detail)
    return detail, summary, throughput


def _run_theoretical_estimation(
    *,
    ops_per_second: float | None,
    calibration: Path | None,
    workers: int | None,
) -> None:
    """Execute the theoretical estimation pipeline and export artefacts."""

    detail, summary, throughput = generate_theoretical_estimates(
        ops_per_second=ops_per_second,
        calibration=calibration,
        workers=workers,
    )

    write_tables(detail, summary)
    if throughput:
        report_totals(detail)
    if not summary.empty:
        plot_runtime_speedups(summary)
        plot_memory_ratio(summary)
    else:
        LOGGER.warning("No supported configurations found for summary plots.")


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = showcase_benchmarks.build_arg_parser()
    parser.description = (
        "Run QuASAr showcase benchmarks and optionally compute theoretical"
        " resource estimates."
    )
    parser.add_argument(
        "--list-circuits",
        action="store_true",
        help="List available showcase circuits and exit.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Run theoretical estimation after executing the benchmarks.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Run only the theoretical estimation pipeline and skip benchmarks.",
    )
    parser.add_argument(
        "--ops-per-second",
        type=float,
        default=OPS_PER_SECOND_DEFAULT,
        help=(
            "Throughput used to convert cost-model operations to seconds when"
            " running theoretical estimation (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with calibrated cost coefficients used for"
            " theoretical estimation."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments using the combined showcase/estimate parser."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.estimate_only:
        args.estimate = True

    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be a positive integer")

    return args


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    args = parse_args(argv)
    _configure_logging(args.verbose)

    if getattr(args, "list_circuits", False):
        print(_list_circuits())
        return

    if getattr(args, "list_groups", False):
        print(_list_groups())
        return

    throughput = args.ops_per_second if args.ops_per_second > 0 else None

    if not args.estimate_only:
        LOGGER.info("Running showcase benchmarks")
        showcase_benchmarks.run_showcase_benchmarks(args)

    if args.estimate:
        LOGGER.info("Running theoretical estimation")
        _run_theoretical_estimation(
            ops_per_second=throughput,
            calibration=args.calibration,
            workers=args.workers,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

