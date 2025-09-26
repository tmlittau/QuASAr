"""Estimate theoretical runtime and memory requirements for benchmark circuits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


if __package__ in {None, ""}:  # pragma: no cover - script execution
    import importlib
    import sys

    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    importlib.import_module("quasar")
    from theoretical_estimation_runner import collect_estimates  # type: ignore[no-redef]
    from theoretical_estimation_utils import (  # type: ignore[no-redef]
        OPS_PER_SECOND_DEFAULT,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )
    import paper_figures as paper_figures  # type: ignore[no-redef]
else:  # pragma: no cover - package import path
    from . import paper_figures
    from .theoretical_estimation_runner import collect_estimates
    from .theoretical_estimation_utils import (
        OPS_PER_SECOND_DEFAULT,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate theoretical runtime and memory requirements for the paper"
            " benchmark circuits."
        )
    )
    parser.add_argument(
        "--ops-per-second",
        type=float,
        default=OPS_PER_SECOND_DEFAULT,
        help=(
            "Throughput used to convert cost-model operations to seconds."
            " Set to zero to disable runtime conversion (default: 1e9)."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with calibrated cost coefficients.  When omitted"
            " the newest calibration from calibration/ is used if available."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for estimate generation (default: auto).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ops_per_second = args.ops_per_second if args.ops_per_second > 0 else None

    estimator = load_estimator(args.calibration)
    records = collect_estimates(
        paper_figures.CIRCUITS,
        paper_figures.BACKENDS,
        estimator,
        max_workers=args.workers,
    )

    detail = build_dataframe(records, ops_per_second)
    summary = build_summary(detail)

    write_tables(detail, summary)
    if ops_per_second is not None:
        report_totals(detail)

    if not summary.empty:
        plot_runtime_speedups(summary)
        plot_memory_ratio(summary)
    else:
        print("No supported configurations found for summary plots.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

