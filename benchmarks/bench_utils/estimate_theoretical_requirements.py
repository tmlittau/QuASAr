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
    from theoretical_estimation_runner import (  # type: ignore[no-redef]
        LARGE_GATE_THRESHOLD_DEFAULT,
        LARGE_PLANNER_OVERRIDES_DEFAULT,
        collect_estimates,
    )
    from theoretical_estimation_selection import (  # type: ignore[no-redef]
        GROUPS as ESTIMATION_GROUPS,
        format_available_circuits,
        format_available_groups,
        resolve_requested_specs,
    )
    from theoretical_estimation_utils import (  # type: ignore[no-redef]
        OPS_PER_SECOND_DEFAULT,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_relative_speedups,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )
    import paper_figures as paper_figures  # type: ignore[no-redef]
else:  # pragma: no cover - package import path
    from . import paper_figures
    from .theoretical_estimation_runner import (
        LARGE_GATE_THRESHOLD_DEFAULT,
        LARGE_PLANNER_OVERRIDES_DEFAULT,
        collect_estimates,
    )
    from .theoretical_estimation_selection import (
        GROUPS as ESTIMATION_GROUPS,
        format_available_circuits,
        format_available_groups,
        resolve_requested_specs,
    )
    from .theoretical_estimation_utils import (
        OPS_PER_SECOND_DEFAULT,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_relative_speedups,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate theoretical runtime and memory requirements for the benchmark"
            " circuits or custom selections."
        )
    )
    parser.add_argument(
        "--group",
        "--groups",
        dest="groups",
        action="append",
        choices=sorted(ESTIMATION_GROUPS),
        metavar="GROUP",
        default=None,
        help="Include all circuits from the named estimation group.",
    )
    parser.add_argument(
        "--circuit",
        "--circuits",
        dest="circuits",
        action="append",
        metavar="SPEC",
        default=None,
        help=(
            "Custom circuit specification in the form name[params]:q1,q2. "
            "Use --list-circuits to discover available builders."
        ),
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available estimation groups and exit.",
    )
    parser.add_argument(
        "--list-circuits",
        action="store_true",
        help="List available estimation circuit builders and exit.",
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
    parser.add_argument(
        "--large-planner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable tuned planner settings for large simplified circuits."
            " Use --no-large-planner to keep the default planner behaviour."
        ),
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=LARGE_GATE_THRESHOLD_DEFAULT,
        help=(
            "Gate count on the simplified circuit that triggers the tuned"
            " planner (set to 0 to disable, default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--large-batch-size",
        type=int,
        default=None,
        help=(
            "Override the tuned planner batch size.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['batch_size']}."
        ),
    )
    parser.add_argument(
        "--large-horizon",
        type=int,
        default=None,
        help=(
            "Override the tuned planner DP horizon.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['horizon']}."
        ),
    )
    parser.add_argument(
        "--large-quick-max-qubits",
        type=int,
        default=None,
        help=(
            "Override the tuned planner quick-path qubit limit.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['quick_max_qubits']}."
        ),
    )
    parser.add_argument(
        "--large-quick-max-gates",
        type=int,
        default=None,
        help=(
            "Override the tuned planner quick-path gate limit.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['quick_max_gates']}."
        ),
    )
    parser.add_argument(
        "--large-quick-max-depth",
        type=int,
        default=None,
        help=(
            "Override the tuned planner quick-path depth limit.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['quick_max_depth']}."
        ),
    )
    args = parser.parse_args(argv)

    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be a positive integer")
    if args.large_threshold is not None and args.large_threshold < 0:
        parser.error("--large-threshold must be non-negative")
    for opt in (
        "large_batch_size",
        "large_horizon",
        "large_quick_max_qubits",
        "large_quick_max_gates",
        "large_quick_max_depth",
    ):
        value = getattr(args, opt, None)
        if value is not None and value <= 0:
            parser.error(f"--{opt.replace('_', '-')} must be positive")

    return args


def _large_planner_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    mapping = {
        "large_batch_size": "batch_size",
        "large_horizon": "horizon",
        "large_quick_max_qubits": "quick_max_qubits",
        "large_quick_max_gates": "quick_max_gates",
        "large_quick_max_depth": "quick_max_depth",
    }
    overrides: dict[str, object] = {}
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    return overrides


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_groups:
        print(format_available_groups())
        return
    if args.list_circuits:
        print(format_available_circuits())
        return

    ops_per_second = args.ops_per_second if args.ops_per_second > 0 else None

    estimator = load_estimator(args.calibration)
    specs = resolve_requested_specs(args.circuits, args.groups, default_group="showcase")
    overrides = _large_planner_overrides_from_args(args)
    records = collect_estimates(
        specs,
        paper_figures.BACKENDS,
        estimator,
        max_workers=args.workers,
        enable_large_planner=args.large_planner,
        large_gate_threshold=args.large_threshold,
        large_planner_kwargs=overrides or None,
    )

    detail = build_dataframe(records, ops_per_second)
    summary = build_summary(detail)

    write_tables(detail, summary)
    if ops_per_second is not None:
        report_totals(detail)

    if not summary.empty:
        plot_runtime_speedups(summary)
        plot_relative_speedups(summary)
        plot_memory_ratio(summary)
    else:
        print("No supported configurations found for summary plots.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

