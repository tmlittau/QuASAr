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

try:
    from quasar.cost import Backend
except ImportError:  # pragma: no cover - script execution fallback
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from quasar.cost import Backend  # type: ignore

if __package__ in {None, ""}:  # pragma: no cover - script execution
    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

try:  # package execution
    from .bench_utils import paper_figures
    from .bench_utils import showcase_benchmarks
    from .bench_utils.showcase_benchmarks import RUN_TIMEOUT_DEFAULT_SECONDS
    from .bench_utils.theoretical_estimation_runner import collect_estimates
    from .bench_utils.theoretical_estimation_selection import (
        GROUPS as ESTIMATION_GROUPS,
        format_available_circuits as format_estimation_circuits,
        format_available_groups as format_estimation_groups,
        resolve_requested_specs as resolve_estimation_specs,
    )
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
    from .bench_utils.database import BenchmarkDatabase, BenchmarkRun, open_database
except ImportError:  # pragma: no cover - script execution fallback
    from bench_utils import paper_figures  # type: ignore
    from bench_utils import showcase_benchmarks  # type: ignore
    from bench_utils.showcase_benchmarks import (  # type: ignore
        RUN_TIMEOUT_DEFAULT_SECONDS,
    )
    from bench_utils.theoretical_estimation_runner import collect_estimates  # type: ignore
    from bench_utils.theoretical_estimation_selection import (  # type: ignore
        GROUPS as ESTIMATION_GROUPS,
        format_available_circuits as format_estimation_circuits,
        format_available_groups as format_estimation_groups,
        resolve_requested_specs as resolve_estimation_specs,
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
    from bench_utils.database import (  # type: ignore
        BenchmarkDatabase,
        BenchmarkRun,
        open_database,
    )


LOGGER = logging.getLogger(__name__)

__all__ = [
    "generate_theoretical_estimates",
    "run_showcase_suite",
    "summarise_partitioning",
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
    include_baselines: bool = True,
    baseline_backends: Iterable[Backend] | None = None,
    quick: bool = False,
    database: BenchmarkDatabase | None = None,
    run: BenchmarkRun | None = None,
    database_path: Path | None = None,
) -> pd.DataFrame:
    """Execute a subset of the showcase suite programmatically.

    This helper is primarily intended for tests and automation where callers
    require direct access to the raw measurements instead of the CSV/Markdown
    artefacts produced by the CLI entry point.  Optional flags allow the
    benchmark to skip baseline simulators (``include_baselines=False``), limit
    the baseline set (``baseline_backends``) or force QuASAr's quick-path
    execution (``quick=True``) which is useful for CI smoke tests.
    """

    if circuit not in showcase_benchmarks.SHOWCASE_CIRCUITS:
        raise ValueError(f"unknown showcase circuit '{circuit}'")
    spec = showcase_benchmarks.SHOWCASE_CIRCUITS[circuit]
    timeout = run_timeout
    if timeout is None:
        timeout = RUN_TIMEOUT_DEFAULT_SECONDS
    managed_db: BenchmarkDatabase | None = None
    if database is None and database_path is not None:
        managed_db = BenchmarkDatabase(database_path)
        database = managed_db
    try:
        return showcase_benchmarks._run_backend_suite(  # type: ignore[attr-defined]
            spec,
            widths,
            repetitions=repetitions,
            run_timeout=None if timeout <= 0 else timeout,
            memory_bytes=memory_bytes,
            classical_simplification=classical_simplification,
            max_workers=workers,
            include_baselines=include_baselines,
            baseline_backends=baseline_backends,
            quasar_quick=quick,
            database=database,
            run=run,
        )
    finally:
        if managed_db is not None:
            managed_db.close()


def generate_theoretical_estimates(
    *,
    ops_per_second: float | None = OPS_PER_SECOND_DEFAULT,
    calibration: Path | None = None,
    workers: int | None = None,
    circuits: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    database: BenchmarkDatabase | None = None,
):
    """Return detailed and summary DataFrames for theoretical estimates."""

    throughput = ops_per_second if ops_per_second and ops_per_second > 0 else None
    estimator = load_estimator(calibration)
    specs = resolve_estimation_specs(circuits, groups)
    records = collect_estimates(
        specs,
        paper_figures.BACKENDS,
        estimator,
        max_workers=workers,
    )
    detail = build_dataframe(records, throughput)
    summary = build_summary(detail)
    if database is not None:
        method = "theoretical_estimate"
        for rec in records:
            database.insert_estimation(
                record={
                    "circuit": rec.circuit,
                    "qubits": rec.qubits,
                    "framework": rec.framework,
                    "backend": rec.backend,
                    "supported": rec.supported,
                    "time_ops": rec.time_ops,
                    "approx_seconds": rec.approx_seconds(throughput),
                    "memory_bytes": rec.memory_bytes,
                    "note": rec.note,
                },
                method=method,
            )
    return detail, summary, throughput


def summarise_partitioning(df: pd.DataFrame) -> pd.DataFrame:
    """Return an aggregate table comparing QuASAr and baseline metrics."""

    if df.empty:
        return df

    relevant = df[df["framework"].isin({"baseline_best", "quasar"})]
    if relevant.empty:
        return relevant

    base_keys = [
        column
        for column in ("scenario", "variant", "circuit", "qubits")
        if column in relevant.columns
    ]
    metadata_columns = [
        column
        for column in (
            "boundary",
            "schmidt_layers",
            "cross_layers",
            "suffix_sparsity",
            "prefix_core_boundary",
            "core_suffix_boundary",
            "dense_qubits",
            "clifford_qubits",
            "core_qubits",
            "suffix_qubits",
            "total_qubits",
            "patch_distance",
            "boundary_width",
            "gadget_width",
            "stabilizer_rounds",
            "gadget",
            "scheme",
            "chi_target",
            "gadget_size",
            "gadget_layers",
            "dense_gadgets",
            "gadget_spacing",
            "ladder_layers",
            "chain_length",
            "w_state_width",
            "oracle_layers",
            "oracle_rotation_gate_count",
            "oracle_rotation_unique",
            "oracle_rotation_density",
            "oracle_rotation_per_layer",
            "oracle_parameterised_rotations",
            "oracle_entangling_count",
            "oracle_entangling_per_layer",
            "oracle_sparsity",
            "rotation_set",
            "partition_count",
            "partition_total_subsystems",
            "partition_unique_backends",
            "partition_max_multiplicity",
            "partition_mean_multiplicity",
            "partition_backend_breakdown",
            "hierarchy_available",
        )
        if column in relevant.columns
    ]

    rows: list[dict[str, object]] = []
    baseline_frameworks = {
        backend.value for backend in showcase_benchmarks.BASELINE_BACKENDS
    }

    for keys, group in relevant.groupby(base_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(base_keys, keys))

        quasar_rows = group[group["framework"] == "quasar"]
        meta_source = quasar_rows if not quasar_rows.empty else group
        if not meta_source.empty:
            head = meta_source.iloc[0]
            for column in metadata_columns:
                row.setdefault(column, head.get(column))

        baseline_rows = group[group["framework"] == "baseline_best"]
        if not baseline_rows.empty:
            baseline_entry = baseline_rows.iloc[0]
            row["baseline_runtime_mean"] = baseline_entry.get("run_time_mean")
            row["baseline_backend"] = baseline_entry.get("backend")
            peak_memory = baseline_entry.get("run_peak_memory_mean")
            if pd.isna(peak_memory):
                mask = df["framework"].isin(baseline_frameworks)
                for name, value in zip(base_keys, keys):
                    mask &= df[name] == value
                candidates = df[mask]
                if (
                    not candidates.empty
                    and "run_time_mean" in candidates.columns
                    and "run_peak_memory_mean" in candidates.columns
                ):
                    idx = candidates["run_time_mean"].idxmin()
                    peak_memory = candidates.loc[idx, "run_peak_memory_mean"]
            row["baseline_peak_memory_mean"] = peak_memory
        else:
            row["baseline_runtime_mean"] = None
            row["baseline_peak_memory_mean"] = None
            row["baseline_backend"] = None

        if not quasar_rows.empty:
            quasar_entry = quasar_rows.iloc[0]
            row["quasar_runtime_mean"] = quasar_entry.get("run_time_mean")
            row["quasar_peak_memory_mean"] = quasar_entry.get("run_peak_memory_mean")
            row["quasar_backend"] = quasar_entry.get("backend")
            row["quasar_conversions"] = quasar_entry.get("conversion_count")
            row["quasar_boundary_mean"] = quasar_entry.get("conversion_boundary_mean")
            row["quasar_rank_mean"] = quasar_entry.get("conversion_rank_mean")
            row["quasar_frontier_mean"] = quasar_entry.get("conversion_frontier_mean")
            row["quasar_conversion_primitives"] = quasar_entry.get(
                "conversion_primitive_summary"
            )
        else:
            row["quasar_runtime_mean"] = None
            row["quasar_peak_memory_mean"] = None
            row["quasar_backend"] = None
            row["quasar_conversions"] = None
            row["quasar_boundary_mean"] = None
            row["quasar_rank_mean"] = None
            row["quasar_frontier_mean"] = None
            row["quasar_conversion_primitives"] = None

        try:
            runtime_baseline = row["baseline_runtime_mean"]
            runtime_quasar = row["quasar_runtime_mean"]
            if runtime_baseline and runtime_quasar:
                row["runtime_speedup"] = runtime_baseline / runtime_quasar
            else:
                row["runtime_speedup"] = None
        except ZeroDivisionError:
            row["runtime_speedup"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def _run_theoretical_estimation(
    *,
    ops_per_second: float | None,
    calibration: Path | None,
    workers: int | None,
    circuits: Sequence[str] | None,
    groups: Sequence[str] | None,
    database_path: Path | None = None,
) -> None:
    """Execute the theoretical estimation pipeline and export artefacts."""

    if database_path is not None:
        with open_database(database_path) as database:
            detail, summary, throughput = generate_theoretical_estimates(
                ops_per_second=ops_per_second,
                calibration=calibration,
                workers=workers,
                circuits=circuits,
                groups=groups,
                database=database,
            )
    else:
        detail, summary, throughput = generate_theoretical_estimates(
            ops_per_second=ops_per_second,
            calibration=calibration,
            workers=workers,
            circuits=circuits,
            groups=groups,
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
        "--list-estimate-groups",
        action="store_true",
        help="List available theoretical estimation groups and exit.",
    )
    parser.add_argument(
        "--list-estimate-circuits",
        action="store_true",
        help="List theoretical estimation circuit builders and exit.",
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
    parser.add_argument(
        "--estimate-group",
        dest="estimate_groups",
        action="append",
        choices=sorted(ESTIMATION_GROUPS),
        metavar="GROUP",
        default=None,
        help="Include an estimation group when generating theoretical results.",
    )
    parser.add_argument(
        "--estimate-circuit",
        dest="estimate_circuits",
        action="append",
        metavar="SPEC",
        default=None,
        help=(
            "Custom circuit specification for estimation in the form"
            " name[params]:q1,q2. Use --list-estimate-circuits for options."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments using the combined showcase/estimate parser."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.estimate_only:
        args.estimate = True
    if getattr(args, "estimate_circuits", None) or getattr(
        args, "estimate_groups", None
    ):
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

    if getattr(args, "list_estimate_groups", False):
        print(format_estimation_groups())
        return
    if getattr(args, "list_estimate_circuits", False):
        print(format_estimation_circuits())
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
            circuits=args.estimate_circuits,
            groups=args.estimate_groups,
            database_path=getattr(args, "database", None),
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

