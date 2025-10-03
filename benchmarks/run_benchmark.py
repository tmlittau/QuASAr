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
from typing import Iterable, Mapping, Sequence

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
    from .bench_utils import circuits as circuit_lib
    from .bench_utils.showcase_benchmarks import RUN_TIMEOUT_DEFAULT_SECONDS
    from .bench_utils.theoretical_estimation_runner import (
        LARGE_GATE_THRESHOLD_DEFAULT,
        LARGE_PLANNER_OVERRIDES_DEFAULT,
        collect_estimates,
    )
    from .bench_utils.theoretical_estimation_selection import (
        GROUPS as ESTIMATION_GROUPS,
        format_available_circuits as format_estimation_circuits,
        format_available_groups as format_estimation_groups,
        resolve_requested_specs as resolve_estimation_specs,
    )
    from .bench_utils.theoretical_estimation_utils import (
        OPS_PER_SECOND_DEFAULT,
        EstimateRecord,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_relative_speedups,
        plot_runtime_speedups,
        report_totals,
        write_tables,
    )
    from .bench_utils.database import BenchmarkDatabase, BenchmarkRun, open_database
except ImportError:  # pragma: no cover - script execution fallback
    from bench_utils import paper_figures  # type: ignore
    from bench_utils import showcase_benchmarks  # type: ignore
    from bench_utils import circuits as circuit_lib  # type: ignore
    from bench_utils.showcase_benchmarks import (  # type: ignore
        RUN_TIMEOUT_DEFAULT_SECONDS,
    )
    from bench_utils.theoretical_estimation_runner import (  # type: ignore
        LARGE_GATE_THRESHOLD_DEFAULT,
        LARGE_PLANNER_OVERRIDES_DEFAULT,
        collect_estimates,
    )
    from bench_utils.theoretical_estimation_selection import (  # type: ignore
        GROUPS as ESTIMATION_GROUPS,
        format_available_circuits as format_estimation_circuits,
        format_available_groups as format_estimation_groups,
        resolve_requested_specs as resolve_estimation_specs,
    )
    from bench_utils.theoretical_estimation_utils import (  # type: ignore
        OPS_PER_SECOND_DEFAULT,
        EstimateRecord,
        build_dataframe,
        build_summary,
        load_estimator,
        plot_memory_ratio,
        plot_relative_speedups,
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
    "run_clifford_random_suite",
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


def _parse_range_spec(spec: str) -> list[int]:
    parts = [int(part) for part in spec.split(":")]
    if len(parts) == 1:
        return parts
    if len(parts) == 2:
        start, stop = parts
        if stop < start:
            raise ValueError("range end must be >= start")
        return list(range(start, stop + 1))
    if len(parts) == 3:
        start, stop, step = parts
        if step <= 0:
            raise ValueError("range step must be positive")
        if stop < start:
            raise ValueError("range end must be >= start")
        return list(range(start, stop + 1, step))
    raise ValueError("range must be start[:end[:step]]")


def _parse_qubit_values(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return tuple()
    qubits: list[int] = []
    for spec in values:
        if not spec:
            continue
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                if ":" in chunk:
                    qubits.extend(_parse_range_spec(chunk))
                else:
                    qubits.append(int(chunk))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"invalid qubit specification '{chunk}': {exc}") from exc
    unique = sorted(dict.fromkeys(qubits))
    if any(q <= 0 for q in unique):
        raise ValueError("qubit widths must be positive integers")
    return tuple(unique)


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


def _large_planner_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Extract tuned planner overrides from CLI arguments."""

    mapping = {
        "estimate_large_batch_size": "batch_size",
        "estimate_large_horizon": "horizon",
        "estimate_large_quick_max_qubits": "quick_max_qubits",
        "estimate_large_quick_max_gates": "quick_max_gates",
        "estimate_large_quick_max_depth": "quick_max_depth",
    }
    overrides: dict[str, object] = {}
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    return overrides


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
    reuse_existing: bool = False,
    database: BenchmarkDatabase | None = None,
    run: BenchmarkRun | None = None,
    database_path: Path | None = None,
    include_theoretical_sv: bool = False,
    theoretical_sv_options: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Execute a subset of the showcase suite programmatically.

    This helper is primarily intended for tests and automation where callers
    require direct access to the raw measurements instead of the CSV/Markdown
    artefacts produced by the CLI entry point.  Optional flags allow the
    benchmark to skip baseline simulators (``include_baselines=False``), limit
    the baseline set (``baseline_backends``) or force QuASAr's quick-path
    execution (``quick=True``) which is useful for CI smoke tests.  When
    ``reuse_existing`` is ``True`` and a SQLite database is supplied, cached
    measurements are loaded instead of rerunning simulations when the
    configuration matches a previous benchmark run.
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
    effective_timeout = None if timeout <= 0 else timeout
    try:
        if reuse_existing and database is not None:
            baseline_selection: Iterable[Backend] | None
            if include_baselines:
                baseline_selection = baseline_backends
            else:
                baseline_selection = ()
            cached = showcase_benchmarks._load_cached_suite_results(  # type: ignore[attr-defined]
                database,
                spec=spec,
                widths=widths,
                repetitions=repetitions,
                run_timeout=effective_timeout,
                memory_bytes=memory_bytes,
                classical_simplification=classical_simplification,
                include_baselines=include_baselines,
                baseline_backends=baseline_selection,
                quasar_quick=quick,
                include_theoretical_sv=include_theoretical_sv,
                theoretical_sv_options=theoretical_sv_options,
            )
            if cached is not None:
                return cached
        return showcase_benchmarks._run_backend_suite(  # type: ignore[attr-defined]
            spec,
            widths,
            repetitions=repetitions,
            run_timeout=effective_timeout,
            memory_bytes=memory_bytes,
            classical_simplification=classical_simplification,
            max_workers=workers,
            include_baselines=include_baselines,
            baseline_backends=baseline_backends,
            quasar_quick=quick,
            database=database,
            run=run,
            include_theoretical_sv=include_theoretical_sv,
            theoretical_sv_options=theoretical_sv_options,
        )
    finally:
        if managed_db is not None:
            managed_db.close()


def run_clifford_random_suite(
    widths: Iterable[int],
    clifford_depths: Iterable[int],
    total_depths: Iterable[int],
    *,
    repetitions: int = 1,
    run_timeout: float | None = None,
    memory_bytes: int | None = None,
    classical_simplification: bool = False,
    workers: int | None = None,
    quasar_quick: bool = False,
    reuse_existing: bool = False,
    database: BenchmarkDatabase | None = None,
    run: BenchmarkRun | None = None,
    database_path: Path | None = None,
    include_theoretical_sv: bool = False,
    theoretical_sv_options: Mapping[str, object] | None = None,
    tail_twoq_prob: float = 0.3,
    tail_angle_eps: float = 1e-3,
    clifford_seed: int = 1337,
    tail_seed: int = 2025,
) -> pd.DataFrame:
    """Execute the Clifford+rotation suite across all depth combinations."""

    width_list = tuple(sorted(dict.fromkeys(int(w) for w in widths if int(w) > 0)))
    if not width_list:
        raise ValueError("at least one positive qubit width is required")

    clifford_list = tuple(sorted(dict.fromkeys(int(d) for d in clifford_depths)))
    total_list = tuple(sorted(dict.fromkeys(int(d) for d in total_depths)))
    if not clifford_list:
        raise ValueError("at least one Clifford depth is required")
    if not total_list:
        raise ValueError("at least one total depth is required")
    if any(depth < 0 for depth in clifford_list):
        raise ValueError("clifford depths must be non-negative")
    if any(depth <= 0 for depth in total_list):
        raise ValueError("total depths must be positive")

    combinations: list[tuple[int, int]] = []
    for clifford_depth in clifford_list:
        for total_depth in total_list:
            if total_depth < clifford_depth:
                raise ValueError(
                    f"total depth {total_depth} cannot be smaller than Clifford depth {clifford_depth}"
                )
            combinations.append((clifford_depth, total_depth))

    managed_db: BenchmarkDatabase | None = None
    if database is None and database_path is not None:
        managed_db = BenchmarkDatabase(database_path)
        database = managed_db

    active_run = run
    if database is not None and active_run is None:
        active_run = database.start_run(
            description="clifford_random_suite",
            parameters={
                "widths": list(width_list),
                "clifford_depths": list(clifford_list),
                "total_depths": list(total_list),
                "repetitions": repetitions,
                "run_timeout": run_timeout,
                "memory_bytes": memory_bytes,
                "classical_simplification": classical_simplification,
                "quasar_quick": quasar_quick,
                "tail_twoq_prob": tail_twoq_prob,
                "tail_angle_eps": tail_angle_eps,
                "clifford_seed": clifford_seed,
                "tail_seed": tail_seed,
            },
        )

    frames: list[pd.DataFrame] = []
    baseline_selection = (Backend.STATEVECTOR, Backend.EXTENDED_STABILIZER)

    try:
        for clifford_depth, total_depth in combinations:
            name = f"clifford_random_cd{clifford_depth}_td{total_depth}"
            description = (
                "Random Clifford prefix with rotation tail "
                f"(clifford_depth={clifford_depth}, total_depth={total_depth}, "
                f"twoq_prob={tail_twoq_prob}, angle_eps={tail_angle_eps})"
            )

            def constructor(
                width: int,
                *,
                _cd: int = clifford_depth,
                _td: int = total_depth,
            ) -> object:
                return circuit_lib.random_clifford_with_tail_circuit(
                    width,
                    clifford_depth=_cd,
                    total_depth=_td,
                    clifford_seed=clifford_seed,
                    tail_seed=tail_seed,
                    tail_twoq_prob=tail_twoq_prob,
                    tail_angle_eps=tail_angle_eps,
                )

            spec = showcase_benchmarks.ShowcaseCircuit(
                name=name,
                display_name=f"Clifford {clifford_depth} / total {total_depth}",
                constructor=constructor,
                default_qubits=width_list,
                description=description,
            )

            LOGGER.info(
                "Benchmarking %s across widths: %s", spec.name, ", ".join(map(str, width_list))
            )

            raw_df: pd.DataFrame | None = None
            if reuse_existing and database is not None:
                raw_df = showcase_benchmarks._load_cached_suite_results(  # type: ignore[attr-defined]
                    database,
                    spec=spec,
                    widths=width_list,
                    repetitions=repetitions,
                    run_timeout=run_timeout,
                    memory_bytes=memory_bytes,
                    classical_simplification=classical_simplification,
                    include_baselines=True,
                    baseline_backends=baseline_selection,
                    quasar_quick=quasar_quick,
                    include_theoretical_sv=include_theoretical_sv,
                    theoretical_sv_options=theoretical_sv_options,
                )

            if raw_df is None:
                raw_df = showcase_benchmarks._run_backend_suite(  # type: ignore[attr-defined]
                    spec,
                    width_list,
                    repetitions=repetitions,
                    run_timeout=run_timeout,
                    memory_bytes=memory_bytes,
                    classical_simplification=classical_simplification,
                    max_workers=workers,
                    include_baselines=True,
                    baseline_backends=baseline_selection,
                    quasar_quick=quasar_quick,
                    include_theoretical_sv=include_theoretical_sv,
                    theoretical_sv_options=theoretical_sv_options,
                    database=database,
                    run=active_run,
                )

            if not raw_df.empty:
                frames.append(raw_df)
    finally:
        if managed_db is not None:
            managed_db.close()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def generate_theoretical_estimates(
    *,
    ops_per_second: float | None = OPS_PER_SECOND_DEFAULT,
    calibration: Path | None = None,
    workers: int | None = None,
    circuits: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    database: BenchmarkDatabase | None = None,
    enable_large_planner: bool = True,
    large_gate_threshold: int | None = LARGE_GATE_THRESHOLD_DEFAULT,
    large_planner_kwargs: dict[str, object] | None = None,
):
    """Return detailed and summary DataFrames for theoretical estimates."""

    throughput = ops_per_second if ops_per_second and ops_per_second > 0 else None
    estimator = load_estimator(calibration)
    specs = resolve_estimation_specs(circuits, groups, default_group="showcase")
    method = "theoretical_estimate"

    def store_record(record: EstimateRecord) -> None:
        if database is None:
            return
        database.insert_estimation(
            record={
                "circuit": record.circuit,
                "qubits": record.qubits,
                "framework": record.framework,
                "backend": record.backend,
                "supported": record.supported,
                "time_ops": record.time_ops,
                "approx_seconds": record.approx_seconds(throughput),
                "memory_bytes": record.memory_bytes,
                "note": record.note,
            },
            method=method,
        )

    records = collect_estimates(
        specs,
        paper_figures.BACKENDS,
        estimator,
        max_workers=workers,
        enable_large_planner=enable_large_planner,
        large_gate_threshold=large_gate_threshold,
        large_planner_kwargs=large_planner_kwargs,
        record_callback=store_record if database is not None else None,
    )
    detail = build_dataframe(records, throughput)
    summary = build_summary(detail)
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
    enable_large_planner: bool,
    large_gate_threshold: int | None,
    large_planner_kwargs: dict[str, object] | None,
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
                enable_large_planner=enable_large_planner,
                large_gate_threshold=large_gate_threshold,
                large_planner_kwargs=large_planner_kwargs,
            )
    else:
        detail, summary, throughput = generate_theoretical_estimates(
            ops_per_second=ops_per_second,
            calibration=calibration,
            workers=workers,
            circuits=circuits,
            groups=groups,
            enable_large_planner=enable_large_planner,
            large_gate_threshold=large_gate_threshold,
            large_planner_kwargs=large_planner_kwargs,
        )

    write_tables(detail, summary)
    if throughput:
        report_totals(detail)
    if not summary.empty:
        plot_runtime_speedups(summary)
        plot_relative_speedups(summary)
        plot_memory_ratio(summary)
    else:
        LOGGER.warning("No supported configurations found for summary plots.")


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = showcase_benchmarks.build_arg_parser()
    suite_names: tuple[str, ...] = ()
    if hasattr(showcase_benchmarks, "available_suite_names"):
        suite_names = showcase_benchmarks.available_suite_names()
    has_suite_arg = any(getattr(action, "dest", None) == "suite" for action in parser._actions)
    if suite_names and not has_suite_arg:
        parser.add_argument(
            "--suite",
            choices=sorted(suite_names),
            metavar="SUITE",
            default=None,
            help="Run a preconfigured showcase suite (e.g. stitched-big).",
        )
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
        "--clifford-random-suite",
        action="store_true",
        help="Run the Clifford + random rotation depth sweep suite.",
    )
    parser.add_argument(
        "--clifford-depths",
        type=int,
        nargs="+",
        metavar="DEPTH",
        default=None,
        help="Clifford prefix depths to evaluate for the Clifford+random suite.",
    )
    parser.add_argument(
        "--total-depths",
        type=int,
        nargs="+",
        metavar="DEPTH",
        default=None,
        help="Total circuit depths to evaluate for the Clifford+random suite.",
    )
    parser.add_argument(
        "--clifford-random-qubits",
        action="append",
        metavar="RANGE",
        help="Qubit widths for the Clifford+random suite (start:end[:step] or comma list).",
    )
    parser.add_argument(
        "--clifford-tail-twoq-prob",
        type=float,
        default=0.3,
        help="Two-qubit gate probability within the random tail (default: %(default)s).",
    )
    parser.add_argument(
        "--clifford-tail-angle-eps",
        type=float,
        default=1e-3,
        help=(
            "Minimum distance from Clifford-compatible angles for tail rotations "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--clifford-seed",
        type=int,
        default=1337,
        help="Base RNG seed for the Clifford prefix (default: %(default)s).",
    )
    parser.add_argument(
        "--clifford-tail-seed",
        type=int,
        default=2025,
        help="Base RNG seed for the non-Clifford tail (default: %(default)s).",
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
    parser.add_argument(
        "--estimate-large-planner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable tuned planner settings when the forced or simplified"
            " circuit is large."
            " Use --no-estimate-large-planner to keep the default planner"
            " behaviour."
        ),
    )
    parser.add_argument(
        "--estimate-large-threshold",
        type=int,
        default=LARGE_GATE_THRESHOLD_DEFAULT,
        help=(
            "Gate count that triggers the tuned planner based on the larger of"
            " the forced and simplified circuits"
            " planner (set to 0 to disable, default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--estimate-large-batch-size",
        type=int,
        default=None,
        help=(
            "Override the tuned planner batch size.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['batch_size']}."
        ),
    )
    parser.add_argument(
        "--estimate-large-horizon",
        type=int,
        default=None,
        help=(
            "Override the tuned planner DP horizon.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['horizon']}."
        ),
    )
    parser.add_argument(
        "--estimate-large-quick-max-qubits",
        type=int,
        default=None,
        help=(
            "Override the tuned planner quick-path qubit limit.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['quick_max_qubits']}."
        ),
    )
    parser.add_argument(
        "--estimate-large-quick-max-gates",
        type=int,
        default=None,
        help=(
            "Override the tuned planner quick-path gate limit.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['quick_max_gates']}."
        ),
    )
    parser.add_argument(
        "--estimate-large-quick-max-depth",
        type=int,
        default=None,
        help=(
            "Override the tuned planner quick-path depth limit.  Defaults to"
            f" {LARGE_PLANNER_OVERRIDES_DEFAULT['quick_max_depth']}."
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
    if args.estimate_large_threshold is not None and args.estimate_large_threshold < 0:
        parser.error("--estimate-large-threshold must be non-negative")
    for opt in (
        "estimate_large_batch_size",
        "estimate_large_horizon",
        "estimate_large_quick_max_qubits",
        "estimate_large_quick_max_gates",
        "estimate_large_quick_max_depth",
    ):
        value = getattr(args, opt, None)
        if value is not None and value <= 0:
            parser.error(f"--{opt.replace('_', '-')} must be positive")
    if getattr(args, "sv_mem_budget_gib", None) is not None and args.sv_mem_budget_gib < 0:
        parser.error("--sv-mem-budget-gib must be non-negative")
    if getattr(args, "sv_scratch_factor", None) is not None and args.sv_scratch_factor <= 0:
        parser.error("--sv-scratch-factor must be positive")
    if getattr(args, "sv_dtype_bytes", None) is not None and args.sv_dtype_bytes <= 0:
        parser.error("--sv-dtype-bytes must be positive")
    if getattr(args, "suite", None):
        if getattr(args, "circuit_names", None):
            parser.error("--suite cannot be combined with --circuit/--circuits")
        if getattr(args, "groups", None):
            parser.error("--suite cannot be combined with --group")
    if getattr(args, "repetitions", None) is not None and args.repetitions <= 0:
        parser.error("--repetitions must be a positive integer")

    if getattr(args, "clifford_random_suite", False):
        if not getattr(args, "clifford_depths", None):
            parser.error("--clifford-random-suite requires --clifford-depths")
        if not getattr(args, "total_depths", None):
            parser.error("--clifford-random-suite requires --total-depths")
        if args.clifford_tail_twoq_prob < 0 or args.clifford_tail_twoq_prob > 1:
            parser.error("--clifford-tail-twoq-prob must lie within [0, 1]")
        if args.clifford_tail_angle_eps < 0:
            parser.error("--clifford-tail-angle-eps must be non-negative")
        try:
            widths = _parse_qubit_values(getattr(args, "clifford_random_qubits", None))
        except ValueError as exc:
            parser.error(str(exc))
        if not widths:
            widths = (20,)
        args.clifford_random_widths = widths
    else:
        if getattr(args, "clifford_depths", None):
            parser.error("--clifford-depths requires --clifford-random-suite")
        if getattr(args, "total_depths", None):
            parser.error("--total-depths requires --clifford-random-suite")
        if getattr(args, "clifford_random_qubits", None):
            parser.error("--clifford-random-qubits requires --clifford-random-suite")
        args.clifford_random_widths = tuple()

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
    large_overrides = _large_planner_overrides_from_args(args)

    run_showcase_flag = True
    if getattr(args, "clifford_random_suite", False):
        run_showcase_flag = bool(
            getattr(args, "circuit_names", None)
            or getattr(args, "groups", None)
            or getattr(args, "suite", None)
        )

    if not args.estimate_only:
        if getattr(args, "clifford_random_suite", False):
            LOGGER.info("Running Clifford + random rotation suite")
            run_timeout = None if args.run_timeout <= 0 else args.run_timeout
            memory_bytes = (
                args.memory_bytes if args.memory_bytes and args.memory_bytes > 0 else None
            )
            include_theoretical_sv = bool(getattr(args, "include_theoretical_sv", False))
            mem_budget_gib = getattr(args, "sv_mem_budget_gib", None)
            mem_budget_bytes: int | None = None
            if mem_budget_gib is not None and mem_budget_gib > 0:
                mem_budget_bytes = int(mem_budget_gib * (1024 ** 3))
            dtype_bytes = max(1, int(getattr(args, "sv_dtype_bytes", 16)))
            theoretical_sv_options = {
                "mem_budget_bytes": mem_budget_bytes,
                "scratch_factor": float(getattr(args, "sv_scratch_factor", 1.5)),
                "dtype_bytes": dtype_bytes,
                "c_1q": float(getattr(args, "sv_c1", 1.0)),
                "c_2q": float(getattr(args, "sv_c2", 2.5)),
                "c_diag2q": float(getattr(args, "sv_cdiag", 0.8)),
                "c_3q": float(getattr(args, "sv_c3", 5.0)),
                "c_other": float(getattr(args, "sv_cother", 2.0)),
            }
            database_path = Path(getattr(args, "database", showcase_benchmarks.DATABASE_PATH))
            run_clifford_random_suite(
                widths=args.clifford_random_widths or (20,),
                clifford_depths=args.clifford_depths,
                total_depths=args.total_depths,
                repetitions=args.repetitions,
                run_timeout=run_timeout,
                memory_bytes=memory_bytes,
                classical_simplification=args.enable_classical_simplification,
                workers=args.workers,
                quasar_quick=bool(getattr(args, "quick", False)),
                reuse_existing=bool(getattr(args, "reuse_existing", False)),
                database_path=database_path,
                include_theoretical_sv=include_theoretical_sv,
                theoretical_sv_options=theoretical_sv_options,
                tail_twoq_prob=args.clifford_tail_twoq_prob,
                tail_angle_eps=args.clifford_tail_angle_eps,
                clifford_seed=args.clifford_seed,
                tail_seed=args.clifford_tail_seed,
            )

        if run_showcase_flag:
            LOGGER.info("Running showcase benchmarks")
            showcase_benchmarks.run_showcase_benchmarks(args)
        else:
            LOGGER.info("Skipping default showcase benchmarks (Clifford suite requested)")

    if args.estimate:
        LOGGER.info("Running theoretical estimation")
        _run_theoretical_estimation(
            ops_per_second=throughput,
            calibration=args.calibration,
            workers=args.workers,
            circuits=args.estimate_circuits,
            groups=args.estimate_groups,
            enable_large_planner=args.estimate_large_planner,
            large_gate_threshold=args.estimate_large_threshold,
            large_planner_kwargs=large_overrides or None,
            database_path=getattr(args, "database", None),
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

