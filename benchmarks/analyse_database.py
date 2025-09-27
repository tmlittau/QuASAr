"""Generate plots and tables from stored benchmark database results."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent

try:  # package execution
    from .bench_utils import showcase_benchmarks
    from .bench_utils.plot_utils import compute_baseline_best
    from .bench_utils.showcase_benchmarks import (
        DATABASE_PATH as DEFAULT_DATABASE_PATH,
        FIGURES_DIR as DEFAULT_FIGURES_DIR,
        _export_plot,
        _merge_results,
        _write_markdown,
        SHOWCASE_CIRCUITS,
        SHOWCASE_GROUPS,
    )
    from .bench_utils.theoretical_estimation_utils import (
        build_summary,
        plot_memory_ratio,
        plot_runtime_speedups,
        write_tables,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from bench_utils import showcase_benchmarks  # type: ignore
    from bench_utils.plot_utils import compute_baseline_best  # type: ignore
    from bench_utils.showcase_benchmarks import (  # type: ignore
        DATABASE_PATH as DEFAULT_DATABASE_PATH,
        FIGURES_DIR as DEFAULT_FIGURES_DIR,
        _export_plot,
        _merge_results,
        _write_markdown,
        SHOWCASE_CIRCUITS,
        SHOWCASE_GROUPS,
    )
    from bench_utils.theoretical_estimation_utils import (  # type: ignore
        build_summary,
        plot_memory_ratio,
        plot_runtime_speedups,
        write_tables,
    )


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PACKAGE_ROOT / "results" / "showcase"


def _resolve_circuits(explicit: Sequence[str] | None, groups: Sequence[str] | None) -> list[str]:
    if explicit is None and groups is None:
        return sorted(SHOWCASE_CIRCUITS)

    selected: set[str] = set()
    if explicit:
        for name in explicit:
            if name not in SHOWCASE_CIRCUITS:
                raise ValueError(f"unknown circuit '{name}'")
            selected.add(name)
    if groups:
        for group in groups:
            if group not in SHOWCASE_GROUPS:
                available = ", ".join(sorted(SHOWCASE_GROUPS))
                raise ValueError(f"unknown group '{group}'. Available: {available}")
            selected.update(SHOWCASE_GROUPS[group])
    return sorted(selected)


def _load_showcase_results(database: Path, circuits: Iterable[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in circuits)
    query = """
        SELECT
            b.circuit_id AS circuit,
            b.circuit_display_name AS display_name,
            sr.qubits,
            sr.framework,
            sr.backend,
            sr.mode,
            sr.repetitions,
            sr.prepare_time_mean,
            sr.prepare_time_std,
            sr.run_time_mean,
            sr.run_time_std,
            sr.total_time_mean,
            sr.total_time_std,
            sr.prepare_peak_memory_mean,
            sr.prepare_peak_memory_std,
            sr.run_peak_memory_mean,
            sr.run_peak_memory_std,
            sr.unsupported,
            sr.failed,
            sr.timeout,
            sr.comment,
            sr.error,
            sr.failed_runs,
            sr.partition_count,
            sr.partition_total_subsystems,
            sr.partition_unique_backends,
            sr.partition_max_multiplicity,
            sr.partition_mean_multiplicity,
            sr.partition_backend_breakdown,
            sr.hierarchy_available,
            sr.result_json,
            b.repetitions AS benchmark_repetitions,
            b.classical_simplification,
            b.include_baselines,
            b.quick,
            b.run_timeout,
            b.memory_bytes,
            b.created_at AS benchmark_created_at,
            br.id AS run_id,
            br.description AS run_description,
            br.parameters AS run_parameters,
            br.created_at AS run_created_at
        FROM simulation_run sr
        JOIN benchmark b ON sr.benchmark_id = b.id
        JOIN benchmark_run br ON b.run_id = br.id
        {where_clause}
    """
    where_clause = ""
    params: list[object] = []
    circuit_list = list(circuits)
    if circuit_list:
        where_clause = f"WHERE b.circuit_id IN ({placeholders})"
        params.extend(circuit_list)
    query = query.format(where_clause=where_clause)
    with sqlite3.connect(str(database)) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    bool_columns = [
        "unsupported",
        "failed",
        "timeout",
        "hierarchy_available",
        "classical_simplification",
        "include_baselines",
        "quick",
    ]
    for column in bool_columns:
        if column in df.columns:
            df[column] = df[column].astype(bool)
    return df


def _write_raw_tables(raw_df: pd.DataFrame, output_dir: Path, metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []
    all_summary_frames: list[pd.DataFrame] = []
    all_speedups: list[pd.DataFrame] = []

    for circuit, circuit_df in raw_df.groupby("circuit", sort=True):
        circuit_df = circuit_df.copy()
        circuit_df.sort_values(["qubits", "framework"], inplace=True)
        display_name = circuit_df["display_name"].iloc[0] if not circuit_df.empty else circuit

        raw_path = output_dir / f"{circuit}_raw.csv"
        circuit_df.to_csv(raw_path, index=False)
        _write_markdown(circuit_df, raw_path.with_suffix(".md"))
        all_frames.append(circuit_df)

        try:
            baseline_best = compute_baseline_best(
                circuit_df,
                metrics=("run_time_mean", "total_time_mean", "run_peak_memory_mean"),
            )
        except ValueError:
            LOGGER.warning("No feasible baseline measurements for %s", circuit)
            baseline_best = pd.DataFrame()

        quasar_df = circuit_df[circuit_df["framework"] == "quasar"].copy()
        if not quasar_df.empty:
            quasar_df["framework"] = "quasar"
        summary_frames = [frame for frame in (baseline_best, quasar_df) if not frame.empty]
        if not summary_frames:
            LOGGER.warning("Skipping summary export for %s due to missing data", circuit)
            continue

        summary_df = pd.concat(summary_frames, ignore_index=True)
        summary_df["circuit"] = circuit
        summary_df["display_name"] = display_name
        summary_path = output_dir / f"{circuit}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        _write_markdown(summary_df, summary_path.with_suffix(".md"))
        all_summary_frames.append(summary_df)

        speedups, figure_path = _export_plot(
            summary_df,
            showcase_benchmarks.SHOWCASE_CIRCUITS[circuit],
            figures_dir=DEFAULT_FIGURES_DIR,
            metric=metric,
        )
        if speedups is not None and not speedups.empty:
            speedups["circuit"] = circuit
            speedups["display_name"] = display_name
            speedups_path = output_dir / f"{circuit}_speedups.csv"
            speedups.to_csv(speedups_path, index=False)
            _write_markdown(speedups, speedups_path.with_suffix(".md"))
            all_speedups.append(speedups)
        if figure_path is not None:
            LOGGER.info("Saved figure for %s to %s", circuit, figure_path)

    if all_frames:
        combined_raw = pd.concat(all_frames, ignore_index=True)
        combined_raw.sort_values(["circuit", "qubits", "framework"], inplace=True)
        raw_path = output_dir / "showcase_raw.csv"
        combined_raw = _merge_results(
            raw_path,
            combined_raw,
            key_columns=("circuit", "framework", "qubits"),
            sort_columns=("circuit", "qubits", "framework"),
        )
        combined_raw.to_csv(raw_path, index=False)
        _write_markdown(combined_raw, raw_path.with_suffix(".md"))

    if all_summary_frames:
        combined_summary = pd.concat(all_summary_frames, ignore_index=True)
        summary_path = output_dir / "showcase_summary.csv"
        combined_summary = _merge_results(
            summary_path,
            combined_summary,
            key_columns=("circuit", "framework", "qubits"),
            sort_columns=("circuit", "qubits", "framework"),
        )
        combined_summary.to_csv(summary_path, index=False)
        _write_markdown(combined_summary, summary_path.with_suffix(".md"))

    if all_speedups:
        combined_speedups = pd.concat(all_speedups, ignore_index=True)
        speedups_path = output_dir / "showcase_speedups.csv"
        combined_speedups = _merge_results(
            speedups_path,
            combined_speedups,
            key_columns=("circuit", "baseline_backend", "qubits"),
            sort_columns=("circuit", "qubits", "baseline_backend"),
        )
        combined_speedups.to_csv(speedups_path, index=False)
        _write_markdown(combined_speedups, speedups_path.with_suffix(".md"))


def _export_estimation_tables(database: Path) -> None:
    with sqlite3.connect(str(database)) as conn:
        df = pd.read_sql_query(
            "SELECT circuit_id AS circuit, qubits, framework, backend, supported, time_ops,"
            " approx_seconds, memory_bytes, note FROM estimation",
            conn,
        )
    if df.empty:
        LOGGER.warning("No estimation data found in %s", database)
        return
    df["supported"] = df["supported"].astype(bool)
    df.sort_values(["circuit", "qubits", "framework"], inplace=True)
    summary = build_summary(df)
    write_tables(df, summary)
    plot_runtime_speedups(summary)
    plot_memory_ratio(summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from a QuASAr benchmark database.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database path (default: %(default)s)",
    )
    parser.add_argument(
        "--circuit",
        "--circuits",
        dest="circuit_names",
        action="append",
        default=None,
        help="Limit output to the specified circuit(s) (repeat flag to add more).",
    )
    parser.add_argument(
        "--group",
        dest="groups",
        action="append",
        default=None,
        choices=sorted(SHOWCASE_GROUPS),
        help="Include all circuits that belong to the named group(s).",
    )
    parser.add_argument(
        "--metric",
        default="run_time_mean",
        help="Metric to use for the generated figures (default: run_time_mean).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for CSV/Markdown outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--estimation",
        action="store_true",
        help="Also export theoretical estimation tables and figures.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use twice for debug output).",
    )
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    circuits = _resolve_circuits(args.circuit_names, args.groups)
    _configure_logging(args.verbose)

    LOGGER.info("Loading showcase results for circuits: %s", ", ".join(circuits))
    raw_df = _load_showcase_results(args.database, circuits)
    if raw_df.empty:
        LOGGER.warning("No simulation runs found for the requested selection")
    else:
        _write_raw_tables(raw_df, args.output_dir, args.metric)

    if args.estimation:
        LOGGER.info("Exporting estimation tables")
        _export_estimation_tables(args.database)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

