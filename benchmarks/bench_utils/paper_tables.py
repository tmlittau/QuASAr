"""Generate publication-ready tables for the QuASAr paper.

The script mirrors :mod:`benchmarks.paper_figures` but focuses on producing
tabular summaries that can be embedded directly into the paper.  Tables are
exported as LaTeX ``.tex`` files and rely on benchmark results emitted by the
benchmark runners.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if __package__ in {None, ""}:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from plot_utils import backend_labels, compute_baseline_best  # type: ignore[no-redef]
    from progress import ProgressReporter  # type: ignore[no-redef]
    from threading_utils import resolve_worker_count  # type: ignore[no-redef]
else:  # pragma: no cover - exercised via runtime execution
    from .plot_utils import backend_labels, compute_baseline_best
    from .progress import ProgressReporter
    from .threading_utils import resolve_worker_count

from quasar.cost import Backend


LOGGER = logging.getLogger(__name__)

RESULTS_DIR = PACKAGE_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TableSpec:
    """Description of a LaTeX table to be generated."""

    name: str
    builder: Callable[[Path], pd.DataFrame]
    caption: str | None = None
    label: str | None = None
    column_format: str | None = None


# Cache human readable backend labels for reuse.
BACKEND_LABELS = backend_labels()

# Partitioning scenarios summarised for publication tables.
PARTITIONING_SUMMARY_FILES: Mapping[str, str] = {
    "dual_magic_injection": "dual_magic_injection_summary.csv",
    "staged_rank": "staged_rank_summary.csv",
    "staged_sparsity": "staged_sparsity_summary.csv",
    "tableau_boundary": "tableau_boundary_summary.csv",
    "w_state_oracle": "w_state_oracle_summary.csv",
}


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _log_written(path: Path) -> None:
    LOGGER.info("Wrote %s", path)


def _format_circuit_name(name: object) -> str:
    if not isinstance(name, str):
        return str(name)
    words = []
    for word in name.replace("_", " ").split():
        lower = word.lower()
        if lower in {"ghz", "qft", "mps", "sv"}:
            words.append(lower.upper())
        elif lower == "quasar":
            words.append("QuASAr")
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _format_scenario_name(name: object) -> str:
    if not isinstance(name, str):
        return str(name)
    words = []
    for word in name.replace("_", " ").split():
        lower = word.lower()
        if lower in {"ghz", "qft"}:
            words.append(lower.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _format_variant(name: object) -> str:
    text = str(name)
    if "_" in text:
        suffix = text.split("_")[-1]
        if suffix.isdigit():
            return suffix
    return text


def _format_integer(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{int(round(float(value)))}"
    except (TypeError, ValueError):
        return str(value)


def _format_duration_seconds(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    magnitude = abs(numeric)
    if magnitude >= 1.0:
        return f"{numeric:.2f}\\,\\mathrm{{s}}"
    if magnitude >= 1e-3:
        return f"{numeric * 1e3:.1f}\\,\\mathrm{{ms}}"
    if magnitude >= 1e-6:
        return f"{numeric * 1e6:.1f}\\,\\mathrm{{\\mu s}}"
    return f"{numeric:.2e}\\,\\mathrm{{s}}"


def _format_speedup(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.2f}\\times"


def _format_memory_bytes(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return ""
    mib = numeric / (1024**2)
    if abs(mib) >= 1024:
        gib = mib / 1024
        return f"{gib:.2f}\\,\\mathrm{{GiB}}"
    return f"{mib:.1f}\\,\\mathrm{{MiB}}"


def _lookup_backend(value: object) -> Backend | None:
    if isinstance(value, Backend):
        return value
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    upper = token.upper()
    try:
        return Backend[upper]
    except KeyError:
        lower = token.lower()
        for backend in Backend:
            if lower == backend.value or lower == backend.name.lower():
                return backend
    return None


def _format_backend(value: object) -> str:
    backend = _lookup_backend(value)
    if backend is None:
        return str(value)
    return BACKEND_LABELS.get(backend, backend.name.title())


def _format_boolean(
    value: object,
    *,
    true_label: str = "\\checkmark",
    false_label: str = "\\texttimes",
) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in {"nan", "none"}:
            return ""
        if text in {"true", "yes", "y", "1"}:
            return true_label
        if text in {"false", "no", "n", "0"}:
            return false_label
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return true_label if bool(value) else false_label
    if numeric == 0:
        return false_label
    return true_label


def _format_rotation_set(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"")
    if not text:
        return ""
    entries = []
    for token in text.split(","):
        label = token.strip().upper()
        if not label:
            continue
        mapped = {
            "RZ": "R_{z}",
            "RY": "R_{y}",
            "RX": "R_{x}",
        }.get(label, label)
        entries.append(mapped)
    return "$\\{" + ", ".join(entries) + "\\}$" if entries else ""


def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {', '.join(missing)}")


def _dataframe_to_latex(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
) -> str:
    """Render ``table`` as a LaTeX tabular environment."""

    if table.empty:
        raise ValueError("Cannot render empty table")

    rendered = table.fillna("")
    columns = [str(column) for column in rendered.columns]
    align = column_format or ("l" * len(columns))
    lines = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\hline")
    lines.append(" & ".join(columns) + r" \\")
    lines.append("\\hline")
    for _, row in rendered.iterrows():
        values = [str(value) for value in row.tolist()]
        lines.append(" & ".join(values) + r" \\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _load_showcase_raw(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "showcase" / "showcase_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"showcase raw results not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        LOGGER.info("Showcase raw results at %s are empty", path)
    return df


def _build_showcase_summary(results_dir: Path) -> pd.DataFrame:
    raw = _load_showcase_raw(results_dir)
    if raw.empty:
        return pd.DataFrame()

    required = {
        "circuit",
        "qubits",
        "framework",
        "backend",
        "run_time_mean",
        "run_peak_memory_mean",
    }
    _require_columns(raw, required, context="showcase summary")

    try:
        baseline_runtime = compute_baseline_best(
            raw,
            metrics=("run_time_mean", "run_peak_memory_mean"),
        )
    except ValueError as exc:
        LOGGER.warning("Unable to compute showcase baseline minima: %s", exc)
        return pd.DataFrame()

    baseline_runtime = baseline_runtime.rename(
        columns={
            "backend": "baseline_runtime_backend",
            "run_time_mean": "baseline_run_time_mean",
            "run_time_std": "baseline_run_time_std",
            "run_peak_memory_mean": "baseline_run_peak_memory_mean",
            "run_peak_memory_std": "baseline_run_peak_memory_std",
        }
    )

    quasar = raw[raw["framework"].astype(str).str.lower() == "quasar"].copy()
    if quasar.empty:
        LOGGER.info("Showcase results do not contain QuASAr measurements")
        return pd.DataFrame()

    if "unsupported" in quasar.columns:
        mask = quasar["unsupported"].astype("boolean", copy=False).fillna(False)
        quasar = quasar[~mask.to_numpy(dtype=bool)]
    if "failed" in quasar.columns:
        mask = quasar["failed"].astype("boolean", copy=False).fillna(False)
        quasar = quasar[~mask.to_numpy(dtype=bool)]
    if quasar.empty:
        LOGGER.info("No successful QuASAr runs available for showcase summary")
        return pd.DataFrame()

    quasar = quasar.rename(
        columns={
            "backend": "quasar_backend",
            "run_time_mean": "quasar_run_time_mean",
            "run_time_std": "quasar_run_time_std",
            "total_time_mean": "quasar_total_time_mean",
            "total_time_std": "quasar_total_time_std",
            "run_peak_memory_mean": "quasar_run_peak_memory_mean",
            "run_peak_memory_std": "quasar_run_peak_memory_std",
        }
    )

    quasar_columns = [
        "circuit",
        "qubits",
        "quasar_backend",
        "quasar_run_time_mean",
        "quasar_run_time_std",
        "quasar_total_time_mean",
        "quasar_total_time_std",
        "quasar_run_peak_memory_mean",
        "quasar_run_peak_memory_std",
    ]
    available_quasar_columns = [
        column for column in quasar_columns if column in quasar.columns
    ]
    summary = baseline_runtime.merge(
        quasar[available_quasar_columns],
        on=["circuit", "qubits"],
        how="inner",
    )
    if summary.empty:
        LOGGER.info("Showcase summary merge produced no rows")
        return summary

    try:
        baseline_memory = compute_baseline_best(
            raw,
            metrics=("run_peak_memory_mean", "run_time_mean"),
        )
    except ValueError:
        baseline_memory = pd.DataFrame()
    else:
        baseline_memory = baseline_memory.rename(
            columns={
                "backend": "baseline_memory_backend",
                "run_peak_memory_mean": "baseline_min_peak_memory_mean",
                "run_peak_memory_std": "baseline_min_peak_memory_std",
                "run_time_mean": "baseline_memory_run_time_mean",
                "run_time_std": "baseline_memory_run_time_std",
            }
        )
        memory_columns = [
            "circuit",
            "qubits",
            "baseline_memory_backend",
            "baseline_min_peak_memory_mean",
            "baseline_min_peak_memory_std",
        ]
        available_memory_columns = [
            column for column in memory_columns if column in baseline_memory.columns
        ]
        summary = summary.merge(
            baseline_memory[available_memory_columns],
            on=["circuit", "qubits"],
            how="left",
        )

    runtime_valid = (
        summary["baseline_run_time_mean"] > 0
    ) & (summary["quasar_run_time_mean"] > 0)
    summary["runtime_speedup"] = summary["baseline_run_time_mean"] / summary[
        "quasar_run_time_mean"
    ]
    summary.loc[~runtime_valid, "runtime_speedup"] = pd.NA

    memory_valid = (
        summary["baseline_run_peak_memory_mean"] > 0
    ) & (summary["quasar_run_peak_memory_mean"] > 0)
    summary["memory_ratio"] = summary["baseline_run_peak_memory_mean"] / summary[
        "quasar_run_peak_memory_mean"
    ]
    summary.loc[~memory_valid, "memory_ratio"] = pd.NA

    runtime_baseline_backend = summary["baseline_runtime_backend"].map(
        _lookup_backend
    )
    quasar_backend = summary["quasar_backend"].map(_lookup_backend)
    summary["runtime_optimal"] = pd.Series(
        runtime_baseline_backend == quasar_backend,
        dtype="boolean",
    )
    summary.loc[runtime_baseline_backend.isna(), "runtime_optimal"] = pd.NA

    if "baseline_memory_backend" in summary.columns:
        memory_baseline_backend = summary["baseline_memory_backend"].map(
            _lookup_backend
        )
        summary["memory_optimal"] = pd.Series(
            memory_baseline_backend == quasar_backend,
            dtype="boolean",
        )
        summary.loc[memory_baseline_backend.isna(), "memory_optimal"] = pd.NA
    else:
        summary["memory_optimal"] = pd.Series(pd.NA, index=summary.index, dtype="boolean")

    scale = float(1024**2)
    summary["baseline_run_peak_memory_mib"] = (
        summary["baseline_run_peak_memory_mean"] / scale
    )
    summary["quasar_run_peak_memory_mib"] = (
        summary["quasar_run_peak_memory_mean"] / scale
    )
    if "baseline_min_peak_memory_mean" in summary.columns:
        summary["baseline_min_peak_memory_mib"] = (
            summary["baseline_min_peak_memory_mean"] / scale
        )

    summary.sort_values(["circuit", "qubits"], inplace=True)
    return summary.reset_index(drop=True)


def _build_showcase_runtime_table(results_dir: Path) -> pd.DataFrame:
    summary = _build_showcase_summary(results_dir)
    if summary.empty:
        return pd.DataFrame()

    table = pd.DataFrame(
        {
            "Circuit": summary["circuit"].map(_format_circuit_name),
            "Qubits": summary["qubits"].map(_format_integer),
            "Baseline backend": summary["baseline_runtime_backend"].map(
                _format_backend
            ),
            "Baseline runtime": summary["baseline_run_time_mean"].map(
                _format_duration_seconds
            ),
            "QuASAr backend": summary["quasar_backend"].map(_format_backend),
            "QuASAr runtime": summary["quasar_run_time_mean"].map(
                _format_duration_seconds
            ),
            "Speedup": summary["runtime_speedup"].map(_format_speedup),
            "Optimal backend": summary["runtime_optimal"].map(_format_boolean),
        }
    )
    return table


def _build_showcase_memory_table(results_dir: Path) -> pd.DataFrame:
    summary = _build_showcase_summary(results_dir)
    if summary.empty:
        return pd.DataFrame()

    min_backend = summary.get(
        "baseline_memory_backend", pd.Series([None] * len(summary))
    )
    min_memory = summary.get(
        "baseline_min_peak_memory_mean", pd.Series([None] * len(summary))
    )

    table = pd.DataFrame(
        {
            "Circuit": summary["circuit"].map(_format_circuit_name),
            "Qubits": summary["qubits"].map(_format_integer),
            "Baseline backend": summary["baseline_runtime_backend"].map(
                _format_backend
            ),
            "Baseline peak memory": summary["baseline_run_peak_memory_mean"].map(
                _format_memory_bytes
            ),
            "QuASAr backend": summary["quasar_backend"].map(_format_backend),
            "QuASAr peak memory": summary["quasar_run_peak_memory_mean"].map(
                _format_memory_bytes
            ),
            "Peak memory ratio": summary["memory_ratio"].map(_format_speedup),
            "Minimum-memory backend": min_backend.map(_format_backend),
            "Minimum peak memory": min_memory.map(_format_memory_bytes),
            "QuASAr memory optimal": summary["memory_optimal"].map(
                _format_boolean
            ),
        }
    )
    return table


def _build_partitioning_summary_table(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for scenario, filename in PARTITIONING_SUMMARY_FILES.items():
        path = results_dir / filename
        if not path.exists():
            LOGGER.debug("Partitioning summary %s missing at %s", scenario, path)
            continue
        df = pd.read_csv(path)
        if df.empty:
            LOGGER.debug("Partitioning summary %s empty", scenario)
            continue
        if "runtime_speedup" not in df.columns:
            LOGGER.debug("Partitioning summary %s lacks runtime_speedup", scenario)
            continue
        speeds = df["runtime_speedup"].astype(float)
        mean_speedup = speeds.mean()
        best_index = speeds.idxmax()
        best = df.loc[best_index]
        backend_value = best.get("quasar_backend") or best.get("baseline_backend")
        rows.append(
            {
                "Scenario": _format_scenario_name(scenario),
                "Best variant": _format_variant(best.get("variant")),
                "Qubits": _format_integer(
                    best.get("total_qubits") or best.get("qubits")
                ),
                "Backend": _format_backend(backend_value),
                "Mean speedup": _format_speedup(mean_speedup),
                "Max speedup": _format_speedup(best.get("runtime_speedup")),
            }
        )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table.sort_values("Scenario", inplace=True)
    return table.reset_index(drop=True)


def _build_w_state_oracle_table(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "w_state_oracle_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"w-state oracle summary not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    required = [
        "variant",
        "oracle_layers",
        "oracle_rotation_gate_count",
        "oracle_rotation_unique",
        "oracle_entangling_count",
        "quasar_backend",
        "runtime_speedup",
    ]
    _require_columns(df, required, context="w-state oracle table")
    df = df.loc[:, required].copy()
    df.sort_values(["oracle_layers", "oracle_rotation_gate_count"], inplace=True)
    table = pd.DataFrame(
        {
            "Variant": df["variant"].map(_format_variant),
            "Layers": df["oracle_layers"].map(_format_integer),
            "Rotations": df["oracle_rotation_gate_count"].map(_format_integer),
            "Rotation set": df["oracle_rotation_unique"].map(_format_rotation_set),
            "Entangling gates": df["oracle_entangling_count"].map(_format_integer),
            "Backend": df["quasar_backend"].map(_format_backend),
            "Speedup": df["runtime_speedup"].map(_format_speedup),
        }
    )
    return table


TABLE_SPECS: Sequence[TableSpec] = (
    TableSpec(
        name="showcase_runtime",
        builder=_build_showcase_runtime_table,
        caption=(
            "Runtime comparison between QuASAr and the strongest baseline for "
            "each showcase circuit."
        ),
        label="tab:showcase-runtime",
    ),
    TableSpec(
        name="showcase_memory",
        builder=_build_showcase_memory_table,
        caption=(
            "Peak memory comparison for the showcase circuits, highlighting "
            "whether QuASAr matches the minimum-memory backend."
        ),
        label="tab:showcase-memory",
    ),
    TableSpec(
        name="partitioning_summary",
        builder=_build_partitioning_summary_table,
        caption=(
            "Summary of partitioning benchmarks highlighting the strongest "
            "observed QuASAr speedups."
        ),
        label="tab:partitioning-summary",
    ),
    TableSpec(
        name="w_state_oracle",
        builder=_build_w_state_oracle_table,
        caption=(
            "Resource requirements and performance impact of W-state phase "
            "oracle variants."
        ),
        label="tab:w-state-oracle",
    ),
)


def generate_tables(
    *,
    results_dir: Path | None = None,
    output_dir: Path | None = None,
    tables: Iterable[str] | None = None,
    max_workers: int | None = None,
) -> dict[str, Path]:
    """Generate LaTeX tables and return their file paths."""

    base_results = Path(results_dir) if results_dir is not None else RESULTS_DIR
    base_output = Path(output_dir) if output_dir is not None else TABLES_DIR
    base_output.mkdir(parents=True, exist_ok=True)

    selected = {name for name in tables} if tables is not None else None
    target_specs = [
        spec for spec in TABLE_SPECS if selected is None or spec.name in selected
    ]
    if selected is not None:
        missing = sorted(selected.difference({spec.name for spec in target_specs}))
        if missing:
            LOGGER.warning("Unknown table(s) requested: %s", ", ".join(missing))
    if not target_specs:
        return {}

    progress = ProgressReporter(len(target_specs), prefix="Table generation")
    written: dict[str, Path] = {}

    worker_count = resolve_worker_count(max_workers, len(target_specs))
    LOGGER.info("Using %d worker thread(s) for table generation", worker_count)

    try:
        if worker_count <= 1:
            for spec in target_specs:
                status = spec.name
                try:
                    table = spec.builder(base_results)
                except FileNotFoundError as exc:
                    LOGGER.warning("Skipping %s: %s", spec.name, exc)
                    progress.advance(f"{status} missing")
                    continue
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.error("Failed to build table %s: %s", spec.name, exc)
                    progress.advance(f"{status} error")
                    continue
                if table.empty:
                    LOGGER.warning("Skipping %s: no rows to tabulate", spec.name)
                    progress.advance(f"{status} empty")
                    continue
                latex_path = base_output / f"{spec.name}.tex"
                latex = _dataframe_to_latex(
                    table,
                    caption=spec.caption,
                    label=spec.label,
                    column_format=spec.column_format,
                )
                latex_path.write_text(latex, encoding="utf-8")
                _log_written(latex_path)
                written[spec.name] = latex_path
                progress.advance(status)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(spec.builder, base_results): spec
                    for spec in target_specs
                }
                for future in as_completed(futures):
                    spec = futures[future]
                    status = spec.name
                    try:
                        table = future.result()
                    except FileNotFoundError as exc:
                        LOGGER.warning("Skipping %s: %s", spec.name, exc)
                        progress.advance(f"{status} missing")
                        continue
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.error("Failed to build table %s: %s", spec.name, exc)
                        progress.advance(f"{status} error")
                        continue
                    if table.empty:
                        LOGGER.warning("Skipping %s: no rows to tabulate", spec.name)
                        progress.advance(f"{status} empty")
                        continue
                    latex_path = base_output / f"{spec.name}.tex"
                    latex = _dataframe_to_latex(
                        table,
                        caption=spec.caption,
                        label=spec.label,
                        column_format=spec.column_format,
                    )
                    latex_path.write_text(latex, encoding="utf-8")
                    _log_written(latex_path)
                    written[spec.name] = latex_path
                    progress.advance(status)
    finally:
        progress.close()

    return written

def _parse_tables(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables summarising QuASAr benchmark results",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for debug output).",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=Path,
        help="Directory containing benchmark result CSV files (default: benchmarks/results).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory where LaTeX tables will be written (default: benchmarks/results/tables).",
    )
    parser.add_argument(
        "-t",
        "--table",
        dest="tables",
        action="append",
        help=(
            "Generate only the specified tables.  Accepts comma separated names "
            "and may be provided multiple times."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for table generation (default: auto).",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    generate_tables(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        tables=_parse_tables(args.tables),
        max_workers=args.workers,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
