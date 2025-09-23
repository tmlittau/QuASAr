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
    from plot_utils import backend_labels  # type: ignore[no-redef]
else:  # pragma: no cover - exercised via runtime execution
    from .plot_utils import backend_labels

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


def _build_backend_speedup_table(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "backend_vs_baseline_speedups.csv"
    if not path.exists():
        raise FileNotFoundError(f"backend speedup summary not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    required = [
        "circuit",
        "qubits",
        "run_time_mean_baseline",
        "run_time_mean_quasar",
        "backend_baseline",
        "backend_quasar",
        "speedup",
    ]
    _require_columns(df, required, context="backend speedup table")
    df = df.loc[:, required].copy()
    df.sort_values(["circuit", "qubits"], inplace=True)
    table = pd.DataFrame(
        {
            "Circuit": df["circuit"].map(_format_circuit_name),
            "Qubits": df["qubits"].map(_format_integer),
            "Baseline backend": df["backend_baseline"].map(_format_backend),
            "Baseline runtime": df["run_time_mean_baseline"].map(
                _format_duration_seconds
            ),
            "QuASAr backend": df["backend_quasar"].map(_format_backend),
            "QuASAr runtime": df["run_time_mean_quasar"].map(
                _format_duration_seconds
            ),
            "Speedup": df["speedup"].map(_format_speedup),
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
        name="backend_speedups",
        builder=_build_backend_speedup_table,
        caption=(
            "Runtime comparison between QuASAr and the strongest baseline per "
            "circuit."
        ),
        label="tab:backend-speedups",
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
) -> dict[str, Path]:
    """Generate LaTeX tables and return their file paths."""

    base_results = Path(results_dir) if results_dir is not None else RESULTS_DIR
    base_output = Path(output_dir) if output_dir is not None else TABLES_DIR
    base_output.mkdir(parents=True, exist_ok=True)

    selected = {name for name in tables} if tables is not None else None
    written: dict[str, Path] = {}
    for spec in TABLE_SPECS:
        if selected is not None and spec.name not in selected:
            continue
        try:
            table = spec.builder(base_results)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping %s: %s", spec.name, exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to build table %s: %s", spec.name, exc)
            continue
        if table.empty:
            LOGGER.warning("Skipping %s: no rows to tabulate", spec.name)
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
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    generate_tables(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        tables=_parse_tables(args.tables),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
