"""Utility helpers for the theoretical cost estimation workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


if __package__ in {None, ""}:  # pragma: no cover - script execution
    import importlib
    import sys

    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    importlib.import_module("quasar")
    from plot_utils import (  # type: ignore[no-redef]
        plot_speedup_bars,
        setup_benchmark_style,
    )
    from quasar.calibration import (  # type: ignore[no-redef]
        apply_calibration,
        latest_coefficients,
        load_coefficients,
    )
    from showcase_benchmarks import SHOWCASE_CIRCUITS  # type: ignore[no-redef]
else:  # pragma: no cover - package import path
    from .plot_utils import plot_speedup_bars, setup_benchmark_style
    from quasar.calibration import apply_calibration, latest_coefficients, load_coefficients
    from .showcase_benchmarks import SHOWCASE_CIRCUITS

from quasar.cost import CostEstimator


FIGURES_DIR = PACKAGE_ROOT / "figures"
RESULTS_DIR = PACKAGE_ROOT / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


_SHOWCASE_CIRCUITS = globals().get("SHOWCASE_CIRCUITS", {})
SHOWCASE_DISPLAY_NAMES = {
    name: getattr(spec, "display_name", name)
    for name, spec in _SHOWCASE_CIRCUITS.items()
    if getattr(spec, "display_name", None)
}


def _format_circuit_name(circuit: str) -> str:
    """Return a human-friendly display name for ``circuit``."""

    if circuit in SHOWCASE_DISPLAY_NAMES:
        return SHOWCASE_DISPLAY_NAMES[circuit]
    return circuit.replace("_", " ")


def _format_speedup_label(circuit: str, qubits: object) -> str:
    """Return descriptive label for speedup plots."""

    base = _format_circuit_name(circuit)
    if qubits is None:
        return base
    try:
        numeric = float(qubits)
    except (TypeError, ValueError):
        return base
    if not math.isfinite(numeric):
        return base
    if abs(numeric - round(numeric)) < 1e-9:
        numeric = round(numeric)
    return f"{base} ({numeric:g}q)"


OPS_PER_SECOND_DEFAULT = 1_000_000_000.0
"""Baseline throughput for converting cost-model operations into seconds.

The cost model counts backend-specific primitive operations.  On the shared
benchmark workstation (Intel i9-13900K, NVIDIA RTX 4090, 64 GB RAM) the
empirical calibration runs show that roughly ``1e9`` primitive operations per
second mirrors the observed runtime of QuASAr's GPU-accelerated simulator.  The
value therefore serves as the default conversion factor while still allowing
users to override it via ``--ops-per-second`` when they want to reflect a
different hardware profile.
"""


@dataclass
class EstimateRecord:
    """Container for theoretical runtime and memory estimates."""

    circuit: str
    qubits: int
    framework: str
    backend: str
    supported: bool
    time_ops: float | None
    memory_bytes: float | None
    note: str | None = None

    def approx_seconds(self, ops_per_second: float | None) -> float | None:
        """Return the runtime in seconds under the supplied throughput."""

        if not self.supported or self.time_ops is None:
            return None
        if ops_per_second in (None, 0):
            return None
        return self.time_ops / ops_per_second

    def as_dict(self, ops_per_second: float | None) -> Mapping[str, object]:
        """Return a serialisable representation of the record."""

        approx = self.approx_seconds(ops_per_second)
        time_ops = float(self.time_ops) if self.time_ops is not None else math.nan
        memory = float(self.memory_bytes) if self.memory_bytes is not None else math.nan
        return {
            "circuit": self.circuit,
            "qubits": self.qubits,
            "framework": self.framework,
            "backend": self.backend,
            "supported": bool(self.supported),
            "time_ops": time_ops,
            "approx_seconds": approx if approx is not None else math.nan,
            "memory_bytes": memory,
            "note": self.note or "",
        }


def load_estimator(calibration: Path | None) -> CostEstimator:
    """Return a cost estimator optionally initialised from calibration data."""

    estimator = CostEstimator()
    record = load_coefficients(calibration) if calibration is not None else latest_coefficients()
    if record:
        apply_calibration(estimator, record)
    return estimator


def format_seconds(seconds: float | None) -> str:
    """Return a human-friendly representation of ``seconds``."""

    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {rem:.0f} s"
    hours, rem = divmod(minutes, 60)
    if hours < 24:
        return f"{int(hours)} h {int(rem)} min"
    days, rem = divmod(hours, 24)
    if days < 365:
        return f"{int(days)} d {int(rem)} h"
    years, rem = divmod(days, 365)
    return f"{int(years)} y {int(rem)} d"


def build_dataframe(records: Sequence[EstimateRecord], ops_per_second: float | None) -> pd.DataFrame:
    """Convert ``records`` into a sorted :class:`~pandas.DataFrame`."""

    rows = [rec.as_dict(ops_per_second) for rec in records]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["circuit", "qubits", "framework"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Return per-circuit summary comparing baseline best to QuASAr."""

    if detail.empty:
        return detail

    baselines = detail[detail["framework"] != "quasar"].copy()
    baselines = baselines[baselines["supported"]]
    baselines = baselines.replace({np.nan: None})
    baselines = baselines.dropna(subset=["time_ops"])
    if baselines.empty:
        return pd.DataFrame(
            columns=[
                "circuit",
                "qubits",
                "baseline_framework",
                "baseline_backend",
                "baseline_time_ops",
                "baseline_seconds",
                "baseline_memory_bytes",
                "baseline_note",
                "quasar_backend",
                "quasar_time_ops",
                "quasar_seconds",
                "quasar_memory_bytes",
                "quasar_note",
                "speedup",
                "memory_ratio",
            ]
        )

    idx = baselines.groupby(["circuit", "qubits"])["time_ops"].idxmin()
    baseline_best = baselines.loc[idx].copy()
    baseline_best.rename(
        columns={
            "framework": "baseline_framework",
            "backend": "baseline_backend",
            "time_ops": "baseline_time_ops",
            "approx_seconds": "baseline_seconds",
            "memory_bytes": "baseline_memory_bytes",
            "note": "baseline_note",
        },
        inplace=True,
    )

    quasar = detail[detail["framework"] == "quasar"].copy()
    quasar = quasar[quasar["supported"]]
    quasar = quasar.dropna(subset=["time_ops"])
    quasar.rename(
        columns={
            "backend": "quasar_backend",
            "time_ops": "quasar_time_ops",
            "approx_seconds": "quasar_seconds",
            "memory_bytes": "quasar_memory_bytes",
            "note": "quasar_note",
        },
        inplace=True,
    )
    quasar = quasar[
        [
            "circuit",
            "qubits",
            "quasar_backend",
            "quasar_time_ops",
            "quasar_seconds",
            "quasar_memory_bytes",
            "quasar_note",
        ]
    ]

    summary = pd.merge(
        baseline_best[
            [
                "circuit",
                "qubits",
                "baseline_framework",
                "baseline_backend",
                "baseline_time_ops",
                "baseline_seconds",
                "baseline_memory_bytes",
                "baseline_note",
            ]
        ],
        quasar,
        on=["circuit", "qubits"],
        how="outer",
    )

    def _safe_ratio(num: float | None, denom: float | None) -> float | None:
        if num is None or denom is None:
            return None
        if denom == 0:
            return None
        return num / denom

    summary["speedup"] = summary.apply(
        lambda row: _safe_ratio(row.get("baseline_seconds"), row.get("quasar_seconds")),
        axis=1,
    )
    summary["memory_ratio"] = summary.apply(
        lambda row: _safe_ratio(row.get("baseline_memory_bytes"), row.get("quasar_memory_bytes")),
        axis=1,
    )

    summary.sort_values(["circuit", "qubits"], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def write_tables(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Persist the detailed and summary tables as CSV (and Markdown) files."""

    detail_path = RESULTS_DIR / "theoretical_requirements.csv"
    summary_path = RESULTS_DIR / "theoretical_requirements_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")

    try:
        markdown_path = RESULTS_DIR / "theoretical_requirements_summary.md"
        md = summary.to_markdown(index=False, floatfmt=".4g")
    except Exception:
        md = None
    if md:
        markdown_path.write_text(md + "\n", encoding="utf-8")
        print(f"Wrote {markdown_path}")


def plot_runtime_speedups(summary: pd.DataFrame) -> None:
    """Generate a bar chart visualising baseline-to-QuASAr runtime ratios."""

    valid = summary.dropna(subset=["speedup"])
    if valid.empty:
        return
    data: dict[str, float] = {}
    for row in valid.itertuples():
        label = _format_speedup_label(row.circuit, row.qubits)
        data[label] = float(row.speedup)

    ax = plot_speedup_bars(data, sort=False)
    ax.set_title("Estimated runtime speedup (baseline best ÷ QuASAr)")
    ax.set_ylabel("Speedup (×)")
    fig = ax.figure
    path = FIGURES_DIR / "theoretical_runtime_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_relative_speedups(summary: pd.DataFrame) -> None:
    """Persist the relative speedup figure mirroring benchmark outputs."""

    valid = summary.dropna(subset=["speedup"])
    if valid.empty:
        return

    records: list[dict[str, object]] = []
    data: dict[str, float] = {}
    for row in valid.itertuples():
        label = _format_speedup_label(row.circuit, row.qubits)
        speedup = float(row.speedup)
        data[label] = speedup

        width: object = row.qubits
        if width is not None and pd.notna(width):
            try:
                width = int(width)
            except (TypeError, ValueError):
                width = float(width)

        records.append(
            {
                "circuit": row.circuit,
                "qubits": width,
                "label": label,
                "speedup": speedup,
            }
        )

    ax = plot_speedup_bars(data, sort=True)
    ax.set_title("Relative speedups (baseline best ÷ QuASAr)")
    fig = ax.figure
    png_path = FIGURES_DIR / "theoretical_relative_speedups.png"
    pdf_path = FIGURES_DIR / "theoretical_relative_speedups.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")

    export = pd.DataFrame.from_records(records)
    export.sort_values("speedup", ascending=False, inplace=True)
    csv_path = RESULTS_DIR / "theoretical_relative_speedups.csv"
    export.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


def plot_memory_ratio(summary: pd.DataFrame) -> None:
    """Generate a bar chart for baseline versus QuASAr memory requirements."""

    valid = summary.dropna(subset=["memory_ratio"])
    if valid.empty:
        return
    labels = [f"{row.circuit}@{row.qubits}" for row in valid.itertuples()]
    values = valid["memory_ratio"].to_list()

    setup_benchmark_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#264653")
    ax.set_ylabel("Baseline ÷ QuASAr peak memory")
    ax.set_title("Estimated memory ratio (baseline best ÷ QuASAr)")
    ax.set_ylim(bottom=0)
    ax.bar_label(bars, fmt="{:.2f}", padding=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    path = FIGURES_DIR / "theoretical_memory_ratio.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def report_totals(detail: pd.DataFrame) -> None:
    """Print aggregate runtime estimates for QuASAr and the baseline best."""

    quasar = detail[(detail["framework"] == "quasar") & detail["supported"]]
    baselines = detail[(detail["framework"] != "quasar") & detail["supported"]]
    if quasar.empty:
        return

    quasar_total = quasar["approx_seconds"].sum(skipna=True)
    baseline_best = baselines.groupby(["circuit", "qubits"])["approx_seconds"].min().sum()
    print(
        "Total estimated runtime — QuASAr:",
        format_seconds(quasar_total),
        "; baseline best:",
        format_seconds(baseline_best),
    )


__all__ = [
    "EstimateRecord",
    "FIGURES_DIR",
    "RESULTS_DIR",
    "OPS_PER_SECOND_DEFAULT",
    "build_dataframe",
    "build_summary",
    "format_seconds",
    "load_estimator",
    "plot_memory_ratio",
    "plot_relative_speedups",
    "plot_runtime_speedups",
    "report_totals",
    "write_tables",
]

