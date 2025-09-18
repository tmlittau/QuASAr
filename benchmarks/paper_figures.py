"""Generate reproducible benchmark figures for the QuASAr paper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if __package__ in {None, ""}:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from plot_utils import (  # type: ignore[no-redef]
        backend_labels,
        plot_backend_timeseries,
        plot_heatmap,
        plot_quasar_vs_baseline_best,
        plot_speedup_bars,
        setup_benchmark_style,
    )
    from runner import BenchmarkRunner  # type: ignore[no-redef]
    import circuits as circuit_lib  # type: ignore[no-redef]
else:  # pragma: no cover - exercised via runtime execution
    from .plot_utils import (
        backend_labels,
        plot_backend_timeseries,
        plot_heatmap,
        plot_quasar_vs_baseline_best,
        plot_speedup_bars,
        setup_benchmark_style,
    )
    from .runner import BenchmarkRunner
    from . import circuits as circuit_lib

from quasar import SimulationEngine
from quasar.cost import Backend


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CircuitSpec:
    name: str
    builder: callable
    qubits: Sequence[int]
    kwargs: dict | None = None


BACKENDS: Sequence[Backend] = (
    Backend.STATEVECTOR,
    Backend.TABLEAU,
    Backend.MPS,
    Backend.DECISION_DIAGRAM,
)


CIRCUITS: Sequence[CircuitSpec] = (
    CircuitSpec("qft", circuit_lib.qft_circuit, (3, 4)),
    CircuitSpec("grover", circuit_lib.grover_circuit, (3, 4), {"n_iterations": 1}),
)


def _build_circuit(spec: CircuitSpec, n_qubits: int, *, use_classical_simplification: bool) -> object | None:
    try:
        circuit = spec.builder(n_qubits, **(spec.kwargs or {}))
    except TypeError:
        # Builders used in notebooks rely on the ``use_classical_simplification``
        # attribute rather than accepting a keyword argument.  Align the circuit
        # behaviour manually when the call signature is rigid.
        circuit = spec.builder(n_qubits)
    enable = getattr(circuit, "enable_classical_simplification", None)
    disable = getattr(circuit, "disable_classical_simplification", None)
    if use_classical_simplification:
        if callable(enable):
            enable()
        else:
            circuit.use_classical_simplification = True
    else:
        if callable(disable):
            disable()
        else:
            circuit.use_classical_simplification = False
    return circuit


def collect_backend_data(
    specs: Iterable[CircuitSpec],
    backends: Sequence[Backend],
    *,
    repetitions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return forced and automatic scheduler results for ``specs``."""

    engine = SimulationEngine()
    forced_records: list[dict[str, object]] = []
    auto_records: list[dict[str, object]] = []

    for spec in specs:
        for n in spec.qubits:
            circuit_forced = _build_circuit(spec, n, use_classical_simplification=False)
            circuit_auto = _build_circuit(spec, n, use_classical_simplification=True)
            if circuit_forced is None or circuit_auto is None:
                continue

            for backend in backends:
                runner = BenchmarkRunner()
                try:
                    rec = runner.run_quasar_multiple(
                        circuit_forced,
                        engine,
                        backend=backend,
                        repetitions=repetitions,
                        quick=True,
                    )
                except Exception as exc:  # pragma: no cover - backend limitations
                    forced_records.append(
                        {
                            "circuit": spec.name,
                            "qubits": n,
                            "framework": backend.name,
                            "backend": backend.name,
                            "unsupported": True,
                            "error": str(exc),
                        }
                    )
                    continue

                rec.pop("result", None)
                rec.update(
                    {
                        "circuit": spec.name,
                        "qubits": n,
                        "framework": backend.name,
                        "backend": backend.name,
                        "mode": "forced",
                    }
                )
                forced_records.append(rec)

            runner = BenchmarkRunner()
            try:
                rec = runner.run_quasar_multiple(
                    circuit_auto,
                    engine,
                    repetitions=repetitions,
                    quick=False,
                )
            except Exception:  # pragma: no cover - skip unsupported mixes
                continue
            rec.pop("result", None)
            rec.update(
                {
                    "circuit": spec.name,
                    "qubits": n,
                    "framework": "quasar",
                    "mode": "auto",
                }
            )
            auto_records.append(rec)

    return pd.DataFrame(forced_records), pd.DataFrame(auto_records)


def generate_backend_comparison() -> None:
    forced, auto = collect_backend_data(CIRCUITS, BACKENDS, repetitions=3)
    forced.to_csv(RESULTS_DIR / "backend_forced.csv", index=False)
    auto.to_csv(RESULTS_DIR / "backend_auto.csv", index=False)

    combined = pd.concat([forced, auto], ignore_index=True)
    ax, summary = plot_quasar_vs_baseline_best(
        combined,
        metric="run_time_mean",
        annotate_backend=True,
        return_table=True,
        show_speedup_table=True,
    )
    ax.set_title("Runtime comparison versus baseline best")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "backend_vs_baseline.png")
    ax.figure.savefig(FIGURES_DIR / "backend_vs_baseline.pdf")
    summary.to_csv(RESULTS_DIR / "backend_vs_baseline_speedups.csv", index=False)
    plt.close(ax.figure)

    if not forced.empty and not auto.empty:
        grid = plot_backend_timeseries(forced, auto, metric="run_time_mean")
        grid.savefig(FIGURES_DIR / "backend_timeseries_runtime.png")
        grid.savefig(FIGURES_DIR / "backend_timeseries_runtime.pdf")
        plt.close(grid.fig)

        for frame in (forced, auto):
            frame["run_peak_memory_mib"] = frame.get("run_peak_memory_mean", 0) / (1024**2)

        grid_mem = plot_backend_timeseries(
            forced.assign(run_peak_memory_mean=forced["run_peak_memory_mib"]),
            auto.assign(run_peak_memory_mean=auto["run_peak_memory_mib"]),
            metric="run_peak_memory_mean",
            annotate_auto=False,
        )
        grid_mem.savefig(FIGURES_DIR / "backend_timeseries_memory.png")
        grid_mem.savefig(FIGURES_DIR / "backend_timeseries_memory.pdf")
        plt.close(grid_mem.fig)


def generate_heatmap() -> None:
    results_path = RESULTS_DIR / "plan_choice_heatmap_results.json"
    if not results_path.exists():
        return
    data = json.loads(results_path.read_text())
    if not data:
        return

    df = pd.DataFrame(data)
    df["selected_backend"] = df["steps"].apply(lambda steps: steps[-1] if steps else None)
    labels = backend_labels(df["selected_backend"].dropna().unique())
    df["label"] = df["selected_backend"].map(lambda name: labels.get(name, name))

    pivot = df.pivot(index="circuit", columns="alpha", values="selected_backend")
    annot = df.pivot(index="circuit", columns="alpha", values="label")
    order = list(labels.keys())
    pivot_numeric = pivot.apply(
        lambda col: pd.Categorical(col, categories=order).codes
    )

    ax = plot_heatmap(pivot_numeric, annot=annot, fmt="")
    ax.figure.savefig(FIGURES_DIR / "plan_choice_heatmap.png")
    ax.figure.savefig(FIGURES_DIR / "plan_choice_heatmap.pdf")
    annot.to_csv(RESULTS_DIR / "plan_choice_heatmap_table.csv")
    plt.close(ax.figure)


def generate_speedup_bars() -> None:
    csv_path = Path(__file__).resolve().parent / "quick_analysis_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df["label"] = df.apply(
        lambda row: f"{int(row['qubits'])}q d{int(row['depth'])}", axis=1
    )
    grouped = df.groupby("label")
    speedups = grouped["speedup"].mean().to_dict()
    ax = plot_speedup_bars(speedups)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "relative_speedups.png")
    ax.figure.savefig(FIGURES_DIR / "relative_speedups.pdf")
    grouped["speedup"].mean().reset_index().to_csv(
        RESULTS_DIR / "relative_speedups.csv", index=False
    )
    plt.close(ax.figure)


def main() -> None:
    setup_benchmark_style()
    generate_backend_comparison()
    generate_heatmap()
    generate_speedup_bars()


if __name__ == "__main__":
    main()
