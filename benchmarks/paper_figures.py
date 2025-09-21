"""Generate reproducible benchmark figures for the QuASAr paper.

Run the module as a script with ``--verbose`` to display progress logs while
figures and result tables are generated.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    from parallel_circuits import many_ghz_subsystems  # type: ignore[no-redef]
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
    from .parallel_circuits import many_ghz_subsystems

from quasar import SimulationEngine
from quasar.cost import Backend


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbosity: int) -> None:
    """Initialise logging for CLI usage."""

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _log_written(path: Path) -> None:
    """Emit a user-friendly message when ``path`` is written."""

    LOGGER.info("Wrote %s", path)


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


def _ghz_ladder_circuit(n_qubits: int, *, group_size: int = 4):
    """Return disjoint GHZ ladders that sum to ``n_qubits``."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if n_qubits % group_size != 0:
        raise ValueError(
            "n_qubits must be divisible by group_size for ghz ladder construction"
        )
    num_groups = n_qubits // group_size
    return many_ghz_subsystems(num_groups=num_groups, group_size=group_size)


def _random_clifford_t_circuit(
    n_qubits: int, *, depth_multiplier: int = 3, base_seed: int = 97
):
    """Return a reproducible Clifford+T hybrid circuit."""

    if depth_multiplier <= 0:
        raise ValueError("depth_multiplier must be positive")
    depth = depth_multiplier * n_qubits
    seed = base_seed + n_qubits
    return circuit_lib.random_hybrid_circuit(n_qubits, depth=depth, seed=seed)


def _large_grover_circuit(n_qubits: int, *, iterations: int = 2):
    """Return a Grover search circuit scaled to ``n_qubits``."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return circuit_lib.grover_circuit(n_qubits, n_iterations=iterations)


CIRCUITS: Sequence[CircuitSpec] = (
    CircuitSpec("qft", circuit_lib.qft_circuit, (3, 4)),
    CircuitSpec("grover", circuit_lib.grover_circuit, (3, 4), {"n_iterations": 1}),
    CircuitSpec(
        "ghz_ladder",
        lambda n, *, group_size=4: _ghz_ladder_circuit(n, group_size=group_size),
        (20, 24, 28, 32),
        {"group_size": 4},
    ),
    CircuitSpec(
        "random_clifford_t",
        lambda n, *, depth_multiplier=3, seed=97: _random_clifford_t_circuit(
            n, depth_multiplier=depth_multiplier, base_seed=seed
        ),
        (20, 24, 28, 32),
        {"depth_multiplier": 3, "seed": 97},
    ),
    CircuitSpec(
        "grover_large",
        lambda n, *, iterations=2: _large_grover_circuit(n, iterations=iterations),
        (20, 24, 28, 32),
        {"iterations": 2},
    ),
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

    spec_list = list(specs)
    LOGGER.info(
        "Collecting backend data for %d circuit family(ies)", len(spec_list)
    )
    if not spec_list:
        LOGGER.warning("No circuit specifications provided; skipping collection")
        return pd.DataFrame(), pd.DataFrame()

    engine = SimulationEngine()
    forced_records: list[dict[str, object]] = []
    auto_records: list[dict[str, object]] = []

    for spec in spec_list:
        LOGGER.info("Processing circuit family '%s'", spec.name)
        for n in spec.qubits:
            LOGGER.info("Preparing circuits for %s with %s qubits", spec.name, n)
            circuit_forced = _build_circuit(spec, n, use_classical_simplification=False)
            circuit_auto = _build_circuit(spec, n, use_classical_simplification=True)
            if circuit_forced is None or circuit_auto is None:
                LOGGER.warning(
                    "Skipping circuit %s with %s qubits because construction failed",
                    spec.name,
                    n,
                )
                continue

            for backend in backends:
                runner = BenchmarkRunner()
                LOGGER.info(
                    "Executing forced run: circuit=%s qubits=%s backend=%s",
                    spec.name,
                    n,
                    backend.name,
                )
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
                    LOGGER.warning(
                        "Forced run failed for circuit=%s qubits=%s backend=%s: %s",
                        spec.name,
                        n,
                        backend.name,
                        exc,
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
                LOGGER.info(
                    "Completed forced run: circuit=%s qubits=%s backend=%s",
                    spec.name,
                    n,
                    backend.name,
                )

            runner = BenchmarkRunner()
            LOGGER.info(
                "Executing automatic run: circuit=%s qubits=%s backend=quasar",
                spec.name,
                n,
            )
            try:
                rec = runner.run_quasar_multiple(
                    circuit_auto,
                    engine,
                    repetitions=repetitions,
                    quick=False,
                )
            except Exception as exc:  # pragma: no cover - skip unsupported mixes
                LOGGER.warning(
                    "Automatic scheduling failed for circuit=%s qubits=%s: %s",
                    spec.name,
                    n,
                    exc,
                )
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
            LOGGER.info(
                "Completed automatic run: circuit=%s qubits=%s backend=quasar",
                spec.name,
                n,
            )

    return pd.DataFrame(forced_records), pd.DataFrame(auto_records)


def generate_backend_comparison() -> None:
    LOGGER.info("Generating backend comparison figures")
    forced, auto = collect_backend_data(CIRCUITS, BACKENDS, repetitions=3)
    forced_path = RESULTS_DIR / "backend_forced.csv"
    auto_path = RESULTS_DIR / "backend_auto.csv"
    forced.to_csv(forced_path, index=False)
    auto.to_csv(auto_path, index=False)
    _log_written(forced_path)
    _log_written(auto_path)

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
    png_path = FIGURES_DIR / "backend_vs_baseline.png"
    pdf_path = FIGURES_DIR / "backend_vs_baseline.pdf"
    csv_path = RESULTS_DIR / "backend_vs_baseline_speedups.csv"
    ax.figure.savefig(png_path)
    ax.figure.savefig(pdf_path)
    _log_written(png_path)
    _log_written(pdf_path)
    summary.to_csv(csv_path, index=False)
    _log_written(csv_path)
    plt.close(ax.figure)

    if not forced.empty and not auto.empty:
        grid = plot_backend_timeseries(forced, auto, metric="run_time_mean")
        runtime_png = FIGURES_DIR / "backend_timeseries_runtime.png"
        runtime_pdf = FIGURES_DIR / "backend_timeseries_runtime.pdf"
        grid.savefig(runtime_png)
        grid.savefig(runtime_pdf)
        _log_written(runtime_png)
        _log_written(runtime_pdf)
        plt.close(grid.fig)

        for frame in (forced, auto):
            frame["run_peak_memory_mib"] = frame.get("run_peak_memory_mean", 0) / (1024**2)

        grid_mem = plot_backend_timeseries(
            forced.assign(run_peak_memory_mean=forced["run_peak_memory_mib"]),
            auto.assign(run_peak_memory_mean=auto["run_peak_memory_mib"]),
            metric="run_peak_memory_mean",
            annotate_auto=False,
        )
        mem_png = FIGURES_DIR / "backend_timeseries_memory.png"
        mem_pdf = FIGURES_DIR / "backend_timeseries_memory.pdf"
        grid_mem.savefig(mem_png)
        grid_mem.savefig(mem_pdf)
        _log_written(mem_png)
        _log_written(mem_pdf)
        plt.close(grid_mem.fig)
    else:
        LOGGER.info(
            "Skipping backend timeseries plots because one of the result tables is empty"
        )


def generate_heatmap() -> None:
    results_path = RESULTS_DIR / "plan_choice_heatmap_results.json"
    if not results_path.exists():
        LOGGER.info(
            "Skipping plan-choice heatmap: results file %s is missing",
            results_path,
        )
        return
    data = json.loads(results_path.read_text())
    if not data:
        LOGGER.info(
            "Skipping plan-choice heatmap: results file %s is empty",
            results_path,
        )
        return

    LOGGER.info("Generating plan-choice heatmap from %s", results_path)
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
    heatmap_png = FIGURES_DIR / "plan_choice_heatmap.png"
    heatmap_pdf = FIGURES_DIR / "plan_choice_heatmap.pdf"
    table_csv = RESULTS_DIR / "plan_choice_heatmap_table.csv"
    ax.figure.savefig(heatmap_png)
    ax.figure.savefig(heatmap_pdf)
    _log_written(heatmap_png)
    _log_written(heatmap_pdf)
    annot.to_csv(table_csv)
    _log_written(table_csv)
    plt.close(ax.figure)


def generate_speedup_bars() -> None:
    csv_path = Path(__file__).resolve().parent / "quick_analysis_results.csv"
    if not csv_path.exists():
        LOGGER.info(
            "Skipping speedup summary: quick-analysis results %s not found",
            csv_path,
        )
        return
    df = pd.read_csv(csv_path)
    LOGGER.info("Generating speedup summary from %s", csv_path)
    df["label"] = df.apply(
        lambda row: f"{int(row['qubits'])}q d{int(row['depth'])}", axis=1
    )
    grouped = df.groupby("label")
    speedups = grouped["speedup"].mean().to_dict()
    ax = plot_speedup_bars(speedups)
    ax.figure.tight_layout()
    speedup_png = FIGURES_DIR / "relative_speedups.png"
    speedup_pdf = FIGURES_DIR / "relative_speedups.pdf"
    speedup_csv = RESULTS_DIR / "relative_speedups.csv"
    ax.figure.savefig(speedup_png)
    ax.figure.savefig(speedup_pdf)
    _log_written(speedup_png)
    _log_written(speedup_pdf)
    grouped["speedup"].mean().reset_index().to_csv(speedup_csv, index=False)
    _log_written(speedup_csv)
    plt.close(ax.figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible benchmark figures for the QuASAr paper",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for debug output).",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    setup_benchmark_style()
    generate_backend_comparison()
    generate_heatmap()
    generate_speedup_bars()


if __name__ == "__main__":
    main()
