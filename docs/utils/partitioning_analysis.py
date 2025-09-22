"""Helpers for scoring synthetic partitioning scenarios.

This module provides a light-weight façade over :class:`quasar.cost.CostEstimator`
so that documentation and interactive tutorials can explore the planner's
behaviour without constructing full :class:`~quasar.circuit.Gate` objects.  The
functions mirror :class:`quasar.method_selector.MethodSelector`'s feasibility
checks and expose aggregate plan cost calculations that include conversion
steps between heterogeneous backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from matplotlib.figure import Figure

from quasar import config
from quasar.calibration import apply_calibration, load_coefficients
from quasar.cost import Backend, Cost, CostEstimator, ConversionEstimate


_ROOT = Path(__file__).resolve().parents[2]
_FIGURE_DIR = _ROOT / "benchmarks" / "figures" / "partitioning"
_CALIB_DIR = _ROOT / "calibration"


def _latest_calibration_path() -> Path | None:
    """Return the most recent calibration JSON file if present."""

    if not _CALIB_DIR.exists():
        return None
    files = sorted(_CALIB_DIR.glob("coeff_v*.json"))
    return files[-1] if files else None


def load_calibrated_estimator(
    calibration_path: str | Path | None = None,
) -> tuple[CostEstimator, Path | None]:
    """Create an estimator aligned with stored calibration coefficients."""

    estimator = CostEstimator()
    path: Path | None
    if calibration_path is not None:
        path = Path(calibration_path)
    else:
        path = _latest_calibration_path()
    if path is not None and path.exists():
        coeff = load_coefficients(path)
        apply_calibration(estimator, coeff)
    else:
        path = None
    return estimator, path


def apply_partitioning_style(
    *, context: str = "talk", font_scale: float = 0.9, palette: str = "colorblind"
) -> None:
    """Apply a shared Seaborn/Matplotlib style for partitioning plots."""

    try:
        import seaborn as sns

        sns.set_theme(context=context, style="whitegrid", palette=palette, font_scale=font_scale)
    except ModuleNotFoundError:  # pragma: no cover - optional styling dependency
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "axes.prop_cycle": plt.cycler(color=plt.cm.tab10.colors),
                "axes.titlesize": "medium",
                "axes.labelsize": "medium",
            }
        )


def export_figure(
    fig: Figure,
    name: str,
    *,
    directory: str | Path | None = None,
    formats: Sequence[str] = ("svg",),
    dpi: int = 300,
) -> list[Path]:
    """Persist ``fig`` under ``benchmarks/figures/partitioning`` for reuse."""

    # Only vector images are exported by default so documentation commits avoid
    # binary payloads.  Additional formats (e.g., ``png``) can be requested by
    # passing them explicitly when running the notebook locally.

    target_dir = Path(directory) if directory is not None else _FIGURE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        outfile = target_dir / f"{name}.{fmt}"
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        saved.append(outfile)
    return saved


@dataclass(slots=True)
class FragmentStats:
    """Summary statistics describing a synthetic circuit fragment."""

    num_qubits: int
    num_1q_gates: int
    num_2q_gates: int
    num_measurements: int = 0
    is_clifford: bool = False
    is_local: bool = False
    frontier: int | None = None
    chi: int | Sequence[int] | None = None

    @property
    def total_gates(self) -> int:
        """Return the total number of operations in the fragment."""

        return self.num_1q_gates + self.num_2q_gates + self.num_measurements


@dataclass(slots=True)
class BoundarySpec:
    """Parameters describing a conversion boundary between fragments."""

    num_qubits: int
    rank: int
    frontier: int
    window: int | None = None
    window_1q_gates: int = 0
    window_2q_gates: int = 0
    s_max: int | None = None
    r_max: int | None = None
    q_max: int | None = None


@dataclass(frozen=True)
class SyntheticPlanScenario:
    """Pre-baked synthetic partitioning scenario used by documentation."""

    key: str
    title: str
    description: str
    fragments: Sequence[Mapping[str, object]]
    boundaries: Sequence[Mapping[str, object]]
    total_qubits: int
    entanglement_entropy: float
    rotation_diversity: float
    sparsity: float

    def evaluate(self, estimator: CostEstimator) -> Mapping[str, object]:
        """Evaluate the plan using :func:`build_partition_plan`."""

        return build_partition_plan(
            estimator,
            self.fragments,
            self.boundaries,
            total_qubits=self.total_qubits,
            entanglement_entropy=self.entanglement_entropy,
            rotation_diversity=self.rotation_diversity,
            sparsity=self.sparsity,
        )


def build_clifford_fragment_curves(
    estimator: CostEstimator,
    *,
    num_qubits: Iterable[int] = range(1, 15),
    gate_density: int = 10,
) -> Mapping[str, np.ndarray | int | None]:
    """Return statevector/tableau runtimes for Clifford-only fragments."""

    n_values = np.asarray(list(num_qubits), dtype=int)
    sv_times: list[float] = []
    tab_times: list[float] = []
    for n in n_values:
        total_gates = gate_density * n
        sv_times.append(estimator.statevector(n, total_gates, 0, 0).time)
        tab_times.append(estimator.tableau(n, total_gates).time)
    sv_arr = np.asarray(sv_times, dtype=float)
    tab_arr = np.asarray(tab_times, dtype=float)
    cheaper = np.where(tab_arr < sv_arr)[0]
    threshold = int(n_values[cheaper[0]]) if cheaper.size else None
    return {
        "num_qubits": n_values,
        "statevector": sv_arr,
        "tableau": tab_arr,
        "threshold": threshold,
    }


def build_statevector_partition_tradeoff(
    estimator: CostEstimator,
    *,
    num_qubits: Iterable[int] = range(4, 33),
    prefix_density_1q: int = 2,
    prefix_density_2q: int = 2,
    clifford_density_1q: int = 6,
    clifford_density_2q: int = 6,
    boundary_cap: int = 6,
    rank_cap: int = 32,
) -> Mapping[str, np.ndarray | int | None]:
    """Compare staying on statevector versus switching to tableau mid-circuit."""

    n_values = np.asarray(list(num_qubits), dtype=int)
    stay_times: list[float] = []
    partition_times: list[float] = []
    boundaries: list[int] = []
    ranks: list[int] = []
    for n in n_values:
        boundary = min(n, boundary_cap)
        rank = min(2**boundary, rank_cap)
        boundaries.append(boundary)
        ranks.append(rank)
        prefix_1q = prefix_density_1q * n
        prefix_2q = prefix_density_2q * max(n - 1, 0)
        clifford_1q = clifford_density_1q * n
        clifford_2q = clifford_density_2q * max(n - 1, 0)
        stay = estimator.statevector(
            n,
            prefix_1q + clifford_1q,
            prefix_2q + clifford_2q,
            0,
        ).time
        prefix_cost = estimator.statevector(n, prefix_1q, prefix_2q, 0).time
        tableau_cost = estimator.tableau(n, clifford_1q + clifford_2q).time
        sv_to_tab = estimator.conversion(
            Backend.STATEVECTOR,
            Backend.TABLEAU,
            num_qubits=boundary,
            rank=rank,
            frontier=boundary,
        ).cost.time
        tab_to_sv = estimator.conversion(
            Backend.TABLEAU,
            Backend.STATEVECTOR,
            num_qubits=boundary,
            rank=rank,
            frontier=boundary,
        ).cost.time
        stay_times.append(stay)
        partition_times.append(prefix_cost + sv_to_tab + tableau_cost + tab_to_sv)
    stay_arr = np.asarray(stay_times, dtype=float)
    partition_arr = np.asarray(partition_times, dtype=float)
    cheaper = np.where(partition_arr < stay_arr)[0]
    threshold = int(n_values[cheaper[0]]) if cheaper.size else None
    return {
        "num_qubits": n_values,
        "statevector": stay_arr,
        "partitioned": partition_arr,
        "boundary": np.asarray(boundaries, dtype=int),
        "rank": np.asarray(ranks, dtype=int),
        "threshold": threshold,
    }


def build_conversion_primitive_examples(
    estimator: CostEstimator,
    *,
    source: Backend = Backend.STATEVECTOR,
    target: Backend = Backend.MPS,
    boundaries: Sequence[int] = (2, 4, 6),
    rank_override: Mapping[int, int] | None = None,
    frontier_fn: Callable[[int], int] | None = None,
    window: int | None = None,
    window_1q_gates: int = 0,
    window_2q_gates: int = 0,
) -> list[dict[str, float | int | str | bool | None]]:
    """Return per-primitive costs for small conversion scenarios."""

    results: list[dict[str, float | int | str | bool | None]] = []
    for boundary in boundaries:
        rank = (
            rank_override[boundary]
            if rank_override is not None and boundary in rank_override
            else min(2**boundary, 64)
        )
        frontier = frontier_fn(boundary) if frontier_fn is not None else max(boundary, 1)
        details = estimator.conversion_candidates(
            source,
            target,
            num_qubits=boundary,
            rank=rank,
            frontier=frontier,
            window=window,
            window_1q_gates=window_1q_gates,
            window_2q_gates=window_2q_gates,
        )
        best = min(details.items(), key=lambda kv: kv[1].cost.time)[0]
        for primitive, info in details.items():
            results.append(
                {
                    "boundary": boundary,
                    "rank": rank,
                    "frontier": frontier,
                    "primitive": primitive,
                    "time": info.cost.time,
                    "memory": info.cost.memory,
                    "window": info.window,
                    "selected": primitive == best,
                }
            )
    return results


def build_statevector_vs_mps(
    estimator: CostEstimator,
    *,
    num_qubits: Iterable[int] = range(2, 15),
    gate_density_1q: int = 5,
    gate_density_2q: int = 5,
    chi: int = 4,
) -> Mapping[str, np.ndarray | int | None]:
    """Return runtimes for dense versus local MPS simulation."""

    n_values = np.asarray(list(num_qubits), dtype=int)
    sv_times: list[float] = []
    mps_times: list[float] = []
    for n in n_values:
        num_1q = gate_density_1q * n
        num_2q = gate_density_2q * max(n - 1, 0)
        sv_times.append(estimator.statevector(n, num_1q, num_2q, 0).time)
        mps_times.append(estimator.mps(n, num_1q, num_2q, chi=chi, svd=True).time)
    sv_arr = np.asarray(sv_times, dtype=float)
    mps_arr = np.asarray(mps_times, dtype=float)
    cheaper = np.where(mps_arr < sv_arr)[0]
    threshold = int(n_values[cheaper[0]]) if cheaper.size else None
    return {
        "num_qubits": n_values,
        "statevector": sv_arr,
        "mps": mps_arr,
        "threshold": threshold,
    }


def build_conversion_aware_mps_paths(
    estimator: CostEstimator,
    *,
    num_qubits: Iterable[int] = range(6, 26),
    boundary: int = 4,
    gate_density_1q: int = 5,
    gate_density_2q: int = 5,
    scenarios: Sequence[Mapping[str, int]] | None = None,
) -> Mapping[str, object]:
    """Return statevector baseline and conversion-aware MPS totals."""

    if scenarios is None:
        scenarios = (
            {"label": r"$\\chi=4$, window=6", "chi": 4, "window": 6},
            {"label": r"$\\chi=6$, window=10", "chi": 6, "window": 10},
        )
    n_values = np.asarray(list(num_qubits), dtype=int)
    sv_times: list[float] = []
    for n in n_values:
        num_1q = gate_density_1q * n
        num_2q = gate_density_2q * max(n - 1, 0)
        sv_times.append(estimator.statevector(n, num_1q, num_2q, 0).time)
    sv_arr = np.asarray(sv_times, dtype=float)
    scenario_results: list[dict[str, object]] = []
    for spec in scenarios:
        chi = int(spec["chi"])
        window = int(spec["window"])
        label = str(spec.get("label", f"chi={chi}"))
        rank = int(spec.get("rank", chi**2))
        window_1q = gate_density_1q * window
        window_2q = gate_density_2q * max(window - 1, 0)
        sv_to_mps = estimator.conversion(
            Backend.STATEVECTOR,
            Backend.MPS,
            num_qubits=boundary,
            rank=rank,
            frontier=0,
            window=window,
            window_1q_gates=window_1q,
            window_2q_gates=window_2q,
        ).cost.time
        mps_to_sv = estimator.conversion(
            Backend.MPS,
            Backend.STATEVECTOR,
            num_qubits=boundary,
            rank=rank,
            frontier=0,
            window=window,
            window_1q_gates=window_1q,
            window_2q_gates=window_2q,
        ).cost.time
        totals: list[float] = []
        for n in n_values:
            num_1q = gate_density_1q * n
            num_2q = gate_density_2q * max(n - 1, 0)
            mps_time = estimator.mps(n, num_1q, num_2q, chi=chi, svd=True).time
            totals.append(sv_to_mps + mps_time + mps_to_sv)
        totals_arr = np.asarray(totals, dtype=float)
        cheaper = np.where(totals_arr < sv_arr)[0]
        threshold = int(n_values[cheaper[0]]) if cheaper.size else None
        scenario_results.append(
            {
                "label": label,
                "chi": chi,
                "window": window,
                "rank": rank,
                "sv_to_mps": sv_to_mps,
                "mps_to_sv": mps_to_sv,
                "total": totals_arr,
                "threshold": threshold,
            }
        )
    return {
        "num_qubits": n_values,
        "statevector": sv_arr,
        "scenarios": scenario_results,
    }


def build_statevector_vs_decision_diagram(
    estimator: CostEstimator,
    *,
    num_qubits: Iterable[int] = range(1, 20),
    gate_density: int = 10,
) -> Mapping[str, np.ndarray | int | None]:
    """Return runtimes for dense versus decision diagram simulation."""

    n_values = np.asarray(list(num_qubits), dtype=int)
    sv_times: list[float] = []
    dd_times: list[float] = []
    for n in n_values:
        gates = gate_density * n
        sv_times.append(estimator.statevector(n, gates, 0, 0).time)
        dd_times.append(
            estimator.decision_diagram(
                num_gates=gates,
                frontier=n,
                sparsity=0.7,
                entanglement_entropy=max(0.0, math.log2(n) if n > 1 else 0.0),
            ).time
        )
    return {
        "num_qubits": n_values,
        "statevector": np.asarray(sv_times, dtype=float),
        "decision_diagram": np.asarray(dd_times, dtype=float),
    }


def build_conversion_primitive_costs(
    estimator: CostEstimator,
    *,
    num_qubits: Iterable[int] = range(1, 10),
    source: Backend = Backend.STATEVECTOR,
    target: Backend = Backend.DECISION_DIAGRAM,
) -> list[dict[str, object]]:
    """Return the cheapest primitive for a range of boundary sizes."""

    entries: list[dict[str, object]] = []
    for q in num_qubits:
        estimate = estimator.conversion(source, target, num_qubits=q, rank=2**q, frontier=q)
        entries.append(
            {
                "boundary": int(q),
                "primitive": estimate.primitive,
                "time": estimate.cost.time,
                "memory": estimate.cost.memory,
            }
        )
    return entries


def documentation_plan_scenarios() -> Sequence[SyntheticPlanScenario]:
    """Return preconfigured scenarios used by partitioning documentation."""

    balanced = SyntheticPlanScenario(
        key="balanced_handoff",
        title="Balanced hand-off across fragments",
        description=(
            "Conversion windows span 15 and 13 qubits, making the tableau→statevector"
            " and statevector→decision-diagram transfers visible alongside the"
            " fragment runtimes without dominating the statevector core."
        ),
        fragments=(
            {
                "name": "Clifford prefix",
                "backend": Backend.TABLEAU,
                "num_qubits": 16,
                "num_1q": 72,
                "num_2q": 36,
                "depth": 18,
            },
            {
                "name": "Non-Clifford core",
                "backend": Backend.STATEVECTOR,
                "num_qubits": 16,
                "num_1q": 18,
                "num_2q": 12,
                "entanglement_entropy": 7.4,
                "rotation_diversity": 0.48,
                "sparsity": 0.38,
            },
            {
                "name": "Sparse suffix",
                "backend": Backend.DECISION_DIAGRAM,
                "num_qubits": 16,
                "num_1q": 40,
                "num_2q": 24,
                "frontier": 7,
                "entanglement_entropy": 4.2,
                "sparsity": 0.82,
                "phase_rotation": 0.08,
                "amplitude_rotation": 0.08,
            },
        ),
        boundaries=(
            {
                "name": "Prefix → core",
                "source": Backend.TABLEAU,
                "target": Backend.STATEVECTOR,
                "num_qubits": 15,
                "rank": 320,
                "frontier": 16,
                "window": 13,
                "window_1q": 32,
                "window_2q": 24,
            },
            {
                "name": "Core → suffix",
                "source": Backend.STATEVECTOR,
                "target": Backend.DECISION_DIAGRAM,
                "num_qubits": 13,
                "rank": 208,
                "frontier": 11,
                "window": 11,
                "window_1q": 26,
                "window_2q": 18,
            },
        ),
        total_qubits=16,
        entanglement_entropy=7.4,
        rotation_diversity=0.48,
        sparsity=0.38,
    )

    conversion_dominated = SyntheticPlanScenario(
        key="conversion_dominated",
        title="Conversion-dominated hand-off",
        description=(
            "Expanded 16- and 14-qubit boundaries trigger the expensive ST"
            " primitive so conversion time rivals or exceeds the fragment runtime."
        ),
        fragments=(
            {
                "name": "Clifford prefix",
                "backend": Backend.TABLEAU,
                "num_qubits": 16,
                "num_1q": 60,
                "num_2q": 28,
                "depth": 16,
            },
            {
                "name": "Non-Clifford core",
                "backend": Backend.STATEVECTOR,
                "num_qubits": 16,
                "num_1q": 18,
                "num_2q": 12,
                "entanglement_entropy": 7.0,
                "rotation_diversity": 0.46,
                "sparsity": 0.4,
            },
            {
                "name": "Sparse suffix",
                "backend": Backend.DECISION_DIAGRAM,
                "num_qubits": 16,
                "num_1q": 32,
                "num_2q": 20,
                "frontier": 6,
                "entanglement_entropy": 3.8,
                "sparsity": 0.84,
                "phase_rotation": 0.06,
                "amplitude_rotation": 0.06,
            },
        ),
        boundaries=(
            {
                "name": "Prefix → core",
                "source": Backend.TABLEAU,
                "target": Backend.STATEVECTOR,
                "num_qubits": 16,
                "rank": 400,
                "frontier": 16,
                "window": 14,
                "window_1q": 34,
                "window_2q": 26,
            },
            {
                "name": "Core → suffix",
                "source": Backend.STATEVECTOR,
                "target": Backend.DECISION_DIAGRAM,
                "num_qubits": 14,
                "rank": 224,
                "frontier": 10,
                "window": 10,
                "window_1q": 22,
                "window_2q": 16,
            },
        ),
        total_qubits=16,
        entanglement_entropy=7.0,
        rotation_diversity=0.46,
        sparsity=0.4,
    )

    return [balanced, conversion_dominated]


def build_partition_plan(
    estimator: CostEstimator,
    fragments: Sequence[Mapping[str, object]],
    boundaries: Sequence[Mapping[str, object]],
    *,
    total_qubits: int,
    entanglement_entropy: float,
    rotation_diversity: float,
    sparsity: float,
) -> Mapping[str, object]:
    """Evaluate a synthetic plan comparing single-backend and partitioned runs."""

    total_1q = sum(int(frag.get("num_1q", 0)) for frag in fragments)
    total_2q = sum(int(frag.get("num_2q", 0)) for frag in fragments)
    single_backend = estimator.statevector(
        num_qubits=total_qubits,
        num_1q_gates=total_1q,
        num_2q_gates=total_2q,
        num_meas=0,
        entanglement_entropy=entanglement_entropy,
        rotation_diversity=rotation_diversity,
        sparsity=sparsity,
    )

    fragment_costs: list[tuple[Backend, Cost]] = []
    fragment_results: list[dict[str, object]] = []
    for frag in fragments:
        backend = frag["backend"]
        if backend == Backend.TABLEAU:
            total_gates = int(frag.get("num_1q", 0)) + int(frag.get("num_2q", 0))
            two_qubit_ratio = (
                int(frag.get("num_2q", 0)) / total_gates if total_gates else 0.0
            )
            cost = estimator.tableau(
                num_qubits=int(frag.get("num_qubits", total_qubits)),
                num_gates=total_gates,
                two_qubit_ratio=two_qubit_ratio,
                depth=int(frag.get("depth", max(total_gates // max(total_qubits, 1), 1))),
                rotation_diversity=float(frag.get("rotation_diversity", 0.2)),
            )
        elif backend == Backend.STATEVECTOR:
            cost = estimator.statevector(
                num_qubits=int(frag.get("num_qubits", total_qubits)),
                num_1q_gates=int(frag.get("num_1q", 0)),
                num_2q_gates=int(frag.get("num_2q", 0)),
                num_meas=int(frag.get("num_meas", 0)),
                entanglement_entropy=float(frag.get("entanglement_entropy", entanglement_entropy)),
                rotation_diversity=float(frag.get("rotation_diversity", rotation_diversity)),
                sparsity=float(frag.get("sparsity", sparsity)),
            )
        elif backend == Backend.DECISION_DIAGRAM:
            total_gates = int(frag.get("num_1q", 0)) + int(frag.get("num_2q", 0))
            two_qubit_ratio = (
                int(frag.get("num_2q", 0)) / total_gates if total_gates else 0.0
            )
            cost = estimator.decision_diagram(
                num_gates=total_gates,
                frontier=int(frag.get("frontier", total_qubits)),
                sparsity=float(frag.get("sparsity", sparsity)),
                entanglement_entropy=float(frag.get("entanglement_entropy", entanglement_entropy)),
                two_qubit_ratio=two_qubit_ratio,
                phase_rotation_diversity=float(frag.get("phase_rotation", 0.0)),
                amplitude_rotation_diversity=float(frag.get("amplitude_rotation", 0.0)),
            )
        else:  # pragma: no cover - defensive guard for new backends
            raise ValueError(f"unsupported backend: {backend}")
        fragment_costs.append((backend, cost))
        frag_entry = dict(frag)
        frag_entry["cost"] = cost
        fragment_results.append(frag_entry)

    boundary_specs = [
        BoundarySpec(
            num_qubits=int(spec["num_qubits"]),
            rank=int(spec["rank"]),
            frontier=int(spec["frontier"]),
            window=int(spec.get("window")) if spec.get("window") is not None else None,
            window_1q_gates=int(spec.get("window_1q", 0)),
            window_2q_gates=int(spec.get("window_2q", 0)),
            s_max=spec.get("s_max"),
            r_max=spec.get("r_max"),
            q_max=spec.get("q_max"),
        )
        for spec in boundaries
    ]

    aggregate = aggregate_partitioned_plan(fragment_costs, boundary_specs, estimator=estimator)

    conversion_results: list[dict[str, object]] = []
    for conv in aggregate["conversions"]:
        spec = dict(boundaries[conv["index"]])
        spec.update(
            {
                "source": conv["source"],
                "target": conv["target"],
                "primitive": conv["primitive"],
                "cost": conv["cost"],
            }
        )
        conversion_results.append(spec)

    return {
        "single_backend": single_backend,
        "fragments": fragment_results,
        "conversions": conversion_results,
        "aggregate": aggregate,
    }


def _update_peak_memory(current: float, candidates: Iterable[float]) -> float:
    """Return the maximum memory footprint seen across ``candidates``."""

    for value in candidates:
        if value > current:
            current = value
    return current


def evaluate_fragment_backends(
    stats: FragmentStats,
    *,
    sparsity: float | None = None,
    phase_rotation_diversity: int | None = None,
    amplitude_rotation_diversity: int | None = None,
    allow_tableau: bool = True,
    max_memory: float | None = None,
    max_time: float | None = None,
    target_accuracy: float | None = None,
    estimator: CostEstimator | None = None,
    selection_metric: str | Callable[[Backend, Cost], float] = "weighted",
    metric_weights: tuple[float, float] = (0.5, 0.5),
) -> tuple[Backend | None, MutableMapping[str, object]]:
    """Evaluate backend feasibility for a synthetic fragment.

    Parameters
    ----------
    stats:
        Summary describing the fragment under consideration.
    sparsity, phase_rotation_diversity, amplitude_rotation_diversity:
        Circuit-level metrics used by the decision diagram heuristic.
    allow_tableau:
        Permit stabiliser simulation when the fragment is Clifford only.
    max_memory, max_time:
        Optional resource limits applied to each backend estimate.
    target_accuracy:
        Desired lower bound on simulation fidelity.  When supplied together
        with ``stats.chi`` the value is surfaced in the diagnostics to mirror
        :class:`~quasar.method_selector.MethodSelector`.
    estimator:
        Optional estimator instance.  A new :class:`CostEstimator` is created
        when omitted.
    selection_metric:
        Strategy used to compare feasible backends.  ``"weighted"`` (default)
        minimises a weighted sum of runtime and memory.  ``"pareto"`` keeps all
        non-dominated backends and falls back to the weighted score.  A callable
        may be supplied for custom scoring.
    metric_weights:
        Pair of ``(time_weight, memory_weight)`` used by the weighted metric.

    Returns
    -------
    tuple
        Selected backend (or ``None`` when all candidates are infeasible) and
        a diagnostics mapping mirroring the structure produced by
        :class:`~quasar.method_selector.MethodSelector`.
    """

    estimator = estimator or CostEstimator()

    total_gates = stats.total_gates
    num_qubits = stats.num_qubits
    two_qubit_ratio = stats.num_2q_gates / total_gates if total_gates else 0.0
    rotation_density = (
        (phase_rotation_diversity or 0) + (amplitude_rotation_diversity or 0)
    ) / max(total_gates, 1)
    depth_est = max(1, math.ceil(total_gates / num_qubits)) if num_qubits else 0
    entanglement = min(
        num_qubits,
        (stats.num_2q_gates / max(num_qubits, 1))
        * math.log2(num_qubits + 1)
        * math.log2(depth_est + 1),
    )

    diag: MutableMapping[str, object] = {
        "metrics": {
            "num_qubits": num_qubits,
            "num_gates": total_gates,
            "sparsity": sparsity if sparsity is not None else 0.0,
            "phase_rotation_diversity": phase_rotation_diversity or 0,
            "amplitude_rotation_diversity": amplitude_rotation_diversity or 0,
            "local": stats.is_local,
            "two_qubit_ratio": two_qubit_ratio,
            "rotation_density": rotation_density,
            "depth_estimate": depth_est,
            "entanglement_entropy": entanglement,
        },
        "backends": {},
    }

    candidates: dict[Backend, Cost] = {}
    frontier = stats.frontier or stats.num_qubits

    # ------------------------------------------------------------------
    # Tableau backend
    # ------------------------------------------------------------------
    if allow_tableau and stats.is_clifford and total_gates:
        table_cost = estimator.tableau(
            num_qubits,
            total_gates,
            two_qubit_ratio=two_qubit_ratio,
            depth=depth_est,
            rotation_diversity=rotation_density,
        )
        feasible = True
        reasons: list[str] = []
        if max_memory is not None and table_cost.memory > max_memory:
            feasible = False
            reasons.append("memory > threshold")
        if max_time is not None and table_cost.time > max_time:
            feasible = False
            reasons.append("time > threshold")
        diag["backends"][Backend.TABLEAU] = {
            "feasible": feasible,
            "reasons": reasons,
            "cost": table_cost,
        }
        if feasible:
            candidates[Backend.TABLEAU] = table_cost
    else:
        reason = "tableau disabled"
        if not allow_tableau:
            reason = "tableau disabled"
        elif not stats.is_clifford:
            reason = "non-clifford fragment"
        elif not total_gates:
            reason = "no gates"
        diag["backends"][Backend.TABLEAU] = {
            "feasible": False,
            "reasons": [reason],
        }

    # ------------------------------------------------------------------
    # Decision diagram backend
    # ------------------------------------------------------------------
    sparse = sparsity if sparsity is not None else 0.0
    phase_rot = phase_rotation_diversity or 0
    amp_rot = amplitude_rotation_diversity or 0
    nnz = int((1 - sparse) * (2**num_qubits))
    s_thresh = config.DEFAULT.dd_sparsity_threshold
    amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(num_qubits, sparsity)

    passes = (
        sparse >= s_thresh
        and nnz <= config.DEFAULT.dd_nnz_threshold
        and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
        and amp_rot <= amp_thresh
    )

    dd_metric = False
    metric_value: float | None = None
    if passes:
        s_score = sparse / s_thresh if s_thresh else 0.0
        nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
        phase_score = 1 - (
            phase_rot / config.DEFAULT.dd_phase_rotation_diversity_threshold
            if config.DEFAULT.dd_phase_rotation_diversity_threshold
            else 0.0
        )
        amp_score = 1 - (amp_rot / amp_thresh if amp_thresh else 0.0)
        weight_sum = (
            config.DEFAULT.dd_sparsity_weight
            + config.DEFAULT.dd_nnz_weight
            + config.DEFAULT.dd_phase_rotation_weight
            + config.DEFAULT.dd_amplitude_rotation_weight
        )
        metric_value = (
            config.DEFAULT.dd_sparsity_weight * s_score
            + config.DEFAULT.dd_nnz_weight * nnz_score
            + config.DEFAULT.dd_phase_rotation_weight * phase_score
            + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
        )
        metric_value = metric_value / weight_sum if weight_sum else 0.0
        dd_metric = metric_value >= config.DEFAULT.dd_metric_threshold
    else:
        reasons = []
        if sparse < s_thresh:
            reasons.append("sparsity below threshold")
        if nnz > config.DEFAULT.dd_nnz_threshold:
            reasons.append("nnz above threshold")
        if phase_rot > config.DEFAULT.dd_phase_rotation_diversity_threshold:
            reasons.append("phase diversity above threshold")
        if amp_rot > amp_thresh:
            reasons.append("amplitude diversity above threshold")
        diag["backends"][Backend.DECISION_DIAGRAM] = {
            "feasible": False,
            "reasons": reasons,
        }

    if dd_metric:
        dd_cost = estimator.decision_diagram(
            num_gates=total_gates,
            frontier=frontier,
            sparsity=sparse,
            phase_rotation_diversity=phase_rotation_diversity,
            amplitude_rotation_diversity=amplitude_rotation_diversity,
            entanglement_entropy=entanglement,
            two_qubit_ratio=two_qubit_ratio,
        )
        feasible = True
        reasons: list[str] = []
        if max_memory is not None and dd_cost.memory > max_memory:
            feasible = False
            reasons.append("memory > threshold")
        if max_time is not None and dd_cost.time > max_time:
            feasible = False
            reasons.append("time > threshold")
        entry: MutableMapping[str, object] = {
            "feasible": feasible,
            "reasons": reasons,
            "cost": dd_cost,
            "metric": metric_value,
            "dd_metric_threshold": config.DEFAULT.dd_metric_threshold,
        }
        diag["backends"][Backend.DECISION_DIAGRAM] = entry
        if feasible:
            candidates[Backend.DECISION_DIAGRAM] = dd_cost

    # ------------------------------------------------------------------
    # Matrix product state backend
    # ------------------------------------------------------------------
    if stats.is_local and num_qubits:
        chosen_chi = stats.chi
        if chosen_chi is None:
            chosen_chi = estimator.chi_max or 4
        chi_limit: int | None = None
        infeasible_chi = False
        if max_memory is not None:
            chi_limit = estimator.chi_from_memory(num_qubits, max_memory)
            if chi_limit <= 0:
                infeasible_chi = True
            else:
                max_chi = (
                    max(chosen_chi)
                    if isinstance(chosen_chi, Sequence) and not isinstance(chosen_chi, (str, bytes))
                    else int(chosen_chi)
                )
                if max_chi > chi_limit:
                    infeasible_chi = True
        mps_cost = estimator.mps(
            num_qubits,
            stats.num_1q_gates + stats.num_measurements,
            stats.num_2q_gates,
            chi=chosen_chi,
            svd=True,
            entanglement_entropy=entanglement,
            sparsity=sparse,
            rotation_diversity=rotation_density,
        )
        feasible = not infeasible_chi
        reasons: list[str] = []
        if infeasible_chi:
            reasons.append("bond dimension exceeds memory limit")
        if max_memory is not None and mps_cost.memory > max_memory:
            feasible = False
            reasons.append("memory > threshold")
        if max_time is not None and mps_cost.time > max_time:
            feasible = False
            reasons.append("time > threshold")
        entry = {
            "feasible": feasible,
            "reasons": reasons,
            "cost": mps_cost,
            "chi": chosen_chi,
        }
        if chi_limit is not None:
            entry["chi_limit"] = chi_limit
        if target_accuracy is not None:
            entry["target_accuracy"] = target_accuracy
        diag["backends"][Backend.MPS] = entry
        if feasible:
            candidates[Backend.MPS] = mps_cost
    else:
        reason = "non-local gates" if stats.num_2q_gates else "no multi-qubit gates"
        diag["backends"][Backend.MPS] = {
            "feasible": False,
            "reasons": [reason],
        }

    # ------------------------------------------------------------------
    # Statevector backend
    # ------------------------------------------------------------------
    sv_cost = estimator.statevector(
        num_qubits,
        stats.num_1q_gates,
        stats.num_2q_gates,
        stats.num_measurements,
        sparsity=sparse,
        two_qubit_ratio=two_qubit_ratio,
        entanglement_entropy=entanglement,
        rotation_diversity=rotation_density,
    )
    sv_feasible = True
    reasons: list[str] = []
    if max_memory is not None and sv_cost.memory > max_memory:
        sv_feasible = False
        reasons.append("memory > threshold")
    if max_time is not None and sv_cost.time > max_time:
        sv_feasible = False
        reasons.append("time > threshold")
    diag["backends"][Backend.STATEVECTOR] = {
        "feasible": sv_feasible,
        "reasons": reasons,
        "cost": sv_cost,
    }
    if sv_feasible:
        candidates[Backend.STATEVECTOR] = sv_cost

    if not candidates:
        diag["selected_backend"] = None
        diag["selected_cost"] = None
        return None, diag

    scores: dict[Backend, float] = {}

    def _weighted_score(cost: Cost) -> float:
        time_weight, mem_weight = metric_weights
        return time_weight * cost.time + mem_weight * cost.memory

    if callable(selection_metric):
        for backend, cost in candidates.items():
            scores[backend] = selection_metric(backend, cost)
        selected = min(scores, key=scores.__getitem__)
    else:
        metric_name = str(selection_metric).lower()
        if metric_name == "weighted":
            for backend, cost in candidates.items():
                scores[backend] = _weighted_score(cost)
            selected = min(scores, key=scores.__getitem__)
        elif metric_name == "pareto":
            pareto_front: list[Backend] = []
            for backend, cost in candidates.items():
                dominated = False
                for other_backend, other_cost in candidates.items():
                    if backend is other_backend:
                        continue
                    better_or_equal = (
                        other_cost.memory <= cost.memory
                        and other_cost.time <= cost.time
                    )
                    strictly_better = (
                        other_cost.memory < cost.memory
                        or other_cost.time < cost.time
                    )
                    if better_or_equal and strictly_better:
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append(backend)
            if not pareto_front:
                pareto_front = list(candidates)
            for backend in pareto_front:
                scores[backend] = _weighted_score(candidates[backend])
            selected = min(pareto_front, key=lambda b: scores[b])
            for backend, entry in diag["backends"].items():
                if isinstance(entry, MutableMapping):
                    entry["pareto_optimal"] = backend in pareto_front
        else:
            raise ValueError(f"unknown selection metric: {selection_metric}")

    diag["selected_backend"] = selected
    diag["selected_cost"] = candidates[selected]
    for backend, entry in diag["backends"].items():
        if isinstance(entry, Mapping):
            entry["selected"] = backend == selected
            if backend in scores:
                entry["score"] = scores[backend]
    return selected, diag


def estimate_conversion(
    source: Backend,
    target: Backend,
    boundary: BoundarySpec,
    *,
    estimator: CostEstimator | None = None,
) -> ConversionEstimate:
    """Return the cheapest conversion primitive for the provided boundary."""

    estimator = estimator or CostEstimator()
    return estimator.conversion(
        source,
        target,
        boundary.num_qubits,
        boundary.rank,
        boundary.frontier,
        boundary.window,
        window_1q_gates=boundary.window_1q_gates,
        window_2q_gates=boundary.window_2q_gates,
        s_max=boundary.s_max,
        r_max=boundary.r_max,
        q_max=boundary.q_max,
    )


def aggregate_single_backend_plan(
    fragments: Sequence[tuple[Backend, Cost]]
) -> Cost:
    """Aggregate costs for a plan that uses a single backend."""

    total_time = sum(cost.time for _, cost in fragments)
    peak_memory = _update_peak_memory(0.0, (cost.memory for _, cost in fragments))
    log_depth = max((cost.log_depth for _, cost in fragments), default=0.0)
    conversion_time = sum(cost.conversion for _, cost in fragments)
    return Cost(
        time=total_time,
        memory=peak_memory,
        log_depth=log_depth,
        conversion=conversion_time,
    )


def aggregate_partitioned_plan(
    fragments: Sequence[tuple[Backend, Cost]],
    boundaries: Sequence[BoundarySpec],
    *,
    estimator: CostEstimator | None = None,
) -> MutableMapping[str, object]:
    """Aggregate costs for a heterogeneous plan with conversions.

    ``boundaries`` must contain ``len(fragments) - 1`` entries describing the
    interfaces between consecutive fragments.  Conversions are skipped when the
    neighbouring fragments already use the same backend.
    """

    if boundaries and len(boundaries) != max(len(fragments) - 1, 0):
        raise ValueError("number of boundaries must match fragment transitions")

    estimator = estimator or CostEstimator()
    total_time = 0.0
    peak_memory = 0.0
    log_depth = 0.0
    conversion_time = 0.0
    conversions: list[dict[str, object]] = []

    for backend, cost in fragments:
        total_time += cost.time
        peak_memory = _update_peak_memory(peak_memory, [cost.memory])
        log_depth = max(log_depth, cost.log_depth)
        conversion_time += cost.conversion

    for idx in range(len(fragments) - 1):
        src_backend, _ = fragments[idx]
        dst_backend, _ = fragments[idx + 1]
        if src_backend == dst_backend:
            continue
        spec = boundaries[idx]
        estimate = estimate_conversion(src_backend, dst_backend, spec, estimator=estimator)
        conversions.append(
            {
                "index": idx,
                "source": src_backend,
                "target": dst_backend,
                "primitive": estimate.primitive,
                "cost": estimate.cost,
            }
        )
        total_time += estimate.cost.time
        conversion_time += estimate.cost.time
        peak_memory = _update_peak_memory(peak_memory, [estimate.cost.memory])
        log_depth = max(log_depth, estimate.cost.log_depth)

    total_cost = Cost(
        time=total_time,
        memory=peak_memory,
        log_depth=log_depth,
        conversion=conversion_time,
    )
    return {
        "total_cost": total_cost,
        "fragments": list(fragments),
        "conversions": conversions,
    }


__all__ = [
    "BoundarySpec",
    "FragmentStats",
    "aggregate_partitioned_plan",
    "aggregate_single_backend_plan",
    "apply_partitioning_style",
    "build_clifford_fragment_curves",
    "build_conversion_aware_mps_paths",
    "build_conversion_primitive_costs",
    "build_conversion_primitive_examples",
    "build_partition_plan",
    "build_statevector_partition_tradeoff",
    "build_statevector_vs_mps",
    "build_statevector_vs_decision_diagram",
    "documentation_plan_scenarios",
    "evaluate_fragment_backends",
    "estimate_conversion",
    "export_figure",
    "load_calibrated_estimator",
    "SyntheticPlanScenario",
]
