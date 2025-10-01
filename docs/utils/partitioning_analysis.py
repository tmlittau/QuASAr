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
from quasar.method_selector import NoFeasibleBackendError, _soft_penalty
from quasar.sparsity import adaptive_dd_sparsity_threshold


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
        record = load_coefficients(path)
        apply_calibration(estimator, record)
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

    @classmethod
    def from_export(cls, metrics: Mapping[str, object]) -> "FragmentStats":
        """Construct :class:`FragmentStats` from exported selector metrics."""

        frontier = metrics.get("frontier")
        return cls(
            num_qubits=int(metrics.get("num_qubits", 0)),
            num_1q_gates=int(metrics.get("num_1q", metrics.get("num_1q_gates", 0))),
            num_2q_gates=int(metrics.get("num_2q", metrics.get("num_2q_gates", 0))),
            num_measurements=int(metrics.get("num_meas", metrics.get("num_measurements", 0))),
            is_clifford=bool(metrics.get("is_clifford", metrics.get("clifford", False))),
            is_local=bool(metrics.get("local", False)),
            frontier=int(frontier) if frontier is not None else None,
            chi=metrics.get("chi"),
        )


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


def _prepare_selector_metrics(
    stats: FragmentStats,
    *,
    sparsity: float | None,
    phase_rotation_diversity: int | None,
    amplitude_rotation_diversity: int | None,
    selector_metrics: Mapping[str, object] | None,
) -> tuple[dict[str, object], int]:
    """Combine heuristic inputs with exported selector metrics."""

    total_gates = stats.total_gates
    base_metrics: dict[str, object] = {}
    if selector_metrics:
        for key, value in selector_metrics.items():
            if key == "fragments":
                continue
            base_metrics[key] = value
    fragments: list[dict[str, object]]
    if selector_metrics and selector_metrics.get("fragments"):
        fragments = [dict(entry) for entry in selector_metrics["fragments"]]  # type: ignore[index]
    else:
        rotation_total = (phase_rotation_diversity or 0) + (amplitude_rotation_diversity or 0)
        rotation_density = rotation_total / max(total_gates, 1) if total_gates else 0.0
        max_distance = 1 if stats.is_local or stats.num_2q_gates == 0 else max(stats.num_qubits - 1, 1)
        fragments = [
            {
                "qubits": list(range(stats.num_qubits)),
                "num_qubits": stats.num_qubits,
                "num_gates": total_gates,
                "num_meas": stats.num_measurements,
                "num_1q": stats.num_1q_gates,
                "num_2q": stats.num_2q_gates,
                "num_t": 0,
                "sparsity": sparsity if sparsity is not None else 0.0,
                "phase_rotation_diversity": phase_rotation_diversity or 0,
                "amplitude_rotation_diversity": amplitude_rotation_diversity or 0,
                "rotation_density": rotation_density,
                "entanglement_entropy": 0.0,
                "long_range_fraction": 0.0 if stats.is_local else 1.0,
                "long_range_extent": 0.0 if stats.is_local else 1.0,
                "max_interaction_distance": max_distance,
                "local": stats.is_local,
                "scope": "fragment",
            }
        ]
    base_metrics["fragments"] = fragments

    def _get_float(name: str, fallback: float) -> float:
        value = base_metrics.get(name, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _get_int(name: str, fallback: int) -> int:
        value = base_metrics.get(name, fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    num_qubits = _get_int("num_qubits", stats.num_qubits)
    num_gates = _get_int("num_gates", total_gates)
    base_metrics["num_qubits"] = num_qubits
    base_metrics["num_gates"] = num_gates
    base_metrics.setdefault("num_meas", stats.num_measurements)
    base_metrics.setdefault("num_1q", stats.num_1q_gates)
    base_metrics.setdefault("num_2q", stats.num_2q_gates)
    base_metrics.setdefault("local", stats.is_local)

    frag = fragments[0] if fragments else {}
    sparse = sparsity if sparsity is not None else frag.get("sparsity")
    base_metrics["sparsity"] = _get_float("sparsity", float(sparse) if sparse is not None else 0.0)
    phase = phase_rotation_diversity if phase_rotation_diversity is not None else frag.get("phase_rotation_diversity")
    base_metrics["phase_rotation_diversity"] = _get_float(
        "phase_rotation_diversity",
        float(phase) if phase is not None else 0.0,
    )
    amp = (
        amplitude_rotation_diversity
        if amplitude_rotation_diversity is not None
        else frag.get("amplitude_rotation_diversity")
    )
    base_metrics["amplitude_rotation_diversity"] = _get_float(
        "amplitude_rotation_diversity",
        float(amp) if amp is not None else 0.0,
    )

    rotation_total = (
        base_metrics["phase_rotation_diversity"] + base_metrics["amplitude_rotation_diversity"]
    )
    rotation_density = rotation_total / max(num_gates, 1) if num_gates else frag.get("rotation_density", 0.0)
    base_metrics["rotation_density"] = _get_float("rotation_density", float(rotation_density))

    ent = base_metrics.get("entanglement_entropy")
    if ent is None:
        depth_est = max(1, math.ceil(num_gates / max(num_qubits, 1))) if num_qubits else 0
        ent = min(
            num_qubits,
            (base_metrics["num_2q"] / max(num_qubits, 1)) * math.log2(num_qubits + 1) * math.log2(depth_est + 1),
        )
    base_metrics["entanglement_entropy"] = float(ent)

    sparse_value = float(base_metrics["sparsity"])
    nnz = base_metrics.get("nnz")
    if nnz is None:
        nnz = max(1, int(round((1 - sparse_value) * (2**num_qubits))))
    base_metrics["nnz"] = int(nnz)

    base_metrics.setdefault("mps_long_range_fraction", frag.get("long_range_fraction", 0.0 if stats.is_local else 1.0))
    base_metrics.setdefault("mps_long_range_extent", frag.get("long_range_extent", 0.0 if stats.is_local else 1.0))
    base_metrics.setdefault(
        "mps_max_interaction_distance",
        frag.get(
            "max_interaction_distance",
            1 if stats.is_local or stats.num_2q_gates == 0 else max(stats.num_qubits - 1, 1),
        ),
    )

    hybrid_frontier = base_metrics.get("dd_hybrid_frontier")
    if hybrid_frontier is None:
        nnz_val = max(1, int(base_metrics["nnz"]))
        if num_qubits > 0 and nnz_val > 0:
            hybrid_frontier = min(num_qubits, int(math.ceil(math.log2(nnz_val))))
        else:
            hybrid_frontier = 0
        if num_gates and hybrid_frontier == 0:
            hybrid_frontier = 1
    base_metrics["dd_hybrid_frontier"] = int(hybrid_frontier)

    hybrid_penalty = base_metrics.get("dd_hybrid_penalty")
    if hybrid_penalty is None:
        hybrid_penalty = max(0, num_qubits - int(hybrid_frontier))
    base_metrics["dd_hybrid_penalty"] = int(hybrid_penalty)

    if "dd_hybrid_replay" in base_metrics:
        base_metrics["dd_hybrid_replay"] = int(base_metrics["dd_hybrid_replay"])
    else:
        replay = 0
        if num_qubits > 0 and num_gates > 0 and hybrid_penalty > 0:
            replay = int(round(num_gates * hybrid_penalty / max(num_qubits, 1)))
        base_metrics["dd_hybrid_replay"] = replay

    if "dd_size_override" not in base_metrics:
        base_metrics["dd_size_override"] = False
    if "effective_dd_sparsity" not in base_metrics:
        base_metrics["effective_dd_sparsity"] = sparse_value

    largest_subsystem = max((int(entry.get("num_qubits", 0)) for entry in fragments), default=0)
    return base_metrics, largest_subsystem


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
    selector_metrics: Mapping[str, object] | None = None,
) -> tuple[Backend | None, MutableMapping[str, object]]:
    """Evaluate backend feasibility for a synthetic fragment or replay diagnostics.

    Parameters
    ----------
    stats:
        Summary describing the fragment under consideration.
    sparsity, phase_rotation_diversity, amplitude_rotation_diversity:
        Circuit-level metrics used by the decision diagram heuristic. When
        ``selector_metrics`` is supplied these values serve as fallbacks for
        missing entries in the exported metrics.
    allow_tableau:
        Permit stabiliser simulation when the fragment is Clifford only.
    max_memory, max_time:
        Optional resource limits applied to each backend estimate.
    target_accuracy:
        Desired lower bound on simulation fidelity.
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
    selector_metrics:
        Optional mapping mirroring ``MethodSelector``'s diagnostics.  When
        provided the function replays the planner's decision using the recorded
        metrics instead of recomputing approximations.

    Returns
    -------
    tuple
        Selected backend (or ``None`` when all candidates are infeasible) and
        a diagnostics mapping mirroring the structure produced by
        :class:`~quasar.method_selector.MethodSelector`.
    """

    estimator = estimator or CostEstimator()

    metrics, largest_subsystem = _prepare_selector_metrics(
        stats,
        sparsity=sparsity,
        phase_rotation_diversity=phase_rotation_diversity,
        amplitude_rotation_diversity=amplitude_rotation_diversity,
        selector_metrics=selector_metrics,
    )
    diag: MutableMapping[str, object] = {"metrics": metrics, "backends": {}}

    fragments = metrics.get("fragments", [])
    total_gates = int(metrics.get("num_gates", stats.total_gates))
    num_qubits = int(metrics.get("num_qubits", stats.num_qubits))
    num_2q = int(metrics.get("num_2q", stats.num_2q_gates))
    two_qubit_ratio = num_2q / total_gates if total_gates else 0.0
    rotation_density = float(metrics.get("rotation_density", 0.0))
    entanglement = float(metrics.get("entanglement_entropy", 0.0))
    sparse = min(max(float(metrics.get("sparsity", 0.0)), 0.0), 1.0)
    phase_rot = float(metrics.get("phase_rotation_diversity", 0.0))
    amp_rot = float(metrics.get("amplitude_rotation_diversity", 0.0))
    nnz = max(1, int(metrics.get("nnz", 1)))
    long_range_fraction = float(metrics.get("mps_long_range_fraction", 0.0))
    long_range_extent = float(metrics.get("mps_long_range_extent", 0.0))
    max_interaction_distance = int(metrics.get("mps_max_interaction_distance", 0))
    local = bool(metrics.get("local", stats.is_local))
    frontier = stats.frontier or num_qubits
    total_multi = int(fragments[0].get("multi_qubit_gates", num_2q)) if fragments else num_2q

    metrics["two_qubit_ratio"] = two_qubit_ratio

    candidates: dict[Backend, Cost] = {}

    # Tableau backend ---------------------------------------------------
    if allow_tableau and stats.is_clifford and total_gates:
        depth_est = max(1, math.ceil(total_gates / max(num_qubits, 1))) if num_qubits else 0
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

    # Decision diagram backend -----------------------------------------
    s_thresh = adaptive_dd_sparsity_threshold(num_qubits)
    amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(num_qubits, sparse)
    softness = max(1.0, config.DEFAULT.dd_rotation_softness)

    structure_override = bool(metrics.get("dd_size_override", False))
    if not selector_metrics and not structure_override:
        size_override = num_qubits <= 10 and nnz <= config.DEFAULT.dd_nnz_threshold
        if (
            size_override
            and sparse < s_thresh
            and amp_rot == 0
            and phase_rot <= 2
            and local
        ):
            structure_override = True
            metrics["dd_size_override"] = True

    effective_sparse = float(metrics.get("effective_dd_sparsity", sparse))
    if structure_override and effective_sparse == sparse:
        bonus = 1.0 + max(0, 10 - num_qubits) / 5.0
        effective_sparse = min(1.0, max(sparse, s_thresh * bonus))
        metrics["effective_dd_sparsity"] = effective_sparse

    passes = effective_sparse >= s_thresh and nnz <= config.DEFAULT.dd_nnz_threshold

    dd_metric = False
    metric_value: float | None = None
    if passes:
        s_score = effective_sparse / s_thresh if s_thresh > 0 else 0.0
        s_score = min(max(s_score, 0.0), 1.2)
        nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
        nnz_score = min(max(nnz_score, -1.0), 1.0)
        phase_score = _soft_penalty(
            phase_rot,
            config.DEFAULT.dd_phase_rotation_diversity_threshold,
            softness,
        )
        amp_score = _soft_penalty(amp_rot, amp_thresh, softness)
        weight_sum = (
            config.DEFAULT.dd_sparsity_weight
            + config.DEFAULT.dd_nnz_weight
            + config.DEFAULT.dd_phase_rotation_weight
            + config.DEFAULT.dd_amplitude_rotation_weight
        )
        weighted = (
            config.DEFAULT.dd_sparsity_weight * s_score
            + config.DEFAULT.dd_nnz_weight * nnz_score
            + config.DEFAULT.dd_phase_rotation_weight * phase_score
            + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
        )
        metric_value = weighted / weight_sum if weight_sum else 0.0
        metrics["decision_diagram_metric"] = metric_value
        metrics["dd_metric_threshold"] = config.DEFAULT.dd_metric_threshold
        metrics["dd_phase_score"] = phase_score
        metrics["dd_amplitude_score"] = amp_score
        metrics["dd_hybrid_replay"] = int(metrics.get("dd_hybrid_replay", 0))
        dd_metric = metric_value >= config.DEFAULT.dd_metric_threshold
        if not dd_metric:
            diag["backends"][Backend.DECISION_DIAGRAM] = {
                "feasible": False,
                "reasons": ["metric below threshold"],
                "metric": metric_value,
                "hybrid_frontier": int(metrics["dd_hybrid_frontier"]),
            }
    else:
        reasons: list[str] = []
        if effective_sparse < s_thresh:
            reasons.append("sparsity below threshold")
        if nnz > config.DEFAULT.dd_nnz_threshold:
            reasons.append("nnz above threshold")
        phase_ratio = phase_rot / max(config.DEFAULT.dd_phase_rotation_diversity_threshold, 1)
        amp_ratio = amp_rot / max(amp_thresh, 1)
        if phase_ratio > 1:
            reasons.append("phase diversity incurs penalty")
        if amp_ratio > 1:
            reasons.append("amplitude diversity incurs penalty")
        diag["backends"][Backend.DECISION_DIAGRAM] = {
            "feasible": False,
            "reasons": reasons,
            "hybrid_frontier": int(metrics["dd_hybrid_frontier"]),
        }

    if dd_metric:
        dd_cost = estimator.decision_diagram(
            num_gates=total_gates,
            frontier=frontier,
            sparsity=sparse,
            phase_rotation_diversity=phase_rot,
            amplitude_rotation_diversity=amp_rot,
            entanglement_entropy=entanglement,
            two_qubit_ratio=two_qubit_ratio,
            converted_frontier=int(metrics["dd_hybrid_frontier"]),
            hybrid_replay_gates=int(metrics.get("dd_hybrid_replay", 0)),
        )
        feasible = True
        reasons = []
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
            "hybrid_frontier": int(metrics["dd_hybrid_frontier"]),
            "hybrid_penalty": int(metrics["dd_hybrid_penalty"]),
        }
        diag["backends"][Backend.DECISION_DIAGRAM] = entry
        if feasible:
            candidates[Backend.DECISION_DIAGRAM] = dd_cost

    # Matrix product state backend -------------------------------------
    if local and num_qubits and total_multi:
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
            long_range_fraction=long_range_fraction,
            long_range_extent=long_range_extent,
        )
        feasible = not infeasible_chi
        reasons: list[str] = []
        over_fraction = long_range_fraction > config.DEFAULT.mps_long_range_fraction_threshold
        over_extent = long_range_extent > config.DEFAULT.mps_long_range_extent_threshold
        enforce_locality = (
            over_fraction
            and over_extent
            and (
                largest_subsystem >= config.DEFAULT.mps_locality_strict_qubits
                or max_interaction_distance >= config.DEFAULT.mps_locality_strict_distance
            )
        )
        locality_reasons: list[str] = []
        if over_fraction:
            locality_reasons.append("non-local interactions exceed fraction threshold")
        if over_extent:
            locality_reasons.append("interaction span exceeds extent threshold")
        if enforce_locality:
            feasible = False
            reasons.extend(locality_reasons)
        if infeasible_chi:
            reasons.append("bond dimension exceeds memory limit")
            feasible = False
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
            "long_range_fraction": long_range_fraction,
            "long_range_extent": long_range_extent,
            "max_interaction_distance": max_interaction_distance,
        }
        if chi_limit is not None:
            entry["chi_limit"] = chi_limit
        if target_accuracy is not None:
            entry["target_accuracy"] = target_accuracy
        if locality_reasons:
            entry["locality_warnings"] = locality_reasons
        diag["backends"][Backend.MPS] = entry
        if feasible:
            candidates[Backend.MPS] = mps_cost
    else:
        reason = "non-local gates" if total_multi else "no multi-qubit gates"
        diag["backends"][Backend.MPS] = {
            "feasible": False,
            "reasons": [reason],
            "long_range_fraction": long_range_fraction,
            "long_range_extent": long_range_extent,
            "max_interaction_distance": max_interaction_distance,
        }

    # Statevector backend ----------------------------------------------
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
        raise NoFeasibleBackendError(
            "No simulation backend satisfies the given constraints"
        )

    if callable(selection_metric):
        def score_fn(backend: Backend) -> float:
            return selection_metric(backend, candidates[backend])

        selected = min(candidates, key=score_fn)
    else:
        metric_name = str(selection_metric).lower()
        if metric_name == "weighted":
            time_weight, mem_weight = metric_weights

            def score_fn(backend: Backend) -> float:
                cost = candidates[backend]
                return time_weight * cost.time + mem_weight * cost.memory

            selected = min(candidates, key=score_fn)
        elif metric_name == "pareto":
            pareto_front: list[Backend] = []
            for backend, cost in candidates.items():
                dominated = False
                for other, other_cost in candidates.items():
                    if other == backend:
                        continue
                    if (
                        other_cost.memory <= cost.memory
                        and other_cost.time <= cost.time
                        and (other_cost.memory < cost.memory or other_cost.time < cost.time)
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append(backend)
            if not pareto_front:
                pareto_front = list(candidates)

            time_weight, mem_weight = metric_weights

            def score_fn(backend: Backend) -> float:
                cost = candidates[backend]
                return time_weight * cost.time + mem_weight * cost.memory

            selected = min(pareto_front, key=score_fn)
            for backend, entry in diag["backends"].items():
                if isinstance(entry, MutableMapping):
                    entry["pareto_optimal"] = backend in pareto_front
        else:
            raise ValueError(f"unknown selection metric: {selection_metric}")
    diag["selected_backend"] = selected
    diag["selected_cost"] = candidates[selected]
    scores = {backend: score_fn(backend) for backend in candidates}
    for backend, entry in diag["backends"].items():
        if isinstance(entry, Mapping):
            entry["selected"] = backend == selected
            if backend in scores:
                entry["score"] = scores[backend]
    return selected, diag


def replay_backend_selection(
    selector_metrics: Mapping[str, object],
    *,
    fragment_index: int = 0,
    allow_tableau: bool = True,
    max_memory: float | None = None,
    max_time: float | None = None,
    target_accuracy: float | None = None,
    estimator: CostEstimator | None = None,
    selection_metric: str | Callable[[Backend, Cost], float] = "weighted",
    metric_weights: tuple[float, float] = (0.5, 0.5),
) -> tuple[Backend | None, MutableMapping[str, object]]:
    """Replay a planner decision from exported :class:`MethodSelector` metrics."""

    fragments = selector_metrics.get("fragments")
    if not fragments:
        raise ValueError("selector metrics must include fragment entries")
    fragment_list = [dict(entry) for entry in fragments]  # type: ignore[assignment]
    if fragment_index < 0 or fragment_index >= len(fragment_list):
        raise IndexError("fragment index out of range")

    metrics_copy: dict[str, object] = {
        key: value
        for key, value in selector_metrics.items()
        if key != "fragments"
    }
    metrics_copy["fragments"] = fragment_list

    stats = FragmentStats.from_export(fragment_list[fragment_index])
    return evaluate_fragment_backends(
        stats,
        allow_tableau=allow_tableau,
        max_memory=max_memory,
        max_time=max_time,
        target_accuracy=target_accuracy,
        estimator=estimator,
        selection_metric=selection_metric,
        metric_weights=metric_weights,
        selector_metrics=metrics_copy,
    )
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
    "replay_backend_selection",
    "estimate_conversion",
    "export_figure",
    "load_calibrated_estimator",
    "SyntheticPlanScenario",
]
