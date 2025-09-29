"""Calculation routines for theoretical runtime and memory estimation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


if __package__ in {None, ""}:  # pragma: no cover - script execution
    import importlib
    import sys

    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    importlib.import_module("quasar")
    import paper_figures as paper_figures  # type: ignore[no-redef]
    from progress import ProgressReporter  # type: ignore[no-redef]
    from threading_utils import resolve_worker_count  # type: ignore[no-redef]
else:  # pragma: no cover - package import path
    from . import paper_figures
    from .progress import ProgressReporter
    from .threading_utils import resolve_worker_count

from quasar.analyzer import CircuitAnalyzer
from quasar.cost import Backend, Cost, CostEstimator
from quasar.partitioner import Partitioner
from quasar.planner import (
    Planner,
    _add_cost,
    _circuit_depth,
    _parallel_simulation_cost,
    _simulation_cost,
)

from .theoretical_estimation_utils import EstimateRecord


def _format_backend_sequence(steps) -> str:
    """Return a human-readable description of backend usage."""

    sequence: list[str] = []
    previous: str | None = None
    for step in steps:
        name = step.backend.name
        if name != previous:
            sequence.append(name)
            previous = name
    return " → ".join(sequence) if sequence else "n/a"


def _estimate_plan_cost(planner: Planner, plan) -> tuple[Cost, bool, int]:
    """Return the aggregated cost of ``plan`` along with metadata.

    The helper replays the plan's segments, accounting for independent
    parallel groups and any conversion layers.  It returns the cumulative
    cost, a flag indicating whether parallel execution was exploited and
    the number of conversion layers encountered.
    """

    estimator = planner.estimator
    part = Partitioner()
    gates = list(plan.gates)
    steps = list(plan.steps)
    total = Cost(time=0.0, memory=0.0)
    used_parallel = False

    for step in steps:
        segment = gates[step.start : step.end]
        if not segment:
            continue
        groups = part.parallel_groups(segment)
        if len(groups) > 1:
            cost = _parallel_simulation_cost(estimator, step.backend, groups)
            used_parallel = True
        else:
            num_meas = sum(1 for g in segment if g.gate.upper() in {"MEASURE", "RESET"})
            num_1q = sum(
                1
                for g in segment
                if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
            )
            num_2q = len(segment) - num_1q - num_meas
            num_qubits = len({q for gate in segment for q in gate.qubits})
            num_t = sum(1 for g in segment if g.gate.upper() in {"T", "TDG"})
            depth = _circuit_depth(segment)
            cost = _simulation_cost(
                estimator,
                step.backend,
                num_qubits,
                num_1q,
                num_2q,
                num_meas,
                num_t_gates=num_t,
                depth=depth,
            )
        total = _add_cost(total, cost)

    conversion_count = 0
    for layer in getattr(plan, "conversions", []):
        total = _add_cost(total, layer.cost)
        conversion_count += 1

    return total, used_parallel, conversion_count


def _estimate_quasar_record(
    *, planner: Planner, circuit, spec: paper_figures.CircuitSpec, n_qubits: int
) -> EstimateRecord:
    """Return the QuASAr estimate for ``circuit`` using ``planner``."""

    if circuit is None:
        return EstimateRecord(
            circuit=spec.name,
            qubits=n_qubits,
            framework="quasar",
            backend="n/a",
            supported=False,
            time_ops=None,
            memory_bytes=None,
            note="circuit construction failed",
        )

    try:
        plan = planner.plan(circuit, use_cache=False)
    except Exception as exc:  # pragma: no cover - defensive fallback
        analyzer = CircuitAnalyzer(circuit, estimator=planner.estimator)
        resources = analyzer.resource_estimates()
        candidates = [
            (backend, cost)
            for backend, cost in resources.items()
            if paper_figures._supports_backend(circuit, backend)
        ]
        if not candidates:
            return EstimateRecord(
                circuit=spec.name,
                qubits=n_qubits,
                framework="quasar",
                backend="n/a",
                supported=False,
                time_ops=None,
                memory_bytes=None,
                note=f"planner failed: {exc}",
            )
        backend, cost = min(candidates, key=lambda item: item[1].time)
        return EstimateRecord(
            circuit=spec.name,
            qubits=n_qubits,
            framework="quasar",
            backend=backend.name,
            supported=True,
            time_ops=cost.time,
            memory_bytes=cost.memory,
            note=f"planner fallback to single backend ({exc})",
        )

    steps = list(plan.steps)
    if not steps:
        return EstimateRecord(
            circuit=spec.name,
            qubits=n_qubits,
            framework="quasar",
            backend="n/a",
            supported=False,
            time_ops=None,
            memory_bytes=None,
            note="planner produced empty plan",
        )

    total_cost, used_parallel, conversion_count = _estimate_plan_cost(planner, plan)
    backend_label = _format_backend_sequence(steps)

    notes: list[str] = [f"{len(steps)} segment plan"]
    if conversion_count:
        plural = "s" if conversion_count != 1 else ""
        notes.append(f"{conversion_count} conversion{plural}")
    if used_parallel:
        notes.append("parallel execution")

    return EstimateRecord(
        circuit=spec.name,
        qubits=n_qubits,
        framework="quasar",
        backend=backend_label,
        supported=True,
        time_ops=total_cost.time,
        memory_bytes=total_cost.memory,
        note="; ".join(notes),
    )

def estimate_circuit(
    spec: paper_figures.CircuitSpec,
    n_qubits: int,
    estimator: CostEstimator,
    backends: Sequence[Backend],
    planner: Planner,
) -> list[EstimateRecord]:
    """Return records for ``spec`` at ``n_qubits`` covering baselines and QuASAr."""

    records: list[EstimateRecord] = []

    forced = paper_figures._build_circuit(
        spec, n_qubits, use_classical_simplification=False
    )
    auto = paper_figures._build_circuit(
        spec, n_qubits, use_classical_simplification=True
    )

    if forced is None or auto is None:
        note = "circuit construction failed"
        records.append(
            EstimateRecord(
                circuit=spec.name,
                qubits=n_qubits,
                framework="quasar",
                backend="n/a",
                supported=False,
                time_ops=None,
                memory_bytes=None,
                note=note,
            )
        )
        for backend in backends:
            records.append(
                EstimateRecord(
                    circuit=spec.name,
                    qubits=n_qubits,
                    framework=backend.name,
                    backend=backend.name,
                    supported=False,
                    time_ops=None,
                    memory_bytes=None,
                    note=note,
                )
            )
        return records

    forced_analyzer = CircuitAnalyzer(forced, estimator=estimator)
    forced_resources = forced_analyzer.resource_estimates()

    for backend in backends:
        supported = paper_figures._supports_backend(forced, backend)
        cost: Cost | None = forced_resources.get(backend)
        records.append(
            EstimateRecord(
                circuit=spec.name,
                qubits=n_qubits,
                framework=backend.name,
                backend=backend.name,
                supported=supported and cost is not None,
                time_ops=None if not supported or cost is None else cost.time,
                memory_bytes=None if not supported or cost is None else cost.memory,
                note=None if supported else "unsupported gate set",
            )
        )

    quasar_record = _estimate_quasar_record(
        planner=planner,
        circuit=auto,
        spec=spec,
        n_qubits=n_qubits,
    )
    records.append(quasar_record)

    return records


def _estimate_width(
    spec: paper_figures.CircuitSpec,
    width: int,
    backends: Sequence[Backend],
    estimator: CostEstimator,
) -> tuple[list[EstimateRecord], list[str]]:
    """Return records and progress messages for a single ``spec`` width."""

    planner = Planner(estimator=estimator, perf_prio="time")
    width_records = estimate_circuit(
        spec,
        width,
        estimator,
        backends,
        planner,
    )
    messages = [f"{rec.circuit}@{rec.qubits} {rec.framework.lower()}" for rec in width_records]
    return width_records, messages


def collect_estimates(
    specs: Iterable[paper_figures.CircuitSpec],
    backends: Sequence[Backend],
    estimator: CostEstimator,
    *,
    max_workers: int | None = None,
) -> list[EstimateRecord]:
    """Return all estimate records for ``specs`` using ``estimator``."""

    spec_list = list(specs)
    total_steps = sum(len(spec.qubits) * (len(backends) + 1) for spec in spec_list)
    progress = ProgressReporter(total_steps, prefix="Estimating") if total_steps else None
    ordered: dict[int, list[tuple[int, list[EstimateRecord]]]] = {}

    backend_list = tuple(backends)
    total_widths = sum(len(spec.qubits) for spec in spec_list)
    worker_count = resolve_worker_count(max_workers, total_widths)

    try:
        if worker_count <= 1:
            for spec_index, spec in enumerate(spec_list):
                for position, width in enumerate(spec.qubits):
                    width_records, messages = _estimate_width(
                        spec, width, backend_list, estimator
                    )
                    ordered.setdefault(spec_index, []).append((position, width_records))
                    if progress:
                        for message in messages:
                            progress.advance(message)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _estimate_width, spec, width, backend_list, estimator
                    ): (spec_index, position)
                    for spec_index, spec in enumerate(spec_list)
                    for position, width in enumerate(spec.qubits)
                }
                for future in as_completed(futures):
                    spec_index, position = futures[future]
                    try:
                        width_records, messages = future.result()
                    except Exception:
                        if progress:
                            progress.close()
                        raise
                    ordered.setdefault(spec_index, []).append((position, width_records))
                    if progress:
                        for message in messages:
                            progress.advance(message)
    finally:
        if progress:
            progress.close()

    records: list[EstimateRecord] = []
    for index in range(len(spec_list)):
        entries = ordered.get(index, [])
        entries.sort(key=lambda item: item[0])
        for _, width_records in entries:
            records.extend(width_records)
    return records


__all__ = ["collect_estimates", "estimate_circuit"]

