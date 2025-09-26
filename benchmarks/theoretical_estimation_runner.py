"""Calculation routines for theoretical runtime and memory estimation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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

from .theoretical_estimation_utils import EstimateRecord


def estimate_circuit(
    spec: paper_figures.CircuitSpec,
    n_qubits: int,
    estimator: CostEstimator,
    backends: Sequence[Backend],
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

    auto_analyzer = CircuitAnalyzer(auto, estimator=estimator)
    auto_resources = auto_analyzer.resource_estimates()
    supported_backends: list[tuple[Backend, Cost]] = []
    for backend, cost in auto_resources.items():
        if paper_figures._supports_backend(auto, backend):
            supported_backends.append((backend, cost))

    if not supported_backends:
        records.append(
            EstimateRecord(
                circuit=spec.name,
                qubits=n_qubits,
                framework="quasar",
                backend="n/a",
                supported=False,
                time_ops=None,
                memory_bytes=None,
                note="no compatible backend available",
            )
        )
    else:
        backend, cost = min(supported_backends, key=lambda item: item[1].time)
        records.append(
            EstimateRecord(
                circuit=spec.name,
                qubits=n_qubits,
                framework="quasar",
                backend=backend.name,
                supported=True,
                time_ops=cost.time,
                memory_bytes=cost.memory,
                note="automatic planner (single-backend approximation)",
            )
        )

    return records


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
    ordered: dict[int, list[EstimateRecord]] = {}

    def _estimate(spec: paper_figures.CircuitSpec) -> tuple[list[EstimateRecord], list[str]]:
        spec_records: list[EstimateRecord] = []
        messages: list[str] = []
        for width in spec.qubits:
            width_records = estimate_circuit(spec, width, estimator, backends)
            spec_records.extend(width_records)
            for rec in width_records:
                label = f"{rec.circuit}@{rec.qubits} {rec.framework.lower()}"
                messages.append(label)
        return spec_records, messages

    worker_count = resolve_worker_count(max_workers, len(spec_list))

    try:
        if worker_count <= 1:
            for index, spec in enumerate(spec_list):
                spec_records, messages = _estimate(spec)
                ordered[index] = spec_records
                if progress:
                    for message in messages:
                        progress.advance(message)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_estimate, spec): index
                    for index, spec in enumerate(spec_list)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        spec_records, messages = future.result()
                    except Exception:
                        if progress:
                            progress.close()
                        raise
                    ordered[index] = spec_records
                    if progress:
                        for message in messages:
                            progress.advance(message)
    finally:
        if progress:
            progress.close()

    records: list[EstimateRecord] = []
    for index in range(len(spec_list)):
        records.extend(ordered.get(index, []))
    return records


__all__ = ["collect_estimates", "estimate_circuit"]

