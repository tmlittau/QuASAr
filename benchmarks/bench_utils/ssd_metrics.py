"""Helpers to summarise SSD partition data for benchmark outputs."""

from __future__ import annotations

from collections import Counter
import statistics
from typing import Any, Dict


def partition_metrics_from_result(result: Any) -> Dict[str, object]:
    """Return aggregate metrics describing the partitions in ``result``."""

    metrics: Dict[str, object] = {
        "partition_count": None,
        "partition_total_subsystems": None,
        "partition_unique_backends": None,
        "partition_max_multiplicity": None,
        "partition_mean_multiplicity": None,
        "partition_backend_breakdown": None,
        "hierarchy_available": None,
    }
    if result is None:
        return metrics

    hierarchy = getattr(result, "hierarchy", None)
    metrics["hierarchy_available"] = bool(hierarchy)

    partitions = getattr(result, "partitions", None)
    if not partitions:
        return metrics

    multiplicities: list[int] = []
    backend_counts: Counter[str] = Counter()

    for part in partitions:
        multiplicity = getattr(part, "multiplicity", None)
        if multiplicity is None:
            subsystems = getattr(part, "subsystems", ())
            multiplicity = len(subsystems) if subsystems else 1
        multiplicity = int(multiplicity)
        multiplicities.append(multiplicity)

        backend = getattr(part, "backend", None)
        backend_name = getattr(backend, "value", getattr(backend, "name", str(backend)))
        backend_counts[str(backend_name)] += multiplicity

    metrics.update(
        partition_count=len(partitions),
        partition_total_subsystems=sum(multiplicities) if multiplicities else None,
        partition_unique_backends=len(backend_counts) if backend_counts else None,
        partition_max_multiplicity=max(multiplicities) if multiplicities else None,
        partition_mean_multiplicity=(
            statistics.fmean(multiplicities) if len(multiplicities) >= 1 else None
        ),
        partition_backend_breakdown=(
            ", ".join(f"{name}:{count}" for name, count in sorted(backend_counts.items()))
            if backend_counts
            else None
        ),
    )
    return metrics


__all__ = ["partition_metrics_from_result"]

