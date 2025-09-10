"""Test decision-diagram frontier extraction runtime scaling."""

from __future__ import annotations

import time

import quasar_convert as qc


def _measure_time(frontier: int, repeats: int = 5) -> float:
    """Return total time to extract a boundary of ``frontier`` qubits."""
    eng = qc.ConversionEngine()
    ssd = qc.SSD()
    ssd.boundary_qubits = list(range(frontier))
    start = time.perf_counter()
    for _ in range(repeats):
        eng.convert_boundary_to_statevector(ssd)
    return time.perf_counter() - start


def test_frontier_extraction_time_scales_with_size() -> None:
    """Extraction time should grow with the frontier size."""
    frontiers = [4, 6, 8, 10]
    times = [_measure_time(f) for f in frontiers]
    for prev, curr in zip(times, times[1:]):
        assert curr > prev * 2
