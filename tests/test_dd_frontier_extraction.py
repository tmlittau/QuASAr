"""Test decision-diagram frontier extraction output size."""

from __future__ import annotations

import quasar_convert as qc


def _extract(frontier: int) -> list[complex]:
    """Return the statevector for a boundary of ``frontier`` qubits."""
    eng = qc.ConversionEngine()
    ssd = qc.SSD()
    ssd.boundary_qubits = list(range(frontier))
    return eng.convert_boundary_to_statevector(ssd)


def test_frontier_extraction_output_size() -> None:
    """Extraction should yield statevectors of the expected dimension."""
    frontiers = [4, 6, 8, 10]
    for f in frontiers:
        vec = _extract(f)
        assert len(vec) == 1 << f
