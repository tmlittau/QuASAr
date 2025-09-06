from __future__ import annotations

"""Regression tests for verbose backend selection output."""

import io
import contextlib

from benchmarks.circuits import qft_circuit, w_state_circuit
from quasar import Backend, Scheduler


def _capture_backend_selection(circuit):
    """Run backend selection with verbose logging and capture output."""
    scheduler = Scheduler(
        quick_max_qubits=10,
        quick_max_gates=100,
        quick_max_depth=100,
        verbose_selection=True,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        backend = scheduler.select_backend(circuit)
    return backend, buf.getvalue().strip()


def test_verbose_selection_qft():
    backend, log = _capture_backend_selection(qft_circuit(3))
    assert backend == Backend.STATEVECTOR
    assert "[backend-selection]" in log
    assert "sparsity=0.000000" in log
    assert "rotation_diversity=2.000000" in log
    assert "nnz=8" in log
    assert "locality=False" in log
    assert "candidates=STATEVECTOR" in log
    assert "selected=STATEVECTOR" in log


def test_verbose_selection_w_state():
    backend, log = _capture_backend_selection(w_state_circuit(3))
    assert backend == Backend.MPS
    assert "[backend-selection]" in log
    assert "sparsity=0.625000" in log
    assert "rotation_diversity=0.000000" in log
    assert "nnz=3" in log
    assert "locality=True" in log
    assert "candidates=MPS>STATEVECTOR" in log
    assert "selected=MPS" in log

