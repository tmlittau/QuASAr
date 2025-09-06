"""Run sample circuits to demonstrate verbose backend-selection logs."""

from __future__ import annotations

import contextlib
import io

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.circuits import qft_circuit, w_state_circuit
from quasar import Backend, Scheduler


def _run_and_capture(circuit):
    scheduler = Scheduler(
        quick_max_qubits=10,
        quick_max_gates=100,
        quick_max_depth=100,
        verbose_selection=True,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        backend = scheduler.select_backend(circuit)
    log = buf.getvalue().strip()
    return backend, log


def main() -> None:
    tests = [
        ("QFT", qft_circuit(3), Backend.STATEVECTOR),
        ("W-state", w_state_circuit(3), Backend.MPS),
    ]
    for name, circuit, expected in tests:
        backend, log = _run_and_capture(circuit)
        print(f"{name} circuit backend: {backend.name}")
        print(log)
        assert backend == expected
        assert f"selected={expected.name}" in log


if __name__ == "__main__":
    main()
