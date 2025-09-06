"""Demonstrate backend selection logging.

This script runs sample circuits with verbose backend selection enabled and
prints the explanatory log lines.  It asserts that the reported backend choice
matches the value returned by :class:`~quasar.scheduler.Scheduler`.
"""
from __future__ import annotations

import io
import contextlib

from benchmarks.circuits import qft_circuit, w_state_circuit
from quasar import Backend
from quasar.scheduler import Scheduler


def run_and_log(circuit, expected: Backend) -> None:
    sched = Scheduler(
        verbose_selection=True,
        quick_max_qubits=10,
        quick_max_gates=100,
        quick_max_depth=10,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        backend = sched.select_backend(circuit)
    log = buf.getvalue().strip()
    print(log)
    assert backend == expected
    assert f"selected={expected.name}" in log


def main() -> None:  # pragma: no cover - example usage
    run_and_log(qft_circuit(5), Backend.STATEVECTOR)
    run_and_log(w_state_circuit(5), Backend.DECISION_DIAGRAM)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
