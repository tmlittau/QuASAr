from __future__ import annotations

import io
import math
import re
import contextlib

from benchmarks.circuits import qft_circuit, w_state_circuit
from quasar import Backend
from quasar.scheduler import Scheduler
from quasar.sparsity import sparsity_estimate
from quasar.symmetry import (
    phase_rotation_diversity,
    amplitude_rotation_diversity,
)


PATTERN = (
    r"sparsity=(?P<s>\d+\.\d+) "
    r"rotation_diversity=(?P<r>\d+\.\d+) "
    r"nnz=(?P<nnz>\d+) locality=(?P<loc>True|False) "
    r"candidates=(?P<cand>[A-Z_>]+) selected=(?P<sel>[A-Z_]+)"
)


def _expected_metrics(circuit):
    sparsity = sparsity_estimate(circuit)
    rotation = max(
        phase_rotation_diversity(circuit),
        amplitude_rotation_diversity(circuit),
    )
    nnz = int((1 - sparsity) * (2 ** circuit.num_qubits))
    multi = [g for g in circuit.gates if len(g.qubits) > 1]
    local = bool(multi) and all(
        len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
    )
    return sparsity, rotation, nnz, local


def _run(circuit):
    sched = Scheduler(
        verbose_selection=True,
        quick_max_qubits=10,
        quick_max_gates=100,
        quick_max_depth=10,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        backend = sched.select_backend(circuit)
    return backend, buf.getvalue().strip()


def test_qft_verbose_selection():
    circuit = qft_circuit(5)
    backend, log = _run(circuit)
    match = re.search(PATTERN, log)
    assert match, log
    s, r, nnz, loc = _expected_metrics(circuit)
    assert math.isclose(float(match.group("s")), s, rel_tol=1e-6)
    assert math.isclose(float(match.group("r")), r, rel_tol=1e-6)
    assert int(match.group("nnz")) == nnz
    assert (match.group("loc") == "True") is loc
    assert match.group("cand") == "STATEVECTOR"
    assert match.group("sel" ) == Backend.STATEVECTOR.name
    assert backend == Backend.STATEVECTOR


def test_w_state_verbose_selection():
    circuit = w_state_circuit(5)
    backend, log = _run(circuit)
    match = re.search(PATTERN, log)
    assert match, log
    s, r, nnz, loc = _expected_metrics(circuit)
    assert math.isclose(float(match.group("s")), s, rel_tol=1e-6)
    assert math.isclose(float(match.group("r")), r, rel_tol=1e-6)
    assert int(match.group("nnz")) == nnz
    assert (match.group("loc") == "True") is loc
    assert match.group("cand") == "DECISION_DIAGRAM>MPS>STATEVECTOR"
    assert match.group("sel" ) == Backend.DECISION_DIAGRAM.name
    assert backend == Backend.DECISION_DIAGRAM
