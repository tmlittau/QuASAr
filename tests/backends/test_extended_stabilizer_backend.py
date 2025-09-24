from __future__ import annotations

import numpy as np

from quasar.backends.statevector import AerStatevectorBackend, ExtendedStabilizerBackend
from quasar.cost import Backend


def _apply_sequence(backend, sequence):
    backend.load(2)
    for name, qubits in sequence:
        backend.apply_gate(name, qubits)


def test_extended_stabilizer_matches_statevector():
    sequence = [
        ("H", (0,)),
        ("CX", (0, 1)),
        ("T", (0,)),
        ("TDG", (1,)),
        ("S", (0,)),
    ]
    ext = ExtendedStabilizerBackend()
    sv = AerStatevectorBackend()
    _apply_sequence(ext, sequence)
    _apply_sequence(sv, sequence)

    ext_state = ext.statevector()
    sv_state = sv.statevector()

    fidelity = abs(np.vdot(sv_state, ext_state))
    assert fidelity > 0.99

    ssd = ext.extract_ssd()
    assert ssd.partitions
    assert ssd.partitions[0].backend == Backend.EXTENDED_STABILIZER
