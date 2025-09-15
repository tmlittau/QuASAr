# Multi-controlled X Decomposition

QuASAr provides a generic expansion for multi-controlled X (MCX) gates through
``decompose_mcx``.  An MCX with ``n`` control qubits is reduced to a sequence of
Toffoli gates and single-qubit rotations.  The algorithm introduces ``n-2``
ancillary qubits that carry partial conjunctions of the controls.  These
ancillas are allocated using new qubit indices larger than any existing control
or target index and are uncomputed at the end of the sequence.

The approach guarantees that all ancilla qubits are returned to ``|0⟩`` but
increases the circuit width by ``n-2`` and does not attempt to minimise the
number of elementary gates.  Circuits that cannot spare additional qubits must
supply their own decomposition strategy.
