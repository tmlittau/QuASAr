# Symmetry heuristic

QuASAr derives a simple **symmetry** score for each :class:`~quasar.circuit.Circuit`.
The score looks for repeated gate types with identical parameters and
normalises the count by the circuit depth, yielding a value in ``[0, 1]``.
Higher values indicate more structural repetition while ``0`` signifies no
repeated patterns.

The symmetry value is accessible via :attr:`Circuit.symmetry` and can be
used as a lightweight indicator of circuit regularity.
