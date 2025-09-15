# HHL Circuit

`benchmarks.circuits.hhl_circuit` builds a quantum circuit that prepares the
normalized solution state of the linear system ``Ax = b``.  The function
computes the solution vector classically and emits an ``INITIALIZE`` gate to
load it into a quantum register.

## Arguments
- `A` – Hermitian matrix of size `2^n × 2^n`.
- `b` – Right hand side vector of length `2^n`.

## Assumptions
- `A` is Hermitian and invertible.

## Example
```python
import numpy as np
from benchmarks.circuits import hhl_circuit

A = np.array([[1, 0.5], [0.5, 1]], dtype=complex)
b = np.array([1, 0], dtype=complex)
qc = hhl_circuit(A, b)
```
