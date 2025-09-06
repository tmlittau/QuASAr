# Classical control simplification

QuASAr tracks whether each qubit in a circuit is known to be in a definite
classical state.  This information allows the `Circuit.simplify_classical_controls`
method to remove gates that have no effect and to reduce some controlled
operations to simpler forms.

## Classicality tracking

Each `Circuit` instance maintains a `classical_state` list with one entry per
qubit.  A value of `0` or `1` indicates that the qubit is classically known to
be in that state, while `None` marks a qubit whose value is unknown or in
superposition.  The list is updated after every gate:

* Pauli `X` and `Y` flip a classical bit.
* Phase gates (`Z`, `S`, `T` and their adjoints) preserve classical bits.
* ``RX``/``RY`` rotations by integer multiples of :math:`\pi` are treated as
  Pauli operations; other rotation angles promote the qubit to a quantum state.
* Any gate introducing superposition or entanglement—such as ``H``, arbitrary
  ``RX``/``RY`` rotations or any multi‑qubit gate—marks the affected qubits as
  non‑classical (`None`).

## Controlled gate reduction

When all control qubits of a gate are known classical bits, QuASAr simplifies
as follows:

* If any control is `0`, the controlled gate never fires and is dropped.
* If every control is `1`, the gate is reduced to its target operation.
  When the target qubit is also classical and the base operation is
  deterministic (`X`, `Y`, phase gates or integer‐π rotations), the gate is
  elided entirely after updating the classical state.
* If a control or the target is still quantum (`None`), the gate is preserved.

## Example

```python
from quasar import Circuit

circ = Circuit.from_dict([
    {"gate": "X", "qubits": [0]},     # qubit 0 becomes classically 1
    {"gate": "CX", "qubits": [0, 1]},  # reduced to X on qubit 1 and elided
])

circ.simplify_classical_controls()
assert circ.gates == []
assert circ.classical_state == [1, 1]
```

The initial `X` marks qubit 0 as classical `1`.  The controlled `CX` then reduces
to a single `X` on the target, which is elided because the target was also
classical.  Both qubits end in known classical states with no remaining gates.

