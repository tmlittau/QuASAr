# Tableau to Decision Diagram Conversion Benchmark

The `benchmarks.partition_circuits.stim_to_dd_circuit` helper prepares a
collection of identical GHZ subsystems that start on the stabiliser (Stim
Tableau) backend and then transition to the decision-diagram simulator.
It is designed to exercise QuASAr's conversion pipeline by keeping the
subsystems independent until the exact moment non-Clifford gates are
introduced.

## Circuit structure

* **Stabiliser preparation** – each group of `group_size` qubits is
  initialised into a GHZ state using only Clifford gates. The method
  selector therefore assigns the Tableau backend and compresses the
  identical groups into a single partition with multiple subsystems.
* **T layer** – a uniform layer of `T` rotations is applied to every
  qubit. This breaks the Clifford structure and forces a conversion to
  the decision-diagram backend. Because the groups remain disjoint, the
  conversion happens independently for each subsystem.
* **Optional entangling layer** – when `entangling_layer=True` a final
  chain of `CX` gates links neighbouring subsystems *after* the `T`
  layer. This demonstrates that cross-group entanglement can be added
  without interfering with the earlier Tableau→DD conversion.

## Inspecting the conversion

```python
from benchmarks.partition_circuits import stim_to_dd_circuit
from quasar.cost import Backend
from quasar.partitioner import Partitioner

circuit = stim_to_dd_circuit(num_groups=3, group_size=3)
ssd = Partitioner().partition(circuit, debug=True)

print([part.backend for part in ssd.partitions])
# → [Backend.TABLEAU, Backend.DECISION_DIAGRAM, ...]

print([(conv.source, conv.target, conv.boundary) for conv in ssd.conversions])
# → [(Backend.TABLEAU, Backend.DECISION_DIAGRAM, (0, 1, 2)), ...]
```

Each conversion record captures the boundary qubits for one GHZ group.
The accompanying unit test `tests/test_stim_to_dd_conversion.py`
verifies that the planner emits the Tableau partition, switches to the
decision-diagram backend after the `T` layer, and records the conversion
layer diagnostics.
