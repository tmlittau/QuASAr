# QuASAr Simulation Engine API

This guide summarises the public surface of `quasar.simulation_engine` and
demonstrates how to execute circuits, inspect the resulting subsystem
descriptor (SSD), and customise the planner.

## Quick start

```python
from quasar import Circuit, Gate
from quasar.simulation_engine import SimulationEngine

engine = SimulationEngine()

circuit = Circuit([
    Gate("H", [0]),
    Gate("CX", [0, 1]),
    Gate("CX", [1, 2]),
])

result = engine.simulate(circuit)
print("Backends:", [part.backend.name for part in result.ssd.partitions])
print("Final amplitudes:", result.final_state(as_numpy=True))
```

The planner automatically selects a suitable backend for each partition.  For
small GHZ-style circuits like the example above a dense statevector backend is
typically chosen, but larger circuits may trigger tensor-network or tableau
simulators instead.  The :class:`SimulationResult` returned by
:meth:`SimulationEngine.simulate` always provides access to the full SSD so that
advanced users can inspect individual partitions, conversions and cost
estimates.

## Retrieving terminal states

`SimulationResult.final_state()` retrieves the state stored on the last SSD
partition.  When `as_numpy=True` the helper converts the state into a dense
`numpy.ndarray` (if the backend supports it).  Passing a specific partition
index returns intermediate states, and setting `as_numpy=False` exposes the
backend-native representation such as stabiliser tableaux or decision diagram
nodes.

```python
native = result.final_state()
amplitudes = result.final_state(as_numpy=True)
```

If the SSD does not contain the requested state the method returns `None`.  In
addition, attempting to use `as_numpy=True` without NumPy installed raises a
`RuntimeError` so that environments without the optional dependency fail
loudly.

## Guiding the planner

`SimulationEngine.simulate` accepts several optional parameters:

- `backend` prefers a specific backend (for example
  `Backend.STATEVECTOR` or `Backend.STIM_TABLEAU`).
- `memory_threshold` overrides the automatically detected memory ceiling for the
  dense statevector backend.
- `target_accuracy`, `max_time` and `optimization_level` are forwarded to the
  planner and scheduler to tune trade-offs between accuracy and runtime.

The returned :class:`SimulationResult` also exposes per-phase timings (`analysis_time`,
`planning_time`, `execution_time`), backend switch counts, conversion durations
and optional fidelity metrics computed against a reference state.

## Inspecting the SSD

The SSD attached to `SimulationResult.ssd` stores each partition together with
its backend, qubits, dependencies, entanglement relations and recorded state.
Utility methods such as :meth:`SSD.by_backend` and :meth:`SSD.to_networkx`
provide convenient entry points for visualisation and deeper analysis.

When you need to persist SSDs for later inspection, consider serialising them
with `quasar.ssd.SSDCache` from the same module.
