# QuASAr

This repository contains experimental utilities for the QuASAr project.  The
`quasar_convert` package optionally provides a native C++ backend for faster
conversion routines.

## Installation

Building from source requires a C++17 compiler and a recent version of Python.
The native extension is built automatically when installing the package:

```bash
pip install .
```

If a compiler is not available the build will fall back to a pure Python stub
implementation with reduced performance but identical APIs.

The project's dependencies are declared in `pyproject.toml`. For development,
including running the test suite, install the package with its testing extras:

```bash
pip install -e .[test]
```

## Retrieving partition states

Running a circuit with :class:`quasar.SimulationEngine` returns a
:class:`~quasar.ssd.SSD` object describing the final state.  Each entry in
``SSD.partitions`` exposes the backend specific state of that subsystem via
``SSD.extract_state``.  Users can iterate over all partitions to access the
raw data:

```python
from quasar import Circuit, SimulationEngine, Backend

engine = SimulationEngine()
result = engine.simulate(Circuit([{"gate": "H", "qubits": [0]}]), backend=Backend.STATEVECTOR)
for part in result.ssd.partitions:
    state = result.ssd.extract_state(part)
    print(part.backend, type(state))
```

Depending on the backend, ``state`` may be a dense NumPy vector, a list of MPS
tensors, a ``stim.Tableau`` or a decision diagram node.

The :func:`SimulationEngine.simulate` method accepts an optional ``backend``
argument to explicitly choose the simulation backend (e.g.,
``Backend.TABLEAU`` for Clifford circuits).  When ``backend`` is ``None`` (the
default), the planner selects a backend automatically based on estimated cost.
Supplying a :class:`~quasar.cost.Backend` instance fixes the simulator choice
and disables this automatic selection.  Clifford circuits default to the
specialised TABLEAU backend, though a general-purpose backend like
``Backend.STATEVECTOR`` can be requested explicitly.

```python
from quasar import Backend, Circuit, SimulationEngine

circ = Circuit([
    {"gate": "H", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]},
])
engine = SimulationEngine()

auto = engine.simulate(circ, backend=None)               # automatic selection
forced = engine.simulate(circ, backend=Backend.STATEVECTOR)  # explicit override
print(auto.plan.final_backend, forced.plan.final_backend)
```
If the circuit is small enough to satisfy the quick-path thresholds
described below, this selection degenerates to running the whole circuit on
a single backend.  Skipping partitioning and scheduling avoids overhead and
speeds up tiny workloads.

Dense backends are powered by Qiskit Aer and accept a ``method`` argument to
select the underlying simulator implementation.  For example::

    from quasar.backends import StatevectorBackend
    backend = StatevectorBackend(method="density_matrix")

Convenience classes :class:`AerStatevectorBackend` and :class:`AerMPSBackend`
preconfigure the common ``statevector`` and ``matrix_product_state`` methods
respectively.  Custom backend instances can be supplied to :class:`Scheduler`
or :class:`SimulationEngine` via their ``backends`` argument to control the
simulation method.

## Sparsity heuristic

Every :class:`~quasar.circuit.Circuit` computes a ``sparsity`` estimate for the
state produced from ``|0…0>``.  The value lies in ``[0, 1]`` with ``0``
representing a fully dense state and ``1`` indicating maximal sparsity.  The
estimator performs a single ``O(num_gates)`` pass over the circuit, treating
``H``, ``RX``, ``RY``, ``U``, ``U2`` and ``U3`` as branching gates.  Uncontrolled
branching gates double the number of non‑zero amplitudes while controlled
versions add one.  Consequently, an ``n``‑qubit W‑state yields a high sparsity
of ``1 - n / 2**n`` whereas the quantum Fourier transform drives the estimate
to ``0``.  See [docs/sparsity.md](docs/sparsity.md) for details.

## Configuration

QuASAr exposes a small set of tunables that influence planning heuristics.  The
default values are defined in ``quasar/config.py`` and may be overridden either
programmatically or via environment variables at import time:

* ``QUASAR_QUICK_MAX_QUBITS`` – maximum number of qubits for the quick-path
  estimate.  Set to ``None`` to disable.
* ``QUASAR_QUICK_MAX_GATES`` – maximum gate count for quick-path planning.
* ``QUASAR_QUICK_MAX_DEPTH`` – maximum circuit depth for quick-path planning.
* ``QUASAR_BACKEND_ORDER`` – comma-separated preference list of backends
  (e.g. ``"MPS,STATEVECTOR"``).

The same parameters can be passed directly to :class:`Planner` and
:class:`Scheduler` constructors to override the defaults on a per-instance
basis, allowing runtime tuning without relying on environment variables.

The default quick-path limits are tuned using
``benchmarks/quick_analysis_benchmark.py`` and currently favour circuits of
approximately 12 qubits, 240 gates and depth 60, offering substantial
speedups for small problems.

### Automatic single-backend selection

When a circuit's size falls below *all* ``QUASAR_QUICK_MAX_*`` thresholds,
the planner skips dynamic programming and schedules the entire circuit on
the cheapest backend in the configured preference order.  These thresholds
can be tuned via the environment variables above or by supplying
``quick_max_qubits``, ``quick_max_gates`` and ``quick_max_depth`` to
:class:`Planner` or :class:`Scheduler`.

```python
import time
from quasar import Circuit, SimulationEngine, Planner

circ = Circuit([
    {"gate": "H", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]},
])

engine = SimulationEngine()
start = time.perf_counter()
engine.simulate(circ)  # quick path uses a single backend
print(f"quick path: {time.perf_counter() - start:.3f}s")

planner = Planner(quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None)
engine = SimulationEngine(planner=planner)
start = time.perf_counter()
engine.simulate(circ)  # full planner is slower on tiny circuits
print(f"full planning: {time.perf_counter() - start:.3f}s")
```

The difference is small but measurable: the quick path avoids planning
entirely and generally selects a dense simulator for such tiny circuits.

### Benchmarking considerations

Quick-path execution hides planning overhead and may select a backend that
does not scale to larger problems.  When collecting benchmark numbers or
comparing backend performance, disable this feature by setting the
``QUASAR_QUICK_MAX_QUBITS``, ``QUASAR_QUICK_MAX_GATES`` and
``QUASAR_QUICK_MAX_DEPTH`` variables to ``None`` (or passing ``None`` to the
corresponding constructor arguments).  This forces the planner to consider
all backends, yielding more representative results.

## Scalable benchmark circuits

QuASAr can simulate parameterized circuits sourced from MQTBench and QASMBench.
The table below lists families from MQTBench that allow adjusting qubit count or
depth.

| Algorithm | Parameters | Notes for QuASAr |
|-----------|------------|-----------------|
| Amplitude Estimation | `num_qubits`, `probability` | Bind probability then convert with `Circuit.from_qiskit`. |
| BMW‑QUARK cardinality/circular ansätze | `num_qubits`, `depth` | Uses `RXX` layers; ensure backend support. |
| Bernstein–Vazirani | `num_qubits`, `dynamic`, `hidden_string` | Includes classical bits; custom string optional. |
| CDKM/Draper/VBE adders | `num_qubits`, `kind` | Select adder style as needed. |
| Deutsch–Jozsa | `num_qubits`, `balanced` | Oracle choice affects circuit structure. |
| GHZ / W state | `num_qubits` | Straightforward conversion. |
| Graph state | `num_qubits`, `degree` | Generates random regular graph. |
| Grover | `num_qubits` | Decompose multi‑controlled phase if backend lacks support. |
| HHL | `num_qubits` | Requires ≥3 qubits for phase estimation. |
| HRS arithmetic circuits | `num_qubits` | Check divisibility/odd‑even constraints before generation. |
| QAOA | `num_qubits`, `repetitions`, `seed` | Depth controlled via repetitions; bind parameters. |
| QNN | `num_qubits` | Uses feature map + ansatz. |
| QPE (exact/inexact) | `num_qubits` | Ancilla qubits handled internally. |
| QFT / QFT with GHZ input | `num_qubits` | Standard or entangled input. |
| Quantum Walk | `num_qubits`, `depth` | Multi‑controlled X may need decomposition. |
| Random circuit | `num_qubits` | Depth = 2 × qubits; seeded. |
| Shor | `circuit_size` | Limited preset sizes. |
| Variational ansätze (RealAmplitudes, EfficientSU2, TwoLocal) | `num_qubits`, `reps`, `entanglement` | Bind ansatz parameters as needed. |

QASMBench supplies OpenQASM‑2 files covering small, medium and large circuit
families. Select the file matching the desired qubit count (e.g.
``adder_n10.qasm``). Some families encode depth in the filename, such as
``QAOA_3SAT_N10000_p1``. Load the file with
``QuantumCircuit.from_qasm_file`` and convert via ``Circuit.from_qiskit``;
decompose any gates unsupported by the chosen backend.


See [benchmarks/README.md](benchmarks/README.md) for instructions on running
and extending the benchmark suite.
Raw parameter listings and JSON results generated by the notebooks are stored in
[`benchmarks/results/`](benchmarks/results/) for inclusion in reports and
appendices.
