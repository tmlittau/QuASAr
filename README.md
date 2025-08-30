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
``Backend.TABLEAU`` for Clifford circuits).  When omitted, the planner selects a
backend automatically based on estimated cost.

Dense backends are powered by Qiskit Aer and accept a ``method`` argument to
select the underlying simulator implementation.  For example::

    from quasar.backends import StatevectorBackend
    backend = StatevectorBackend(method="density_matrix")

Convenience classes :class:`AerStatevectorBackend` and :class:`AerMPSBackend`
preconfigure the common ``statevector`` and ``matrix_product_state`` methods
respectively.  Custom backend instances can be supplied to :class:`Scheduler`
or :class:`SimulationEngine` via their ``backends`` argument to control the
simulation method.

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
