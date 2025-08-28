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
from quasar import Circuit, SimulationEngine

engine = SimulationEngine()
result = engine.simulate(Circuit([{"gate": "H", "qubits": [0]}]))
for part in result.ssd.partitions:
    state = result.ssd.extract_state(part)
    print(part.backend, type(state))
```

Depending on the backend, ``state`` may be a dense NumPy vector, a list of MPS
tensors, a ``stim.Tableau`` or a decision diagram node.

