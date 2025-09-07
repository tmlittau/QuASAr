# Benchmark Suite

This directory provides utilities for measuring the performance of the
various QuASAr simulation backends and conversion primitives.

## Installation

The benchmarks require the core project dependencies plus a few optional
packages for data analysis and notebooks:

```bash
pip install -e .[test]
pip install pandas jupyter
```

## Running `run_benchmarks.py`

The `run_benchmarks.py` script executes parameterised circuit families and
records average runtimes:

```bash
python benchmarks/run_benchmarks.py --circuit ghz --qubits 4:12:2 --repetitions 5 --output results/ghz
```

- `--circuit` selects a family from [circuits.py](circuits.py) using the
  `<name>_circuit` naming pattern.
- `--qubits` specifies a `start:end[:step]` range.
- `--repetitions` repeats each configuration to compute a mean and variance.
- `--output` is the base path for the generated `.json` and `.csv` files.

Each benchmark records separate timings for the preparation and execution
phases as well as their sum:

- **prepare_time** – conversion, circuit construction and any backend specific
  compilation that happens before execution.
- **run_time** – execution of the prepared circuit on the backend.
- **total_time** – combined runtime of both phases.
- For QuASAr runs, **backend** – the simulator backend selected by the
  scheduler.



## Comparing with baseline backends

When results for several baseline simulators have been collected alongside
QuASAr timings, the helper functions in [`plot_utils.py`](plot_utils.py) can
highlight how QuASAr compares to the fastest individual backend. The function
`compute_baseline_best` determines the per‑circuit minimum of `run_time_mean`
and `total_time_mean` across all non‑QuASAr frameworks. `plot_quasar_vs_baseline_best`
then overlays this "baseline_best" curve with the QuASAr measurements and can
annotate QuASAr points with the backend chosen by the scheduler:

```python
from benchmarks.plot_utils import plot_quasar_vs_baseline_best

df = runner.dataframe()
plot_quasar_vs_baseline_best(df, annotate_backend=True)
```

## Using notebooks

Benchmark results can be explored with the Jupyter notebooks in
[notebooks/](notebooks):

```bash
jupyter notebook benchmarks/notebooks/comparison.ipynb
```

## Adding circuit families

New circuit generators are added to
[circuits.py](circuits.py).  Implement a function returning a
`Circuit` object and follow the `<name>_circuit` convention so that the
CLI can discover it automatically.  The existing functions, such as
[`ghz_circuit`](circuits.py), illustrate the expected structure.

## Running specific backends

Benchmarks can force a particular simulator without using adapter classes.
Invoke :func:`benchmarks.runner.BenchmarkRunner.run_quasar_multiple` and pass
the desired :class:`quasar.cost.Backend` value:

```python
from benchmarks.runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend

runner = BenchmarkRunner()
engine = SimulationEngine()
rec = runner.run_quasar_multiple(circuit, engine, backend=Backend.STATEVECTOR, repetitions=3)
```

This leverages the scheduler while still collecting timings for a single
backend implementation.  The returned record includes the final simulation
state under the ``result`` key.  Both forced and automatically selected runs
populate this field so that downstream scripts or notebooks can inspect the
produced state.  The state is deliberately omitted from the generated CSV/JSON
files and therefore must be consumed from the in‑memory result.

## Feeding results into the cost model

The JSON results produced by the runner can calibrate
`CostEstimator` coefficients.  Compute per‑gate or per‑primitive times
from the measurements and update the estimator in place:

```python
import json
from quasar import CostEstimator

with open("results/ghz.json") as f:
    data = json.load(f)
coeff = {
    "sv_gate_1q": data[0]["avg_time"] / len(circuit.gates),
    "sv_gate_2q": data[0]["avg_time"] / len(circuit.gates),
    "sv_meas": data[0]["avg_time"] / len(circuit.gates),
}

est = CostEstimator()
est.update_coefficients(coeff)
```

Persist updated coefficients with `est.to_file("coeff.json")` and load
them in future runs via `CostEstimator.from_file`.

## Results and parameters

Each benchmarking notebook concludes with a cell that records the exact
parameters used during execution. The cell saves these parameters and any
available raw results as JSON files in [`benchmarks/results/`](results/).
These artifacts accompany generated plots and can be referenced in reports
or appendices for reproducibility.
