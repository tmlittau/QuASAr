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

Adapters are expected to perform heavy translation work in the preparation
phase so that `run_time` reflects only the actual simulation cost.

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

## Adding backends

Backend adapters live in [backends.py](backends.py).  Subclass
`BackendAdapter` and provide a `name` along with the underlying backend
class:

```python
class MyBackendAdapter(BackendAdapter):
    def __init__(self) -> None:
        super().__init__(name="my_backend", backend_cls=MyBackend)
```

Include the new adapter in the module's `__all__` list so that it can be
imported by `run_benchmarks.py`.

## Feeding results into the cost model

The JSON results produced by the runner can calibrate
`CostEstimator` coefficients.  Compute per‑gate or per‑primitive times
from the measurements and update the estimator in place:

```python
import json
from quasar import CostEstimator

with open("results/ghz.json") as f:
    data = json.load(f)
coeff = {"sv_gate": data[0]["avg_time"] / len(circuit.gates)}

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
