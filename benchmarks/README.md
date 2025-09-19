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

The `run_benchmarks.py` script executes parameterised circuit families on
QuASAr and all available single‑method simulators. For each configuration the
fastest non‑QuASAr backend is determined and only this aggregate baseline is
stored alongside the QuASAr result:

```bash
python benchmarks/run_benchmarks.py --circuit ghz --qubits 4:12:2 --repetitions 5 --output benchmarks/results/ghz
```

- `--circuit` selects a family from [circuits.py](circuits.py) using the
  `<name>_circuit` naming pattern.
- `--qubits` specifies a `start:end[:step]` range.
- `--repetitions` repeats each configuration to compute a mean and variance.
- `--output` is the base path for the generated `.json` and `.csv` files.
- Circuit families composed solely of Clifford gates (`H`, `S`, `CX`, `CZ`, etc.)
  are skipped to avoid benchmarking workloads that are trivial for stabiliser
  simulators.

The new `--scenario` flag drives the partitioning experiments derived from the
`docs/partitioning_*.ipynb` notebooks.  Each scenario enumerates a deterministic
parameter sweep defined in [`partitioning_workloads.py`](partitioning_workloads.py)
and maps the settings onto the large-scale circuit generators.  For example,
the boundary-width sweep is executed with:

```bash
python benchmarks/run_benchmarks.py --scenario tableau_boundary --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/tableau_boundary
```

Scenario runs emit two sets of artefacts: the raw measurement table
(`tableau_boundary.csv`/`.json`) and an aggregated summary table in CSV and
Markdown form (`tableau_boundary_summary.csv`/`.md`).  The Markdown output is
formatted for the paper and lists QuASAr/backbone runtimes, memory usage,
conversion counts and the dominant conversion primitives.

Each benchmark records separate timings for the preparation and execution
phases as well as their sum:

- **prepare_time** – conversion, circuit construction and any backend specific
  compilation that happens before execution.
- **run_time** – execution of the prepared circuit on the backend.
- **total_time** – combined runtime of both phases.
- **backend** – for QuASAr rows, the simulator chosen by the scheduler; for
  baseline entries, the backend that achieved the minimum runtime.

Statevector simulations are skipped when the circuit width exceeds the
available memory. A default budget of 64 GiB (about 32 qubits) is assumed but
can be adjusted via the ``QUASAR_STATEVECTOR_MAX_MEMORY_BYTES`` environment
variable, the global ``--memory-bytes`` flag on `run_benchmarks.py`, or by
passing ``memory_bytes`` to ``BenchmarkRunner.run_quasar``/``run_quasar_multiple``.

To regenerate the partitioning sweeps used in the paper, execute the three
scenarios with identical parameters so that QuASAr and the baseline backends
share the same 256 MiB budget and single-shot timing window:

```bash
python benchmarks/run_benchmarks.py --scenario tableau_boundary --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/tableau_boundary
python benchmarks/run_benchmarks.py --scenario staged_rank --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/staged_rank
python benchmarks/run_benchmarks.py --scenario staged_sparsity --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/staged_sparsity
```

Each invocation writes CSV/JSON pairs plus a Markdown summary directly under
`benchmarks/results/`.  The tables are deterministic because the circuit
generators fix their random seeds.

### Timing semantics

The recorded times capture only backend execution and the minimal preparation
steps that occur after a plan has been produced. Circuit analysis, planning,
and method selection are completed before timing begins for both forced and
automatically selected runs.

When a backend is forced, the scheduler honours the requested simulator but
the measurement window remains the same—it excludes planning and records just
the chosen backend's execution. In automatic mode, the scheduler picks a
backend prior to the timed region.

Quick‑path shortcuts (`quick=True` or the `QUASAR_QUICK_*` environment
variables) bypass planning entirely. In auto‑selection mode the scheduler uses
simple heuristics; in forced mode the specified backend runs immediately. These
shortcuts reduce overall wall‑clock time but the reported `run_time` still
reflects only the backend's execution.



## Comparing with baseline backends

`run_benchmarks.py` already aggregates the baseline measurements into a single
"baseline_best" curve. The helper functions in
[`plot_utils.py`](plot_utils.py) centralise the styling used throughout the
paper and expose dedicated entry points for the common notebook figures:

```python
from benchmarks.plot_utils import (
    setup_benchmark_style,
    plot_quasar_vs_baseline_best,
    plot_backend_timeseries,
)

setup_benchmark_style()
ax, speedups = plot_quasar_vs_baseline_best(
    df,
    annotate_backend=True,
    return_table=True,
    show_speedup_table=True,
)
print(speedups)

forced = df[df["mode"] == "forced"]
auto = df[df["mode"] == "auto"]
plot_backend_timeseries(forced, auto, metric="run_time_mean")
```

The module also provides `plot_metric_trend`, `plot_heatmap` and
`plot_speedup_bars` for the ancillary figures used in the appendix. All
functions share a palette that maps QuASAr backends to consistent colours and
markers and they automatically annotate selected backends with short labels.

### Regenerating paper figures

Run [`paper_figures.py`](paper_figures.py) after collecting benchmark data to
generate the publication figures and their CSV summaries:

```bash
python benchmarks/paper_figures.py
```

The script requires `seaborn` in addition to the core dependencies. It writes
publication-ready PNG/PDF pairs to [`benchmarks/figures/`](figures/) and stores
the tabular data, including per-circuit speedups, in
[`benchmarks/results/`](results/). Generated figures are ignored by Git so that
repositories do not accumulate large binary artefacts; rerun the script whenever
you need fresh images. The CSV outputs remain versioned to provide reproducible
numeric references for the paper.

## Using notebooks

Benchmark results can be explored with the Jupyter notebooks in
[notebooks/](notebooks):

```bash
jupyter notebook benchmarks/notebooks/comparison.ipynb
```

- `partitioning_workloads_results.ipynb` regenerates the deterministic
  partitioning sweeps, persists the CSV/JSON artefacts, and produces the paper
  figures inline for quick inspection.

## Adding circuit families

New circuit generators are added to
[circuits.py](circuits.py).  Implement a function returning a
`Circuit` object and follow the `<name>_circuit` convention so that the
CLI can discover it automatically.  The existing functions, such as
[`ghz_circuit`](circuits.py), illustrate the expected structure.

## Parallel subsystem templates

[`parallel_circuits.py`](parallel_circuits.py) contains helpers for
benchmarks that emphasise independent subsystems.  The
``many_ghz_subsystems(num_groups, group_size)`` generator creates
``num_groups`` disjoint GHZ chains with ``group_size`` qubits each.  No
gates cross subsystem boundaries, allowing the planner and scheduler to
identify parallel partitions immediately:

```python
from benchmarks.parallel_circuits import many_ghz_subsystems

circuit = many_ghz_subsystems(num_groups=8, group_size=6)
```

This circuit family is useful when measuring the benefits of QuASAr's
parallel execution heuristics or validating partitioning behaviour on
larger disjoint systems.

## Running specific backends

Benchmarks can force a particular simulator by selecting a QuASAr backend
directly.  Invoke :func:`benchmarks.runner.BenchmarkRunner.run_quasar_multiple`
and pass the desired :class:`quasar.cost.Backend` value (for example,
``Backend.STATEVECTOR`` or ``Backend.TABLEAU``).  Set ``quick=True`` to bypass
planning and execute immediately on the chosen backend:

```python
from benchmarks.runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend

runner = BenchmarkRunner()
engine = SimulationEngine()
rec = runner.run_quasar_multiple(
    circuit,
    engine,
    backend=Backend.STATEVECTOR,
    repetitions=3,
    quick=True,
)
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
