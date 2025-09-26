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
- `--workers` bounds the number of worker threads used to benchmark qubit
  widths and scenarios in parallel.  When omitted the script auto-detects a
  suitable level of concurrency based on the available CPU cores.
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
available memory. QuASAr now derives this ceiling from the
``QUASAR_STATEVECTOR_MAX_MEMORY_BYTES`` environment variable and, when
``psutil`` is installed, the system's currently available memory, falling back
to 64 GiB (about 32 qubits). The resulting budget is cached by both the
benchmark runner and the :class:`~quasar.simulation_engine.SimulationEngine`,
ensuring the planner enforces the limit even when the quick path is bypassed.
You can override the ceiling via the environment variable, the global
``--memory-bytes`` flag on `run_benchmarks.py`, the ``memory_bytes`` keyword on
``BenchmarkRunner.run_quasar``/``run_quasar_multiple``, or the
``SimulationEngine(memory_threshold=...)`` and ``simulate(memory_threshold=...)``
APIs. Supplying a non-positive threshold disables the guard for that call.

### Showcase benchmark suites

The following circuit families were added to highlight situations where QuASAr
delivers large wins over single-method simulators. Each generator exposes rich
metadata on the returned :class:`~quasar.circuit.Circuit` objects under the
``metadata`` attribute so downstream scripts can recover the layer structure,
transition depths and control layouts.

| Circuit family | Key characteristics | When to use |
| -------------- | ------------------- | ----------- |
| `clustered_ghz_random_circuit`, `clustered_w_random_circuit` | 50-qubit (configurable) circuits that prepare GHZ or W states on disjoint blocks of five qubits before applying ~1000 layers of random hybrid gates. The metadata records the block size, number of blocks, and random-layer offsets. | Stress QuASAr's ability to keep subsystems independent during deep hybrid evolution. |
| `clustered_ghz_qft_circuit`, `clustered_w_qft_circuit`, `clustered_ghz_random_qft_circuit` | Reuse the clustered preparation but follow it with a global QFT or a QFT after the random evolution. | Demonstrate QuASAr's ability to switch between sparse subsystem handling and dense global transforms. |
| `layered_clifford_delayed_magic_circuit`, `layered_clifford_midpoint_circuit` | Depth-5000 workloads with configurable Clifford-only prefixes (80% and 60% respectively) before transitioning to non-Clifford layers. | Measure QuASAr's planning decisions when only part of the circuit demands non-Clifford simulation techniques. |
| `layered_clifford_ramp_circuit` | Similar to the layered transition circuits but gradually increases the non-Clifford density between two fractions. Metadata lists the per-layer Clifford/non-Clifford flag. | Evaluate how quickly QuASAr reacts to gradual changes in gate character. |
| `classical_controlled_circuit`, `dynamic_classical_control_circuit`, `classical_controlled_fanout_circuit` | 28-qubit (configurable) circuits that initialise subsets of qubits into classical basis states and reuse them as controls across thousands of layers. `dynamic_classical_control_circuit` flips controls frequently while `classical_controlled_fanout_circuit` increases fan-out. The generators default to `use_classical_simplification=False` so benchmarks can toggle the optimisation on demand. | Quantify the savings from QuASAr's classical-control simplification pass and stress the planner's handling of mixed classical/quantum regions. |

All of the showcase circuits adhere to the `<name>_circuit` naming convention
and are available through the benchmarking CLI via `--circuit <name>`. They use
fixed seeds by default to keep gate patterns reproducible; override the seed
argument when exploring stochastic behaviour.

### Showcase benchmark runner

The [`showcase_benchmarks.py`](showcase_benchmarks.py) helper executes the
showcase circuits on QuASAr as well as the baseline backends, mirroring the
timeout handling used for the paper figures.  It exports raw measurements,
per-circuit summaries, derived speedup tables and comparative figures under
`benchmarks/results/showcase/` and `benchmarks/figures/showcase/`:

```bash
python benchmarks/showcase_benchmarks.py --repetitions 3 --run-timeout 900
```

Use `--circuits` to select a subset of workloads, `--qubits` to override the
default width selections (e.g. `--qubits clustered_ghz_random=40:60:10`) and
`--reuse-existing` to skip rerunning configurations with cached results.
Pass `--workers <n>` to control how many threads execute circuit widths in
parallel; omit the flag to let the runner auto-detect a sensible default.

### Theoretical cost estimates

When empirical benchmarking is too time-consuming you can approximate the
runtime and memory requirements of the paper circuits via
[`estimate_theoretical_requirements.py`](estimate_theoretical_requirements.py).
The helper relies purely on QuASAr's static cost model and therefore finishes
within seconds even for large workloads:

```bash
python benchmarks/estimate_theoretical_requirements.py --workers 8
```

The script produces two CSV tables under `benchmarks/results/` and companion
bar charts in `benchmarks/figures/` contrasting QuASAr with the best single
backend.  Use `--ops-per-second` to supply a custom throughput for converting
cost-model operations into wall-clock seconds and `--calibration` to point at a
specific coefficient file.

#### Benchmark hardware and throughput reference

All empirical experiments and theoretical projections referenced in this
directory were produced on a workstation equipped with:

- **CPU:** Intel i9-13900K
- **GPU:** NVIDIA RTX 4090
- **Memory:** 64 GB RAM

The theoretical estimator uses a default throughput of `1e9` primitive
operations per second when converting cost-model operation counts into
wall-clock time.  This constant mirrors the sustained performance observed for
QuASAr's GPU-accelerated simulator on the workstation above; pass a different
value via `--ops-per-second` to reflect other hardware profiles.

### Reproducing paper figures

Execute the commands below in order to rebuild every artefact used by
[`paper_figures.py`](paper_figures.py). Check that each step produces the
described files before moving to the next command.
The `--workers` flag mirrors the benchmark runner and enables threaded circuit
execution for the forced/automatic comparisons.

1. **Partitioning sweeps** – regenerate the benchmark tables that back the
   mainline runtime and memory figures:

   ```bash
   python benchmarks/run_benchmarks.py --scenario tableau_boundary --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/tableau_boundary
   python benchmarks/run_benchmarks.py --scenario staged_rank --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/staged_rank
   python benchmarks/run_benchmarks.py --scenario staged_sparsity --repetitions 1 --memory-bytes 268435456 --output benchmarks/results/staged_sparsity
   ```

   Each run emits `<scenario>.csv`, `<scenario>.json`, and
   `<scenario>_summary.{csv,md}` under `benchmarks/results/`. These files are
   deterministic because the generators fix their random seeds.

2. **Quick-analysis benchmarking** – measure planner speedups for the relative
   speedup bar chart:

   ```bash
   python benchmarks/quick_analysis_benchmark.py
   ```

   This script saves `benchmarks/quick_analysis_results.csv` and displays a
   diagnostic plot. Without this CSV the final stage skips the
   `relative_speedups.{png,pdf,csv}` outputs.

3. **Optional planner heatmap** – capture the plan-choice sweep used for the
   heatmap figure by executing the
   [`notebooks/plan_choice_heatmap.ipynb`](notebooks/plan_choice_heatmap.ipynb)
   notebook. One way to run it non-interactively is:

   ```bash
   jupyter nbconvert --to notebook --execute benchmarks/notebooks/plan_choice_heatmap.ipynb --output plan_choice_heatmap_executed.ipynb
   ```

   The notebook writes `benchmarks/results/plan_choice_heatmap_results.json`
   and `plan_choice_heatmap_params.json`. If these artefacts are absent the
   heatmap (`plan_choice_heatmap.{png,pdf}` plus `plan_choice_heatmap_table.csv`)
   will be omitted from the final outputs.

4. **Generate paper figures** – once all required data are present, produce the
   publication graphics and tables:

   ```bash
   python benchmarks/paper_figures.py
   ```

   The script collects the benchmark summaries into
   `benchmarks/results/backend_{forced,auto}.csv`, saves derived tables such as
   `backend_vs_baseline_speedups.csv`, and exports publication-ready
   `*.png`/`*.pdf` files to `benchmarks/figures/`. Figures that rely on missing
   prerequisites—such as the planner heatmap or quick-analysis speedup bars—are
   skipped automatically so you can rerun the earlier steps and execute this
   command again.

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

Use ``--repetitions`` to change the number of samples per circuit/backend pair,
``--run-timeout`` to cap the runtime of individual simulations (default: 600
seconds per run, configurable via ``RUN_TIMEOUT_DEFAULT_SECONDS`` in
``paper_figures.py``; pass ``0`` to disable), and
``--reuse-existing`` to filter previously recorded CSVs when the raw data does
not need to be regenerated.  The script writes publication-ready PNG/PDF pairs
to [`benchmarks/figures/`](figures/) and stores the tabular data, including
per-circuit speedups, in [`benchmarks/results/`](results/).  Timeseries and
heatmap plots require the optional `seaborn` dependency; when it is not
available the corresponding artefacts are skipped with a warning. Generated
figures are ignored by Git so that repositories do not accumulate large binary
artefacts; rerun the script whenever you need fresh images. The CSV outputs
remain versioned to provide reproducible numeric references for the paper.

``paper_figures.py`` exercises every backend on large, non-Clifford circuits and
can take *days* when dense statevector simulations are forced (e.g., Grover at
24 qubits). Use [`estimate_paper_figures.py`](estimate_paper_figures.py) to
approximate the runtime and memory footprint before launching the full sweep:

```bash
python benchmarks/estimate_paper_figures.py --ops-per-second 5e8
```

The estimator reuses the cost model from the planner to report per-run operation
counts, approximate wall-clock time (based on the supplied throughput) and peak
memory for every circuit/backend combination.  It highlights unsupported pairs
and gives a rough upper bound for the automatic QuASAr schedule so you can spot
unreasonable workloads ahead of time.

### High-qubit workloads derived from the partitioning notebooks

`paper_figures.py` now includes three circuit families that mirror the
partitioning notebooks.  Each entry specifies its parameter sweep directly in
`CIRCUITS` so the figures can be regenerated without consulting the notebooks.
The configurations below assume the default repetition count of three runs per
point and should be combined with `--memory-bytes` limits that keep statevector
simulations feasible on the target hardware.  The script trims each sweep to
widths supported by :func:`benchmarks.memory_utils.max_qubits_statevector`
based on the smaller of ``QUASAR_STATEVECTOR_MAX_MEMORY_BYTES`` and the detected
available memory.  A 25% headroom is reserved to keep the host responsive.  With
the 64 GiB default the cap becomes 29 qubits on a machine with at least that
much RAM, so the predefined sweeps top out at 28 qubits until you raise the
environment override.

#### GHZ ladder partitions

- **Builder** – `parallel_circuits.many_ghz_subsystems` via the
  `ghz_ladder` entry in `paper_figures.py`.
- **Qubit counts** – `20, 24, 28` by default.  With the fixed `group_size=4`
  this sweeps five through seven independent ladders.  Increase
  ``QUASAR_STATEVECTOR_MAX_MEMORY_BYTES`` to reintroduce wider ladders on
  machines with additional RAM.
- **Knobs** – `group_size` may be increased to stress wider ladders provided the
  qubit counts remain multiples of the chosen value.  Keep
  `use_classical_simplification=False` so the tableau backend handles each
  ladder before non-Clifford gates appear.
- **Resources** – Tableau and MPS runs remain lightweight, but forcing the
  statevector backend at the 28-qubit default consumes roughly 32 GiB.  Raising
  the memory budget above 64 GiB allows the quick-path sweeps to include
  32-qubit ladders again.  Disable the quick path when measuring plan cache
  warm-up so partition reuse is visible.

#### Random Clifford+T hybrids

- **Builder** – `random_hybrid_circuit` via the `random_clifford_t` entry.
- **Qubit counts** – `20, 24, 28` with a depth of `3 × n_qubits` per instance
  at the default memory limit.  Larger widths appear automatically once the
  statevector budget exceeds their amplitude footprint.
- **Knobs** – The deterministic seed is `97 + n_qubits`; adjust the
  `depth_multiplier` to scale circuit depth and raise `base_seed` if multiple
  independent sweeps are required.
- **Resources** – Expect dense partitions and substantial T-counts.  Allocate at
  least 96 GiB when forcing statevector backends beyond 28 qubits or cap the run
  with ``--memory-bytes`` to skip infeasible baselines.  Adaptive QuASAr runs
  favour MPS/DD mixes.

#### Larger Grover instances

- **Builder** – `grover_circuit` via the `grover_large` entry.
- **Qubit counts** – `20, 24, 28` with two Grover iterations per problem size
  when using the default budget.  Additional sizes become available as soon as
  the memory ceiling allows their statevector realisations.
- **Knobs** – Increase the `iterations` keyword to probe longer amplitude
  amplification phases.  All runs keep `use_classical_simplification=True` in
  automatic mode so the optimiser prunes redundant Clifford layers.
- **Resources** – Forcing a statevector backend at 28 qubits requires roughly
  32 GiB.  Increase the budget beyond 64 GiB to benchmark the 32-qubit variant.
  Disable the quick path when contrasting plan cache behaviour to ensure the
  planner runs on every configuration.

### Scheduled experiments

To extend the partitioning study, schedule the following batches once the new
workloads have been generated:

- Random-hybrid circuits at 24 and 28 qubits using adaptive backend selection to
  contrast QuASAr with the best fixed-method baseline.  Increase the memory
  override if you need the 32-qubit point for comparison.
- Surface-corrected QAOA at 24 and 28 qubits comparing QuASAr versus runs
  forced onto pure MPS and pure statevector backends.
- GHZₙ and Groverₙ for `n ≥ 6` with the plan cache warm-up measured while the
  quick path is disabled so cache misses are observable.

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
