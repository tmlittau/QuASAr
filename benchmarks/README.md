# Benchmark utilities

The benchmarking entry points live in :mod:`benchmarks.bench_utils` with a thin
CLI wrapper in :mod:`benchmarks.run_benchmark`.  The helpers provide two
complementary workflows:

* **Showcase execution** – run the curated benchmark circuits across QuASAr and
  the baseline simulators.  Circuits can be targeted individually, via named
  groups, or by running the entire suite.
* **Theoretical estimation** – evaluate the cost model for the same circuits
  without executing the simulators in order to obtain analytical runtime and
  memory predictions.

Both workflows reuse the shared benchmarking utilities so that the behaviour is
identical between the CLI and programmatic usage.

## Installation

Install the project and optional analysis dependencies:

```bash
pip install -e .[test]
pip install pandas jupyter
```

## Running benchmarks

Launch the showcase suite with the default configuration:

```bash
python benchmarks/run_benchmark.py
```

The command iterates over all showcase circuits defined in
``benchmarks/bench_utils/showcase_benchmarks.py`` and writes the raw results,
summaries and figures to ``benchmarks/bench_utils/results/showcase``.  Useful
flags include:

*By default the suite now uses narrower qubit ranges (24–40 qubits for the
clustered and layered families, 16–28 qubits for the classical-control
workloads) and shallower random sections to keep workstation runs under the
timeout budget while preserving the characteristic gate patterns.*

Useful flags include:

* ``--suite <name>`` – execute a preconfigured stitched showcase suite such as
  ``stitched-big``.  Suites expand to a fixed list of circuit factories and
  qubit widths defined in ``benchmarks/bench_utils/stitched_suite.py``.  The
  option is mutually exclusive with ``--circuit`` and ``--group`` so that the
  stitched specification stays consistent.
* ``--circuit <name>`` – run a single circuit (repeat the flag to add more).
* ``--group <name>`` – run every circuit in the named group.
* ``--list-circuits`` / ``--list-groups`` – inspect available names and exit.
* ``--workers <n>`` – limit the number of worker threads (defaults to the
  auto-detected concurrency used by :mod:`benchmarks.bench_utils`).
* ``--qubits name=4:10:2`` – override the qubit widths for a specific circuit.
* ``--enable-classical-simplification`` – enable classical-control
  simplification for all generated circuits.
* ``--estimate`` – generate theoretical estimates after the showcase run.

Legacy automation (including the CI smoke test) previously targeted circuit
families such as ``ghz`` or ``w_state``.  The smoke-test wrapper now maps those
names to the closest showcase circuits so that existing pipelines keep
functioning while the main CLI only advertises the curated showcase set.  Use
``--list-circuits`` to discover the canonical identifiers.

When no circuits or groups are specified the full suite is executed.  Reuse the
``--reuse-existing`` flag to skip recomputation if the CSV files already exist.

### Stitched showcase suites

Use ``--suite`` to launch one of the stitched benchmark suites without needing
to enumerate every circuit manually.  The suites combine representative circuits
from the clustered, layered and classical-control families with wider qubit
widths and longer random sections than the default workstation-friendly ranges.
For example, the stitched-big suite covers nine circuits split across
``stitched_clustered_hybrid`` (40/48/56 qubits),
``stitched_layered_magic_islands`` (28/36/44 qubits) and
``stitched_classical_diag_windows`` (32/40/48 qubits).  The CLI advertises the
available suite names in ``--help`` and accepts the usual overrides:

```bash
python benchmarks/run_benchmark.py --suite stitched-big --workers 16 \
  --reuse-existing
```

Combine ``--suite`` with ``--qubits`` to tailor the widths for individual
entries or with ``--enable-classical-simplification`` to exercise the stitched
classical-control variants.  Programmatic callers can pass ``suite="stitched-big"``
to :func:`benchmarks.run_benchmark.run_showcase_suite` to obtain the same
behaviour.

### Visualising stitched benchmark runs

The helper :mod:`benchmarks.plots_stitched` renders stacked runtime and
peak-memory bars for stitched benchmark suites.  It consumes either the JSON
summary emitted by :mod:`benchmarks.run_benchmark` or the consolidated SQLite
database used by the showcase CLI.  After running a stitched suite, call the
script with the location of the results file:

```bash
python benchmarks/run_benchmark.py --suite stitched-2x --out out/stitched-2x \
  --repeats 3 --choose-best-baseline

python benchmarks/plots_stitched.py \
  --results out/stitched-2x/results.json \
  --out-dir out/stitched-2x/plots \
  --title "QuASAr stitched-2x"
```

When the showcase CLI stores results in the default SQLite database, the plots
can be generated directly from the database file.  The script automatically
selects the most recent benchmark run unless a specific run identifier is
provided via ``--run-id``:

```bash
python benchmarks/plots_stitched.py \
  --database benchmarks/bench_utils/results/benchmarks.sqlite \
  --out-dir out/stitched-2x/plots \
  --title "QuASAr stitched-2x"
```

Useful flags:

* ``--show-all-baselines`` plots every baseline backend discovered in the
  results instead of only the fastest baseline.
* ``--circuit`` restricts the plots to the named circuits (repeat the flag to
  select multiple circuits).
* ``--csv`` writes a compact CSV summary comparing the best baseline against
  QuASAr for each circuit.
* ``--dpi`` adjusts the output resolution.

The figures are saved as ``runtime_by_circuit.png`` and
``memory_by_circuit.png`` inside the requested output directory.

### Programmatic access

The CLI delegates to :func:`benchmarks.run_benchmark.run_showcase_suite` for
automation and testing.  The helper returns a :class:`pandas.DataFrame` with the
same structure as the persisted CSV output:

```python
from benchmarks.run_benchmark import run_showcase_suite

df = run_showcase_suite(
    "classical_controlled",
    widths=(2,),
    repetitions=1,
    workers=1,
    include_baselines=False,  # skip baseline simulators for quick smoke tests
    quick=True,               # use QuASAr's quick-path execution
)
```

Use ``include_baselines=False`` in automation or CI environments to avoid
running the slower baseline simulators.  Combine it with ``quick=True`` to
exercise QuASAr's quick-path execution which greatly reduces runtime while
still validating the planner and scheduler plumbing.

## Theoretical estimation

Add the ``--estimate`` flag to ``run_benchmark.py`` to generate analytical
runtime and memory predictions after executing the benchmarks.  Use
``--estimate-only`` to skip the simulator runs entirely.  The theoretical
estimates reuse the cost-estimator helpers in
``benchmarks/bench_utils/theoretical_estimation_utils.py`` and produce CSV
tables and figures in ``benchmarks/bench_utils/results``.  By default the
estimator analyses the paper circuits, but the CLI now accepts custom
selections:

* ``--estimate-group`` – include one of the predefined estimation groups such as
  ``paper`` or ``showcase`` (repeat to add more).
* ``--estimate-circuit`` – add a custom circuit specification using the format
  ``name[params]:q1,q2``.  Circuit builders come from
  ``benchmarks/bench_utils/circuits.py`` and
  ``benchmarks/bench_utils/large_scale_circuits.py``.  Parameters map to the
  builder keywords, for example ``grover_circuit[n_iterations=2]:20``.
* ``--list-estimate-groups`` / ``--list-estimate-circuits`` – display available
  groups and builders.
* ``--ops-per-second`` – set the conversion factor from model operations to
  seconds (use ``0`` to omit runtime conversion).
* ``--calibration`` – provide a JSON file with calibrated cost coefficients.
* ``--estimate-large-planner`` / ``--no-estimate-large-planner`` – toggle the
  tuned planner configuration that accelerates estimates when either the forced
  or classically simplified circuits are very large.
* ``--estimate-large-threshold`` – gate-count threshold for enabling the tuned
  planner using the larger of the forced and simplified circuits (``0``
  disables the heuristic).
* ``--estimate-large-batch-size`` / ``--estimate-large-horizon`` /
  ``--estimate-large-quick-max-*`` – override the coarse planner parameters used
  once the threshold is exceeded.

When enabled (the default) the runner inspects both the forced and simplified
circuits and, for gate counts above the threshold, instantiates a planner with a
wider batch size, a finite DP horizon and explicit quick-path limits.  This
keeps small and medium-sized circuits on the exhaustive search while ensuring
that highly-clustered showcase variants – even those that only shrink after
classical simplification – complete promptly.

The standalone helper ``benchmarks/bench_utils/estimate_theoretical_requirements.py``
offers the same flags when you only need analytical estimates.  For example:

```bash
python benchmarks/bench_utils/estimate_theoretical_requirements.py \
  --group showcase \
  --circuit qft_circuit:16,20 \
  --circuit "grover_circuit[n_iterations=3]:12"
```

Programmatic access is exposed via
:func:`benchmarks.run_benchmark.generate_theoretical_estimates`, which returns
both the detailed and summary :class:`pandas.DataFrame` objects.

## Quick inspection

For ad-hoc inspection of existing results launch the notebooks in
``benchmarks/notebooks``:

```bash
jupyter notebook benchmarks/notebooks/comparison.ipynb
```

The notebooks expect the CSV artefacts generated by the showcase suite.

