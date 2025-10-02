# QuASAr Test Suite Usage

This guide explains how to run the benchmarking utilities that ship with the
`test_suite/` package. They generate random Clifford circuits, time the
Tableau-based stabilizer simulation path against statevector baselines, and plot
speedup trends as a function of circuit depth.

## 1. Install dependencies

The suite relies on the following Python packages in addition to QuASAr's base
dependencies:

- [`stim`](https://github.com/quantumlib/Stim) for fast Clifford circuit
  simulation.
- [`qiskit`](https://qiskit.org/) and `qiskit-aer` for statevector benchmarks.
- [`mqt.ddsim`](https://ddsimgate.readthedocs.io/en/latest/) (optional) for the
  decision-diagram baseline.
- [`psutil`](https://psutil.readthedocs.io/) (optional) to report process RSS.
- `matplotlib` and `seaborn` for visualization.

Install them into your active environment:

```bash
pip install stim qiskit qiskit-aer matplotlib seaborn mqt.ddsim psutil
```

> **Note:** If `qiskit-aer` is unavailable the runner automatically falls back to the theoretical statevector estimator. Real execution is strongly recommended for small problem sizes to calibrate the estimator constants.

## 2. Generate cutoff measurements

The cutoff harness searches for the minimum circuit depth where the statevector
backend becomes slower than the Tableau + conversion pipeline by a target
speedup factor. Invoke it from the repository root:

```bash
python test_suite/cutoff_suite.py \
  --out out/cutoff \
  --ns 20 22 24 26 \
  --depth-min 200 \
  --depth-max 100000 \
  --target-speedup 2.0 \
  --sv-timeout-sec 60
```

### Key arguments

- `--out`: Directory where JSON results are written. It is created when needed.
- `--ns`: List of qubit counts to benchmark. Small `n` keeps Aer runnable while
  large depths expose QuASAr's strengths.
- `--depth-min` / `--depth-max`: Bounds for the binary search over circuit depth.
- `--target-speedup`: Desired ratio of statevector time to Tableau + conversion.
- `--sv-timeout-sec`: Wall-clock timeout for Aer jobs. On timeout the harness
  switches to the theoretical estimator and records the trial as a timeout.

Each run logs individual trial measurements and prints a summary line per `n`:

```
[n=24] cutoff≈7344 (target 2.0x). Any SV timeouts among trials? True
```

The JSON payload saved as `<out>/results.json` has the structure:

```json
{
  "params": { ... command arguments ... },
  "cutoffs": { "24": 7344, ... },
  "runs": [
    {
      "n": 24,
      "depth": 4096,
      "tableau_time": 0.19,
      "convert_time": 0.02,
      "sv_time": 0.38,
      "sv_peak_bytes": 402653184,
      "sv_oom": false,
      "sv_timed_out": false,
      "sv_mode": "measured",
      "speedup_vs_sv": 1.82,
      "dd_runtime": 0.17,
      "dd_peak_mem": 58511360,
      "dd_timed_out": false,
      "dd_error": null,
      "dd_mode": "measured",
      "es_runtime": null,
      "es_peak_mem": null,
      "es_timed_out": null,
      "es_error": null,
      "es_mode": null
    },
    ...
  ]
}
```

When the extended-stabilizer backend cannot accept a circuit (for example when
arbitrary rotation gates appear), it records `es_mode: "unsupported"` and keeps
the other ES fields `null`.

### Optional decision-diagram and extended-stabilizer baselines

Enable additional baselines by passing the corresponding flags. Both baselines
run in isolated subprocesses with hard wall-clock timeouts so they are safe to
use inside batch jobs:

```bash
python test_suite/cutoff_suite.py \
  --out out/cutoff_dd_es \
  --ns 20 22 \
  --depth-min 200 \
  --depth-max 20000 \
  --target-speedup 2.0 \
  --sv-timeout-sec 60 \
  --run-dd --dd-timeout-sec 60 \
  --run-es --es-timeout-sec 60
```

The JSON rows then include timing and memory estimates (RSS at completion when
`psutil` is available) for each requested baseline. Timeouts are reported with
`*_timed_out: true`; crashes surface through the `*_error` field.

The `runs` array records every depth that was evaluated during the search so you
can post-process or replay the measurements.

## 3. Plot runtime and speedup curves

Use the plotting helper to visualize the JSON output:

```bash
python test_suite/plots_cutoff.py \
  --results out/cutoff/results.json \
  --out-dir out/cutoff/plots
```

This command produces two figures per qubit count:

- `runtime_n{n}.png`: Log–log runtime comparison between Tableau + conversion
  and the statevector baseline. Red "×" markers highlight trials where the
  statevector simulator hit the timeout and the theoretical estimate was used.
- `speedup_n{n}.png`: Ratio of statevector runtime to Tableau + conversion with
  reference lines at 2×, 3×, and 4× speedups. The vertical dashed line marks the
  estimated cutoff depth.

## 4. Calibrate theoretical estimators (optional)

To obtain realistic statevector numbers, edit the coefficients in
`test_suite/theoretical_baselines.py` after collecting measured runs that do not
hit the timeout. Fit the constants so the estimator matches measured runtimes at
small `n`, then re-run the cutoff search. The conversion time helper follows the
same pattern and can be replaced with your production converter timing routine.

## 5. Next steps

Once you have identified a depth that guarantees the desired speedup, construct
a benchmark circuit with a Clifford prefix at that depth followed by a shallow
non-Clifford window. This keeps the statevector baseline runnable while
showcasing QuASAr's performance advantage in the deep regime.
