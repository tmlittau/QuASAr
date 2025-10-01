# Calibration

The cost estimator used by QuASAr relies on a set of coefficients that
scale simple analytical models to the actual performance of the machine
where conversions are executed.  The values can be tuned using small
micro‑benchmarks bundled with the project.

The matrix product state (MPS) calibration distinguishes between costs
for single- and two-qubit gates and an additional optional
singular-value-decomposition (SVD) truncation step.  Separate
coefficients for these components are produced by the calibration
script.  Small benchmark circuits also record fixed runtime and memory
overheads for the statevector and MPS simulators, exposed as
``sv_base_time``, ``sv_base_mem``, ``mps_base_time`` and ``mps_base_mem``.

## Default coefficients

Baseline coefficients are derived from operation counts and memory
requirements reported in established simulators and textbooks:

- **Statevector** numbers follow QuEST's flop counts for unitary
  updates (Jones et al., 2019) and the Qiskit Aer performance guide for
  memory overheads.
- **Tableau** costs are based on the Aaronson & Gottesman (2004)
  stabilizer formalism, which stores ``2n^2`` bits and applies roughly
  ``2n^2`` bit operations per Clifford gate.
- **Matrix product state** scaling uses the expressions from
  Schollwöck's review of the density matrix renormalisation group
  (2011) with each tensor element taking 16 bytes.
- **Decision diagram** parameters originate from the QMDD analysis by
  Zulehner & Wille (2019), assuming 32-byte nodes and a 20% unique table
  cache overhead.

These defaults provide reasonable first-order estimates and can be
further refined with the calibration workflow below.

## Conversion overhead

Backend switches incur additional fixed and ingestion costs.  The default
coefficients model per-amplitude transfer times of `5.0` for statevector
ingestion, `3.0` for tableau, `4.0` for MPS and `2.0` for decision
diagram backends.  Ingestion memory overheads are set to zero by default.
A separate `conversion_base` value of `5.0` adds a constant penalty to
every backend transition.  These parameters can be measured and adjusted
using the calibration utilities in this project to better match the
characteristics of the target hardware.

## Running the benchmarks

Run the benchmarking utility in ``tools/`` to measure the cost of the
supported simulation backends and conversion primitives:

```bash
python tools/benchmark_coefficients.py
```

Each invocation writes a new versioned JSON file under the top-level
``calibration/`` directory, e.g. ``calibration/coeff_v1.json``.  The
latest available file is loaded automatically by
:class:`~quasar.cost.CostEstimator` when instantiated with default
arguments.  Calibration outputs now bundle provenance metadata alongside
the coefficients to make measurements auditable and reproducible.

Existing estimators can be updated in place using the calibration
helpers:

```python
from quasar import CostEstimator, latest_coefficients, apply_calibration

estimator = CostEstimator()
record = latest_coefficients()
if record:
    apply_calibration(estimator, record)
    print("Loaded calibration from", record["metadata"].get("date", "unknown"))
```

These utilities allow coefficients to be tuned once and reused across
runs on the same hardware.

## Calibration provenance

Calibration files follow a structured schema::

    {
      "coeff": {
        "sv_gate_1q": 0.000123,
        "mps_mem": 8.0,
        ...
      },
      "variance": {
        "sv_gate_1q": 1.2e-10,
        "mps_mem": 0.0,
        ...
      },
      "metadata": {
        "hardware": {
          "node": "quasar-dev",
          "processor": "11th Gen Intel(R) Core(TM) i7-1185G7",
          "system": "Linux",
          "python": "3.11.7",
          ...
        },
        "date": "2024-03-17T12:04:55Z",
        "software_commit": "e3f3c0d6f5d54bcba30d1d756f81a1e7c2c92c3e"
      }
    }

- ``coeff`` stores the mean value recorded across repeated benchmark
  samples for each coefficient.
- ``variance`` captures the (population) variance of the observed
  samples.  Coefficients with deterministic microbenchmarks naturally
  show a variance of ``0.0``.
- ``metadata`` aggregates identifying information to ensure future runs
  can be compared on the same hardware and software revision.  The
  ``hardware`` block reports lightweight identifiers such as the host
  name, CPU model and Python runtime, ``date`` is written in UTC using
  ISO-8601 format and ``software_commit`` mirrors the repository commit
  hash that produced the calibration.

The provenance block is optional when loading historical calibration
files that pre-date this schema; missing fields simply fall back to
empty dictionaries.  Downstream utilities that only require the numeric
coefficients can continue to call :func:`quasar.calibration.apply_calibration`
with the complete record, which extracts the ``coeff`` mapping
automatically.  Consumers that wish to inspect the measurement quality
or trace calibration lineage should consult the ``variance`` and
``metadata`` entries directly.
