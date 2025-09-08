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

Run the calibration script to measure the cost of the supported
simulation backends and conversion primitives:

```bash
python -m quasar.calibration --output coeff.json
```

A JSON file with the measured coefficients will be written to
`coeff.json`.  The file can be supplied to :class:`~quasar.cost.CostEstimator`
when creating an estimator instance:

```python
from quasar import CostEstimator
est = CostEstimator.from_file("coeff.json")
```

Existing estimators can be updated in place using the measured values:

```python
from quasar import run_calibration

coeff = run_calibration()
estimator = CostEstimator()
estimator.update_coefficients(coeff)
```

To persist coefficients from an existing estimator:

```python
estimator.to_file("coeff.json")
```

These utilities allow coefficients to be tuned once and reused across
runs on the same hardware.
