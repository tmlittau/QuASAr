# Calibration

The cost estimator used by QuASAr relies on a set of coefficients that
scale simple analytical models to the actual performance of the machine
where conversions are executed.  The values can be tuned using small
micro‑benchmarks bundled with the project.

The matrix product state (MPS) calibration distinguishes between costs
for single- and two-qubit gates and an additional optional
singular-value-decomposition (SVD) truncation step.  Separate
coefficients for these components are produced by the calibration
script.

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
