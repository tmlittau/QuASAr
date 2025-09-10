# Cost model

QuASAr predicts runtime and memory for each backend using analytical
expressions scaled by calibration coefficients. The coefficients encode
constant factors measured on real hardware while the equations capture
asymptotic behaviour. The sections below list the equations and the
meaning of each tunable parameter.

## Statevector simulation

For a register of ``n`` qubits the estimator uses

\[
T = c_{1q} N_{1q} + c_{2q} N_{2q} + c_m N_m, \qquad
M = c_{bpa} 2^n b_{amp}
\]

where ``N_{1q}``, ``N_{2q}`` and ``N_m`` are the counts of single- and
two-qubit gates and measurements, and ``b_{amp}`` is the raw bytes per
amplitude (``16`` for ``complex128``).

| Coefficient | Meaning |
|-------------|---------|
| ``sv_gate_1q`` | Cost per single-qubit gate (~12 FLOPs per amplitude) [^quest]. |
| ``sv_gate_2q`` | Cost per two-qubit gate (~80 FLOPs per amplitude) [^quest]. |
| ``sv_meas`` | Measurement overhead per amplitude. |
| ``sv_bytes_per_amp`` | Extra memory beyond raw amplitude storage [^aer]. |

## Stabilizer tableau

Clifford-only circuits use the Aaronson–Gottesman tableau formalism. For
``n`` qubits and ``N`` Clifford gates:

\[
T = c_{tab} N n^2, \qquad
M = c_{tab\_mem} n^2 + c_{phase} (2n) + c_{meas} N_m
\]

| Coefficient | Meaning |
|-------------|---------|
| ``tab_gate`` | Per-gate bit operations (≈2``n^2``) [^ag04]. |
| ``tab_mem`` | Bytes per ``n^2`` tableau elements. |
| ``tab_phase_mem`` | Phase bit storage per row. |
| ``tab_meas_mem`` | Memory per recorded measurement. |

## Matrix product state

For an ``n``-qubit chain with site costs ``l_i r_i`` and bond dimensions
``χ_i`` the estimator uses

\[
T = c_{1q} N_{1q} \sum l_i r_i + c_{2q} N_{2q} n \overline{χ_i l_i r_i} +
    c_{trunc} N_{2q} n τ,
\]
\[
M = c_{mem} \sum l_i r_i + c_{tmp} \max χ_i l_i r_i
\]

| Coefficient | Meaning |
|-------------|---------|
| ``mps_gate_1q`` | Single-qubit tensor updates (4``χ^2`` multiplies) [^scholl]. |
| ``mps_gate_2q`` | Two-qubit tensor updates (16``χ^3`` operations) [^scholl]. |
| ``mps_trunc`` | Optional SVD truncation ~32``χ^3\log χ``. |
| ``mps_mem`` | Bytes per tensor element (``16`` for ``complex128``). |
| ``mps_temp_mem`` | Temporary workspace for SVD. |

## Decision diagrams

Decision diagram (QMDD) simulation scales with the active node count
``r``:

\[
T = c_{dd\_gate} N r, \qquad
M = c_{dd\_mem} r b_{node} (1 + c_{cache})
\]

| Coefficient | Meaning |
|-------------|---------|
| ``dd_gate`` | Node operations per gate [^qmdd]. |
| ``dd_mem`` | Memory multiplier for nodes and cache. |
| ``dd_node_bytes`` | Bytes per QMDD node (four edges + terminal index). |
| ``dd_cache_overhead`` | Fractional unique-table cache overhead. |

## Conversion and ingestion

Switching backends adds a fixed cost ``conversion_base`` and a
per-amplitude ingestion term ``ingest_*``. Conversion primitives use
polynomials in the SSD parameters:

| Coefficient | Meaning |
|-------------|---------|
| ``b2b_svd``, ``b2b_copy`` | Boundary-to-boundary SVD and copying time. |
| ``b2b_svd_mem`` | Temporary memory for the B2B SVD. |
| ``lw_extract``, ``lw_temp_mem`` | Local window extraction time and extra memory. |
| ``st_stage`` | Staged conversion time with bond cap ``st_chi_cap``. |
| ``full_extract`` | Full extraction time. |
| ``ingest_sv`` etc. | Per-amplitude ingestion cost for each backend; ``ingest_*_mem`` controls extra memory. |
| ``conversion_base`` | Fixed overhead added to every backend transition. |

## Overriding coefficients

Coefficients can be customised in two ways:

1. **Configuration files** – run the calibration benchmarks and load the
   resulting JSON file:

   ```python
   from quasar import CostEstimator
   est = CostEstimator.from_file("coeff.json")
   ```

2. **Environment variables** – applications may point to a coefficient
   file via an environment variable and load it explicitly:

   ```python
   import os
   from quasar import CostEstimator

   coeff_path = os.getenv("QUASAR_COEFF_FILE")
   est = CostEstimator.from_file(coeff_path) if coeff_path else CostEstimator()
   ```

Existing estimators also expose
``est.update_coefficients({"sv_gate_1q": 0.9}, decay=0.2)`` for fine-grained
adjustments at runtime.  The ``decay`` parameter applies an exponential moving
average and defaults to the value of ``QUASAR_COEFF_EMA_DECAY``.

## References

[^quest]: Jonathan A. Jones *et al.*, "QuEST and High Performance Simulation of Quantum Computers", 2019. [arXiv:1904.06343](https://arxiv.org/abs/1904.06343)
[^aer]: *Qiskit Aer performance guide*. [https://docs.quantum.ibm.com/api/qiskit/aer#performance](https://docs.quantum.ibm.com/api/qiskit/aer#performance)
[^ag04]: Scott Aaronson and Daniel Gottesman, "Improved Simulation of Stabilizer Circuits", 2004. [arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)
[^scholl]: Ulrich Schollwöck, "The density-matrix renormalization group in the age of matrix product states", 2011. [doi:10.1016/j.aop.2010.09.012](https://doi.org/10.1016/j.aop.2010.09.012)
[^qmdd]: Robert Wille and Jonas Zulehner, "Advanced Simulation of Quantum Computations", 2019. [arXiv:1801.00112](https://arxiv.org/abs/1801.00112)
