# Cost model

QuASAr predicts runtime and memory for each backend using analytical
expressions scaled by calibration coefficients. The revised model
captures the effects of gate mix, sparsity, frontier width, entanglement
entropy and rotation diversity. Constants are derived from published
complexity analyses together with large-scale HPC benchmarks reported for
QuEST, qFlex, cuStateVec, cuTensorNet and GPU-accelerated QMDD solvers
[^quest][^qflex][^custatevec][^cutensornet][^qmddbench].

The sections below list the equations and the meaning of each tunable
parameter.

## Statevector simulation

For a register of ``n`` qubits the estimator uses

\[
T = \bigl(c_{1q} N_{1q} + c_{2q} N_{2q} + c_m N_m\bigr)
    \Bigl(1 + \alpha_{mix} \rho + \alpha_{rot} \delta + \alpha_{ent} H
    - \alpha_{s} S\Bigr) + T_0,
\]
\[
M = M_0 + c_{bpa} 2^n b_{amp}
    \Bigl(1 + \beta_{rot} \delta + \beta_{ent} H\Bigr)
\]

where ``ρ`` is the two-qubit gate ratio, ``δ`` the rotation diversity,
``H`` the normalised entanglement entropy, ``S`` the sparsity estimate and
``b_{amp}`` the raw bytes per amplitude (``16`` for ``complex128``). The
baseline coefficients come from QuEST's FLOP counts while the modifiers
are fitted to cuStateVec and qFlex throughput measurements.

| Coefficient | Meaning |
|-------------|---------|
| ``sv_gate_1q`` | Cost per single-qubit gate (~12 FLOPs per amplitude) [^quest]. |
| ``sv_gate_2q`` | Cost per two-qubit gate (~80 FLOPs per amplitude) [^quest]. |
| ``sv_meas`` | Measurement overhead per amplitude. |
| ``sv_bytes_per_amp`` | Extra memory beyond raw amplitude storage [^aer]. |
| ``sv_two_qubit_weight`` | Runtime penalty for two-qubit dominated mixes informed by qFlex [^qflex]. |
| ``sv_rotation_weight`` | Slowdown from diverse rotation angles observed in cuStateVec tuning [^custatevec]. |
| ``sv_entropy_weight`` | Cache pressure from high entanglement entropy. |
| ``sv_sparsity_discount`` | Small speed-up for sparse kernels due to skipped fusion. |
| ``sv_memory_*`` | Memory expansion driven by rotation diversity and entropy. |

## Stabilizer tableau

Clifford-only circuits use the Aaronson–Gottesman tableau formalism. For
``n`` qubits and ``N`` Clifford gates the adaptive model is

\[
T = c_{tab} N n^2 \Bigl(1 + \gamma_{mix} \rho + \gamma_{depth} D
    + \gamma_{rot} \delta\Bigr),
\]
\[
M = c_{tab\_mem} n^2 \Bigl(1 + \gamma_{mix} \rho + \gamma_{depth} D
    + \gamma_{rot} \delta\Bigr) + c_{phase} (2n) + c_{meas} N_m.
\]

``D`` denotes the depth estimate ``N / n``. Rotation diversity models
the overhead of temporarily injected non-Clifford phases before
stabiliser reduction.

| Coefficient | Meaning |
|-------------|---------|
| ``tab_gate`` | Per-gate bit operations (≈2``n^2``) [^ag04]. |
| ``tab_mem`` | Bytes per ``n^2`` tableau elements. |
| ``tab_phase_mem`` | Phase bit storage per row. |
| ``tab_meas_mem`` | Memory per recorded measurement. |
| ``tab_two_qubit_weight`` | Depth multiplier for entangling Cliffords. |
| ``tab_depth_weight`` | Depth-normalised slowdown for deep circuits. |
| ``tab_rotation_weight`` | Overhead from non-Pauli rotations entering the tableau. |

## Matrix product state

For an ``n``-qubit chain with site costs ``l_i r_i`` and bond dimensions
``χ_i`` the estimator uses

\[
T = \Bigl(c_{1q} N_{1q} \sum l_i r_i + c_{2q} N_{2q} n \overline{χ_i l_i r_i}
    + c_{trunc} N_{2q} n τ\Bigr)
    \Bigl(1 + \eta_{ent} H + \eta_{rot} \delta - \eta_{s} S\Bigr) + T_0,
\]
\[
M = M_0 + c_{mem} \sum l_i r_i \Bigl(1 + \eta_{ent} H + \eta_{rot} \delta
    - \eta_{s} S\Bigr),
\]

where ``H`` is the normalised entropy implied by the chosen ``χ_i`` or
derived from the circuit structure. Sparsity discounts and rotation
penalties are fitted against cuTensorNet and tensor-network HPC studies
[^scholl][^cutensornet].

| Coefficient | Meaning |
|-------------|---------|
| ``mps_gate_1q`` | Single-qubit tensor updates (4``χ^2`` multiplies) [^scholl]. |
| ``mps_gate_2q`` | Two-qubit tensor updates (16``χ^3`` operations) [^scholl]. |
| ``mps_trunc`` | Optional SVD truncation ~32``χ^3\log χ``. |
| ``mps_mem`` | Bytes per tensor element (``16`` for ``complex128``). |
| ``mps_temp_mem`` | Temporary workspace for SVD. |
| ``mps_entropy_weight`` | Runtime growth with increasing bond entropy. |
| ``mps_rotation_weight`` | Sensitivity to rotation diversity limiting canonicalisation. |
| ``mps_sparsity_discount`` | Benefit from aggressive truncation on sparse amplitudes. |

## Decision diagrams

Decision diagram (QMDD) simulation scales with the active node count
``r``. Recent GPU-accelerated benchmarks highlight sensitivity to
frontier width, sparsity and rotation diversity [^qmdd][^qmddbench]. The
estimator therefore models

\[
T = c_{dd\_gate} N r \Bigl(1 - \sigma_s S + \sigma_f \log_2 r
    + \sigma_{rot} \delta + \sigma_{ent} H + \sigma_{mix} \rho\Bigr) + T_0,
\]
\[
M = M_0 + c_{dd\_mem} r b_{node} (1 + c_{cache})
    \Bigl(1 - \sigma_s S + \sigma_f \log_2 r + \sigma_{rot} \delta
    + \sigma_{ent} H + \sigma_{mix} \rho\Bigr).
\]

| Coefficient | Meaning |
|-------------|---------|
| ``dd_gate`` | Node operations per gate [^qmdd]. |
| ``dd_mem`` | Memory multiplier for nodes and cache. |
| ``dd_node_bytes`` | Bytes per QMDD node (four edges + terminal index). |
| ``dd_cache_overhead`` | Fractional unique-table cache overhead. |
| ``dd_sparsity_discount`` | Savings from highly sparse amplitude vectors. |
| ``dd_frontier_weight`` | Growth with the logarithm of the frontier width. |
| ``dd_rotation_penalty`` | Penalty for diverse rotations causing node splits. |
| ``dd_entropy_penalty`` | Additional branching for entangled cuts. |
| ``dd_two_qubit_weight`` | Increased node churn from entangling layers. |

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

### Conversion primitive glossary

Planning reports and notebooks (for example ``partitioning_tradeoffs``)
surface a ``primitive`` column indicating which conversion strategy the
estimator selected for the boundary between fragments. The available
values correspond to the following behaviours:

| Primitive | Description |
|-----------|-------------|
| ``None`` | No conversion is required because both fragments run on the same backend. |
| ``B2B`` | Boundary-to-boundary extraction: performs an SVD across the cut, truncates according to the allowed rank, and copies the resulting tensors into the target backend. |
| ``LW`` | Local-window extraction: simulates a dense window of the boundary qubits (bounded by ``window_1q_gates`` / ``window_2q_gates`` if configured) before handing the reduced state to the target backend. |
| ``ST`` | Staged transfer: routes the state through an intermediate representation capped by ``st_chi_cap`` to limit memory before converting back to the destination backend. |
| ``Full`` | Full extraction: materialises the complete source state (or the maximum allowed by boundary/rank constraints) prior to ingestion by the destination backend. |

## Calibration sweeps

The ``calibration/cost_model_sweeps.py`` module bundles synthetic
profiles extracted from the referenced HPC studies. Running
``fit_all_coefficients`` produces a coefficient dictionary compatible with
``CostEstimator``:

```python
from calibration.cost_model_sweeps import fit_all_coefficients
from quasar.cost import CostEstimator

est = CostEstimator()
est.update_coefficients(fit_all_coefficients())
```

The accompanying notebook ``calibration/cost_model_calibration.ipynb``
demonstrates how sparsity, depth and entanglement sweeps alter backend
boundaries using the updated metric-aware estimator.

Coefficients can still be customised via configuration files or
environment variables. Existing estimators also expose
``est.update_coefficients({"sv_gate_1q": 0.9}, decay=0.2)`` for
fine-grained adjustments at runtime. The ``decay`` parameter applies an
exponential moving average and defaults to the value of
``QUASAR_COEFF_EMA_DECAY``.

## References

[^quest]: Jonathan A. Jones *et al.*, "QuEST and High Performance Simulation of Quantum Computers", 2019. [arXiv:1904.06343](https://arxiv.org/abs/1904.06343)
[^aer]: *Qiskit Aer performance guide*. [https://docs.quantum.ibm.com/api/qiskit/aer#performance](https://docs.quantum.ibm.com/api/qiskit/aer#performance)
[^ag04]: Scott Aaronson and Daniel Gottesman, "Improved Simulation of Stabilizer Circuits", 2004. [arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)
[^scholl]: Ulrich Schollwöck, "The density-matrix renormalization group in the age of matrix product states", 2011. [doi:10.1016/j.aop.2010.09.012](https://doi.org/10.1016/j.aop.2010.09.012)
[^qflex]: Francisco Villalonga *et al.*, "Faster than classical simulation of quantum supremacy circuits", 2019. [arXiv:1905.00444](https://arxiv.org/abs/1905.00444)
[^custatevec]: NVIDIA, *cuStateVec performance guide*, 2023. [https://docs.nvidia.com/cuda/custatevec](https://docs.nvidia.com/cuda/custatevec)
[^cutensornet]: NVIDIA, *cuTensorNet library performance*, 2024. [https://docs.nvidia.com/cuda/cutensornet](https://docs.nvidia.com/cuda/cutensornet)
[^qmdd]: Robert Wille and Jonas Zulehner, "Advanced Simulation of Quantum Computations", 2019. [arXiv:1801.00112](https://arxiv.org/abs/1801.00112)
[^qmddbench]: Thomas Grurl *et al.*, "Accelerating QMDD-based simulation using GPUs", 2022. [arXiv:2203.08281](https://arxiv.org/abs/2203.08281)
