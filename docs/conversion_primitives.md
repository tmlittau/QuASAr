# Conversion primitives

QuASAr evaluates four conversion primitives—boundary-to-boundary (B2B), local window (LW), staged transfer (ST), and full extraction—whenever the planner switches between simulation backends. These primitives share the same calibration coefficients as the core cost model and appear directly in planner traces and SSD metadata. This note consolidates the authoritative definitions, parameters, and diagnostics for each primitive, complementing the high-level overview in [docs/cost_model.md](cost_model.md) and the [partitioning theory reference](partitioning_theory.md).

## Parameter glossary

Conversion estimates depend on the boundary descriptors emitted by the planner:

- **Boundary size ``q``** – the number of qubits along the cut. Logged as ``boundary``/``boundary_size`` on :class:`~quasar.ssd.PartitionTraceEntry` and ``ConversionLayer.boundary`` on the SSD.【F:quasar/ssd.py†L18-L66】【F:quasar/ssd.py†L118-L160】
- **Rank ``s``** – the Schmidt-rank bound used for SVD-based primitives. Stored as ``rank`` in planner traces and SSD conversion layers. When no upstream estimator supplies a refined value the partitioner defaults to ``2**q`` before scoring primitives.【F:quasar/ssd.py†L30-L66】【F:quasar/ssd.py†L118-L160】【F:quasar/partitioner.py†L143-L201】
- **Frontier ``r``** – the decision-diagram frontier estimate recorded alongside ``rank`` to size DD-style conversions. The sequential backlog logic uses the boundary size as the default frontier when queuing pending conversions.【F:quasar/ssd.py†L30-L66】【F:quasar/ssd.py†L118-L160】【F:quasar/partitioner.py†L143-L201】
- **Window ``w``** – the optional dense window size for LW extractions. When not specified by the planner the estimator derives an entanglement-aware window (clamped to at least ``min(q, 4)``) before scoring the primitives.【F:quasar/cost.py†L812-L881】

The planner always threads ``q``, ``s`` and ``r`` into the cost estimator when emitting trace diagnostics. The LW window parameter remains implicit in the traces but can be reproduced analytically through the estimator helpers described below.

## Primitive breakdown

All primitives include a fixed ``conversion_base`` overhead and per-amplitude ingestion cost ``ingest_*`` for the destination backend.【F:quasar/cost.py†L660-L736】 The remaining terms differ per primitive:

- **B2B** – performs an SVD across the boundary (``b2b_svd`` term) and copies the resulting tensors (``b2b_copy``), allocating optional SVD workspace (``b2b_svd_mem``).【F:quasar/cost.py†L708-L722】 B2B scales polynomially with ``q`` and ``s`` and tends to win when ranks stay small.
- **LW** – simulates a dense window of size ``w`` using statevector kernels (``lw_extract`` plus any supplied window gate counts) before handing the reduced state to the target backend.【F:quasar/cost.py†L724-L737】 LW is sensitive to the chosen window and dominates when the cut can be isolated to four qubits or fewer.
- **ST** – routes the state through an intermediate representation capped by ``st_chi_cap`` with runtime controlled by ``st_stage`` and memory by ``chi_tilde^2``. It provides a middle ground between LW and full extraction when windows are large but Schmidt ranks remain within the staging cap.【F:quasar/cost.py†L739-L749】
- **Full extraction** – materialises the complete source state (``full_extract``) before ingestion, growing exponentially with ``q`` but using no intermediate compression beyond the destination backend itself.【F:quasar/cost.py†L751-L758】

`CostEstimator.conversion_candidates` evaluates all four primitives, returning their time/memory estimates and component breakdowns, while `CostEstimator.conversion` selects the primitive with minimal runtime.【F:quasar/cost.py†L700-L759】 This keeps planner diagnostics, SSD metadata, and documentation tables aligned with the calibrated coefficients.

## Example cost sweeps

The helper :func:`docs.utils.partitioning_analysis.build_conversion_primitive_examples` uses the calibrated estimator to tabulate per-primitive costs for small scenarios, matching the planner's ``debug=True`` traces.【F:docs/utils/partitioning_analysis.py†L188-L224】 Running the helper with ``calibration/coeff_v1.json`` yields the following illustrative tables.【7a22ef†L1-L20】

Scenario 1 – statevector→tableau with heuristic LW windows (derived ``w``):

| Boundary q | Rank s | Frontier r | Primitive | Time (a.u.) | Memory (a.u.) | Selected? |
|---|---|---|---|---|---|---|
| 2 | 4 | 2 | B2B | 113 | 32 | |
| 2 | 4 | 2 | LW | 25 | 4 | |
| 2 | 4 | 2 | ST | 209 | 32 | |
| 2 | 4 | 2 | Full | 21 | 4 | ✓|
| 4 | 16 | 4 | B2B | 2101 | 1024 | |
| 4 | 16 | 4 | LW | 85 | 16 | |
| 4 | 16 | 4 | ST | 12341 | 1024 | |
| 4 | 16 | 4 | Full | 69 | 16 | ✓|
| 6 | 64 | 6 | B2B | 33989 | 24576 | |
| 6 | 64 | 6 | LW | 229 | 64 | ✓|
| 6 | 64 | 6 | ST | 12485 | 1536 | |
| 6 | 64 | 6 | Full | 261 | 64 | |

Scenario 2 – statevector→MPS with a forced 14-qubit LW window (representing a dense, non-local boundary):

| Boundary q | Rank s | Frontier r | Primitive | Time (a.u.) | Memory (a.u.) | Selected? |
|---|---|---|---|---|---|---|
| 12 | 64 | 12 | B2B | 102405 | 49152 | |
| 12 | 64 | 12 | LW | 49157 | 16384 | |
| 12 | 64 | 12 | ST | 28677 | 4096 | |
| 12 | 64 | 12 | Full | 20485 | 4096 | ✓|
| 14 | 64 | 14 | B2B | 173061 | 57344 | |
| 14 | 64 | 14 | LW | 98309 | 16384 | |
| 14 | 64 | 14 | ST | 77829 | 16384 | ✓|
| 14 | 64 | 14 | Full | 81925 | 16384 | |

These sweeps highlight three regimes: full extraction dominates when ranks stay tiny; LW wins while the window remains narrow; and ST becomes competitive once LW is forced to operate on the full boundary. The planner surfaces the chosen primitive and its estimated `Cost(time, memory, log_depth)` inside `PartitionTraceEntry.cost`, enabling direct comparison with these tables.【F:quasar/ssd.py†L18-L38】

## Planner diagnostics and SSD metadata

Planner traces emitted with ``debug=True`` populate :class:`~quasar.ssd.PartitionTraceEntry` entries containing ``boundary_size``, ``rank``, ``frontier``, ``primitive`` and the associated :class:`~quasar.cost.Cost`. Conversion layers stored on the SSD mirror the same boundary, rank and frontier metadata.【F:quasar/partitioner.py†L132-L201】【F:quasar/ssd.py†L18-L160】 These fields line up with the parameters used in the estimator, ensuring documentation, traces and downstream analysis scripts share a common vocabulary.

## Reproducing the tables

To regenerate the examples or explore alternative calibrations, run the helper in a Python shell:

```python
from docs.utils.partitioning_analysis import (
    load_calibrated_estimator,
    build_conversion_primitive_examples,
)
from quasar.cost import Backend

est, _ = load_calibrated_estimator()
rows = build_conversion_primitive_examples(
    est,
    source=Backend.STATEVECTOR,
    target=Backend.TABLEAU,
    boundaries=(2, 4, 6),
)
```

Adjust ``boundaries``, ``rank_override`` or ``window`` to study alternative crossover points. The helper wraps :meth:`~quasar.cost.CostEstimator.conversion_candidates`, so the emitted numbers always track the planner's internal scoring.【F:docs/utils/partitioning_analysis.py†L188-L224】【F:quasar/cost.py†L700-L759】 When writing documentation or analysing traces, prefer these utilities to guarantee consistency with runtime diagnostics.
