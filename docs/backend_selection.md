# Backend selection

QuASAr's planner compares simulation backends for each circuit fragment.  It
examines basic structural metrics to guide this choice:

1. **Clifford detection** – if all gates in a fragment are Clifford operations
   the specialised TABLEAU backend is used exclusively, bypassing other
   candidates.
2. **Heuristic metrics** – overall circuit [sparsity](sparsity.md), an
   estimate of the number of non‑zero amplitudes (*nnz*), and both *phase* and
   *amplitude* rotation diversity are consulted. Each metric is normalised by
   its corresponding threshold – ``adaptive_dd_sparsity_threshold(n_qubits)``,
   ``dd_nnz_threshold``, ``dd_phase_rotation_diversity_threshold`` and
   ``adaptive_dd_amplitude_rotation_threshold(n_qubits, sparsity)`` – and
   combined using weights ``dd_sparsity_weight``, ``dd_nnz_weight``,
   ``dd_phase_rotation_weight`` and ``dd_amplitude_rotation_weight``.  The
   combined score is

   ``(w_s*(s/s_thr) + w_n*(1 - nnz/nnz_thr) + w_p*(1 - p/p_thr) + w_a*(1 - a/a_thr)) / (w_s+w_n+w_p+w_a)``

   where ``s`` is the sparsity, ``nnz`` the non‑zero estimate, ``p`` the phase
   rotation diversity and ``a`` the amplitude rotation diversity. Normalising
   each term by its threshold keeps the score in the ``[0,1]`` range,
   maintaining comparability with dense backends.  The score must exceed
   ``dd_metric_threshold`` for the decision‑diagram backend to be considered.
   Circuits with a high diversity of rotations are suppressed regardless of
   score to avoid decision‑diagram blow‑ups.
3. **Entanglement heuristic** – an upper bound on the maximal Schmidt rank is
   derived from the gate sequence.  This estimate combines with the fidelity
   target ``mps_target_fidelity`` (default ``1.0`` and overrideable via
   ``QUASAR_MPS_TARGET_FIDELITY``) to determine the bond dimension ``χ`` for
   matrix‑product state simulation.  The resulting ``χ`` is further capped by
   the memory threshold supplied to the planner; if even ``χ = 1`` exceeds the
   available memory the MPS backend is skipped.
4. **Fallback** – remaining candidates such as the dense STATEVECTOR simulator
   are considered based on estimated runtime and memory cost.

The planner compares candidate costs using a configurable priority between
runtime and memory.  By default memory consumption takes precedence to avoid
backends with excessive requirements when a slower but leaner alternative
exists.  This behaviour can be overridden via the planner's ``perf_prio``
option, setting it to ``"time"`` to prioritise runtime instead.

The default weights for sparsity, nnz and the two rotation metrics are ``1.0``
each with a ``dd_metric_threshold`` of ``0.8``.  Phase rotation diversity is
limited to ``16`` distinct angles while the amplitude rotation threshold uses
the same base value but scales with circuit width and sparsity. These values –
along with ``dd_sparsity_threshold`` and ``dd_nnz_threshold`` – may be tuned via
the ``QUASAR_DD_*`` environment variables or by overriding ``config.DEFAULT`` at
runtime.  Lowering ``mps_target_fidelity`` reduces the required bond dimension
and can therefore make the MPS backend applicable to more circuits.

This lightweight heuristic steers the planner towards specialised backends for
circuits exhibiting repeated structure, large zero‑amplitude regions or limited
entanglement while favouring dense methods otherwise.

## Fidelity-driven χ estimation

When the selector entertains an MPS candidate it traces a deterministic pipeline
from fidelity requirements to a concrete bond dimension.  The steps mirror the
helpers in :mod:`quasar.cost` and are exposed by the documentation utilities in
``docs/utils/partitioning_analysis.py``:

1. :meth:`~quasar.cost.CostEstimator.bond_dimensions` walks the gate sequence
   and doubles the Schmidt rank of every cut crossed by a two-qubit gate.  This
   yields the maximal entanglement that the fragment could exhibit along the
   linearised qubit order.【F:quasar/cost.py†L268-L287】
2. :meth:`~quasar.cost.CostEstimator.chi_for_fidelity` scales the maximal rank
   by the requested fidelity.  The method raises the rank when the target
   exceeds the default ``mps_target_fidelity`` and lowers it when the caller
   accepts truncation error; circuit depth influences the scaling exponent so
   deeper fragments require larger ``χ`` even at identical fidelity targets.【F:quasar/cost.py†L312-L324】
3. :meth:`~quasar.cost.CostEstimator.chi_from_memory` compares the candidate
   rank with the available memory budget (``max_memory`` passed to
   :class:`Planner.plan`).  The heuristic models the quadratic storage cost of
   MPS tensors using the calibrated ``mps_mem`` coefficient and rejects
   configurations where the requested ``χ`` would overflow the cap.【F:quasar/cost.py†L326-L345】
4. :meth:`~quasar.cost.CostEstimator.chi_for_constraints` combines the previous
   stages.  When the fidelity-driven ``χ`` violates the memory ceiling the
   method returns ``0`` so :class:`~quasar.method_selector.MethodSelector`
   records the MPS backend as infeasible (“bond dimension exceeds memory
   limit”). Otherwise the selector evaluates the MPS cost model with the
   computed ``χ`` and still applies explicit time and memory checks against the
   calibrated estimates, ensuring base overheads such as ``mps_base_mem`` are
   respected.【F:quasar/cost.py†L347-L366】【F:quasar/method_selector.py†L260-L318】

``docs/utils/partitioning_analysis.load_calibrated_estimator`` exposes the same
helpers used by the planner so tutorials and notebooks can reproduce the
fidelity-to-``χ`` trace without instantiating full circuits.【F:docs/utils/partitioning_analysis.py†L24-L81】

### Worked example: six-qubit ladder fragment

Consider a six-qubit fragment containing two layers of nearest-neighbour CX
gates (ten entangling operations) interleaved with twelve single-qubit
rotations.  With the default calibration table the selector performs the
following calculations when the caller asks for ``97%`` fidelity and a
``75 000`` byte memory ceiling:

| Stage | Calculation | Result |
| --- | --- | --- |
| 1 | ``bond_dimensions`` over the gate list | ``[4, 4, 4, 4, 4]`` → ``χ_max = 4`` |
| 2 | ``chi_for_fidelity(num_qubits=6, fidelity=0.97)`` | ``χ_fid = 3`` |
| 3 | ``chi_from_memory(num_qubits=6, max_memory=75_000)`` | ``χ_mem = 111`` |
| 4 | ``chi_for_constraints(...)`` | ``χ = 3`` (feasible) |

Once ``χ`` is fixed, the cost model predicts runtime and memory as shown below
(``svd=True`` includes truncation overheads):

| χ | Runtime (model units) | Peak memory (bytes) |
| --- | --- | --- |
| 2 | 1.10 × 10³ | 5.60 × 10⁴ |
| 3 | 4.09 × 10³ | 5.61 × 10⁴ |
| 4 | 1.06 × 10⁴ | 5.61 × 10⁴ |

Lowering the memory ceiling to ``50 000`` bytes keeps ``χ`` at ``3`` but the
final memory check rejects the fragment because the calibrated base footprint
``mps_base_mem`` already consumes ``56 000`` bytes.  This interaction explains
why the planner may skip MPS even when ``chi_from_memory`` appears permissive.

### Configuration knobs

Two configuration layers control this behaviour:

* ``config.DEFAULT.mps_target_fidelity`` (environment variable
  ``QUASAR_MPS_TARGET_FIDELITY``) defaults to ``1.0`` and defines the fidelity
  used when the caller does not supply ``target_accuracy``. Tightening the
  target increases ``χ`` globally.【F:quasar/config.py†L63-L112】
* :meth:`Planner.plan` accepts ``max_memory`` and ``target_accuracy``.  The
  planner forwards these caps to ``chi_for_constraints`` to derive
  ``estimator.chi_max`` for the full circuit, ensuring subsequent fragment-level
  checks reuse the same limit.【F:quasar/planner.py†L1248-L1296】

Additional planner options such as ``max_time`` and ``perf_prio`` continue to
filter backends after the ``χ`` calculation, but they do not alter the fidelity
trace described above.

Example: planning a small quantum Fourier transform selects the MPS backend:

```python
from benchmarks.circuits import qft_circuit
from quasar import Backend, SimulationEngine

engine = SimulationEngine()
plan = engine.planner.plan(qft_circuit(5))
assert plan.final_backend == Backend.MPS
```

## Classical control simplification

The planner analyses the circuit as it is presented.  For accurate backend
selection, classical control simplification must run before planning so that
gates conditioned on known classical bits are removed or reduced.  Circuits
constructed with ``use_classical_simplification=False`` retain their original
structure and need explicit reinitialisation via
``circuit.enable_classical_simplification()`` (or reconstruction) to benefit.

Simplifying a quantum Fourier transform initialised in ``|0\rangle`` removes all
controlled phase rotations, leaving only Hadamards.  The planner then detects a
pure Clifford circuit and selects the ``STIM`` backend:

```python
from benchmarks.circuits import qft_circuit
from quasar import Backend, SimulationEngine

circ = qft_circuit(5)
plan = SimulationEngine().planner.plan(circ)
assert plan.final_backend == Backend.STIM
```

## Telemetry

Backend decisions made on the quick path can be recorded for later analysis.
Set ``QUASAR_VERBOSE_SELECTION=1`` to emit the evaluated metrics together with
per-backend cost estimates and rejection reasons; the planner prints the same
diagnostics for each segment it analyses.  For persistent logs set the environment variable
``QUASAR_BACKEND_SELECTION_LOG`` or override
``config.DEFAULT.backend_selection_log`` with a filesystem path.  Each quick
selection appends a CSV row with

``sparsity,nnz,phase_rot,amplitude_rot,locality,backend,score,ranking``

where ``locality`` is ``1`` when all multi‑qubit gates act on adjacent qubits
and ``ranking`` lists candidates in descending preference separated by ``>``.
Use the helper script to aggregate results across benchmark runs:

```bash
python tools/analyze_backend_selection.py logs/*.csv
```

When ``Planner.plan`` is invoked with ``explain=True`` the returned
``PlanDiagnostics`` object exposes a ``backend_selection`` mapping with the
same diagnostic payload.  Each entry captures the heuristics, estimated costs
and the reasons why candidates were accepted or rejected, making it easier to
trace planning decisions without relying solely on console output.  The
``reason`` field reflects the planner's backlog-aware workflow: repeated
``deferred_switch_candidate`` entries track a pending backend change, while
``deferred_backend_switch`` marks the point where the conversion becomes
cheaper than continuing with the current backend.  Additional reasons such as
``statevector_lock`` and ``single_qubit_preamble`` capture fast-path rejections
for dense fragments and single-qubit prefixes respectively.【F:quasar/partitioner.py†L283-L399】【F:quasar/partitioner.py†L495-L508】【F:quasar/partitioner.py†L536-L547】

Rank and frontier diagnostics in these traces come from the conversion helper
inside the partitioner.  Boundaries fall back to ``2**|boundary|`` when no
external rank model is supplied, and the estimator's chosen primitive plus its
cost are carried over to any emitted `ConversionLayer`, keeping the diagnostic
output aligned with the conversion glossary.【F:quasar/partitioner.py†L143-L201】【F:quasar/partitioner.py†L298-L308】 Refer to the
[Partitioning Theory reference](partitioning_theory.md#single-method-backlog-and-deferred-conversions)
for a narrative walkthrough of the backlog logic and how it interacts with the
sequential fragment tracker.
