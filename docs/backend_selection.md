# Backend selection

QuASAr's planner compares simulation backends for each circuit fragment.  It
examines basic structural metrics to guide this choice:

1. **Clifford detection** – if all gates in a fragment are Clifford operations
   the specialised TABLEAU backend is used exclusively, bypassing other
   candidates.
2. **Heuristic metrics** – overall circuit [symmetry](symmetry.md),
   [sparsity](sparsity.md) and *rotation diversity* are consulted.  Symmetry and
   sparsity form a weighted sum; when this exceeds ``dd_metric_threshold`` and
   the number of distinct phase rotations stays below
   ``dd_rotation_diversity_threshold`` the planner includes the decision‑diagram
   backend as a candidate.  The weights ``dd_symmetry_weight`` and
   ``dd_sparsity_weight`` control each metric's contribution.  Decision diagrams
   benefit from repeated phase angles; circuits with highly diverse rotations
   tend to blow up the decision structure, hence the diversity guard.
3. **Entanglement heuristic** – an upper bound on the maximal Schmidt rank is
   derived from the gate sequence.  This estimate combines with the fidelity
   target ``mps_target_fidelity`` (default ``1.0`` and overrideable via
   ``QUASAR_MPS_TARGET_FIDELITY``) to determine the bond dimension ``χ`` for
   matrix‑product state simulation.  The resulting ``χ`` is further capped by
   the memory threshold supplied to the planner; if even ``χ = 1`` exceeds the
   available memory the MPS backend is skipped.
4. **Fallback** – remaining candidates such as the dense STATEVECTOR simulator
   are considered based on estimated runtime and memory cost.

The weights and thresholds of step 2 default to ``1.0`` for both symmetry and
sparsity with a ``dd_metric_threshold`` of ``0.8`` and a
``dd_rotation_diversity_threshold`` of ``16`` distinct phase angles.  They may
be tuned via the ``QUASAR_DD_SYMMETRY_WEIGHT``, ``QUASAR_DD_SPARSITY_WEIGHT``,
``QUASAR_DD_METRIC_THRESHOLD`` and
``QUASAR_DD_ROTATION_DIVERSITY_THRESHOLD`` environment variables or by
overriding ``config.DEFAULT`` at runtime.  Lowering ``mps_target_fidelity``
reduces the required bond dimension and can therefore make the MPS backend
applicable to more circuits.

This lightweight heuristic steers the planner towards specialised backends for
circuits exhibiting repeated structure, large zero‑amplitude regions or limited
entanglement while favouring dense methods otherwise.

Example: planning a small quantum Fourier transform selects the MPS backend:

```python
from benchmarks.circuits import qft_circuit
from quasar import Backend, SimulationEngine

engine = SimulationEngine()
plan = engine.planner.plan(qft_circuit(5))
assert plan.final_backend == Backend.MPS
```
