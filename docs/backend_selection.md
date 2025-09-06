# Backend selection

QuASAr's planner compares simulation backends for each circuit fragment.  It
examines basic structural metrics to guide this choice:

1. **Clifford detection** – if all gates in a fragment are Clifford operations
   the specialised TABLEAU backend is used exclusively, bypassing other
   candidates.
2. **Heuristic metrics** – overall circuit [symmetry](symmetry.md) and
   [sparsity](sparsity.md) scores are combined into a weighted sum.  When this
   weighted score exceeds the ``dd_metric_threshold`` the planner includes the
   decision‑diagram backend as a candidate.  The weights
   ``dd_symmetry_weight`` and ``dd_sparsity_weight`` control each metric's
   contribution.
3. **Entanglement heuristic** – an upper bound on the maximal Schmidt rank is
   derived from the gate sequence.  If this estimate does not exceed the
   ``mps_chi_threshold`` (``64`` by default, configurable via the
   ``QUASAR_MPS_CHI_THRESHOLD`` environment variable) the MPS backend is
   considered.
4. **Fallback** – remaining candidates such as the dense STATEVECTOR simulator
   are considered based on estimated runtime and memory cost.

The weights and threshold of step 2 default to ``1.0`` for both symmetry and
sparsity with a ``dd_metric_threshold`` of ``0.8``.  They may be tuned via the
``QUASAR_DD_SYMMETRY_WEIGHT``, ``QUASAR_DD_SPARSITY_WEIGHT`` and
``QUASAR_DD_METRIC_THRESHOLD`` environment variables or by overriding
``config.DEFAULT`` at runtime.  The MPS threshold mentioned in step 3 defaults
to ``64`` and is configurable through ``QUASAR_MPS_CHI_THRESHOLD``.

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
