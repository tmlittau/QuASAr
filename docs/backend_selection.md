# Backend selection

QuASAr's planner compares simulation backends for each circuit fragment.  It
examines basic structural metrics to guide this choice:

1. **Clifford detection** – if all gates in a fragment are Clifford operations
   the specialised TABLEAU backend is preferred.
2. **Heuristic metrics** – overall circuit [symmetry](symmetry.md) and
   [sparsity](sparsity.md) scores are evaluated against configurable thresholds.
   When either score exceeds its threshold the planner includes the
   decision‑diagram backend as a candidate.
3. **Fallback** – remaining candidates such as the dense STATEVECTOR simulator
   are considered based on estimated runtime and memory cost.

The thresholds controlling step 2 default to ``0.3`` for symmetry and ``0.8``
for sparsity.  They can be tuned via the
``QUASAR_DD_SYMMETRY_THRESHOLD`` and ``QUASAR_DD_SPARSITY_THRESHOLD``
environment variables or by overriding ``config.DEFAULT`` at runtime.

This lightweight heuristic steers the planner towards the decision‑diagram
backend for circuits exhibiting repeated structure or large zero‑amplitude
regions, while favouring dense methods otherwise.
