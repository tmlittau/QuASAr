# Backend selection

QuASAr's planner compares simulation backends for each circuit fragment.  It
examines basic structural metrics to guide this choice:

1. **Clifford detection** – if all gates in a fragment are Clifford operations
   the specialised TABLEAU backend is used exclusively, bypassing other
   candidates.
2. **Heuristic metrics** – overall circuit [sparsity](sparsity.md), an
   estimate of the number of non‑zero amplitudes (*nnz*), and *rotation
   diversity* are consulted.  Each metric is normalised by its corresponding
   threshold – ``adaptive_dd_sparsity_threshold(n_qubits)``,
   ``dd_nnz_threshold`` and ``dd_rotation_diversity_threshold`` – and combined
   using weights ``dd_sparsity_weight``, ``dd_nnz_weight`` and
   ``dd_rotation_weight``.  The combined score is

   ``(w_s*(s/s_thr) + w_n*(1 - nnz/nnz_thr) + w_r*(1 - rot/rot_thr)) / (w_s+w_n+w_r)``

   where ``s`` is the sparsity, ``nnz`` the non‑zero estimate and ``rot`` the
   rotation diversity.  Normalising each term by its threshold keeps the score
   in the ``[0,1]`` range, maintaining comparability with dense backends.  The
   score must exceed ``dd_metric_threshold`` for the decision‑diagram backend to
   be considered.  Circuits with a high diversity of rotations are suppressed
   regardless of score to avoid decision‑diagram blow‑ups.
3. **Entanglement heuristic** – an upper bound on the maximal Schmidt rank is
   derived from the gate sequence.  This estimate combines with the fidelity
   target ``mps_target_fidelity`` (default ``1.0`` and overrideable via
   ``QUASAR_MPS_TARGET_FIDELITY``) to determine the bond dimension ``χ`` for
   matrix‑product state simulation.  The resulting ``χ`` is further capped by
   the memory threshold supplied to the planner; if even ``χ = 1`` exceeds the
   available memory the MPS backend is skipped.
4. **Fallback** – remaining candidates such as the dense STATEVECTOR simulator
   are considered based on estimated runtime and memory cost.

The default weights for sparsity, nnz and rotation are ``1.0`` each with a
``dd_metric_threshold`` of ``0.8`` and a ``dd_rotation_diversity_threshold`` of
``16`` distinct phase angles.  These values – along with
``dd_sparsity_threshold`` and ``dd_nnz_threshold`` – may be tuned via the
``QUASAR_DD_*`` environment variables or by overriding ``config.DEFAULT`` at
runtime.  Lowering ``mps_target_fidelity`` reduces the required bond dimension
and can therefore make the MPS backend applicable to more circuits.

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

## Telemetry

Backend decisions made on the quick path can be recorded for later analysis.
Set ``QUASAR_VERBOSE_SELECTION=1`` to emit the evaluated metrics and candidate
ranking to standard output; the planner prints similar information for each
segment it analyses.  For persistent logs set the environment variable
``QUASAR_BACKEND_SELECTION_LOG`` or override
``config.DEFAULT.backend_selection_log`` with a filesystem path.  Each quick
selection appends a CSV row with

``sparsity,nnz,rotation,locality,backend,score,ranking``

where ``locality`` is ``1`` when all multi‑qubit gates act on adjacent qubits
and ``ranking`` lists candidates in descending preference separated by ``>``.
Use the helper script to aggregate results across benchmark runs:

```bash
python tools/analyze_backend_selection.py logs/*.csv
```
