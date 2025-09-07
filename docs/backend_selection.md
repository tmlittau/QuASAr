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
   ``dd_amplitude_rotation_diversity_threshold`` – and combined using weights
   ``dd_sparsity_weight``, ``dd_nnz_weight``, ``dd_phase_rotation_weight`` and
   ``dd_amplitude_rotation_weight``.  The combined score is

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
each with a ``dd_metric_threshold`` of ``0.8`` and rotation‑diversity thresholds
of ``16`` distinct angles for both phase and amplitude rotations. These values
– along with ``dd_sparsity_threshold`` and ``dd_nnz_threshold`` – may be tuned
via the ``QUASAR_DD_*`` environment variables or by overriding
``config.DEFAULT`` at runtime.  Lowering ``mps_target_fidelity`` reduces the
required bond dimension
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

circ = qft_circuit(5, use_classical_simplification=True)
circ.simplify_classical_controls()  # must run before planning
plan = SimulationEngine().planner.plan(circ)
assert plan.final_backend == Backend.STIM
```

## Telemetry

Backend decisions made on the quick path can be recorded for later analysis.
Set ``QUASAR_VERBOSE_SELECTION=1`` to emit the evaluated metrics and candidate
ranking to standard output; the planner prints similar information for each
segment it analyses.  For persistent logs set the environment variable
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
