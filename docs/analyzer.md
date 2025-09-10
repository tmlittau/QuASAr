# Circuit Analyzer

QuASAr's `CircuitAnalyzer` inspects quantum circuits and derives statistics
used by the planner and scheduler. The `AnalysisResult` returned by
``CircuitAnalyzer.analyze()`` now includes two pieces of per-gate metadata:

* **`gate_entanglement`** – how each gate affects entanglement. Entries are
  `"none"`, `"creates"`, or `"modifies"` depending on whether a gate acts on a
  single qubit, introduces new entanglement between qubits, or operates on an
  already entangled set.
* **`method_compatibility`** – a list of simulation methods that can execute
  each gate. Method names correspond to the lower‑case backend identifiers
  (``"statevector"``, ``"tableau"``, ``"mps"``, ``"decision_diagram"``).

These annotations are aligned with the circuit's topological order and are also
attached to the gate objects themselves for downstream consumers.
