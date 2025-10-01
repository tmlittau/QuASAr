# Cost estimator playground

The `docs/cost_estimator_playground.ipynb` notebook provides an interactive UI for
exploring how the :class:`~quasar.cost.CostEstimator` reacts to changes in
fragment metrics.  It combines a set of sliders with scenario presets that are
populated from the exported fragment metrics located in
`docs/data/conversion_scenarios.json`.

## Loading real-circuit presets

Use the **Scenario** dropdown at the top of the widget column to load metrics
captured from representative circuits:

| Scenario      | Backend            | q | s | Sparsity | Rotation diversity |
|---------------|-------------------|---|---|----------|--------------------|
| GHZ 8q        | TABLEAU            | 8 | 2 | 0.992    | 0.0                |
| QFT 6q        | TABLEAU            | 6 | 1 | 0.031    | 0.0                |
| W state 5q    | DECISION_DIAGRAM   | 5 | 4 | 0.844    | 4.0                |
| Grover 4q     | DECISION_DIAGRAM   | 4 | 4 | 0.063    | 2.0                |

Selecting a preset synchronises the sliders with the stored metrics so the
conversion and simulation estimates mirror the recorded fragment.  The **Custom**
option keeps the current slider values, allowing further experimentation around a
scenario.

## Conversion primitive table

The **Conversion primitive estimates** table evaluates
:meth:`~quasar.cost.CostEstimator.conversion_candidates` for the selected source
and target backends using the boundary size ``q``, Schmidt rank ``s`` and staging
cap ``χ_cap`` from the sliders.  Each row corresponds to one of the conversion
primitives (B2B, LW, ST, Full) and reports the runtime and memory estimates, the
resolved dense window (for LW) and the staging cap that would be applied by ST.
The table highlights when staging would require multiple phases (`Stages` > 1),
which is useful for checking whether lowering ``χ_cap`` causes repeated
extractions.

## Simulation cost table

The **Simulation cost estimates** table recomputes backend costs for the current
metrics.  When a preset is active the gate counts, rotation diversity split and
long-range interaction metrics are taken directly from the exported diagnostics.
For ad-hoc scenarios the notebook derives a conservative gate mix from ``q`` and
assumes an even split between phase and amplitude rotation diversity.  This
provides a quick way to gauge how runtime and memory scale across the
statevector, tableau, MPS and decision diagram backends for the same fragment.

## Worked example: W state fragment

Loading the *W state 5q* preset demonstrates how the metrics drive backend
selection.  The simulation table shows that the decision diagram backend remains
memory-efficient despite the moderately high Schmidt rank, while the conversion
summary reveals that staged extraction would require a two-pass ingestion when
converting to MPS unless ``χ_cap`` is increased.  Reducing the sparsity slider
immediately increases the decision diagram cost, highlighting how sparsity
thresholds influence the planner's backend ranking.
