# Small circuit heuristic

QuASAr performs a lightweight planning pre-pass that compares the
estimated cost of executing a circuit on a single backend against a
coarsely partitioned multi-backend plan.  If the single-backend
estimate, including a small planning overhead, is cheaper the planner
skips dynamic partitioning and executes the circuit as a single step.

This replaces the previous `force_single_backend_below` option and the
`QUASAR_FORCE_SINGLE_BACKEND_BELOW` environment variable.
