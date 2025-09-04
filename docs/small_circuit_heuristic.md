# Small circuit heuristic

QuASAr's planner and scheduler expose a `force_single_backend_below` option
that disables multi-backend partitioning for small circuits.  When set to an
integer *N*, any circuit whose number of qubits **or** depth does not exceed *N*
executes entirely on a single backend.  No conversion layers are inserted in
this mode, even if the circuit is slightly larger than the quick-path limits.

The setting is available through the `Planner` and `Scheduler` constructors and
can also be configured via the `QUASAR_FORCE_SINGLE_BACKEND_BELOW` environment
variable.
