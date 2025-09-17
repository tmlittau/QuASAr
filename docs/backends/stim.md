# Stim backend

QuASAr's Stim backend wraps [`stim.TableauSimulator`](https://github.com/quantumlib/Stim)
to evolve stabilizer states.  The scheduler may execute disjoint Clifford
subsystems on separate simulator instances before merging them back together for
subsequent gates that span their qubits.

## Subsystem merging

Independent stabilizer subsystems are combined via a **direct sum** of their
underlying tableaus.  The helper `quasar.backends.stim_backend.direct_sum`
appends the second tableau to the first using Stim's in-place `+=` operator,
producing a block-diagonal tableau without ever materialising a dense
statevector.  Scheduler merge routines use the same helper when consolidating
parallel stabilizer simulations, ensuring later cross-subsystem Clifford gates
operate on a single tableau representation.

Developers should favour the direct-sum helper when composing stabilizer
partitions.  Treating the merge as a tensor product would incorrectly entangle
identical subsystems and break the Clifford invariants that Stim relies on.
