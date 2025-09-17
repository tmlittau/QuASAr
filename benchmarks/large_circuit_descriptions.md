# Large Circuit Benchmarks

This document outlines composite circuits designed to stress QuASAr's
partitioning engine. Each circuit blends independent subroutines that eventually
interact so the scheduler must juggle different simulation backends.

## Mixed Backend Subsystems

`mixed_backend_subsystems(ghz_width, qaoa_width, qaoa_layers, random_width, seed)`
assembles three contiguous regions that naturally prefer different simulators.
A Clifford-only GHZ prefix is followed by a low-entanglement QAOA block and a
dense non-local suffix seeded with a Toffoli gadget. Cross-register connectors
force QuASAr to introduce conversion layers between tableau, MPS, and
statevector (or decision-diagram) backends as the execution traverses the
hybrid structure.

## Stim-to-DD Conversion Circuit

`stim_to_dd_circuit(num_groups, group_size, entangling_layer)` prepares several
independent GHZ subsystems using only Clifford gates, prompting the planner to
group them into tableau-friendly partitions. A global layer of non-Clifford
``T`` rotations then breaks the stabiliser structure so every subsystem must
convert to a decision-diagram backend. Optional post-conversion entanglers
couple neighbouring groups without disturbing the staged conversion behaviour.
