# Large Circuit Benchmarks

This document outlines composite circuits designed to stress QuASAr's
partitioning engine. Each circuit blends independent subroutines that eventually
interact so the scheduler must juggle different simulation backends.

## Surface-Code QAOA Hybrid

`surface_code_qaoa_circuit(bit_width, distance, rounds)` interleaves low-degree
ring-graph QAOA layers with stabiliser cycles drawn from a distance-`distance`
surface code. QAOA segments favour MPS-style simulation on the problem qubits,
while the inserted Clifford stabiliser rounds operate on additional ancilla
registers well suited to tableau backends. The alternating structure forces
QuASAr to transition between representations as it steps through the layered
schedule.

## GHZ–Grover Fusion Circuit

`ghz_grover_fusion_circuit(ghz_qubits, grover_qubits, iterations)` prepares a
GHZ state and a Grover search routine on disjoint registers. Both prefixes are
independent, enabling the planner to schedule them concurrently on specialised
backends: stabiliser simulation for the GHZ branch and statevector simulation
for the Grover oracle. A single cross-register entangling gate at the end fuses
the partitions, demonstrating how QuASAr coordinates the union of incompatible
substructures.

## QAOA with Toffoli Gadget

`qaoa_toffoli_gadget_circuit(width, rounds_before, rounds_after)` surrounds a
central Toffoli gate with layers of ring-graph QAOA. The shallow QAOA segments
admit efficient tensor-network simulation, but the non-Clifford three-qubit
gadget temporarily breaks the low-degree pattern and prompts a backend switch.
This circuit spotlights QuASAr's ability to introduce conversion layers around
isolated nonlocal operations while resuming the original representation
afterwards.
